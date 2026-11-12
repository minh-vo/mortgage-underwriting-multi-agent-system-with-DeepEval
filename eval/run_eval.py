"""Evaluate the system with DeepEval: run each case end to end, then score the run.

Run with: uv run python eval/run_eval.py [--case CASE-001]
This calls Claude for the runs and for the judge, so it costs money.
"""

import argparse
import asyncio
import json
import os
import uuid
from datetime import date

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import (ArgumentCorrectnessMetric, PlanAdherenceMetric,  # noqa: E402
                              PlanQualityMetric, StepEfficiencyMetric, TaskCompletionMetric,
                              ToolCorrectnessMetric)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams  # noqa: E402

from underwriter import a2a_client, graph, llm, metrics  # noqa: E402
from underwriter.config import (AREAS, EVAL_RESULTS, get_settings, load_cases,  # noqa: E402
                                usage_cost)
from underwriter.mcp_client import tool_session  # noqa: E402
from underwriter.sanitize import sanitize  # noqa: E402

TASK = ("Underwrite mortgage application {case_id}: analyse credit, income, assets and "
        "collateral against the underwriting policy using the calculator and policy tools, "
        "have the analyses quality-reviewed, and reach a decision of APPROVED, "
        "CONDITIONAL_APPROVAL or DENIED with a decision memo.")
JUDGE_SYSTEM = "You are an impartial evaluator of an AI agent system. Follow the instructions exactly."
POLICY_TOOLS = ("get_policy", "search_policy")


class ClaudeJudge(DeepEvalBaseLLM):
    """The DeepEval judge, on the Anthropic SDK. It uses JUDGE_MODEL, not the agents' model."""

    def __init__(self):
        self.usage: list[dict] = []
        super().__init__(get_settings().judge_model)

    def load_model(self):
        return None  # the client is opened per run by llm.use_client()

    def get_model_name(self) -> str:
        return self.name

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            answer, usage = await llm.structured(JUDGE_SYSTEM, prompt, schema, agent="judge",
                                                 model=self.name)
        else:
            answer, usage = await llm.text(JUDGE_SYSTEM, prompt, agent="judge", model=self.name)
        self.usage += usage
        return answer

    def generate(self, prompt: str, schema=None):
        async def call():
            async with llm.use_client():
                return await self.a_generate(prompt, schema)
        return asyncio.run(call())


# ---- What DeepEval is given: built from the real run, nothing else ----------------

def build_trace(case_id: str, events: list[dict]) -> dict:
    """The run as nested spans, in the shape DeepEval's own tracer produces."""
    children = []
    for e in events:
        kind = e["type"]
        if kind == "supervisor":
            children.append({"name": "supervisor", "type": "agent",
                             "input": {"allowed_moves": e["allowed"]},
                             "output": {"next_step": e["action"], "areas": e["areas"],
                                        "reason": e["reason"], "is_rework": e["rework"],
                                        "areas_run_in_parallel": len(e["areas"]) > 1},
                             "children": []})
        elif kind == "specialist":
            a = e["analysis"]
            children.append({
                "name": f"{e['area']}_analyst", "type": "agent",
                "input": {"area": e["area"], "rework_feedback": e["feedback"] or None},
                "output": ({"status": "failed", "error": a["error"]} if a["status"] == "failed" else
                           {"status": "complete", "risk_level": a["risk_level"], "summary": a["summary"],
                            "findings": a["findings"], "policy_refs": a["policy_refs"]}),
                "children": [{"name": c["tool"], "type": "tool", "input": c["input"],
                              "output": c["output"], "children": []} for c in e["tools"]]})
        elif kind == "critic":
            children.append({"name": "critic", "type": "agent",
                             "output": {"verdict": e["verdict"], "defects": e["defects"],
                                        "rework_targets": e["rework_targets"], "summary": e["summary"]},
                             "children": []})
        elif kind == "decision":
            children.append({"name": "decision_agent", "type": "agent",
                             "output": {"decision": e["decision"], "risk_score": e["risk_score"],
                                        "conditions": e["conditions"], "memo": e["memo"],
                                        "human_review_reasons": e["reasons"]},
                             "children": []})
        elif kind == "final" and e["human"]:
            children.append({"name": "human_review", "type": "base",
                             "output": {"decision": e["final_decision"]}, "children": []})
    return {"name": "underwriting_workflow", "type": "agent",
            "input": TASK.format(case_id=case_id), "children": children}


def expected_tools(application: dict) -> list[ToolCall]:
    """The seven calculator calls with the arguments metrics.py says they should receive."""
    return [ToolCall(name=name, input_parameters=args)
            for name, args in metrics.tool_inputs(application).items()]


def checks(expected: dict, state: dict, events: list[dict], paused: bool) -> dict[str, bool]:
    """Deterministic checks: no judge involved."""
    rework_ok, last_critic = True, None
    for e in events:
        if e["type"] == "critic":
            last_critic = e
        elif e["type"] == "supervisor" and e["rework"]:
            rework_ok &= bool(last_critic) and any(d["severity"] == "major"
                                                   for d in last_critic["defects"])
    policy_areas = {c["area"] for c in state.get("tool_trace", []) if c["tool"] in POLICY_TOOLS}
    return {
        "decision in acceptable set": state["decision"]["decision"] in expected["acceptable_outcomes"],
        "human review as expected": paused == expected["human_review_expected"],
        "no bias flags": not state.get("bias_flags"),
        "no rework without a major defect": rework_ok,
        "every analyst looked up the policy": policy_areas == set(AREAS),
    }


# ---- Running ---------------------------------------------------------------------

async def score(case_id: str, application: dict, state: dict, events: list[dict],
                judge: ClaudeJudge) -> dict:
    """Measure the six DeepEval metrics on one finished run."""
    async with tool_session() as tools:
        available = [ToolCall(name=t["name"], description=t["description"])
                     for t in await tools.anthropic_tools()]
    called = [ToolCall(name=c["tool"], input_parameters=c["input"], output=c["output"])
              for c in state.get("tool_trace", [])]
    task = TASK.format(case_id=case_id)
    test_case = LLMTestCase(input=task, actual_output=state["decision"]["memo"],
                            tools_called=called, expected_tools=expected_tools(application))
    test_case._trace_dict = build_trace(case_id, events)

    suite = [
        TaskCompletionMetric(model=judge),
        StepEfficiencyMetric(model=judge),
        ToolCorrectnessMetric(model=judge, available_tools=available,
                              evaluation_params=[ToolCallParams.INPUT_PARAMETERS]),
        ArgumentCorrectnessMetric(model=judge),
        PlanQualityMetric(model=judge),
        PlanAdherenceMetric(model=judge),
    ]
    results = {}
    async with llm.use_client():
        for metric in suite:
            try:
                await metric.a_measure(test_case, _show_indicator=False)
                results[metric.__name__] = {"score": round(float(metric.score), 3),
                                            "passed": bool(metric.is_successful()),
                                            "reason": str(metric.reason or "")}
            except Exception as error:  # reported, never approximated
                results[metric.__name__] = {"score": None, "passed": False,
                                            "reason": f"Not measured: {type(error).__name__}: {error}"}
    return results


async def evaluate_case(case_id: str, case: dict, judge: ClaudeJudge) -> dict:
    thread_id = f"eval-{case_id}-{uuid.uuid4().hex[:8]}"
    print(f"\n{case_id}: running ...", flush=True)
    result = await graph.run_case(case_id, thread_id)
    events, paused = result["events"], result["status"] == "paused"
    if paused:  # concur with the AI so the evaluation needs no human
        ai = result["state"]["decision"]
        result = await graph.resume_case(thread_id, {
            "decision": ai["decision"], "conditions": ai["conditions"],
            "notes": "Concurred automatically by the evaluation run."})
        events = events + result["events"]
    if result["status"] != "done":
        message = next((e["message"] for e in reversed(events) if e["type"] == "error"), "unknown")
        raise RuntimeError(f"{case_id} did not finish: {message}")

    state = result["state"]
    application = sanitize(case["application"])
    judged_before = len(judge.usage)
    print(f"{case_id}: scoring ...", flush=True)
    scores = await score(case_id, application, state, events, judge)
    return {"decision": state["decision"]["decision"], "risk_score": state["decision"]["risk_score"],
            "human_review": paused, "reworks": state.get("rework_count", 0),
            "run_cost_usd": usage_cost(state.get("usage", [])),
            "judge_cost_usd": usage_cost(judge.usage[judged_before:]),
            "metrics": scores, "checks": checks(case["expected"], state, events, paused)}


def report(results: dict) -> bool:
    all_ok = True
    for case_id, r in results.items():
        cost = "n/a" if r["run_cost_usd"] is None else f"${r['run_cost_usd']:.2f}"
        judge_cost = "n/a" if r["judge_cost_usd"] is None else f"${r['judge_cost_usd']:.2f}"
        print(f"\n{case_id}: {r['decision']} (risk score {r['risk_score']}) · "
              f"human review: {'yes' if r['human_review'] else 'no'} · reworks: {r['reworks']} · "
              f"run {cost} · judge {judge_cost}")
        for name, m in r["metrics"].items():
            shown = "  n/a" if m["score"] is None else f"{m['score']:.2f}"
            print(f"  {'PASS' if m['passed'] else 'FAIL'}  {shown}  {name}")
            all_ok &= m["passed"]
        for name, ok in r["checks"].items():
            print(f"  {'PASS' if ok else 'FAIL'}        {name}")
            all_ok &= ok
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def save(results: dict, judge_model: str) -> None:
    """Write eval/results.json. A --case run replaces only that case's entry."""
    saved = json.loads(EVAL_RESULTS.read_text()) if EVAL_RESULTS.exists() else {}
    cases = {**saved.get("cases", {}), **results}
    EVAL_RESULTS.write_text(json.dumps({
        "run_date": date.today().isoformat(), "judge_model": judge_model,
        "model": get_settings().model, "cases": dict(sorted(cases.items()))}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run one case, for example CASE-001")
    args = parser.parse_args()
    cases = load_cases()
    if args.case and args.case not in cases:
        parser.error(f"unknown case {args.case}; choose from {', '.join(cases)}")
    if not get_settings().api_key:
        parser.error("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")

    a2a_client.ensure_critic_running()
    judge = ClaudeJudge()
    results = {}
    for case_id in ([args.case] if args.case else cases):
        results[case_id] = asyncio.run(evaluate_case(case_id, cases[case_id], judge))
        save({case_id: results[case_id]}, judge.name)  # keep finished cases if a later one fails
    ok = report(results)
    print(f"Saved to {EVAL_RESULTS}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
