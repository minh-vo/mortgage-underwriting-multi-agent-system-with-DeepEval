"""Streamlit UI: run a case, watch the agents work, review the decision.

Run with: uv run streamlit run app.py
"""

import asyncio
import json
import uuid

import streamlit as st

from underwriter import a2a_client, graph, metrics
from underwriter.agents import decision
from underwriter.config import AREAS, EVAL_RESULTS, PRICES, ROOT, get_settings, load_cases, usage_cost
from underwriter.sanitize import pii_table, sanitize

NODES = {"measure": "Sanitize and measure", "supervisor": "Supervisor", "credit": "Credit analyst",
         "income": "Income analyst", "assets": "Assets analyst", "collateral": "Collateral analyst",
         "critic": "Critic service", "decision": "Decision Agent", "human": "Human review",
         "final": "Final decision"}
EDGES = """
  measure -> supervisor
  subgraph cluster_specialists { label="Specialists, in parallel" color="#888888" fontcolor="#888888"
                                 fontname="Helvetica" fontsize=10 credit income assets collateral }
  supervisor -> {credit income assets collateral} [dir=both]
  {credit income assets collateral} -> tools [style=dashed label="MCP"]
  supervisor -> critic [dir=both label="A2A"]
  supervisor -> decision
  decision -> human [label="review trigger"]
  decision -> final
  human -> final
"""
KEY_HELP = f"""

**What to do**
1. **Get a key.** Sign in at https://console.anthropic.com, open **API Keys** and create one.
   A Claude subscription (Pro or Max) is not an API key.
2. **Open the file `{ROOT / ".env"}`.** If it does not exist, copy `.env.example`
   (same folder) and name the copy `.env`.
3. **Put the key on this line** and save, with no quotes or spaces: `ANTHROPIC_API_KEY=your-key-here`
4. **Restart the app.** In the terminal where it runs, press Ctrl+C, then run
   `uv run streamlit run app.py` again. Refreshing the browser is not enough.
"""

st.set_page_config(page_title="Mortgage underwriting agents", layout="wide")
settings, cases = get_settings(), load_cases()
runs = st.session_state.setdefault("runs", {})  # case_id -> {thread_id, events, status, state}


@st.cache_resource
def start_critic():
    return a2a_client.ensure_critic_running()


def show_error(message: str) -> None:
    st.error(message)
    if "authentication" in message.lower():
        st.warning("⚠ **Please replace your Claude API key before running the program.** Claude "
                   "rejected the key in your `.env` file: it is mistyped, expired or revoked." + KEY_HELP)


# ---- Drawing -----------------------------------------------------------------

def key_figures(m: dict, score: int) -> None:
    dti_status = {"not_needed": "within limit", "allowed": "exception allowed",
                  "not_allowed": "exception not allowed", "over_max": "over the maximum"}
    tiles = [
        ("Back-end DTI", f"{m['back_end_dti']}%", f"limit 43% (2.5) · {dti_status[m['dti_exception']]}"),
        ("LTV", f"{m['ltv']}%", "max 95% (4.3) · " + ("over the maximum" if m["ltv_over_max"] else
                                                      "MI required" if m["mi_required"] else "no MI")),
        ("Reserves", f"{m['reserve_months']} months",
         f"need {m['required_reserve_months']} (3.1) · "
         + (f"shortfall ${m['funds_shortfall']:,.0f}" if m["funds_shortfall"] else
            "meets requirement" if m["reserve_months"] >= m["required_reserve_months"]
            else "below requirement")),
        ("Credit score", str(score), f"min 620 (1.1) · {m['credit_tier']} tier"),
    ]
    for column, (label, value, note) in zip(st.columns(4), tiles):
        column.metric(label, value)
        column.caption(note)


def flow(events: list[dict]) -> None:
    """Draw the architecture, marking the steps that are working now and the ones that finished."""
    active, done, waiting = {"measure"}, set(), set()
    for e in events:
        kind = e["type"]
        if kind == "sanitized":
            done.add("measure")
            active = {"supervisor"}
        elif kind == "supervisor":  # its pick is what runs next
            waiting = set(e["areas"]) if e["action"] == "run_specialists" else set()
            active = set(waiting) or {e["action"]}
            done -= active
        elif kind == "specialist":
            waiting.discard(e["area"])
            if e["analysis"]["status"] == "complete":
                done.add(e["area"])
            active = set(waiting) or {"supervisor"}
        elif kind == "critic":
            done.add("critic")
            active = {"supervisor"}
        elif kind == "decision":
            done |= {"supervisor", "decision"}
            active = set()
        elif kind == "paused":
            active = {"human"}
        elif kind == "final":
            done |= active | {"final"}
            active = set()
        elif kind == "error":
            active = set()
    lines = []
    for node, label in NODES.items():
        if node in active:
            lines.append(f'{node} [label="▶ {label}" fillcolor="#ffd54f" penwidth=3]')
        elif node in done:
            lines.append(f'{node} [label="{label} ✓" fillcolor="#c8e6c9"]')
        else:
            lines.append(f'{node} [label="{label}" fillcolor="#ffffff"]')
    st.graphviz_chart(
        'digraph { rankdir=LR bgcolor="transparent" '
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 fontcolor="#000000"] '
        'edge [color="#888888" fontcolor="#888888" fontname="Helvetica" fontsize=9] '
        'tools [label="Tool server\n7 calculators + policy lookup" fillcolor="#eeeeee" shape=cylinder] '
        + " ".join(lines) + EDGES + "}")
    st.caption("▶ working now · ✓ finished")


def timeline(events: list[dict]) -> None:
    for e in events:
        kind = e["type"]
        if kind == "sanitized":
            st.expander(f"✓ {e['at']} · Application sanitized, figures computed")
        elif kind == "supervisor":
            what = e["action"] + (f": {', '.join(e['areas'])}" if e["areas"] else "")
            with st.expander(f"✓ {e['at']} · Supervisor → {'rework ' if e['rework'] else ''}{what}"):
                st.write(e["reason"])
                st.caption("Allowed moves: " + "; ".join(e["allowed"]))
        elif kind == "specialist":
            a, failed = e["analysis"], e["analysis"]["status"] == "failed"
            head = f"FAILED (attempt {a['failed_attempts']})" if failed else f"{a['risk_level']} risk"
            with st.expander(f"{'✗' if failed else '✓'} {e['at']} · {e['area'].title()} analyst · {head}"):
                if e["feedback"]:
                    st.markdown("**Rework. Feedback it was given:**")
                    st.text(e["feedback"])
                if failed:
                    show_error(a["error"])
                for call in e["tools"]:
                    st.markdown(f"`{call['tool']}({json.dumps(call['input'])})`")
                    st.caption(call["output"][:300])
                for flag in e["bias_flags"]:
                    st.warning(f"Fair-lending flag: {flag}")
        elif kind == "critic":
            passed = e["verdict"] == "pass"
            with st.expander(f"{'✓' if passed else '✗'} {e['at']} · Critic: {e['verdict'].upper()} · "
                             f"{len(e['defects'])} defect(s)"):
                critic_review(e)
        elif kind == "decision":
            st.expander(f"✓ {e['at']} · Decision: {e['decision']} · risk score {e['risk_score']}")
        elif kind == "paused":
            st.expander(f"⏸ {e['at']} · Paused for human review")
        elif kind == "final":
            st.expander(f"✓ {e['at']} · Final decision: {e['final_decision']}")
        elif kind == "error":
            show_error(f"{e['at']} · The run stopped: {e['message']}")


def critic_review(review: dict) -> None:
    st.write(review["summary"])
    st.markdown("**Defects (problems with the analyses)**")
    for d in review["defects"]:
        st.markdown(f"- {d['area']} · {d['severity']} · {d['statement']} *Should be:* {d['should_be']}")
    if not review["defects"]:
        st.caption("None.")
    st.markdown("**Borrower risks (not defects)**")
    for risk in review["application_risks"] + review["policy_violations"]:
        st.markdown(f"- {risk}")


def usage_table(usage: list[dict]) -> None:
    rows = {}
    for u in usage:
        row = rows.setdefault(u["agent"], {"agent": u["agent"], "calls": 0, "input tokens": 0,
                                           "output tokens": 0, "cost (USD)": 0.0})
        row["calls"] += 1
        row["input tokens"] += u["input_tokens"]
        row["output tokens"] += u["output_tokens"]
        price = PRICES.get(u["model"])
        row["cost (USD)"] = (None if price is None or row["cost (USD)"] is None else round(
            row["cost (USD)"] + u["input_tokens"] / 1e6 * price[0] + u["output_tokens"] / 1e6 * price[1], 4))
    st.dataframe(list(rows.values()), hide_index=True)
    total = usage_cost(usage)
    st.caption("Total: " + (f"${total:.2f}" if total is not None else "tokens only (no price for this model)"))


def evaluation_tab(case_id: str) -> None:
    results = json.loads(EVAL_RESULTS.read_text()) if EVAL_RESULTS.exists() else {}
    case = results.get("cases", {}).get(case_id)
    if not case:
        st.info("No evaluation results for this case yet. Produce them with "
                f"`uv run python eval/run_eval.py --case {case_id}` (this calls Claude and costs money).")
        return
    st.caption(f"DeepEval results from the evaluation run on {results['run_date']}, judge "
               f"{results['judge_model']}. These scores are from that evaluation run, not from "
               "the run on this page.")
    st.dataframe([{"metric": name, "score": r["score"], "result": "PASS" if r["passed"] else "FAIL",
                   "reason": r["reason"]} for name, r in case["metrics"].items()], hide_index=True)
    st.markdown("**Checks:** " + " · ".join(f"{name} {'✓' if ok else '✗'}"
                                            for name, ok in case["checks"].items()))


def result_tabs(case_id: str, state: dict) -> None:
    tabs = st.tabs(["Analyses", "Critic review", "Decision memo", "Privacy",
                    "Audit and usage", "Evaluation"])
    with tabs[0]:
        for area, tab in zip(AREAS, st.tabs([a.title() for a in AREAS])):
            a = state.get("analyses", {}).get(area)
            if not a:
                tab.caption("Not available yet.")
            elif a["status"] == "failed":
                tab.error(f"This analysis could not be completed: {a['error']}")
            else:
                tab.markdown(f"**{a['risk_level']} risk.** {a['summary']}")
                tab.markdown("\n".join(f"- {f}" for f in a["findings"]))
                tab.caption("Policy sections: " + ", ".join(a["policy_refs"]))
                if a["conditions"]:
                    tab.markdown("**Conditions**\n" + "\n".join(f"- {c}" for c in a["conditions"]))
                tab.markdown(f"**Recommendation:** {a['recommendation']}")
    with tabs[1]:
        critic_review(state["review"]) if state.get("review") else st.caption("Not available yet.")
    with tabs[2]:
        st.markdown(state["decision"]["memo"]) if state.get("decision") else st.caption("Not available yet.")
    with tabs[3]:
        raw = cases[case_id]["application"]
        st.dataframe(pii_table(raw, sanitize(raw)), hide_index=True)
        st.caption("Only the masked version is sent to the agents or saved.")
    with tabs[4]:
        st.markdown("\n".join(f"- {line}" for line in state.get("audit", [])))
        usage_table(state.get("usage", []))
    with tabs[5]:
        evaluation_tab(case_id)


def review_block(case_id: str, run: dict) -> None:
    paused = next(e for e in reversed(run["events"]) if e["type"] == "paused")
    st.warning("⏸ **Paused for human review.** Why: " + " ".join(paused["reasons"]))
    left, right = st.columns(2)
    with left:
        st.markdown(f"**AI recommendation:** {paused['decision']} · risk score {paused['risk_score']}")
        st.markdown(paused["memo"])
    with right:
        choice = st.radio("Your decision", decision.LABELS, index=decision.LABELS.index(paused["decision"]),
                          format_func=lambda d: f"{d} (concur)" if d == paused["decision"] else d,
                          key=f"choice-{case_id}")
        conditions = st.text_area("Conditions, one per line", "\n".join(paused["conditions"]),
                                  key=f"conditions-{case_id}")
        notes = st.text_input("Notes", key=f"notes-{case_id}")
        justification = ""
        if choice != paused["decision"]:
            justification = st.text_area("Justification for override", key=f"why-{case_id}")
        if st.button("Submit review", type="primary", key=f"submit-{case_id}"):
            answer = {"decision": choice, "notes": notes, "override_justification": justification,
                      "conditions": [c.strip() for c in conditions.splitlines() if c.strip()]}
            error = decision.validate_human_review(answer, paused["decision"])
            if error:
                st.error(error)
            else:
                result = asyncio.run(graph.resume_case(run["thread_id"], answer))
                run["events"] += result["events"]
                run.update(status=result["status"], state=result["state"])
                st.rerun()


def final_panel(run: dict) -> None:
    final = next(e for e in reversed(run["events"]) if e["type"] == "final")
    human = final["human"]
    tag = " · **OVERRIDDEN BY REVIEWER**" if human and human["overridden"] else ""
    st.success(f"**Final decision: {final['final_decision']}**{tag}")
    st.markdown(f"AI recommended: {final['ai_decision']} · "
                + (f"Human decided: {human['decision']}" if human else "No human review was required."))
    if human and human["overridden"]:
        st.markdown(f"Justification: {human['override_justification']}")
    with st.expander("Decision memo", expanded=True):
        st.markdown(run["state"]["decision"]["memo"])


# ---- Page ----------------------------------------------------------------------

case_id = st.sidebar.radio("Case", list(cases), format_func=lambda c: f"{c} · {cases[c]['profile']}")
try:
    start_critic()
    critic_error = ""
except a2a_client.CriticUnavailable as error:
    critic_error = str(error)
ready = bool(settings.api_key) and not critic_error
st.sidebar.markdown("**Before you run:** add your Claude API key to the `.env` file in the project folder.")
clicked = st.sidebar.button("Run underwriting", type="primary", disabled=not ready)
st.sidebar.caption("API key: " + ("found ✓" if settings.api_key else "missing ✗ (see the steps on the page)"))
st.sidebar.caption(f"Model: {settings.model}")
st.sidebar.caption("Critic service: " + ("unreachable" if critic_error else "running"))
if critic_error:
    st.sidebar.error(critic_error)

application = sanitize(cases[case_id]["application"])
st.title(f"{case_id} · {cases[case_id]['profile']}")
if not settings.api_key:
    st.warning("⚠ **Please add your Claude API key before running the program.** "
               "No key was found, so the Run button is switched off." + KEY_HELP)
key_figures(metrics.case_metrics(application), application["credit"]["credit_score"])

if clicked:
    run = runs[case_id] = {"thread_id": uuid.uuid4().hex, "events": [], "status": "running", "state": {}}
    st.caption("A run takes a few minutes and can't be cancelled from this page.")
    box = st.empty()

    def show(event: dict) -> None:
        run["events"].append(event)
        with box.container():
            flow(run["events"])
            timeline(run["events"])

    with box.container():
        flow([])

    result = asyncio.run(graph.run_case(case_id, run["thread_id"], on_event=show))
    run.update(status=result["status"], state=result["state"])
    st.rerun()

run = runs.get(case_id)
if not run:
    st.info("Pick a case and press **Run underwriting** to watch the agents work.")
else:
    if run["status"] == "paused":
        review_block(case_id, run)
    elif run["status"] == "done":
        final_panel(run)
    flow(run["events"])
    left, right = st.columns(2)
    with left:
        st.subheader("Run timeline")
        timeline(run["events"])
    with right:
        st.subheader("Results")
        result_tabs(case_id, run["state"])
