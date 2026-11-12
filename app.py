"""Streamlit UI: run a case, watch the agents work, review the decision.

Run with: uv run streamlit run app.py
"""

import asyncio
import json
import uuid

import streamlit as st

from underwriter import a2a_client, graph, metrics
from underwriter.agents import decision
from underwriter.config import AREAS, EVAL_RESULTS, PRICES, get_settings, load_cases, usage_cost
from underwriter.sanitize import pii_table, sanitize

STAGES = ["Sanitize", "Specialists", "Critic", "Decision", "Human review"]
STAGE_OF = {"sanitized": 0, "supervisor": 1, "specialist": 1, "critic": 2, "decision": 3,
            "paused": 4, "final": 4}

st.set_page_config(page_title="Mortgage underwriting agents", layout="wide")
settings, cases = get_settings(), load_cases()
runs = st.session_state.setdefault("runs", {})  # case_id -> {thread_id, events, status, state}


@st.cache_resource
def start_critic():
    return a2a_client.ensure_critic_running()


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


def progress(events: list[dict]) -> None:
    current = max((STAGE_OF.get(e["type"], 0) for e in events), default=-1)
    done = any(e["type"] == "final" for e in events)
    parts = [f"**[{name}]**" if i == current and not done else f"{name} ✓" if i < current or done
             else name for i, name in enumerate(STAGES)]
    st.markdown(" → ".join(parts))


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
                    st.error(a["error"])
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
            st.error(f"{e['at']} · The run stopped: {e['message']}")


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
clicked = st.sidebar.button("Run underwriting", type="primary", disabled=not ready)
st.sidebar.caption(f"Model: {settings.model}")
st.sidebar.caption("Critic service: " + ("unreachable" if critic_error else "running"))
if critic_error:
    st.sidebar.error(critic_error)
if not settings.api_key:
    st.sidebar.info("Add ANTHROPIC_API_KEY to a .env file to run a case (see .env.example). "
                    "A Claude subscription is not an API key.")

application = sanitize(cases[case_id]["application"])
st.title(f"{case_id} · {cases[case_id]['profile']}")
key_figures(metrics.case_metrics(application), application["credit"]["credit_score"])

if clicked:
    run = runs[case_id] = {"thread_id": uuid.uuid4().hex, "events": [], "status": "running", "state": {}}
    st.caption("A run takes a few minutes and can't be cancelled from this page.")
    box = st.empty()

    def show(event: dict) -> None:
        run["events"].append(event)
        with box.container():
            progress(run["events"])
            timeline(run["events"])

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
    progress(run["events"])
    left, right = st.columns(2)
    with left:
        st.subheader("Run timeline")
        timeline(run["events"])
    with right:
        st.subheader("Results")
        result_tabs(case_id, run["state"])
