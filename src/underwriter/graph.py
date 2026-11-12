"""The LangGraph workflow, and the two functions that run it."""

from datetime import datetime

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from . import a2a_client, llm, metrics
from .agents import critic, decision, specialists, supervisor
from .config import get_settings, load_application
from .models import State
from .sanitize import sanitize

RECURSION_LIMIT = 50


# ---- Nodes -------------------------------------------------------------------

async def measure_node(state: State) -> dict:
    return {"metrics": metrics.case_metrics(state["application"]),
            "audit": ["application sanitized; figures computed"]}


async def supervisor_node(state: State) -> dict:
    step, usage = await supervisor.choose_next(state)
    update = {"next_step": step, "supervisor_log": [step], "usage": usage,
              "audit": [f"supervisor: {step['action']} {', '.join(step['areas'])}".strip()]}
    review = state.get("review")
    if step["action"] == "run_specialists" and review is not None:
        # Dispatching a rework: the old review no longer describes the analyses.
        update["review"] = None
        update["rework_count"] = state.get("rework_count", 0) + 1
    return update


def route(state: State):
    """Run exactly what the Supervisor picked. Chosen specialists run in parallel."""
    step = state["next_step"]
    if step["action"] == "run_specialists":
        return [Send("specialist", specialists.build_payload(area, state)) for area in step["areas"]]
    return step["action"]


async def specialist_node(payload: dict) -> dict:
    return await specialists.run_specialist(payload)


async def critic_node(state: State) -> dict:
    result = await a2a_client.review({
        "case_id": state["case_id"], "application": state["application"],
        "metrics": state["metrics"], "analyses": state.get("analyses", {})})
    usage = result.pop("usage", [])
    majors = sum(1 for d in result["defects"] if d["severity"] == "major")
    return {"review": result, "rework_feedback": critic.feedback_by_area(result), "usage": usage,
            "audit": [f"critic: {result['verdict']} ({majors} major defect(s))"]}


async def decision_node(state: State) -> dict:
    result, usage, notes = await decision.decide(state)
    reasons = decision.needs_human_review(result["decision"], result["risk_score"],
                                          state.get("review"), state.get("analyses", {}))
    update = {"decision": result, "review_reasons": reasons, "usage": usage, "audit": notes}
    if not reasons:
        update["final_decision"] = result["decision"]
    return update


def after_decision(state: State) -> str:
    return "human_review" if state["review_reasons"] else END


async def human_review_node(state: State) -> dict:
    ai = state["decision"]
    review = state.get("review")
    answer = interrupt({
        "decision": ai["decision"], "risk_score": ai["risk_score"],
        "conditions": ai["conditions"], "memo": ai["memo"],
        "review_summary": review["summary"] if review else "No quality review was possible.",
        "reasons": state["review_reasons"]})
    error = decision.validate_human_review(answer, ai["decision"])
    if error:
        raise ValueError(error)
    overridden = answer["decision"] != ai["decision"]
    human = {"decision": answer["decision"], "notes": answer.get("notes", ""),
             "conditions": answer.get("conditions", []), "overridden": overridden,
             "override_justification": answer.get("override_justification", "")}
    note = (f"human override: {ai['decision']} -> {answer['decision']}" if overridden
            else f"human concurred: {answer['decision']}")
    return {"human": human, "final_decision": answer["decision"], "audit": [note]}


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    graph.add_node("measure", measure_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("critic", critic_node)
    graph.add_node("decision", decision_node)
    graph.add_node("human_review", human_review_node)
    graph.add_edge(START, "measure")
    graph.add_edge("measure", "supervisor")
    graph.add_conditional_edges("supervisor", route, ["specialist", "critic", "decision"])
    graph.add_edge("specialist", "supervisor")
    graph.add_edge("critic", "supervisor")
    graph.add_conditional_edges("decision", after_decision, ["human_review", END])
    graph.add_edge("human_review", END)
    return graph


# ---- Events: what the UI shows ----------------------------------------------

def _events(node: str, update, context: dict) -> list[dict]:
    """Turn one node's state update into the events the UI renders.

    `context` carries the little that an event needs from earlier steps.
    """
    if node == "__interrupt__":
        return [{"type": "paused", **update[0].value}]
    if node == "measure":
        return [{"type": "sanitized", "metrics": update["metrics"]}]
    if node == "supervisor":
        context["rework"] = "review" in update  # this step dispatches a rework
        return [{"type": "supervisor", **update["next_step"], "rework": context["rework"]}]
    if node == "specialist":
        (area, analysis), = update["analyses"].items()
        feedback = context["feedback"].get(area, "") if context["rework"] else ""
        return [{"type": "specialist", "area": area, "analysis": analysis,
                 "tools": update["tool_trace"], "bias_flags": update["bias_flags"],
                 "feedback": feedback}]
    if node == "critic":
        context["feedback"] = update["rework_feedback"]
        return [{"type": "critic", **update["review"], "feedback": update["rework_feedback"]}]
    if node == "decision":
        context["ai_decision"] = update["decision"]["decision"]
        events = [{"type": "decision", **update["decision"], "reasons": update["review_reasons"]}]
        if "final_decision" in update:
            events.append({"type": "final", "final_decision": update["final_decision"],
                           "ai_decision": context["ai_decision"], "human": None})
        return events
    if node == "human_review":
        return [{"type": "final", "final_decision": update["final_decision"],
                 "ai_decision": context["ai_decision"], "human": update["human"]}]
    return []


async def _stream(graph_input, thread_id: str, on_event) -> dict:
    """Run the graph until it finishes or pauses. Everything opened here is closed here."""
    settings = get_settings()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    events: list[dict] = []

    def emit(event: dict) -> None:
        event["at"] = datetime.now().strftime("%H:%M:%S")
        events.append(event)
        if on_event:
            on_event(event)

    async with llm.use_client(), AsyncSqliteSaver.from_conn_string(settings.db_path) as saver:
        graph = build_graph().compile(checkpointer=saver)
        saved = (await graph.aget_state(config)).values
        context = {"rework": False, "feedback": saved.get("rework_feedback", {}),
                   "ai_decision": (saved.get("decision") or {}).get("decision")}
        status = "done"
        try:
            async for chunk in graph.astream(graph_input, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        status = "paused"
                    for event in _events(node, update, context):
                        emit(event)
        except Exception as error:
            status = "error"
            emit({"type": "error", "message": f"{type(error).__name__}: {error}"})
        state = dict((await graph.aget_state(config)).values)
    return {"status": status, "state": state, "events": events}


async def run_case(case_id: str, thread_id: str, on_event=None) -> dict:
    """Underwrite one case. Returns {"status": "done" | "paused" | "error", "state", "events"}."""
    application = sanitize(load_application(case_id))  # raw PII never enters the graph
    return await _stream({"case_id": case_id, "application": application}, thread_id, on_event)


async def get_state(thread_id: str) -> dict:
    """The saved state of a run (empty if the thread does not exist)."""
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(get_settings().db_path) as saver:
        graph = build_graph().compile(checkpointer=saver)
        return dict((await graph.aget_state(config)).values)


async def resume_case(thread_id: str, review: dict, on_event=None) -> dict:
    """Resume a paused case with the human review {decision, notes, conditions, override_justification}.

    An invalid review raises ValueError before anything is resumed, so the case stays paused.
    """
    state = await get_state(thread_id)
    if not state.get("decision") or state.get("final_decision"):
        raise ValueError("This case is not waiting for a human review.")
    error = decision.validate_human_review(review, state["decision"]["decision"])
    if error:
        raise ValueError(error)
    return await _stream(Command(resume=review), thread_id, on_event)
