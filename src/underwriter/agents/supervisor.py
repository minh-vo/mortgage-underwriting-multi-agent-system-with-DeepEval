"""The Supervisor: code decides which moves are allowed, the model picks among them."""

import json

from .. import llm
from ..config import AREAS, MAX_FAILED_ATTEMPTS, MAX_REWORKS
from ..models import SupervisorPick

SYSTEM = """You are the Supervisor of a mortgage underwriting team. You decide the next \
step for one loan file. On each turn you see where the file stands and the moves that are \
allowed right now, and you pick one.

The team:
- Four specialist analysts (credit, income, assets, collateral). They are independent, so \
several can run at the same time.
- A Critic that reviews the finished analyses for quality.
- A Decision Agent that scores the file and writes the decision memo.

Pick only from the allowed moves. For run_specialists, `areas` must come from the areas \
that move lists; where the move says any subset is allowed you may run all of them \
together or only some. For critic and decision, leave `areas` empty.

Give your reason in one or two sentences that someone reading the audit log would \
understand."""

USER = """Case {case_id}.

<status>
{status}
</status>

<allowed_moves>
{allowed}
</allowed_moves>"""


def _complete(state: dict) -> list[str]:
    return [a for a in AREAS if state.get("analyses", {}).get(a, {}).get("status") == "complete"]


def _outstanding(state: dict) -> list[str]:
    """Areas never attempted, or failed and still allowed another attempt."""
    result = []
    for area in AREAS:
        analysis = state.get("analyses", {}).get(area)
        if analysis is None or (analysis["status"] == "failed"
                                and analysis["failed_attempts"] < MAX_FAILED_ATTEMPTS):
            result.append(area)
    return result


def allowed_moves(state: dict) -> list[dict]:
    """The guardrails. Pure code: the model cannot route outside what this returns."""
    review = state.get("review")
    if review is None:
        outstanding = _outstanding(state)
        if outstanding:
            return [{"action": "run_specialists", "areas": outstanding, "any_subset": True}]
        if _complete(state):
            return [{"action": "critic", "areas": []}]
        return [{"action": "decision", "areas": []}]
    if review["verdict"] == "pass":
        return [{"action": "decision", "areas": []}]
    if state.get("rework_count", 0) < MAX_REWORKS:
        return [{"action": "run_specialists", "areas": review["rework_targets"], "any_subset": False}]
    return [{"action": "decision", "areas": []}]


def is_allowed(action: str, areas: list[str], moves: list[dict]) -> bool:
    for move in moves:
        if move["action"] != action:
            continue
        if action != "run_specialists":
            return True
        chosen = set(areas)
        if chosen and chosen <= set(move["areas"]) and (move["any_subset"] or chosen == set(move["areas"])):
            return True
    return False


def describe(move: dict) -> str:
    if move["action"] != "run_specialists":
        return move["action"]
    scope = "any subset of" if move["any_subset"] else "all of"
    return f"run_specialists on {scope} {', '.join(move['areas'])}"


def _status(state: dict) -> dict:
    analyses = state.get("analyses", {})
    review = state.get("review")
    return {
        "analyses": {area: (f"{analyses[area]['status']}"
                            + (f", {analyses[area]['risk_level']} risk"
                               if analyses[area]["status"] == "complete"
                               else f", failed {analyses[area]['failed_attempts']} time(s)")
                            if area in analyses else "not started")
                     for area in AREAS},
        "rework_rounds_used": f"{state.get('rework_count', 0)} of {MAX_REWORKS}",
        "latest_review": ("none since the last analyses were written" if review is None else
                          {"verdict": review["verdict"], "rework_targets": review["rework_targets"],
                           "summary": review["summary"]}),
    }


async def choose_next(state: dict) -> tuple[dict, list[dict]]:
    """Ask the model for the next step and enforce the guardrails.

    Returns (step, usage). The step is logged; what it says is what runs.
    """
    moves = allowed_moves(state)
    user = USER.format(case_id=state["case_id"], status=json.dumps(_status(state), indent=2),
                       allowed=json.dumps(moves, indent=2))
    override, usage = "", []
    try:
        pick, usage = await llm.structured(SYSTEM, user, SupervisorPick, effort="low",
                                           agent="supervisor")
        action, areas, reason = pick.action, list(dict.fromkeys(pick.areas)), pick.reason
        if action != "run_specialists":
            areas = []
        if not is_allowed(action, areas, moves):
            override = f"guardrail override: the model picked {action} {areas}, which is not allowed."
    except llm.LLMError as error:
        override = f"guardrail override: the Supervisor call failed ({error})."
    if override:
        action, areas = moves[0]["action"], list(moves[0]["areas"])
        reason = f"{override} Using the first allowed move."
    step = {"action": action, "areas": areas, "reason": reason,
            "allowed": [describe(move) for move in moves], "override": bool(override)}
    return step, usage
