"""The four specialist analysts: one runner, four roles."""

import json

from .. import bias, llm, metrics
from ..mcp_client import tool_session
from ..models import Analysis, AnalysisDraft, failed_analysis

ROLES = {
    "credit": {
        "title": "Credit Analyst", "chapter": "1 (sections 1.1 to 1.5)",
        "slice": ["credit"],
        "facts": ["credit_tier", "derogatory_flags", "loe_required"],
    },
    "income": {
        "title": "Income Analyst", "chapter": "2 (sections 2.1 to 2.5)",
        "slice": ["employment", "debts", "loan"],
        "facts": ["qualifying_income", "qualifying_income_note", "existing_debt", "piti",
                  "front_end_ratio", "front_end_over_limit", "back_end_dti",
                  "compensating_factors", "dti_exception", "income_flags", "loe_required"],
    },
    "assets": {
        "title": "Asset Analyst", "chapter": "3 (sections 3.1 to 3.4)",
        "slice": ["assets", "loan"],
        "facts": ["liquid_assets", "cash_to_close", "post_closing_liquid", "funds_shortfall",
                  "reserve_months", "required_reserve_months", "large_deposits_unsourced"],
    },
    "collateral": {
        "title": "Collateral Analyst", "chapter": "4 (sections 4.1 to 4.5)",
        "slice": ["property", "loan"],
        "facts": ["ltv_basis", "ltv", "ltv_over_max", "mi_required", "repair_flags",
                  "holdback_limit", "property_type_eligible", "project_review_required"],
    },
}

SYSTEM = """You are the {title} on a mortgage underwriting team. You analyse one area of a \
loan file, and your analysis is read by a quality reviewer and then by the person who \
decides the loan. They rely on it being accurate and on every figure being traceable.

Your area is policy chapter {chapter}. Stay inside it; other analysts cover the rest.

How to work:
1. Call each calculator tool listed under "Authoritative tool inputs" once, with exactly \
the arguments given. Those arguments were computed from the file and are the agreed \
figures for this case, so do not recompute or adjust them.
2. Read the policy sections you rely on with get_policy (or search_policy if you are \
unsure which section applies), and cite their numbers.
3. Apply the policy to the facts. "Authoritative facts" lists what has already been \
established for your area; your analysis must agree with it.

Report the borrower's weaknesses plainly. A weak file needs an accurate analysis, not a \
favourable one. The applicant's identity is masked: never guess at it, and never refer to \
personal characteristics such as age, sex, race, religion, national origin, marital or \
family status, or disability.

Your final message is the analysis itself, in the required JSON format. Every finding \
names its figure and its policy section."""

USER = """Case {case_id}, application date {application_date}.

<application>
{application}
</application>

<authoritative_tool_inputs>
{tool_inputs}
</authoritative_tool_inputs>

<authoritative_facts>
{facts}
</authoritative_facts>
{rework}"""

REWORK = """
<rework>
A quality reviewer found a problem with your previous analysis. Fix exactly this problem \
and keep everything that was correct.

Reviewer's finding:
{feedback}

Your previous analysis:
{previous}
</rework>
"""


def build_payload(area: str, state: dict) -> dict:
    """Everything one specialist run needs. This is what `Send` carries."""
    app, case_metrics = state["application"], state["metrics"]
    role = ROLES[area]
    all_inputs = metrics.tool_inputs(app)
    return {
        "area": area,
        "case_id": state["case_id"],
        "application_date": app["application_date"],
        "application": {key: app[key] for key in role["slice"]},
        "tool_inputs": {tool: all_inputs[tool] for tool in metrics.AREA_TOOLS[area]},
        "facts": {key: case_metrics[key] for key in role["facts"]},
        "feedback": state.get("rework_feedback", {}).get(area, ""),
        "previous": state.get("analyses", {}).get(area),
    }


def build_prompt(payload: dict) -> tuple[str, str]:
    role = ROLES[payload["area"]]
    previous = payload.get("previous")
    rework = ""
    if payload.get("feedback") and previous and previous["status"] == "complete":
        shown = {key: previous[key] for key in AnalysisDraft.model_fields}
        rework = REWORK.format(feedback=payload["feedback"], previous=json.dumps(shown, indent=2))
    user = USER.format(
        case_id=payload["case_id"], application_date=payload["application_date"],
        application=json.dumps(payload["application"], indent=2),
        tool_inputs=json.dumps(payload["tool_inputs"], indent=2),
        facts=json.dumps(payload["facts"], indent=2), rework=rework)
    return SYSTEM.format(title=role["title"], chapter=role["chapter"]), user


async def run_specialist(payload: dict) -> dict:
    """Run one specialist and return its state update (reducer keys only)."""
    area = payload["area"]
    previous = payload.get("previous") or {}
    system, user = build_prompt(payload)
    try:
        async with tool_session() as tools:
            result = await llm.tool_loop(system, user, await tools.anthropic_tools(), tools.call,
                                         schema=AnalysisDraft, agent=area)
    except Exception as error:  # the tool server could not start or dropped
        result = llm.ToolLoopResult(status="failed", error=f"Tool session failed: {error}")

    trace = [{"area": area, **call} for call in result.trace]
    if result.status == "complete":
        analysis = Analysis(area=area, **result.output.model_dump()).model_dump()
        flags = [f"{area}: {flag}" for flag in bias.screen(json.dumps(result.output.model_dump()))]
        note = f"{area} analysis complete ({analysis['risk_level']} risk, {len(trace)} tool calls)"
    else:
        attempts = previous.get("failed_attempts", 0) + 1
        analysis = failed_analysis(area, attempts, result.error)
        flags = []
        note = f"{area} analysis FAILED (attempt {attempts}): {result.error}"
    return {"analyses": {area: analysis}, "tool_trace": trace, "bias_flags": flags,
            "usage": result.usage, "audit": [note]}
