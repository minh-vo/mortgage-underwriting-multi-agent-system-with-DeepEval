"""The Decision Agent, the score bands, and the human-review rules."""

import json

from .. import llm
from ..config import AREAS
from ..models import DecisionDraft

LABELS = ["APPROVED", "CONDITIONAL_APPROVAL", "DENIED"]
BAND_EDGE_WINDOWS = [(26, 35), (56, 65)]  # within 5 points of a band edge

SYSTEM = """You are the Decision Agent on a mortgage underwriting team. You weigh the \
specialists' analyses, the quality review and the authoritative figures, give the file a \
risk score, and write the decision memo. A human underwriter may review your \
recommendation, and the memo is kept as the record of why the decision was made.

Risk score (policy 5.1), from 0 (lowest risk) to 100 (highest risk). The score sets the decision:
- 0 to 30, APPROVED: the file meets every requirement. Requirements that are routine to \
note, such as mortgage insurance above 80% LTV, do not move a file out of this band.
- 31 to 60, CONDITIONAL_APPROVAL: the file can close once specific, achievable conditions \
are met, such as a letter of explanation, a condominium project review, an escrow \
holdback, paying a collection or documenting a deposit.
- 61 to 100, DENIED: the file fails a requirement that a condition cannot cure before \
closing, such as a score below the minimum, a waiting period not met, a debt ratio above \
the maximum or above 43% without the compensating factors, an LTV above the maximum, or \
funds that do not cover closing.
Place the score inside the band according to how many issues there are and how serious \
they are. Your `decision` must be the one the score's band gives.

The authoritative figures were computed in code and are correct. Where an analysis \
disagrees with them, use the figures. If an area's analysis is missing, say so in the \
memo and treat the gap as unresolved risk.

`conditions`: for a conditional approval, each condition the borrower must meet. \
Otherwise an empty list.

`memo`, in markdown, with these parts in this order:
1. Decision: the score and the decision.
2. Rationale: the reasoning in a short paragraph.
3. Key figures: each with its policy section number.
4. Findings by area: credit, income, assets, collateral.
5. Quality review: the Critic's summary and any unresolved defect.
6. Conditions: each one with the weakness it fixes (conditional approval only).
7. Adverse action reasons: the specific reasons for denial (denial only).

Base the decision only on the financial facts and the policy. Never refer to personal \
characteristics such as age, sex, race, religion, national origin, marital or family \
status, or disability."""

USER = """Case {case_id}.

<authoritative_figures>
{metrics}
</authoritative_figures>

<analyses>
{analyses}
</analyses>

<missing_areas>
{missing}
</missing_areas>

<quality_review>
{review}
</quality_review>"""


def band(risk_score: int) -> str:
    """The decision the policy assigns to a risk score (5.1)."""
    if risk_score <= 30:
        return "APPROVED"
    if risk_score <= 60:
        return "CONDITIONAL_APPROVAL"
    return "DENIED"


def needs_human_review(decision: str, risk_score: int, review: dict | None,
                       analyses: dict) -> list[str]:
    """Why a human must review this file (policy 5.2). An empty list means no review."""
    reasons = []
    if decision == "CONDITIONAL_APPROVAL":
        reasons.append("The decision is a conditional approval (5.2).")
    for low, high in BAND_EDGE_WINDOWS:
        if low <= risk_score <= high:
            edge = "approval" if low == 26 else "denial"
            reasons.append(f"Risk score {risk_score} is within 5 points of the {edge} threshold (5.2).")
    if review is not None and review.get("verdict") == "fail":
        reasons.append("The quality review still reports a major defect in: "
                       f"{', '.join(review['rework_targets'])} (5.2).")
    missing = [a for a in AREAS if analyses.get(a, {}).get("status") != "complete"]
    if missing:
        reasons.append(f"Analysis missing or not completed for: {', '.join(missing)} (5.2).")
    return reasons


def validate_human_review(resume: dict, ai_decision: str) -> str | None:
    """Return an error message if the human review cannot be accepted, else None (5.3)."""
    if resume.get("decision") not in LABELS:
        return f"Choose a decision: {', '.join(LABELS)}."
    overriding = resume["decision"] != ai_decision
    if overriding and not (resume.get("override_justification") or "").strip():
        return "A justification is required to override the recommendation (5.3)."
    return None


async def decide(state: dict) -> tuple[dict, list[dict], list[str]]:
    """Score the file and write the memo. Returns (decision, usage, audit notes)."""
    analyses = state.get("analyses", {})
    complete = {a: v for a, v in analyses.items() if v.get("status") == "complete"}
    shown = {a: {k: x for k, x in v.items() if k not in ("status", "failed_attempts", "error")}
             for a, v in complete.items()}
    review = state.get("review")
    review_shown = ("No quality review was possible." if review is None else
                    json.dumps({k: review[k] for k in ("verdict", "summary", "defects",
                                                       "application_risks", "policy_violations")},
                               indent=2))
    user = USER.format(case_id=state["case_id"], metrics=json.dumps(state["metrics"], indent=2),
                       analyses=json.dumps(shown, indent=2),
                       missing=", ".join(a for a in AREAS if a not in complete) or "none",
                       review=review_shown)
    draft, usage = await llm.structured(SYSTEM, user, DecisionDraft, agent="decision")

    score = max(0, min(100, draft.risk_score))
    label = band(score)
    notes = []
    if label != draft.decision:
        notes.append(f"decision label corrected: the model said {draft.decision}, "
                     f"but score {score} is in the {label} band")
    decision = {"risk_score": score, "decision": label, "memo": draft.memo,
                "conditions": draft.conditions if label == "CONDITIONAL_APPROVAL" else []}
    notes.append(f"decision: {label} (risk score {score})")
    return decision, usage, notes
