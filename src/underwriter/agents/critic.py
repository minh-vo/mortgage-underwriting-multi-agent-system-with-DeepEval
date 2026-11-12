"""The Critic: reviews the quality of the analyses. It does not decide the loan."""

import json

from .. import llm
from ..models import CriticReview

SYSTEM = """You are the quality reviewer on a mortgage underwriting team. Specialist \
analysts have each written an analysis of one area of a loan file (credit, income, assets, \
collateral). You review those analyses before a decision is made, so the decision maker \
can trust them. You are not deciding the loan, and you are not underwriting it again.

You are given the authoritative figures for the case. They were computed in code from the \
file and are correct by definition.

A defect is a problem with an analysis:
- a figure that differs from the authoritative figures;
- a material fact in its own area left out (for example a flag listed in the \
authoritative figures that the analysis never mentions);
- policy applied or cited wrongly;
- a statement that contradicts another analysis;
- language that refers to a protected personal characteristic.

Severity: "major" means the defect would mislead the decision (a wrong figure, a missed \
disqualifying fact, a wrong policy conclusion). "minor" means the conclusion still stands \
(wording, a missing citation, an imprecise label).

These are NOT defects:
- A conclusion, risk level or recommendation you would have set differently, when the \
analysis states the facts correctly and applies policy correctly.
- A weakness of the borrower or the property. A weak file analysed accurately has no \
defects. Put borrower weaknesses in application_risks, and unmet policy requirements in \
policy_violations.
- An area listed as missing. Its absence is handled elsewhere; do not hold it against the \
other analyses and do not write a defect for it.

For each defect give the area, the severity, `statement` (what the analysis says or \
omits) and `should_be` (the correct statement, with the figure and policy section). \
`should_be` is handed to the analyst as the instruction for the rework, so make it \
specific enough to act on.

`summary` is two or three sentences for the decision maker on how far the analyses can \
be relied on."""

USER = """Case {case_id}.

<authoritative_figures>
{metrics}
</authoritative_figures>

<analyses>
{analyses}
</analyses>

<missing_areas>
{missing}
</missing_areas>"""


def compute_verdict(review: CriticReview, reviewed_areas: list[str]) -> tuple[str, list[str]]:
    """The verdict is decided here, in code: fail if and only if there is a major defect.

    Returns (verdict, rework targets). Targets are the reviewed areas with a major defect.
    """
    targets = sorted({d.area for d in review.defects
                      if d.severity == "major" and d.area in reviewed_areas})
    return ("fail" if targets else "pass"), targets


def feedback_by_area(review: dict) -> dict[str, str]:
    """The defect text each rework target receives."""
    feedback = {}
    for area in review["rework_targets"]:
        lines = [f"- [{d['severity']}] {d['statement']} It should be: {d['should_be']}"
                 for d in review["defects"] if d["area"] == area]
        feedback[area] = "\n".join(lines)
    return feedback


async def review(case_id: str, application: dict, metrics: dict, analyses: dict) -> dict:
    """Review the complete analyses. Returns the review, the computed verdict and usage."""
    complete = {area: a for area, a in analyses.items() if a.get("status") == "complete"}
    missing = [area for area in ("credit", "income", "assets", "collateral") if area not in complete]
    shown = {area: {k: v for k, v in a.items() if k not in ("status", "failed_attempts", "error")}
             for area, a in complete.items()}
    user = USER.format(case_id=case_id, metrics=json.dumps(metrics, indent=2),
                       analyses=json.dumps(shown, indent=2),
                       missing=", ".join(missing) or "none")
    result, usage = await llm.structured(SYSTEM, user, CriticReview, agent="critic")
    verdict, targets = compute_verdict(result, list(complete))
    return {**result.model_dump(), "verdict": verdict, "rework_targets": targets,
            "reviewed_areas": list(complete), "usage": usage}
