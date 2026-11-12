"""Data shapes shared across modules: Pydantic models and the LangGraph state."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Area = Literal["credit", "income", "assets", "collateral"]
DecisionLabel = Literal["APPROVED", "CONDITIONAL_APPROVAL", "DENIED"]


# ---- What the model is asked to produce -----------------------------------

class AnalysisDraft(BaseModel):
    """A specialist's analysis of one area of the loan file."""
    summary: str = Field(description="Two to four sentences: what you found and what it means.")
    findings: list[str] = Field(description="One fact per item, with its figure and policy section number.")
    policy_refs: list[str] = Field(description='Policy section numbers you relied on, such as "2.5".')
    risk_level: Literal["LOW", "MODERATE", "HIGH"]
    conditions: list[str] = Field(description="What must happen before closing. Empty if nothing.")
    recommendation: str = Field(description="One sentence for the decision maker.")


class Defect(BaseModel):
    """A problem with an analysis (not a weakness of the borrower)."""
    area: Area
    severity: Literal["major", "minor"]
    statement: str = Field(description="What the analysis says or omits.")
    should_be: str = Field(description="What a correct analysis would say.")


class CriticReview(BaseModel):
    """The Critic's review of the specialists' analyses."""
    defects: list[Defect] = Field(description="Problems with the analyses themselves.")
    application_risks: list[str] = Field(description="Weaknesses of the borrower or the property.")
    policy_violations: list[str] = Field(description="Policy requirements the application does not meet.")
    summary: str


class DecisionDraft(BaseModel):
    """The Decision Agent's recommendation."""
    risk_score: int = Field(description="0 (lowest risk) to 100 (highest risk).")
    decision: DecisionLabel
    conditions: list[str]
    memo: str = Field(description="The decision memo, in markdown.")


class SupervisorPick(BaseModel):
    """The Supervisor's choice of the next step."""
    action: Literal["run_specialists", "critic", "decision"]
    areas: list[Area] = Field(description="Areas to run. Empty unless action is run_specialists.")
    reason: str = Field(description="One or two sentences on why this is the right next step.")


# ---- What the program stores (model output plus fields set in code) --------

class Analysis(AnalysisDraft):
    area: Area
    status: Literal["complete", "failed"] = "complete"
    failed_attempts: int = 0  # failures in a row; reset to 0 on success
    error: str = ""


class ToolCall(BaseModel):
    area: str
    tool: str
    input: dict
    output: str
    is_error: bool = False


def failed_analysis(area: str, failed_attempts: int, error: str) -> dict:
    """The record stored when a specialist could not finish. It is treated as missing."""
    return Analysis(area=area, status="failed", failed_attempts=failed_attempts, error=error,
                    summary="", findings=[], policy_refs=[], risk_level="HIGH", conditions=[],
                    recommendation="").model_dump()


# ---- LangGraph state --------------------------------------------------------

def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class State(TypedDict, total=False):
    """Values are plain dicts and lists, so checkpoints always serialize.

    Keys with a reducer may be written by nodes that run in parallel.
    Every other key has exactly one writer per step.
    """
    case_id: str
    application: dict  # sanitized
    metrics: dict
    analyses: Annotated[dict, merge_dicts]  # area -> Analysis
    tool_trace: Annotated[list, operator.add]  # ToolCall
    bias_flags: Annotated[list, operator.add]
    usage: Annotated[list, operator.add]
    audit: Annotated[list, operator.add]
    supervisor_log: Annotated[list, operator.add]
    next_step: dict  # the Supervisor's latest pick
    review: dict | None  # CriticReview plus verdict and rework_targets
    rework_feedback: dict  # area -> defect text
    rework_count: int
    decision: dict | None
    review_reasons: list  # why a human must review; empty when not needed
    human: dict | None
    final_decision: str | None
