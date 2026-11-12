"""Score bands, the human-review triggers (P7) and override validation."""

import pytest

from underwriter.agents import decision
from underwriter.config import AREAS
from underwriter.fake_llm import FakeLLM

DONE = {area: {"status": "complete"} for area in AREAS}
PASS = {"verdict": "pass", "rework_targets": []}


@pytest.mark.parametrize("score,label", [(0, "APPROVED"), (30, "APPROVED"),
                                         (31, "CONDITIONAL_APPROVAL"), (60, "CONDITIONAL_APPROVAL"),
                                         (61, "DENIED"), (100, "DENIED")])
def test_band(score, label):
    assert decision.band(score) == label


def test_a_clean_approval_or_denial_needs_no_review():
    assert decision.needs_human_review("APPROVED", 15, PASS, DONE) == []
    assert decision.needs_human_review("DENIED", 85, PASS, DONE) == []


def test_trigger_1_conditional_approval():
    reasons = decision.needs_human_review("CONDITIONAL_APPROVAL", 45, PASS, DONE)
    assert reasons == ["The decision is a conditional approval (5.2)."]


@pytest.mark.parametrize("score,needed", [(25, False), (26, True), (30, True), (35, True),
                                          (55, True), (56, True), (65, True), (66, False)])
def test_trigger_2_score_near_a_band_edge(score, needed):
    reasons = decision.needs_human_review(decision.band(score), score, PASS, DONE)
    assert bool(reasons) == needed


def test_a_one_point_change_in_score_no_longer_removes_the_review():
    """P7: 61 is a denial, and it is still reviewed because it sits next to the edge."""
    assert decision.needs_human_review("DENIED", 61, PASS, DONE) == [
        "Risk score 61 is within 5 points of the denial threshold (5.2)."]
    assert decision.needs_human_review("APPROVED", 28, PASS, DONE) == [
        "Risk score 28 is within 5 points of the approval threshold (5.2)."]


def test_trigger_3_the_critic_is_still_failing():
    review = {"verdict": "fail", "rework_targets": ["assets"]}
    reasons = decision.needs_human_review("DENIED", 90, review, DONE)
    assert reasons == ["The quality review still reports a major defect in: assets (5.2)."]


def test_trigger_4_an_analysis_is_missing_or_failed():
    analyses = {**DONE, "credit": {"status": "failed"}}
    del analyses["income"]
    reasons = decision.needs_human_review("APPROVED", 10, PASS, analyses)
    assert reasons == ["Analysis missing or not completed for: credit, income (5.2)."]


def test_no_quality_review_at_all_is_not_itself_a_trigger():
    assert decision.needs_human_review("APPROVED", 10, None, DONE) == []


def test_override_needs_a_justification():
    ai = "CONDITIONAL_APPROVAL"
    assert decision.validate_human_review({"decision": ai}, ai) is None
    assert "justification is required" in decision.validate_human_review({"decision": "DENIED"}, ai)
    assert "justification is required" in decision.validate_human_review(
        {"decision": "DENIED", "override_justification": "   "}, ai)
    assert decision.validate_human_review(
        {"decision": "DENIED", "override_justification": "Reserves too thin."}, ai) is None
    assert "Choose a decision" in decision.validate_human_review({"decision": "MAYBE"}, ai)


def state():
    return {"case_id": "CASE-001", "metrics": {"back_end_dti": 30.1}, "analyses": {}, "review": None}


async def test_the_band_overrides_a_label_that_disagrees_with_the_score(monkeypatch):
    FakeLLM(score=70, decision_label="APPROVED").install(monkeypatch)
    result, usage, notes = await decision.decide(state())
    assert result["decision"] == "DENIED" and result["risk_score"] == 70
    assert "decision label corrected" in notes[0]
    assert usage[0]["agent"] == "decision"


async def test_conditions_are_kept_only_for_a_conditional_approval(monkeypatch):
    FakeLLM(score=45).install(monkeypatch)
    result, _, notes = await decision.decide(state())
    assert result["decision"] == "CONDITIONAL_APPROVAL" and result["conditions"]
    assert notes == ["decision: CONDITIONAL_APPROVAL (risk score 45)"]

    FakeLLM(score=20, decision_label="CONDITIONAL_APPROVAL").install(monkeypatch)
    result, _, _ = await decision.decide(state())
    assert result["decision"] == "APPROVED" and result["conditions"] == []


async def test_an_out_of_range_score_is_clamped(monkeypatch):
    FakeLLM(score=140).install(monkeypatch)
    result, _, _ = await decision.decide(state())
    assert result["risk_score"] == 100 and result["decision"] == "DENIED"


async def test_the_decision_prompt_names_missing_areas(monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    await decision.decide(state())
    prompt = fake.prompts("decision")[0]
    assert "<missing_areas>\ncredit, income, assets, collateral\n</missing_areas>" in prompt
    assert "No quality review was possible." in prompt
