"""The specialist runner with the fake LLM and the real MCP tool server."""

import pytest

from underwriter import metrics
from underwriter.agents import specialists
from underwriter.config import AREAS, load_application
from underwriter.fake_llm import FakeLLM
from underwriter.models import Analysis
from underwriter.sanitize import sanitize


def state_for(case_id: str) -> dict:
    app = sanitize(load_application(case_id))
    return {"case_id": case_id, "application": app, "metrics": metrics.case_metrics(app)}


@pytest.mark.parametrize("area", AREAS)
def test_prompt_has_the_slice_the_tool_inputs_and_the_facts(area):
    payload = specialists.build_payload(area, state_for("CASE-002"))
    system, user = specialists.build_prompt(payload)
    assert specialists.ROLES[area]["title"] in system
    assert set(payload["tool_inputs"]) == set(metrics.AREA_TOOLS[area])
    for tag in ("application", "authoritative_tool_inputs", "authoritative_facts"):
        assert f"<{tag}>" in user
    assert "<rework>" not in user
    assert "[APPLICANT]" not in user and "Lindqvist" not in user  # identity is not sent at all


async def test_a_specialist_returns_a_complete_analysis_and_a_real_tool_trace(monkeypatch):
    FakeLLM().install(monkeypatch)
    state = state_for("CASE-002")
    update = await specialists.run_specialist(specialists.build_payload("income", state))
    analysis = Analysis(**update["analyses"]["income"])
    assert analysis.status == "complete" and analysis.failed_attempts == 0
    called = {call["tool"]: call for call in update["tool_trace"]}
    assert set(called) == {"total_obligations", "dti_ratio", "housing_ratio", "get_policy"}
    assert called["dti_ratio"]["input"] == metrics.tool_inputs(state["application"])["dti_ratio"]
    assert called["dti_ratio"]["output"].startswith("46.11%")
    assert all(call["area"] == "income" for call in update["tool_trace"])
    assert set(update) == {"analyses", "tool_trace", "bias_flags", "usage", "audit"}


async def test_a_failed_run_is_stored_as_failed_and_counts_attempts(monkeypatch):
    """P5: a failed analysis is marked failed, never passed off as finished."""
    FakeLLM(fail={"credit": 2}).install(monkeypatch)
    state = state_for("CASE-001")
    first = await specialists.run_specialist(specialists.build_payload("credit", state))
    assert first["analyses"]["credit"]["status"] == "failed"
    assert first["analyses"]["credit"]["failed_attempts"] == 1
    state["analyses"] = first["analyses"]
    second = await specialists.run_specialist(specialists.build_payload("credit", state))
    assert second["analyses"]["credit"]["failed_attempts"] == 2
    assert "FAILED (attempt 2)" in second["audit"][0]


async def test_a_rework_prompt_carries_the_feedback_and_the_previous_analysis(monkeypatch):
    """P3: the reworked specialist sees the defect and its own earlier work."""
    fake = FakeLLM().install(monkeypatch)
    state = state_for("CASE-003")
    first = await specialists.run_specialist(specialists.build_payload("assets", state))
    state["analyses"] = first["analyses"]
    state["rework_feedback"] = {"assets": "States reserves of 9 months; the figure is 3.9."}
    second = await specialists.run_specialist(specialists.build_payload("assets", state))
    prompt = fake.prompts("assets")[1]
    assert "<rework>" in prompt
    assert "States reserves of 9 months; the figure is 3.9." in prompt
    assert '"summary": "assets analysis"' in prompt  # its previous analysis
    assert second["analyses"]["assets"]["summary"] == "assets analysis (reworked)"


async def test_protected_terms_in_an_analysis_are_flagged(monkeypatch):
    fake = FakeLLM().install(monkeypatch)
    original = fake.tool_loop

    async def biased(*args, **kwargs):
        result = await original(*args, **kwargs)
        result.output.summary = "Given the borrower's age and marital status, risk is low."
        return result

    monkeypatch.setattr(specialists.llm, "tool_loop", biased)
    update = await specialists.run_specialist(specialists.build_payload("credit", state_for("CASE-001")))
    assert len(update["bias_flags"]) == 2 and update["bias_flags"][0].startswith("credit:")
