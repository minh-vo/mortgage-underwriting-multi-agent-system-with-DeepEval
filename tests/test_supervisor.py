"""The Supervisor's guardrails (P2): every row of the allowed-moves table."""

import pytest

from underwriter import llm
from underwriter.agents import supervisor
from underwriter.config import AREAS, MAX_REWORKS
from underwriter.fake_llm import FakeLLM
from underwriter.models import SupervisorPick


def done(area):
    return {"area": area, "status": "complete", "failed_attempts": 0, "risk_level": "LOW"}


def failed(area, attempts):
    return {"area": area, "status": "failed", "failed_attempts": attempts, "risk_level": "HIGH"}


ALL_DONE = {area: done(area) for area in AREAS}
FAIL_REVIEW = {"verdict": "fail", "rework_targets": ["assets", "income"], "summary": "s"}
PASS_REVIEW = {"verdict": "pass", "rework_targets": [], "summary": "s"}


def test_at_the_start_any_subset_of_the_four_areas_may_run():
    assert supervisor.allowed_moves({}) == [
        {"action": "run_specialists", "areas": AREAS, "any_subset": True}]


def test_missing_and_once_failed_areas_are_outstanding():
    state = {"analyses": {"credit": done("credit"), "income": failed("income", 1)}}
    assert supervisor.allowed_moves(state) == [
        {"action": "run_specialists", "areas": ["income", "assets", "collateral"], "any_subset": True}]


def test_all_complete_and_no_review_means_critic():
    assert supervisor.allowed_moves({"analyses": ALL_DONE}) == [{"action": "critic", "areas": []}]


def test_an_area_that_failed_twice_is_not_retried():
    state = {"analyses": {**ALL_DONE, "credit": failed("credit", 2)}}
    assert supervisor.allowed_moves(state) == [{"action": "critic", "areas": []}]


def test_nothing_complete_and_everything_exhausted_goes_to_decision():
    state = {"analyses": {area: failed(area, 2) for area in AREAS}}
    assert supervisor.allowed_moves(state) == [{"action": "decision", "areas": []}]


def test_a_passed_review_means_decision():
    state = {"analyses": ALL_DONE, "review": PASS_REVIEW}
    assert supervisor.allowed_moves(state) == [{"action": "decision", "areas": []}]


@pytest.mark.parametrize("rework_count", range(MAX_REWORKS))
def test_a_failed_review_under_the_cap_means_rework_of_exactly_the_targets(rework_count):
    state = {"analyses": ALL_DONE, "review": FAIL_REVIEW, "rework_count": rework_count}
    assert supervisor.allowed_moves(state) == [
        {"action": "run_specialists", "areas": ["assets", "income"], "any_subset": False}]


def test_a_failed_review_at_the_cap_means_decision():
    state = {"analyses": ALL_DONE, "review": FAIL_REVIEW, "rework_count": MAX_REWORKS}
    assert supervisor.allowed_moves(state) == [{"action": "decision", "areas": []}]


def test_is_allowed_checks_the_action_and_the_areas():
    start = supervisor.allowed_moves({})
    assert supervisor.is_allowed("run_specialists", ["credit", "income"], start)
    assert not supervisor.is_allowed("run_specialists", [], start)
    assert not supervisor.is_allowed("critic", [], start)
    rework = supervisor.allowed_moves({"analyses": ALL_DONE, "review": FAIL_REVIEW})
    assert supervisor.is_allowed("run_specialists", ["income", "assets"], rework)
    assert not supervisor.is_allowed("run_specialists", ["assets"], rework)  # must be all targets
    assert not supervisor.is_allowed("run_specialists", ["credit"], rework)


async def test_the_models_pick_is_what_runs_and_is_logged_with_the_allowed_moves(monkeypatch):
    def pick_two(allowed):
        return SupervisorPick(action="run_specialists", areas=["credit", "assets"],
                              reason="Start with credit and assets.")

    fake = FakeLLM(supervisor=pick_two).install(monkeypatch)
    step, usage = await supervisor.choose_next({"case_id": "CASE-001"})
    assert step == {"action": "run_specialists", "areas": ["credit", "assets"],
                    "reason": "Start with credit and assets.", "override": False,
                    "allowed": ["run_specialists on any subset of credit, income, assets, collateral"]}
    assert usage[0]["agent"] == "supervisor"
    assert "<allowed_moves>" in fake.prompts("supervisor")[0]


async def test_a_pick_that_is_not_allowed_is_replaced_and_logged(monkeypatch):
    def skip_ahead(allowed):
        return SupervisorPick(action="decision", areas=[], reason="Looks fine, decide now.")

    FakeLLM(supervisor=skip_ahead).install(monkeypatch)
    step, _ = await supervisor.choose_next({"case_id": "CASE-001"})
    assert step["action"] == "run_specialists" and step["areas"] == AREAS
    assert step["override"] and step["reason"].startswith("guardrail override")


async def test_a_failed_supervisor_call_falls_back_to_the_first_allowed_move(monkeypatch):
    async def broken(*args, **kwargs):
        raise llm.LLMError("refusal")

    monkeypatch.setattr(llm, "structured", broken)
    step, usage = await supervisor.choose_next({"case_id": "CASE-001", "analyses": ALL_DONE})
    assert step["action"] == "critic" and step["override"] and usage == []
