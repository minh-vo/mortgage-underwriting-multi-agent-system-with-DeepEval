"""The whole workflow with the fake LLM, the real MCP server and a real SQLite checkpoint."""

import asyncio
import json

import pytest

from underwriter import a2a_client, graph
from underwriter.config import AREAS, MAX_REWORKS, load_application
from underwriter.fake_llm import FakeLLM
from underwriter.models import Defect


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    path = tmp_path / "underwriter.db"
    monkeypatch.setenv("DB_PATH", str(path))
    return path


def install(monkeypatch, **options) -> FakeLLM:
    return FakeLLM(**options).install(monkeypatch, fake_critic_service=True)


def major(area="assets"):
    return Defect(area=area, severity="major", statement="States reserves of 9 months.",
                  should_be="Reserves are 3.9 months (3.1).")


def actions(result):
    return [(s["action"], s["areas"]) for s in result["state"]["supervisor_log"]]


def types(result):
    return [e["type"] for e in result["events"]]


async def test_a_clean_case_runs_specialists_critic_decision_and_finishes(monkeypatch):
    install(monkeypatch, score=20)
    seen = []
    result = await graph.run_case("CASE-001", "t1", on_event=seen.append)
    state = result["state"]
    assert result["status"] == "done" and state["final_decision"] == "APPROVED"
    assert actions(result) == [("run_specialists", AREAS), ("critic", []), ("decision", [])]
    assert {a: v["status"] for a, v in state["analyses"].items()} == dict.fromkeys(AREAS, "complete")
    assert types(result) == ["sanitized", "supervisor", "specialist", "specialist", "specialist",
                             "specialist", "supervisor", "critic", "supervisor", "decision", "final"]
    assert seen == result["events"]
    assert state["review"]["verdict"] == "pass"
    assert state.get("rework_count", 0) == 0 and state.get("human") is None


async def test_raw_pii_never_enters_the_state_or_the_events(monkeypatch):
    install(monkeypatch)
    result = await graph.run_case("CASE-001", "t1")
    person = load_application("CASE-001")["applicant"]
    stored = json.dumps(result["state"]) + json.dumps(result["events"])
    for secret in (person["ssn"], person["phone"], person["email"], "Priya", "Raman"):
        assert secret not in stored


async def test_usage_covers_every_agent_including_the_critic(monkeypatch):
    install(monkeypatch)
    result = await graph.run_case("CASE-001", "t1")
    agents = [u["agent"] for u in result["state"]["usage"]]
    assert sorted(set(agents)) == sorted([*AREAS, "critic", "decision", "supervisor"])
    assert agents.count("supervisor") == 3 and agents.count("critic") == 1


async def test_specialists_chosen_together_run_in_parallel_and_all_results_are_kept(monkeypatch):
    """P8: four specialists overlap in time, and their writes merge."""
    fake = install(monkeypatch)
    original, running, peak = fake.tool_loop, 0, 0

    async def tracked(*args, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.2)
        try:
            return await original(*args, **kwargs)
        finally:
            running -= 1

    monkeypatch.setattr(graph.specialists.llm, "tool_loop", tracked)
    result = await graph.run_case("CASE-002", "t1")
    assert peak == 4
    state = result["state"]
    assert set(state["analyses"]) == set(AREAS)
    assert {call["area"] for call in state["tool_trace"]} == set(AREAS)
    assert len(state["tool_trace"]) == 7 + 4  # seven calculators, one policy lookup per area


async def test_a_major_defect_sends_only_that_analyst_back_with_the_feedback(monkeypatch):
    """P3: the reworked specialist is given the defect and its previous analysis."""
    fake = install(monkeypatch, score=20, defect_rounds=[[major("assets")], []])
    result = await graph.run_case("CASE-003", "t1")
    state = result["state"]
    assert actions(result) == [("run_specialists", AREAS), ("critic", []),
                               ("run_specialists", ["assets"]), ("critic", []), ("decision", [])]
    assert state["rework_count"] == 1 and state["review"]["verdict"] == "pass"
    assert [len(fake.prompts(area)) for area in AREAS] == [1, 1, 2, 1]
    rework_prompt = fake.prompts("assets")[1]
    assert "Reserves are 3.9 months (3.1)." in rework_prompt
    assert '"summary": "assets analysis"' in rework_prompt
    assert state["analyses"]["assets"]["summary"] == "assets analysis (reworked)"
    rework_events = [e for e in result["events"] if e["type"] == "specialist" and e["feedback"]]
    assert [e["area"] for e in rework_events] == ["assets"]
    assert [e["rework"] for e in result["events"] if e["type"] == "supervisor"] == [
        False, False, True, False, False]


async def test_at_the_rework_cap_the_case_goes_to_decision_and_then_to_a_human(monkeypatch):
    install(monkeypatch, score=20, defect_rounds=[[major()]] * (MAX_REWORKS + 1))
    result = await graph.run_case("CASE-003", "t1")
    state = result["state"]
    assert state["rework_count"] == MAX_REWORKS
    assert [a for a, _ in actions(result)].count("critic") == MAX_REWORKS + 1
    assert actions(result)[-1] == ("decision", [])
    assert result["status"] == "paused" and state["review"]["verdict"] == "fail"
    assert state["review_reasons"] == [
        "The quality review still reports a major defect in: assets (5.2)."]


async def test_a_specialist_that_fails_twice_ends_in_human_review_not_a_loop(monkeypatch):
    """P5: a failed analysis is retried once, then treated as missing."""
    install(monkeypatch, score=20, fail={"credit": 5})
    result = await graph.run_case("CASE-001", "t1")
    state = result["state"]
    assert actions(result) == [("run_specialists", AREAS), ("run_specialists", ["credit"]),
                               ("critic", []), ("decision", [])]
    assert state["analyses"]["credit"]["status"] == "failed"
    assert state["analyses"]["credit"]["failed_attempts"] == 2
    assert state["review"]["reviewed_areas"] == ["income", "assets", "collateral"]
    assert result["status"] == "paused"
    assert state["review_reasons"] == ["Analysis missing or not completed for: credit (5.2)."]


async def test_a_conditional_approval_pauses_and_a_concurring_review_finishes_it(monkeypatch):
    install(monkeypatch, score=45)
    paused = await graph.run_case("CASE-002", "t1")
    assert paused["status"] == "paused" and types(paused)[-2:] == ["decision", "paused"]
    payload = paused["events"][-1]
    assert payload["decision"] == "CONDITIONAL_APPROVAL" and payload["risk_score"] == 45
    assert payload["memo"].startswith("## Decision memo") and payload["reasons"]
    assert "final_decision" not in paused["state"] or paused["state"]["final_decision"] is None

    done = await graph.resume_case("t1", {"decision": "CONDITIONAL_APPROVAL", "notes": "Agreed.",
                                          "conditions": ["Provide a letter of explanation."]})
    assert done["status"] == "done" and types(done) == ["final"]
    assert done["state"]["final_decision"] == "CONDITIONAL_APPROVAL"
    assert done["state"]["human"]["overridden"] is False
    assert done["state"]["audit"][-1] == "human concurred: CONDITIONAL_APPROVAL"
    assert done["events"][-1]["ai_decision"] == "CONDITIONAL_APPROVAL"


async def test_an_override_without_justification_is_rejected_and_the_case_stays_paused(monkeypatch):
    install(monkeypatch, score=45)
    await graph.run_case("CASE-002", "t1")
    with pytest.raises(ValueError, match="justification is required"):
        await graph.resume_case("t1", {"decision": "DENIED"})
    state = await graph.get_state("t1")
    assert state.get("final_decision") is None and state.get("human") is None

    done = await graph.resume_case("t1", {"decision": "DENIED",
                                          "override_justification": "Reserves are too thin."})
    assert done["state"]["final_decision"] == "DENIED"
    assert done["state"]["human"]["overridden"] is True
    assert done["state"]["audit"][-1] == "human override: CONDITIONAL_APPROVAL -> DENIED"
    with pytest.raises(ValueError, match="not waiting"):
        await graph.resume_case("t1", {"decision": "DENIED", "override_justification": "again"})


def test_a_paused_case_survives_a_restart(monkeypatch, database):
    """P9: each call below gets a new event loop, a new graph and a new database connection."""
    install(monkeypatch, score=45)
    paused = asyncio.run(graph.run_case("CASE-002", "restart"))
    assert paused["status"] == "paused" and database.exists()

    saved = asyncio.run(graph.get_state("restart"))
    assert saved["decision"]["risk_score"] == 45 and len(saved["analyses"]) == 4
    assert asyncio.run(graph.get_state("no-such-thread")) == {}

    done = asyncio.run(graph.resume_case("restart", {"decision": "CONDITIONAL_APPROVAL"}))
    assert done["state"]["final_decision"] == "CONDITIONAL_APPROVAL"


async def test_a_critic_outage_stops_the_run_with_a_readable_error_and_keeps_the_work(monkeypatch):
    install(monkeypatch)

    async def down(payload, url=None, http_client=None):
        raise a2a_client.CriticUnavailable("The Critic service is not reachable at http://127.0.0.1:9105.")

    monkeypatch.setattr(a2a_client, "review", down)
    result = await graph.run_case("CASE-001", "t1")
    assert result["status"] == "error"
    assert result["events"][-1]["type"] == "error"
    assert "Critic service is not reachable" in result["events"][-1]["message"]
    assert len(result["state"]["analyses"]) == 4  # finished work is still there
