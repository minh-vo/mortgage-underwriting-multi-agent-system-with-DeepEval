"""One smoke test of the Streamlit page, with the fake LLM and a fake Critic client."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from underwriter import a2a_client, config
from underwriter.fake_llm import FakeLLM


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "underwriter.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(a2a_client, "ensure_critic_running", lambda *a, **k: None)
    monkeypatch.setattr(config, "EVAL_RESULTS", tmp_path / "results.json")  # absent by default
    FakeLLM(score=45).install(monkeypatch, fake_critic_service=True)
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=60)


def texts(at) -> str:
    parts = [e.value for group in (at.markdown, at.warning, at.success, at.error, at.info, at.caption)
             for e in group]
    return "\n".join(str(p) for p in parts)


def test_run_pause_reject_unjustified_override_then_finish(app, tmp_path, monkeypatch):
    at = app.run()
    assert not at.exception
    assert "CASE-001" in at.title[0].value
    assert [m.label for m in at.metric] == ["Back-end DTI", "LTV", "Reserves", "Credit score"]
    assert "No evaluation results" not in texts(at)  # nothing has run yet

    at.sidebar.button[0].click().run()
    assert not at.exception
    assert "Paused for human review" in texts(at)
    diagram = at.get("graphviz_chart")[0].proto.spec  # the architecture, marked with progress
    assert "▶ Human review" in diagram and "Decision Agent ✓" in diagram and "Credit analyst ✓" in diagram
    assert "## Decision memo" in texts(at)  # the memo is shown in full beside the form
    assert "No evaluation results for this case yet" in texts(at)

    at.radio(key="choice-CASE-001").set_value("DENIED").run()
    at.button(key="submit-CASE-001").click().run()
    assert "A justification is required to override" in texts(at)
    assert "Final decision" not in texts(at)

    at.text_area(key="why-CASE-001").set_value("Reserves are too thin.").run()
    at.button(key="submit-CASE-001").click().run()
    diagram = at.get("graphviz_chart")[0].proto.spec
    assert "Human review ✓" in diagram and "Final decision ✓" in diagram and "▶" not in diagram
    assert not at.exception
    page = texts(at)
    assert "Final decision: DENIED" in page and "OVERRIDDEN BY REVIEWER" in page
    assert "Justification: Reserves are too thin." in page
    assert "## Decision memo" in page  # the memo is in the final panel too
    assert "human override: CONDITIONAL_APPROVAL -> DENIED" in page


def test_evaluation_tab_shows_scores_from_the_results_file(app, tmp_path):
    metric = {"score": 0.9, "passed": True, "reason": "The plan was followed."}
    (tmp_path / "results.json").write_text(json.dumps({
        "run_date": "2026-11-12", "judge_model": "claude-opus-5-5",
        "cases": {"CASE-001": {"metrics": {"Plan Adherence": metric},
                               "checks": {"decision in acceptable set": True, "no bias flags": False}}}}))
    at = app.run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    page = texts(at)
    assert "No evaluation results" not in page
    assert "evaluation run on 2026-11-12, judge claude-opus-5-5" in page
    assert "These scores are from that evaluation run, not from the run on this page." in page
    assert "decision in acceptable set ✓" in page and "no bias flags ✗" in page
    tables = [str(d.value) for d in at.dataframe]
    assert any("Plan Adherence" in t and "PASS" in t for t in tables)


def test_missing_key_shows_a_warning_and_disables_run(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    at = app.run()
    assert not at.exception
    warning = next(w.value for w in at.warning if "Please add your Claude API key before running the program." in w.value)
    for step in ("console.anthropic.com", str(config.ROOT / ".env"), "ANTHROPIC_API_KEY=", "Restart the app"):
        assert step in warning
    assert at.sidebar.button[0].disabled
    sidebar = "\n".join(str(e.value) for e in list(at.sidebar.markdown) + list(at.sidebar.caption))
    assert "Before you run:" in sidebar and "API key: missing" in sidebar
