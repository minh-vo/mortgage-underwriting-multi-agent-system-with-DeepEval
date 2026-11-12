"""The Critic: the verdict rule (P4) and the A2A service, run in-process."""

import httpx
import pytest

from underwriter import a2a_client, critic_service, metrics
from underwriter.agents import critic
from underwriter.config import load_application
from underwriter.fake_llm import FakeLLM
from underwriter.models import CriticReview, Defect
from underwriter.sanitize import sanitize

AREAS = ["credit", "income", "assets", "collateral"]


def defect(area="assets", severity="major"):
    return Defect(area=area, severity=severity, statement="States reserves of 9 months.",
                  should_be="Reserves are 3.9 months (3.1).")


def review_with(*defects, risks=("DTI above 43%.",)):
    return CriticReview(defects=list(defects), application_risks=list(risks),
                        policy_violations=["Reserves below 6 months (3.1)."], summary="s")


def test_borrower_weaknesses_alone_never_fail_the_review():
    """P4: a weak file that was analysed correctly passes."""
    assert critic.compute_verdict(review_with(), AREAS) == ("pass", [])


def test_minor_defects_alone_pass():
    assert critic.compute_verdict(review_with(defect(severity="minor")), AREAS) == ("pass", [])


def test_a_major_defect_fails_and_names_the_areas_to_rework():
    review = review_with(defect("income"), defect("assets"), defect("assets"),
                         defect("credit", "minor"))
    assert critic.compute_verdict(review, AREAS) == ("fail", ["assets", "income"])


def test_a_defect_against_an_area_that_was_not_reviewed_is_ignored():
    assert critic.compute_verdict(review_with(defect("credit")), ["income"]) == ("pass", [])


def test_feedback_gives_each_target_all_of_its_defects():
    review = {"rework_targets": ["assets"],
              "defects": [defect("assets").model_dump(), defect("assets", "minor").model_dump(),
                          defect("income", "minor").model_dump()]}
    feedback = critic.feedback_by_area(review)
    assert list(feedback) == ["assets"]
    assert feedback["assets"].count("It should be: Reserves are 3.9 months (3.1).") == 2


def payload(analyses):
    app = sanitize(load_application("CASE-003"))
    return {"case_id": "CASE-003", "application": app,
            "metrics": metrics.case_metrics(app), "analyses": analyses}


def analysis(area, status="complete"):
    return {"area": area, "status": status, "failed_attempts": 0, "error": "", "summary": f"{area} ok",
            "findings": [], "policy_refs": [], "risk_level": "LOW", "conditions": [],
            "recommendation": "Proceed."}


@pytest.fixture
def service():
    app = critic_service.build_app("http://critic.test")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://critic.test")


async def test_service_returns_the_review_with_verdict_targets_and_usage(monkeypatch, service):
    fake = FakeLLM(defect_rounds=[[defect("assets")]]).install(monkeypatch)
    analyses = {area: analysis(area) for area in AREAS}
    result = await a2a_client.review(payload(analyses), url="http://critic.test", http_client=service)
    assert result["verdict"] == "fail" and result["rework_targets"] == ["assets"]
    assert result["defects"][0]["should_be"] == "Reserves are 3.9 months (3.1)."
    assert result["application_risks"] == ["Thin reserves."]
    assert result["usage"] == [{"agent": "critic", "model": "fake",
                                "input_tokens": 100, "output_tokens": 50}]
    prompt = fake.prompts("critic")[0]
    assert '"back_end_dti": 47.33' in prompt and "assets ok" in prompt


async def test_service_tells_the_critic_which_areas_are_missing(monkeypatch, service):
    fake = FakeLLM(defect_rounds=[[defect("credit")]]).install(monkeypatch)
    analyses = {"income": analysis("income"), "credit": analysis("credit", status="failed")}
    result = await a2a_client.review(payload(analyses), url="http://critic.test", http_client=service)
    prompt = fake.prompts("critic")[0]
    assert "<missing_areas>\ncredit, assets, collateral\n</missing_areas>" in prompt
    assert "credit ok" not in prompt  # a failed analysis is treated as missing
    assert result["verdict"] == "pass"  # the defect names an area that was not reviewed


async def test_a_failure_inside_the_service_reaches_the_caller_as_a_readable_error(monkeypatch, service):
    fake = FakeLLM().install(monkeypatch)

    async def broken(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(critic.llm, "structured", broken)
    with pytest.raises(a2a_client.CriticUnavailable, match="Critic failed: RuntimeError: model unavailable"):
        await a2a_client.review(payload({}), url="http://critic.test", http_client=service)


async def test_a_clear_error_when_the_service_is_not_running():
    with pytest.raises(a2a_client.CriticUnavailable, match="not reachable"):
        await a2a_client.review(payload({}), url="http://127.0.0.1:9")


def test_status_reports_down_when_nothing_is_listening():
    assert a2a_client.critic_status("http://127.0.0.1:9") == "down"
