"""The policy facts each synthetic case is designed to show (CLAUDE.md section 6.2)."""

import copy
import datetime

import pytest

from underwriter import metrics
from underwriter.config import load_application, load_cases


@pytest.fixture(scope="module")
def m():
    return {cid: metrics.case_metrics(load_application(cid)) for cid in load_cases()}


def test_case_001_strong(m):
    c = m["CASE-001"]
    assert c["qualifying_income"] == 12125.0
    assert c["front_end_ratio"] == 23.92 and c["back_end_dti"] == 30.1
    assert c["dti_exception"] == "not_needed"
    assert c["ltv"] == 76.92 and not c["mi_required"]
    assert c["reserve_months"] == 15.2 and c["reserve_months"] >= 12
    assert c["required_reserve_months"] == 2
    assert c["credit_tier"] == "Excellent"
    assert c["derogatory_flags"] == [] and c["loe_required"] == []
    assert c["large_deposits_unsourced"] == []


def test_case_002_borderline_curable(m):
    c = m["CASE-002"]
    assert c["qualifying_income"] == 9000.0  # one-year bonus excluded
    assert 43 < c["back_end_dti"] <= 50 and c["back_end_dti"] == 46.11
    assert len(c["compensating_factors"]) == 2 and c["dti_exception"] == "allowed"
    assert c["ltv_basis"] == 356000 and c["ltv"] == 74.16  # appraisal below price
    assert c["required_reserve_months"] == 6 and c["reserve_months"] == 9.0
    assert len(c["loe_required"]) == 1 and "30-day late" in c["loe_required"][0]
    assert c["derogatory_flags"] == []
    assert c["project_review_required"]
    assert c["repair_flags"] == {"must_complete_before_closing": [], "cosmetic_total": 3200.0,
                                 "holdback_eligible": True}


def test_case_003_near_deny_edge(m):
    c = m["CASE-003"]
    assert 43 < c["back_end_dti"] <= 50 and c["back_end_dti"] == 47.33
    assert c["compensating_factors"] == [] and c["dti_exception"] == "not_allowed"
    assert c["required_reserve_months"] == 6 and c["reserve_months"] == 3.9
    assert c["ltv"] == 90.0 and c["mi_required"] and not c["ltv_over_max"]
    assert c["credit_tier"] == "Fair"
    assert any("gap" in item and "on file" in item for item in c["loe_required"])


def test_case_004_weak(m):
    c = m["CASE-004"]
    assert c["credit_tier"] == "Below minimum"
    assert c["qualifying_income"] == 5833.33
    assert "fell 27.1%" in c["qualifying_income_note"]
    assert c["dti_exception"] == "over_max" and c["front_end_over_limit"]
    assert c["ltv_over_max"]
    flags = " | ".join(c["derogatory_flags"])
    assert "below the minimum 620" in flags
    assert "Chapter 13 bankruptcy discharged 8 months ago" in flags
    assert "60 or more days late" in flags
    assert "collections of $2,600" in flags
    assert [d["amount"] for d in c["large_deposits_unsourced"]] == [12000]
    assert c["liquid_assets"] == 7000.0
    assert len(c["repair_flags"]["must_complete_before_closing"]) == 2


def test_reserves_never_negative_when_closing_cannot_be_funded(m):
    """P6: a shortfall is reported as a shortfall, not as negative reserve months."""
    c = m["CASE-004"]
    assert c["funds_shortfall"] == 24000.0
    assert c["reserve_months"] == 0.0
    assert c["post_closing_liquid"] == -24000.0
    text = metrics.reserves(7000, 2300, 31000, 6)
    assert text.startswith("0 months") and "shortfall of $24,000.00" in text


def test_results_do_not_depend_on_todays_date(monkeypatch):
    app = load_application("CASE-004")
    before = metrics.case_metrics(app)

    class FarFuture(datetime.date):
        @classmethod
        def today(cls):
            return cls(2040, 1, 1)

    monkeypatch.setattr(metrics, "date", FarFuture)
    assert metrics.case_metrics(app) == before


def test_waiting_period_passes_once_enough_time_has_gone():
    app = copy.deepcopy(load_application("CASE-004"))
    app["credit"]["bankruptcy"]["discharge_date"] = "2024-03-01"  # 24 months before
    flags = metrics.case_metrics(app)["derogatory_flags"]
    assert not any("bankruptcy" in f for f in flags)


def test_documented_gift_counts_and_undocumented_does_not():
    app = copy.deepcopy(load_application("CASE-003"))
    app["assets"]["gift_funds"] = {"amount": 10000, "documented": True}
    assert metrics.case_metrics(app)["liquid_assets"] == 55500.0
    app["assets"]["gift_funds"]["documented"] = False
    assert metrics.case_metrics(app)["liquid_assets"] == 45500.0


@pytest.mark.parametrize("case_id", ["CASE-001", "CASE-002", "CASE-003", "CASE-004"])
def test_tool_inputs_reproduce_the_metrics(case_id):
    """P10: the reference tool arguments, run through the calculators, give the same figures."""
    app = load_application(case_id)
    m, inputs = metrics.case_metrics(app), metrics.tool_inputs(app)
    assert set(inputs) == {fn.__name__ for fn in metrics.CALCULATORS}
    assert metrics.dti_ratio(**inputs["dti_ratio"]).startswith(f"{m['back_end_dti']}%")
    assert metrics.housing_ratio(**inputs["housing_ratio"]).startswith(f"{m['front_end_ratio']}%")
    assert metrics.ltv_ratio(**inputs["ltv_ratio"]).startswith(f"{m['ltv']}%")
    expected_reserves = f"{m['reserve_months']} months" if m["reserve_months"] else "0 months"
    assert metrics.reserves(**inputs["reserves"]).startswith(expected_reserves)
    assert metrics.credit_score_tier(**inputs["credit_score_tier"]).startswith(
        str(app["credit"]["credit_score"]))
    assert metrics.large_deposits(**inputs["large_deposits"]).startswith(
        str(len([d for d in app["assets"]["recent_deposits"]
                 if not d["payroll"] and d["amount"] > m["qualifying_income"] / 2])))
    total = m["existing_debt"] + m["piti"]
    assert metrics.total_obligations(**inputs["total_obligations"]).startswith(f"${total:,.2f}")


def test_every_area_tool_is_a_calculator():
    names = {fn.__name__ for fn in metrics.CALCULATORS}
    assert {t for tools in metrics.AREA_TOOLS.values() for t in tools} == names
