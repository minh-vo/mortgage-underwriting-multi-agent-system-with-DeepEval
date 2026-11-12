import json

from underwriter import bias
from underwriter.config import load_application, load_cases
from underwriter.sanitize import pii_table, sanitize


def test_no_raw_pii_survives_anywhere_in_the_application():
    for case_id in load_cases():
        raw = load_application(case_id)
        clean = json.dumps(sanitize(raw))
        person = raw["applicant"]
        for secret in (person["ssn"], person["phone"], person["email"], person["address"],
                       *person["name"].split()):
            assert secret not in clean, (case_id, secret)


def test_masks_keep_the_last_four_digits():
    clean = sanitize(load_application("CASE-001"))["applicant"]
    assert clean == {"name": "[APPLICANT]", "ssn": "***-**-0142", "phone": "***-***-0142",
                     "email": "[EMAIL]", "address": "[ADDRESS]"}


def test_pii_in_free_text_is_masked_and_the_input_is_untouched():
    raw = load_application("CASE-001")
    raw["credit"]["notes"] = "Priya Raman (SSN 900-55-0142) called from 614-555-0142, priya.raman@example.com"
    clean = sanitize(raw)
    assert clean["credit"]["notes"] == "[APPLICANT] (SSN ***-**-0142) called from ***-***-0142, [EMAIL]"
    assert raw["applicant"]["name"] == "Priya Raman"


def test_figures_are_unchanged_by_sanitizing():
    raw = load_application("CASE-002")
    clean = sanitize(raw)
    for key in ("credit", "debts", "loan", "property"):
        assert clean[key] == raw[key]


def test_pii_table_has_before_and_after():
    raw = load_application("CASE-003")
    rows = pii_table(raw, sanitize(raw))
    assert rows[0] == {"field": "name", "before": "Denise Okafor", "after": "[APPLICANT]"}
    assert len(rows) == 5


def test_expected_outcome_is_not_part_of_the_application():
    for case in load_cases().values():
        assert "expected" not in case["application"]
        assert "acceptable_outcomes" not in json.dumps(case["application"])
        assert case["expected"]["acceptable_outcomes"]


def test_bias_screen_flags_protected_terms_but_not_benign_ones():
    assert bias.screen("The borrower's marital status and age suggest stability.") != []
    assert len(bias.screen("Applicant is married and pregnant.")) == 2
    assert bias.screen("The age of the oldest tradeline is 11 years. Average age of accounts is 6.") == []
    assert bias.screen("Single-family home with strong reserves and a 76.9% LTV.") == []
