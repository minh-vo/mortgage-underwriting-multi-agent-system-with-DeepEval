"""Every calculation and policy threshold. Pure functions, no I/O.

All periods of time are measured from the application's own `application_date`,
never from today's date, so results never change with the calendar.
"""

from datetime import date

# ---- Policy thresholds (data/policies.md). tests/test_policy.py checks that
# ---- each number appears in the matching section of the policy text.
MIN_CREDIT_SCORE = 620
TIER_EXCELLENT = 760
TIER_GOOD = 700
TIER_FAIR = 660
BANKRUPTCY_WAIT_YEARS = {7: 4, 13: 2}
FORECLOSURE_WAIT_YEARS = 7
MAX_30_DAY_LATES = 2
COLLECTIONS_LIMIT = 2000
MIN_YEARS_LINE_OF_WORK = 2
GAP_LOE_DAYS = 30
LONG_GAP_MONTHS = 6
SELF_EMPLOYED_DECLINE_PCT = 20
MAX_FRONT_END = 28
MAX_BACK_END = 43
MAX_BACK_END_EXCEPTION = 50
FACTOR_MIN_SCORE = 720
FACTOR_MIN_RESERVES = 12
FACTOR_MAX_LTV = 75
FACTORS_NEEDED = 2
RESERVES_STANDARD = 2
RESERVES_HIGHER = 6
RESERVES_SCORE_CUTOFF = 680
LARGE_DEPOSIT_PCT = 50
HOLDBACK_LIMIT = 5000
MAX_LTV = 95
MI_LTV = 80
ELIGIBLE_PROPERTY_TYPES = ["single_family", "townhouse", "condominium"]

THRESHOLDS_BY_SECTION = {
    "1.1": [MIN_CREDIT_SCORE, TIER_EXCELLENT, TIER_GOOD, TIER_FAIR],
    "1.2": [BANKRUPTCY_WAIT_YEARS[7], BANKRUPTCY_WAIT_YEARS[13]],
    "1.3": [FORECLOSURE_WAIT_YEARS],
    "1.4": [MAX_30_DAY_LATES, 60, 12],
    "1.5": [COLLECTIONS_LIMIT],
    "2.1": [MIN_YEARS_LINE_OF_WORK],
    "2.2": [GAP_LOE_DAYS, LONG_GAP_MONTHS],
    "2.3": [SELF_EMPLOYED_DECLINE_PCT],
    "2.5": [MAX_FRONT_END, MAX_BACK_END, MAX_BACK_END_EXCEPTION, FACTOR_MIN_SCORE,
            FACTOR_MIN_RESERVES, FACTOR_MAX_LTV, FACTORS_NEEDED],
    "3.1": [RESERVES_STANDARD, RESERVES_HIGHER, MAX_BACK_END, RESERVES_SCORE_CUTOFF],
    "3.3": [LARGE_DEPOSIT_PCT],
    "4.2": [HOLDBACK_LIMIT],
    "4.3": [MAX_LTV, MI_LTV],
}

# The calculator tools each specialist is expected to call.
AREA_TOOLS = {
    "credit": ["credit_score_tier"],
    "income": ["total_obligations", "dti_ratio", "housing_ratio"],
    "assets": ["reserves", "large_deposits"],
    "collateral": ["ltv_ratio"],
}


# ---- Small helpers ---------------------------------------------------------

def _months_between(earlier: str, later: str) -> int:
    """Whole months from one ISO date to another."""
    a, b = date.fromisoformat(earlier), date.fromisoformat(later)
    months = (b.year - a.year) * 12 + (b.month - a.month)
    return months - 1 if b.day < a.day else months


def _days_between(earlier: str, later: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def _pct(part: float, whole: float) -> float:
    return round(part / whole * 100, 2) if whole > 0 else 0.0


def qualifying_income(app: dict) -> tuple[float, str]:
    """Monthly qualifying income and a note on how it was built (policy 2.3, 2.4)."""
    emp = app["employment"]
    if emp["type"] == "self_employed":
        years = sorted(emp["self_employed_net_income"])
        earlier, recent = (emp["self_employed_net_income"][y] for y in years[-2:])
        decline = _pct(earlier - recent, earlier) if earlier > 0 else 0.0
        if decline > SELF_EMPLOYED_DECLINE_PCT:
            return round(recent / 12, 2), (
                f"Self-employed net income fell {decline:.1f}% year over year, so the "
                f"lower recent year (${recent:,.0f}) is used (2.3).")
        return round((earlier + recent) / 2 / 12, 2), (
            "Self-employed: average of the two most recent years (2.3).")
    monthly = emp["base_salary"] / 12
    years = sorted(emp["bonus"])
    last_two = [emp["bonus"][y] for y in years[-2:]]
    if len(last_two) == 2 and all(b > 0 for b in last_two) and emp["bonus_expected_to_continue"]:
        monthly += sum(last_two) / 2 / 12
        note = "Base salary plus the 2-year average bonus (2.4)."
    elif any(b > 0 for b in last_two):
        note = "Base salary only. Bonus excluded: not received for 2 years or not expected to continue (2.4)."
    else:
        note = "Base salary only."
    return round(monthly, 2), note


def monthly_debts(app: dict) -> float:
    return float(sum(app["debts"].values()))


def piti(app: dict) -> float:
    return float(sum(app["loan"]["piti"].values()))


def _large_deposit_list(deposits: list[dict], monthly_income: float) -> list[dict]:
    limit = monthly_income * LARGE_DEPOSIT_PCT / 100
    return [d for d in deposits if not d.get("payroll") and d["amount"] > limit]


def _credit_tier(score: int) -> str:
    if score >= TIER_EXCELLENT:
        return "Excellent"
    if score >= TIER_GOOD:
        return "Good"
    if score >= TIER_FAIR:
        return "Fair"
    if score >= MIN_CREDIT_SCORE:
        return "Marginal"
    return "Below minimum"


def _credit_flags(app: dict) -> tuple[list[str], list[str]]:
    """Return (derogatory flags, items that need a letter of explanation)."""
    credit, today = app["credit"], app["application_date"]
    flags, loe = [], []
    if credit["credit_score"] < MIN_CREDIT_SCORE:
        flags.append(f"Credit score {credit['credit_score']} is below the minimum {MIN_CREDIT_SCORE} (1.1)")
    bk = credit.get("bankruptcy")
    if bk:
        need = BANKRUPTCY_WAIT_YEARS[bk["chapter"]] * 12
        passed = _months_between(bk["discharge_date"], today)
        if passed < need:
            flags.append(f"Chapter {bk['chapter']} bankruptcy discharged {passed} months ago; "
                         f"{need} months required (1.2)")
    if credit.get("foreclosure_date"):
        passed = _months_between(credit["foreclosure_date"], today)
        if passed < FORECLOSURE_WAIT_YEARS * 12:
            flags.append(f"Foreclosure {passed} months ago; {FORECLOSURE_WAIT_YEARS * 12} months required (1.3)")
    recent = [l for l in credit["late_payments"] if _months_between(l["date"], today) < 12]
    serious = [l for l in recent if l["days_late"] >= 60]
    thirty = [l for l in recent if l["days_late"] < 60]
    if serious:
        flags.append(f"{len(serious)} payment(s) 60 or more days late in the last 12 months (1.4)")
    if len(thirty) > MAX_30_DAY_LATES:
        flags.append(f"{len(thirty)} payments 30 days late in the last 12 months; maximum {MAX_30_DAY_LATES} (1.4)")
    for late in thirty:
        loe.append(f"30-day late payment on {late['account']} dated {late['date']} (1.4)")
    non_medical = sum(c["amount"] for c in credit["collections"] if c["type"] == "non_medical")
    if non_medical > COLLECTIONS_LIMIT:
        flags.append(f"Non-medical collections of ${non_medical:,.0f} must be paid before closing (1.5)")
    return flags, loe


def _income_flags(app: dict) -> tuple[list[str], list[str]]:
    """Return (income flags, items that need a letter of explanation)."""
    emp, today = app["employment"], app["application_date"]
    flags, loe = [], []
    if emp["years_in_line_of_work"] < MIN_YEARS_LINE_OF_WORK:
        flags.append(f"Only {emp['years_in_line_of_work']} years in the line of work; "
                     f"{MIN_YEARS_LINE_OF_WORK} required (2.1)")
    for gap in emp["gaps"]:
        if _months_between(gap["end"], today) >= 24:
            continue
        days = _days_between(gap["start"], gap["end"])
        if days > GAP_LOE_DAYS:
            status = "on file" if gap.get("documented") else "missing"
            loe.append(f"Employment gap of {days} days ending {gap['end']}; letter of explanation {status} (2.2)")
        if _months_between(gap["start"], gap["end"]) > LONG_GAP_MONTHS:
            on_job = _months_between(emp["start_date"], today)
            if on_job < LONG_GAP_MONTHS:
                flags.append(f"Gap longer than {LONG_GAP_MONTHS} months and only {on_job} months "
                             f"on the current job (2.2)")
    return flags, loe


# ---- The authoritative figures for one application -------------------------

def case_metrics(app: dict) -> dict:
    """All figures and policy facts for an application. The single source of truth."""
    income, income_note = qualifying_income(app)
    debt, payment = monthly_debts(app), piti(app)
    score = app["credit"]["credit_score"]
    loan, prop, assets = app["loan"], app["property"], app["assets"]

    front = _pct(payment, income)
    back = _pct(payment + debt, income)
    ltv_basis = float(min(prop["purchase_price"], prop["appraised_value"]))
    ltv = _pct(loan["amount"], ltv_basis)

    unsourced = [d for d in _large_deposit_list(assets["recent_deposits"], income)
                 if not d.get("source_documented")]
    gift = assets.get("gift_funds") or {}
    gift_counted = gift.get("amount", 0) if gift.get("documented") else 0
    liquid = max(0.0, float(assets["checking"] + assets["savings"] + gift_counted
                            - sum(d["amount"] for d in unsourced)))
    cash_to_close = float(loan["down_payment"] + loan["closing_costs"])
    post_closing = liquid - cash_to_close
    shortfall = max(0.0, -post_closing)
    reserve_months = round(max(0.0, post_closing) / payment, 1) if payment > 0 else 0.0

    self_employed = app["employment"]["type"] == "self_employed"
    required = (RESERVES_HIGHER if back > MAX_BACK_END or self_employed
                or score < RESERVES_SCORE_CUTOFF else RESERVES_STANDARD)

    factors = []
    if score >= FACTOR_MIN_SCORE:
        factors.append(f"credit score {score} is {FACTOR_MIN_SCORE} or higher")
    if reserve_months >= FACTOR_MIN_RESERVES:
        factors.append(f"reserves of {reserve_months} months are {FACTOR_MIN_RESERVES} or more")
    if ltv <= FACTOR_MAX_LTV:
        factors.append(f"LTV {ltv}% is {FACTOR_MAX_LTV}% or less")
    if back <= MAX_BACK_END:
        dti_exception = "not_needed"
    elif back > MAX_BACK_END_EXCEPTION:
        dti_exception = "over_max"
    else:
        dti_exception = "allowed" if len(factors) >= FACTORS_NEEDED else "not_allowed"

    credit_flags, credit_loe = _credit_flags(app)
    income_flags, income_loe = _income_flags(app)

    must_complete = [r["description"] for r in prop["repairs"]
                     if r["category"] in ("structural", "safety")]
    cosmetic_total = float(sum(r["cost"] for r in prop["repairs"] if r["category"] == "cosmetic"))
    if cosmetic_total > HOLDBACK_LIMIT:
        must_complete += [r["description"] for r in prop["repairs"] if r["category"] == "cosmetic"]

    return {
        "qualifying_income": income,
        "qualifying_income_note": income_note,
        "existing_debt": debt,
        "piti": payment,
        "front_end_ratio": front,
        "front_end_over_limit": front > MAX_FRONT_END,
        "back_end_dti": back,
        "ltv_basis": ltv_basis,
        "ltv": ltv,
        "ltv_over_max": ltv > MAX_LTV,
        "mi_required": ltv > MI_LTV,
        "cash_to_close": cash_to_close,
        "liquid_assets": liquid,
        "post_closing_liquid": post_closing,
        "funds_shortfall": shortfall,
        "reserve_months": reserve_months,
        "required_reserve_months": required,
        "compensating_factors": factors,
        "dti_exception": dti_exception,
        "large_deposits_unsourced": [
            {"date": d["date"], "amount": d["amount"], "description": d["description"]}
            for d in unsourced],
        "holdback_limit": HOLDBACK_LIMIT,
        "credit_tier": _credit_tier(score),
        "derogatory_flags": credit_flags,
        "income_flags": income_flags,
        "loe_required": credit_loe + income_loe,
        "repair_flags": {
            "must_complete_before_closing": must_complete,
            "cosmetic_total": cosmetic_total,
            "holdback_eligible": 0 < cosmetic_total <= HOLDBACK_LIMIT,
        },
        "property_type_eligible": prop["type"] in ELIGIBLE_PROPERTY_TYPES,
        "project_review_required": prop["type"] == "condominium",
    }


def tool_inputs(app: dict) -> dict[str, dict]:
    """The exact arguments each calculator tool should receive for this application.

    Used in the specialist prompts and as the reference arguments in the evaluation.
    """
    m = case_metrics(app)
    return {
        "credit_score_tier": {"credit_score": app["credit"]["credit_score"]},
        "total_obligations": {"debts": dict(app["debts"]), "proposed_payment": m["piti"]},
        "dti_ratio": {"monthly_debt": m["existing_debt"] + m["piti"],
                      "monthly_income": m["qualifying_income"]},
        "housing_ratio": {"monthly_payment": m["piti"], "monthly_income": m["qualifying_income"]},
        "reserves": {"liquid_assets": m["liquid_assets"], "monthly_payment": m["piti"],
                     "cash_to_close": m["cash_to_close"],
                     "required_months": m["required_reserve_months"]},
        "large_deposits": {"deposits": list(app["assets"]["recent_deposits"]),
                           "monthly_income": m["qualifying_income"]},
        "ltv_ratio": {"loan_amount": float(app["loan"]["amount"]), "property_value": m["ltv_basis"]},
    }


# ---- Calculator functions. mcp_server.py registers these as MCP tools. -----

def dti_ratio(monthly_debt: float, monthly_income: float) -> str:
    """Back-end debt-to-income ratio (policy 2.5).

    Args:
        monthly_debt: All monthly debt payments INCLUDING the proposed housing payment (PITI).
        monthly_income: Monthly qualifying income.
    """
    if monthly_income <= 0:
        return "Error: monthly_income must be greater than 0"
    ratio = _pct(monthly_debt, monthly_income)
    if ratio <= MAX_BACK_END:
        status = f"within the {MAX_BACK_END}% limit"
    elif ratio <= MAX_BACK_END_EXCEPTION:
        status = f"above {MAX_BACK_END}%; needs {FACTORS_NEEDED} compensating factors"
    else:
        status = f"above the {MAX_BACK_END_EXCEPTION}% maximum"
    return f"{ratio}% back-end DTI ({status}). Debt ${monthly_debt:,.2f} / income ${monthly_income:,.2f}"


def housing_ratio(monthly_payment: float, monthly_income: float) -> str:
    """Front-end (housing) ratio (policy 2.5).

    Args:
        monthly_payment: Proposed housing payment (PITI).
        monthly_income: Monthly qualifying income.
    """
    if monthly_income <= 0:
        return "Error: monthly_income must be greater than 0"
    ratio = _pct(monthly_payment, monthly_income)
    status = (f"within the {MAX_FRONT_END}% limit" if ratio <= MAX_FRONT_END
              else f"above the {MAX_FRONT_END}% limit")
    return f"{ratio}% front-end ratio ({status}). Payment ${monthly_payment:,.2f} / income ${monthly_income:,.2f}"


def ltv_ratio(loan_amount: float, property_value: float) -> str:
    """Loan-to-value ratio (policy 4.1, 4.3).

    Args:
        loan_amount: The loan amount.
        property_value: The LOWER of the purchase price and the appraised value.
    """
    if property_value <= 0:
        return "Error: property_value must be greater than 0"
    ratio = _pct(loan_amount, property_value)
    if ratio > MAX_LTV:
        status = f"above the {MAX_LTV}% maximum"
    elif ratio > MI_LTV:
        status = f"above {MI_LTV}%; mortgage insurance required"
    else:
        status = f"{MI_LTV}% or less; no mortgage insurance"
    return f"{ratio}% LTV ({status}). Loan ${loan_amount:,.2f} / value ${property_value:,.2f}"


def reserves(liquid_assets: float, monthly_payment: float, cash_to_close: float,
             required_months: int) -> str:
    """Post-closing reserves in months of PITI (policy 3.1).

    Args:
        liquid_assets: Verified liquid funds before closing.
        monthly_payment: Proposed housing payment (PITI).
        cash_to_close: Down payment plus closing costs.
        required_months: Months of reserves this file must have.
    """
    if monthly_payment <= 0:
        return "Error: monthly_payment must be greater than 0"
    left = liquid_assets - cash_to_close
    if left < 0:
        return (f"0 months of reserves. Funds shortfall of ${-left:,.2f}: liquid assets "
                f"${liquid_assets:,.2f} do not cover cash to close ${cash_to_close:,.2f}")
    months = round(left / monthly_payment, 1)
    status = "meets" if months >= required_months else "below"
    return (f"{months} months of reserves ({status} the {required_months}-month requirement). "
            f"${left:,.2f} left after closing")


def credit_score_tier(credit_score: int) -> str:
    """Credit tier for a score (policy 1.1).

    Args:
        credit_score: The borrower's credit score.
    """
    tier = _credit_tier(credit_score)
    if tier == "Below minimum":
        return f"{credit_score}: below the minimum score of {MIN_CREDIT_SCORE}; not eligible"
    extra = f"; needs {FACTORS_NEEDED} compensating factors" if tier == "Marginal" else ""
    return f"{credit_score}: {tier} tier{extra}"


def large_deposits(deposits: list[dict], monthly_income: float) -> str:
    """Find large deposits that need source documents (policy 3.3).

    Args:
        deposits: Recent deposits. Each has amount, description, payroll (bool)
            and source_documented (bool).
        monthly_income: Monthly qualifying income.
    """
    large = _large_deposit_list(deposits, monthly_income)
    limit = monthly_income * LARGE_DEPOSIT_PCT / 100
    if not large:
        return f"0 large deposits (threshold ${limit:,.2f}, non-payroll only)"
    lines = [f"${d['amount']:,.2f} '{d['description']}': "
             + ("documented" if d.get("source_documented") else "NOT documented; deduct from verified funds")
             for d in large]
    return f"{len(large)} large deposit(s) over ${limit:,.2f}: " + "; ".join(lines)


def total_obligations(debts: dict[str, float], proposed_payment: float) -> str:
    """Total monthly obligations: existing debts plus the proposed housing payment.

    Args:
        debts: Monthly payment for each existing debt, by name.
        proposed_payment: Proposed housing payment (PITI).
    """
    existing = float(sum(debts.values()))
    return (f"${existing + proposed_payment:,.2f} total monthly obligations "
            f"(existing debts ${existing:,.2f} + housing ${proposed_payment:,.2f})")


CALCULATORS = [dti_ratio, housing_ratio, ltv_ratio, reserves, credit_score_tier,
               large_deposits, total_obligations]
