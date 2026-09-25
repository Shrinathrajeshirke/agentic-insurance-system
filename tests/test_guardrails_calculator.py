import pytest
from src.schema.guardrails import evaluate_underwriting_guardrails
from src.schema.user_profile import UserProfile
from src.tools.calculator import calculate_policy_rates


# =====================================================================
# Underwriting Guardrail Unit Tests (IRDAI HLV & Age Multipliers)
# =====================================================================

def test_guardrail_age_under_18():
    """Applicants under 18 years must produce a CRITICAL entry age violation."""
    profile = UserProfile(
        age=17,
        gender="Male",
        annual_income="3 - 6 Lakhs",
        desired_sum_assured="1 Crore",
        policy_term_years=30
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert not is_valid
    assert any("CRITICAL: Minimum entry age" in f for f in flags)


def test_guardrail_age_over_65():
    """Applicants over 65 years must produce a CRITICAL entry age violation."""
    profile = UserProfile(
        age=66,
        gender="Male",
        annual_income="15 - 25 Lakhs",
        desired_sum_assured="1 Crore",
        policy_term_years=15
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert not is_valid
    assert any("CRITICAL: Maximum entry age" in f for f in flags)


def test_guardrail_hlv_multiplier_tier_1_valid():
    """Age <= 35 allows up to 25x annual income. 10-15 Lakhs bracket (12.5L * 25 = 3.125 Cr)."""
    profile = UserProfile(
        age=28,
        gender="Male",
        annual_income="10 - 15 Lakhs",
        desired_sum_assured="2 Crore",  # 2 Cr is within 3.125 Cr
        policy_term_years=35
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert is_valid
    assert not any("UNDERWRITING WARNING" in f for f in flags)


def test_guardrail_hlv_multiplier_tier_1_exceeded():
    """Age <= 35 with '3 - 6 Lakhs' (5L base * 25 = 1.25 Cr) requesting 2 Crore must warn."""
    profile = UserProfile(
        age=25,
        gender="Male",
        annual_income="3 - 6 Lakhs",
        desired_sum_assured="2 Crore",  # Exceeds 1.25 Cr cap
        policy_term_years=35
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert is_valid  # Warnings do not invalidate the application
    assert any("UNDERWRITING WARNING" in f and "HLV limit" in f for f in flags)


def test_guardrail_hlv_multiplier_tier_3_exceeded():
    """Age 46-55 allows up to 15x annual income. '6 - 10 Lakhs' (8L * 15 = 1.2 Cr). Requesting 2 Cr must warn."""
    profile = UserProfile(
        age=50,
        gender="Male",
        annual_income="6 - 10 Lakhs",
        desired_sum_assured="2 Crore",  # Exceeds 1.2 Cr cap
        policy_term_years=25
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert is_valid
    assert any("UNDERWRITING WARNING" in f and "15x HLV limit" in f for f in flags)


def test_guardrail_maturity_age_adjustment():
    """Requested maturity age (age + term) > 85 must trigger term adjustment advice."""
    profile = UserProfile(
        age=60,
        gender="Male",
        annual_income="15 - 25 Lakhs",
        desired_sum_assured="1 Crore",
        policy_term_years=30  # 60 + 30 = 90 (> 85)
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert is_valid
    assert any("ADJUSTMENT:" in f and "exceeds typical 85-year ceiling" in f for f in flags)


def test_guardrail_financial_gate_low_income():
    """Cover >= 1 Crore with income 'Below 3 Lakhs' must trigger FINANCIAL GATE."""
    profile = UserProfile(
        age=28,
        gender="Male",
        annual_income="Below 3 Lakhs",
        desired_sum_assured="1 Crore",
        policy_term_years=30
    )
    is_valid, flags = evaluate_underwriting_guardrails(profile.model_dump())
    assert is_valid
    assert any("FINANCIAL GATE:" in f for f in flags)


# =====================================================================
# Actuarial Calculator Unit Tests (Pricing, Riders & 0% GST)
# =====================================================================

def test_calculator_output_structure():
    """Calculator must output verified pricing records for top 4 benchmark insurers."""
    rates = calculate_policy_rates(
        age=30,
        smoker=False,
        sum_assured_crores=1.0,
        gender="Male",
        include_ci_rider=False,
        include_adb_rider=False,
        return_of_premium=False
    )
    assert isinstance(rates, list)
    assert len(rates) == 4
    plan_names = [p["plan_name"] for p in rates]
    assert any("HDFC Life" in name for name in plan_names)
    assert any("Max Life" in name for name in plan_names)


def test_calculator_smoker_loading_penalty():
    """Smoker profile must incur higher annual and monthly premiums than non-smokers."""
    non_smoker_rates = calculate_policy_rates(
        age=30, smoker=False, sum_assured_crores=1.0, gender="Male"
    )
    smoker_rates = calculate_policy_rates(
        age=30, smoker=True, sum_assured_crores=1.0, gender="Male"
    )

    for ns, s in zip(non_smoker_rates, smoker_rates):
        assert s["annual_premium"] > ns["annual_premium"], f"Smoker loading missing on {s['plan_name']}"
        assert s["monthly_premium"] > ns["monthly_premium"]


def test_calculator_female_applicant_discount():
    """Female applicants must receive preferred underwriting rates compared to males."""
    male_rates = calculate_policy_rates(
        age=30, smoker=False, sum_assured_crores=1.0, gender="Male"
    )
    female_rates = calculate_policy_rates(
        age=30, smoker=False, sum_assured_crores=1.0, gender="Female"
    )

    for m, f in zip(male_rates, female_rates):
        assert f["annual_premium"] <= m["annual_premium"], f"Female discount missing on {f['plan_name']}"


def test_calculator_zero_percent_gst():
    """Individual term life policies must enforce 0% GST policyholder tax calculation."""
    rates = calculate_policy_rates(
        age=30, smoker=False, sum_assured_crores=1.0, gender="Male"
    )
    for rate in rates:
        assert rate.get("gst_applied_pct", 0) == 0
        assert rate["final_annual_payable"] == rate["annual_premium"]


def test_calculator_return_of_premium_multiplier():
    """TROP (Return of Premium) option must apply a ~2.2x - 2.3x premium multiplier."""
    base_rates = calculate_policy_rates(
        age=28, smoker=False, sum_assured_crores=1.0, gender="Male", return_of_premium=False
    )
    trop_rates = calculate_policy_rates(
        age=28, smoker=False, sum_assured_crores=1.0, gender="Male", return_of_premium=True
    )

    for b, t in zip(base_rates, trop_rates):
        multiplier = t["annual_premium"] / b["annual_premium"]
        assert 2.0 <= multiplier <= 2.5, f"Unexpected TROP multiplier: {multiplier} on {t['plan_name']}"