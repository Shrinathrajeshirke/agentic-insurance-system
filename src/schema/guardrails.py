from typing import Dict, Any, Tuple

INCOME_MAPPING = {
    "Below 3 Lakhs": 250000,
    "3 - 6 Lakhs": 500000,
    "6 - 10 Lakhs": 800000,
    "10 - 15 Lakhs": 1250000,
    "15 - 25 Lakhs": 2000000,
    "25+ Lakhs": 3000000
}

COVER_MAPPING = {
    "50 Lakhs": 5000000,
    "75 Lakhs": 7500000,
    "1 Crore": 10000000,
    "1.5 Crore": 15000000,
    "2 Crore": 20000000,
    "2.5 Crore": 25000000,
    "3 Crore": 30000000,
    "5 Crore+": 50000000
}

def evaluate_underwriting_guardrails(profile_dict: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Evaluates applicant profile against non-negotiable IRDAI life underwriting rules.
    Returns (is_acceptable, list_of_flags_or_warnings).
    """
    flags = []
    age = profile_dict.get("age", 28)
    income_str = profile_dict.get("annual_income", "6 - 10 Lakhs")
    cover_str = profile_dict.get("desired_sum_assured", "1 Crore")
    term = profile_dict.get("policy_term_years", 30)

    # 1. Entry Age Hard Boundary
    if age < 18:
        flags.append("CRITICAL: Minimum entry age for term life insurance in India is 18 years.")
    elif age > 65:
        flags.append("CRITICAL: Maximum entry age for standard pure term plans is 65 years. Senior citizen or specialized whole-life plans apply.")

    # 2. Maximum Maturity Age Limit (IRDAI ceiling typically 75 - 85 years for regular term)
    if (age + term) > 85:
        max_allowed_term = max(5, 85 - age)
        flags.append(f"ADJUSTMENT: Requested maturity age ({age + term}) exceeds typical 85-year ceiling. Maximum suggested policy term is {max_allowed_term} years.")

    # 3. Human Life Value (HLV) Multiplier Check
    numeric_income = INCOME_MAPPING.get(income_str, 800000)
    numeric_cover = COVER_MAPPING.get(cover_str, 10000000)

    if age <= 35:
        max_multiplier = 25
    elif age <= 45:
        max_multiplier = 20
    elif age <= 55:
        max_multiplier = 15
    else:
        max_multiplier = 10

    max_eligible_cover = numeric_income * max_multiplier

    if numeric_cover > max_eligible_cover:
        eligible_in_crores = max_eligible_cover / 10000000
        flags.append(
            f"UNDERWRITING WARNING: Requested cover (₹{numeric_cover/10000000:.2f} Cr) exceeds standard {max_multiplier}x HLV limit "
            f"(₹{eligible_in_crores:.2f} Cr) for income bracket '{income_str}'. Insurers will mandate 3-year audited ITR-V, Form 16, and financial justification."
        )

    # 4. Income Threshold for High Sum Assured (>= 1 Crore)
    if numeric_cover >= 10000000 and numeric_income < 300000:
        flags.append("FINANCIAL GATE: Pure term cover of ₹1 Cr+ is generally restricted for annual income below ₹3 Lakhs under standard underwriting guidelines.")

    is_valid = not any("CRITICAL:" in f for f in flags)
    return is_valid, flags