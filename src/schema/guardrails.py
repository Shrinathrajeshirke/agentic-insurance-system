from typing import Dict, Any, Tuple, List
from src.logger import logger


def evaluate_underwriting_guardrails(profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Evaluates an applicant's profile against non-negotiable IRDAI life underwriting
    regulations, entry age criteria (18-65), and HLV multiplier income ceilings.

    Returns:
        Tuple[bool, List[str]]:
            - bool: True if hard non-negotiable criteria pass; False if critical boundary fails.
            - List[str]: List of raised underwriting flags, warnings, or critical rejection reasons.
    """
    flags: List[str] = []

    if not profile:
        return True, flags

    # 1. Age Validation & Boundaries
    age = profile.get("age")
    if age is None:
        flags.append("CRITICAL: Applicant age is missing or not provided.")
        return False, flags

    try:
        age_int = int(age)
    except (ValueError, TypeError):
        flags.append(f"CRITICAL: Invalid age format provided ('{age}'). Must be an integer.")
        return False, flags

    if age_int < 18:
        flags.append(f"CRITICAL: Age {age_int} is below the statutory IRDAI minimum entry age of 18 years.")
        return False, flags

    if age_int > 65:
        flags.append(f"CRITICAL: Age {age_int} exceeds standard term life maximum entry age of 65 years.")
        return False, flags

    # 2. Parse Income and Desired Sum Assured for Human Life Value (HLV) Validation
    income_val = profile.get("annual_income")
    cover_val = profile.get("desired_sum_assured")

    # Map categorical income brackets to approximate numeric base amounts (in Lakhs)
    income_lakhs: float = 0.0
    if isinstance(income_val, (int, float)):
        income_lakhs = float(income_val) / 100000.0
    elif isinstance(income_val, str):
        income_lower = income_val.lower()
        if "3 - 6" in income_lower or "4 lakh" in income_lower:
            income_lakhs = 4.0
        elif "6 - 10" in income_lower:
            income_lakhs = 8.0
        elif "10 - 15" in income_lower:
            income_lakhs = 12.0
        elif "15 - 25" in income_lower:
            income_lakhs = 20.0
        elif "25+" in income_lower:
            income_lakhs = 30.0

    # Map cover strings to numeric Crores
    cover_crores: float = 0.0
    if isinstance(cover_val, (int, float)):
        cover_crores = float(cover_val) / 10000000.0
    elif isinstance(cover_val, str):
        cover_lower = cover_val.lower()
        if "50 lakh" in cover_lower:
            cover_crores = 0.5
        elif "75 lakh" in cover_lower:
            cover_crores = 0.75
        elif "1.5 crore" in cover_lower or "1.5cr" in cover_lower:
            cover_crores = 1.5
        elif "2.5 crore" in cover_lower or "2.5cr" in cover_lower:
            cover_crores = 2.5
        elif "2 crore" in cover_lower or "2cr" in cover_lower:
            cover_crores = 2.0
        elif "3 crore" in cover_lower or "3cr" in cover_lower:
            cover_crores = 3.0
        elif "5 crore" in cover_lower or "5cr" in cover_lower:
            cover_crores = 5.0
        elif "1 crore" in cover_lower or "1cr" in cover_lower:
            cover_crores = 1.0

    # 3. Determine Statutory HLV Multiplier Ceiling by Age
    # Age <= 35: max 25x annual income
    # Age 36 - 45: max 20x annual income
    # Age 46 - 55: max 15x annual income
    # Age 56 - 65: max 10x annual income
    if age_int <= 35:
        max_multiplier = 25
    elif age_int <= 45:
        max_multiplier = 20
    elif age_int <= 55:
        max_multiplier = 15
    else:
        max_multiplier = 10

    if income_lakhs > 0 and cover_crores > 0:
        max_allowed_cover_cr = (income_lakhs * max_multiplier) / 100.0
        if cover_crores > max_allowed_cover_cr:
            flags.append(
                f"UNDERWRITING ALERT: Desired sum assured (₹{cover_crores:.2f} Cr) exceeds statutory HLV multiplier "
                f"({max_multiplier}x annual income, max ₹{max_allowed_cover_cr:.2f} Cr). Mandatory 3-year ITR-V, Form 16, "
                f"and financial justification required."
            )

    # 4. Tobacco / Smoker Advisory Flag
    if profile.get("is_smoker", False):
        flags.append("UNDERWRITING NOTE: Smoker loading applies (~40% to 50% premium increase over standard non-smoker rates).")

    return True, flags