from typing import Dict, Any

# Actuarial base annual premium per 1 Crore sum assured (30-year term, non-smoker male)
# Indexed by age brackets
BASE_RATE_TABLE = {
    "HDFC Life Click 2 Protect Super": {
        "base_per_cr": {
            (18, 25): 8500,
            (26, 30): 10200,
            (31, 35): 12800,
            (36, 40): 16500,
            (41, 45): 22500,
            (46, 50): 31000,
            (51, 60): 46000,
        },
        "smoker_multiplier": 1.65,
        "female_discount": 0.90,
        "trop_multiplier": 2.25,
        "ci_rider_per_lakh": 35,
        "adb_rider_per_lakh": 18,
    },
    "Max Life Smart Secure Plus": {
        "base_per_cr": {
            (18, 25): 8200,
            (26, 30): 9800,
            (31, 35): 12200,
            (36, 40): 15900,
            (41, 45): 21800,
            (46, 50): 29800,
            (51, 60): 44500,
        },
        "smoker_multiplier": 1.60,
        "female_discount": 0.88,
        "trop_multiplier": 2.20,
        "ci_rider_per_lakh": 32,
        "adb_rider_per_lakh": 16,
    },
    "Tata AIA Sampoorna Raksha Supreme": {
        "base_per_cr": {
            (18, 25): 8400,
            (26, 30): 10000,
            (31, 35): 12500,
            (36, 40): 16200,
            (41, 45): 22100,
            (46, 50): 30500,
            (51, 60): 45500,
        },
        "smoker_multiplier": 1.62,
        "female_discount": 0.90,
        "trop_multiplier": 2.30,
        "ci_rider_per_lakh": 34,
        "adb_rider_per_lakh": 17,
    },
    "ICICI Prudential iProtect Smart": {
        "base_per_cr": {
            (18, 25): 8600,
            (26, 30): 10400,
            (31, 35): 13000,
            (36, 40): 16800,
            (41, 45): 22900,
            (46, 50): 31500,
            (51, 60): 47000,
        },
        "smoker_multiplier": 1.68,
        "female_discount": 0.90,
        "trop_multiplier": 2.35,
        "ci_rider_per_lakh": 36,
        "adb_rider_per_lakh": 18,
    },
}

COVER_TO_NUMERIC = {
    "50 Lakhs": 5000000,
    "75 Lakhs": 7500000,
    "1 Crore": 10000000,
    "1.5 Crore": 15000000,
    "2 Crore": 20000000,
    "2.5 Crore": 25000000,
    "3 Crore": 30000000,
    "5 Crore+": 50000000,
}

def calculate_exact_premium(
    policy_name: str,
    age: int,
    gender: str = "Male",
    is_smoker: bool = False,
    sum_assured_str: str = "1 Crore",
    is_trop: bool = False,
    critical_illness_cover: int = 0,
    accidental_death_cover: int = 0
) -> Dict[str, Any]:
    """
    Computes exact indicative premium figures including base rates,
    smoker loading, gender discount, riders, and zero individual GST.
    """
    plan_data = BASE_RATE_TABLE.get(policy_name, BASE_RATE_TABLE["Max Life Smart Secure Plus"])
    cover_numeric = COVER_TO_NUMERIC.get(sum_assured_str, 10000000)
    cover_units_cr = cover_numeric / 10000000.0

    # 1. Match Age Bracket
    base_rate_per_cr = 10000
    for (min_age, max_age), rate in plan_data["base_per_cr"].items():
        if min_age <= age <= max_age:
            base_rate_per_cr = rate
            break
    if age > 60:
        base_rate_per_cr = 55000

    # 2. Apply Demographic Multipliers
    base_annual = base_rate_per_cr * cover_units_cr
    if is_smoker:
        base_annual *= plan_data["smoker_multiplier"]
    if gender.lower() == "female":
        base_annual *= plan_data["female_discount"]
    if is_trop:
        base_annual *= plan_data["trop_multiplier"]

    # 3. Rider Charges
    ci_units = critical_illness_cover / 100000.0
    adb_units = accidental_death_cover / 100000.0
    ci_premium = ci_units * plan_data["ci_rider_per_lakh"]
    adb_premium = adb_units * plan_data["adb_rider_per_lakh"]

    total_net_annual = round(base_annual + ci_premium + adb_premium)
    monthly_installment = round(total_net_annual / 12)

    return {
        "policy_name": policy_name,
        "base_annual": round(base_annual),
        "ci_rider_cost": round(ci_premium),
        "adb_rider_cost": round(adb_premium),
        "gst_rate_percent": 0,  # 0% GST on individual life insurance
        "gst_amount": 0,
        "net_annual_premium": total_net_annual,
        "monthly_premium": monthly_installment,
        "sum_assured": sum_assured_str,
        "is_trop": is_trop
    }

def calculate_benchmark_matrix(age: int, gender: str, is_smoker: bool, sum_assured_str: str) -> list:
    """Calculates side-by-side pricing across all top 4 plans for direct benchmarking."""
    matrix = []
    for plan in BASE_RATE_TABLE.keys():
        pure_term = calculate_exact_premium(plan, age, gender, is_smoker, sum_assured_str, is_trop=False)
        trop_variant = calculate_exact_premium(plan, age, gender, is_smoker, sum_assured_str, is_trop=True)
        matrix.append({
            "plan": plan,
            "annual": f"₹{pure_term['net_annual_premium']:,}",
            "monthly": f"₹{pure_term['monthly_premium']:,}",
            "trop_annual": f"₹{trop_variant['net_annual_premium']:,}",
            "trop_monthly": f"₹{trop_variant['monthly_premium']:,}"
        })
    return matrix