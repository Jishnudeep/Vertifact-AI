from typing import List, Dict, Any
from app.services.bias_lookup import UNRATED_BRIDGING_DOMAINS, VALID_BIAS_RATINGS
from app.utils.text_processing import extract_domain

# Below this many inspected sources a clean result is not evidence of anything (R8.3).
MIN_SOURCES_FOR_PASS = 3

def evaluate_bias_leakage(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check bias leakage across a list of citation source dictionaries:
    - 0% tolerance: no bridging domain may have a political rating.
    - No unknown domain may have an invented bias label.
    - All bias labels must be in VALID_BIAS_RATINGS or 'Unrated'.

    A CLEAN result over fewer than MIN_SOURCES_FOR_PASS sources is reported as
    "not_evaluated" (`passed=None`), never True — a zero-tolerance guardrail that inspected
    one or two citations has not been exercised, and reporting it as green is the false
    green R7.B3/R8.3 flagged.

    A DETECTED violation is always reported (`passed=False`) regardless of sample size:
    suppressing an observed zero-tolerance breach because the sample was small would be
    strictly worse than the false green this rule exists to prevent.
    """
    leaked_count = 0
    total_checked = len(sources)
    violations = []

    if total_checked == 0:
        return {
            "total_sources": 0,
            "leaked_count": 0,
            "leakage_rate": None,
            "passed": None,
            "status": "not_evaluated",
            "violations": [],
        }

    for src in sources:
        url = src.get("url", "")
        domain = src.get("domain", "")
        if not domain and url:
            domain = extract_domain(url)
            
        bias_label = src.get("bias_label", "")
        
        # Check bridging org leakage
        if domain in UNRATED_BRIDGING_DOMAINS and bias_label != "Unrated":
            leaked_count += 1
            violations.append(f"Bridging domain '{domain}' has bias '{bias_label}' (expected 'Unrated')")
            
        # Check unknown / invalid ratings
        if bias_label not in VALID_BIAS_RATINGS and bias_label != "Unrated":
            leaked_count += 1
            violations.append(f"Domain '{domain}' has invalid bias label '{bias_label}'")

    leakage_rate = round(leaked_count / total_checked, 4)

    if leaked_count > 0:
        passed: bool | None = False
        status = "evaluated"
    elif total_checked < MIN_SOURCES_FOR_PASS:
        # Clean, but on too small a sample to call the guardrail exercised.
        passed = None
        status = "not_evaluated"
        violations.append(
            f"Only {total_checked} source(s) inspected (< {MIN_SOURCES_FOR_PASS}); "
            f"bias-leakage guardrail was not meaningfully exercised."
        )
    else:
        passed = True
        status = "evaluated"

    return {
        "total_sources": total_checked,
        "leaked_count": leaked_count,
        "leakage_rate": leakage_rate,
        "passed": passed,
        "status": status,
        "violations": violations
    }
