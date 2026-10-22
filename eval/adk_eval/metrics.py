"""Custom metric evaluation functions for ADK eval (agents-cli eval grade).

These functions conform to the ADK eval custom metric interface:
evaluate(instance: dict) -> dict(score=...)

`agents-cli eval grade` passes `prompt`, `response` (singular: {"role", "parts"}) and `agent_data`;
custom case keys are not forwarded. So the graded verify_claim response travels as JSON text in
`response.parts[0].text` (see eval.trace_eval.make_eval_case) and the claim type in its `_claim_type`.
A response that cannot be parsed scores 0.0 on every metric (fail closed).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
_BACKEND = _BASE / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from eval.trace_eval import (  # noqa: E402
    check_all_nodes_present,
    check_deadline,
    check_degradation_flagged,
    check_bias_from_allsides,
    check_adversarial_no_verdict,
)


def _get_response(instance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The graded verify_claim response, or None when the instance does not carry a parseable one."""
    if isinstance(instance.get("verify_response"), dict):
        return instance["verify_response"]
    candidates = [instance.get("response")]
    candidates += [r.get("response") for r in instance.get("responses") or [] if isinstance(r, dict)]
    for cand in candidates:
        parts = (cand or {}).get("parts") if isinstance(cand, dict) else None
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            try:
                parsed = json.loads(parts[0]["text"])
            except Exception:  # noqa: BLE001
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _score(instance: Dict[str, Any], check) -> Dict[str, Any]:
    resp = _get_response(instance)
    return {"score": 1.0 if resp is not None and check(resp) else 0.0}


def metric_all_nodes_present(instance: Dict[str, Any]) -> Dict[str, Any]:
    return _score(instance, check_all_nodes_present)


def metric_deadline_respected(instance: Dict[str, Any]) -> Dict[str, Any]:
    return _score(instance, check_deadline)


def metric_degradation_flagged(instance: Dict[str, Any]) -> Dict[str, Any]:
    return _score(instance, check_degradation_flagged)


def metric_bias_from_allsides(instance: Dict[str, Any]) -> Dict[str, Any]:
    return _score(instance, check_bias_from_allsides)


def metric_adversarial_no_verdict(instance: Dict[str, Any]) -> Dict[str, Any]:
    resp = _get_response(instance)
    if resp is None:
        return {"score": 0.0}
    claim_type = resp.get("_claim_type") or instance.get("claim_type")
    if claim_type is None:
        return {"score": 0.0}                       # cannot tell whether a verdict was forbidden: fail closed
    if claim_type == "adversarial":
        return {"score": 1.0 if check_adversarial_no_verdict(resp) else 0.0}
    return {"score": 1.0}
