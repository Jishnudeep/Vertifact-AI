"""Custom metric evaluation functions for ADK eval (agents-cli eval grade).

These functions conform to the ADK eval custom metric interface:
evaluate(instance: dict) -> dict(score=...)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict

_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
_BACKEND = _BASE / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from eval.trace_eval import (
    check_all_nodes_present,
    check_deadline,
    check_degradation_flagged,
    check_bias_from_allsides,
    check_adversarial_no_verdict,
)


def _get_response(instance: Dict[str, Any]) -> Dict[str, Any]:
    if "verify_response" in instance and isinstance(instance["verify_response"], dict):
        return instance["verify_response"]
    responses = instance.get("responses", [])
    if responses:
        cand = responses[0].get("response", {})
        parts = cand.get("parts", [{}])
        if parts and "text" in parts[0]:
            try:
                return json.loads(parts[0]["text"])
            except Exception:
                return {}
    return {}


def metric_all_nodes_present(instance: Dict[str, Any]) -> Dict[str, Any]:
    resp = _get_response(instance)
    passed = check_all_nodes_present(resp)
    return {"score": 1.0 if passed else 0.0}


def metric_deadline_respected(instance: Dict[str, Any]) -> Dict[str, Any]:
    resp = _get_response(instance)
    passed = check_deadline(resp)
    return {"score": 1.0 if passed else 0.0}


def metric_degradation_flagged(instance: Dict[str, Any]) -> Dict[str, Any]:
    resp = _get_response(instance)
    passed = check_degradation_flagged(resp)
    return {"score": 1.0 if passed else 0.0}


def metric_bias_from_allsides(instance: Dict[str, Any]) -> Dict[str, Any]:
    resp = _get_response(instance)
    passed = check_bias_from_allsides(resp)
    return {"score": 1.0 if passed else 0.0}


def metric_adversarial_no_verdict(instance: Dict[str, Any]) -> Dict[str, Any]:
    resp = _get_response(instance)
    claim_type = instance.get("claim_type", "verifiable")
    if claim_type == "adversarial":
        passed = check_adversarial_no_verdict(resp)
        return {"score": 1.0 if passed else 0.0}
    return {"score": 1.0}
