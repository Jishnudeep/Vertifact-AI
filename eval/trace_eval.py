from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.config import settings
from app.services.bias_lookup import get_bias_label
from app.utils.errors import degradation_message

AGENT_LABELS = {
    "factcheck": "Fact-check",
    "crossref": "Cross-reference",
    "linguistic": "Linguistic",
}

EXPECTED_AGENTS = {"factcheck", "crossref", "linguistic"}


@dataclass
class TraceAuditReport:
    passed: bool
    failures: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)


def check_all_nodes_present(response: Dict[str, Any]) -> bool:
    agents = response.get("agents") or []
    names = {a.get("name") for a in agents if isinstance(a, dict)}
    return names == EXPECTED_AGENTS


def check_deadline(response: Dict[str, Any], deadline_ms: Optional[float] = None) -> bool:
    if deadline_ms is None:
        deadline_ms = settings.request_deadline_s * 1000
    processing_time_ms = response.get("processing_time_ms")
    if processing_time_ms is None:
        return False
    return processing_time_ms <= deadline_ms


def check_degradation_flagged(response: Dict[str, Any]) -> bool:
    agents = response.get("agents") or []
    confidence = response.get("confidence") or {}
    warnings = confidence.get("warnings") or []

    for agent in agents:
        if agent.get("status") == "degraded":
            name = agent.get("name")
            label = AGENT_LABELS.get(name, name)
            error_code = agent.get("error_code")
            expected_msg = degradation_message(label, error_code)
            if expected_msg not in warnings:
                return False
    return True


def check_bias_from_allsides(response: Dict[str, Any]) -> bool:
    sources = response.get("sources") or []
    for src in sources:
        if not isinstance(src, dict):
            return False
        domain = src.get("domain")
        if not domain:
            return False
        bias_label = src.get("bias_label")
        expected_bias = get_bias_label(domain)
        if bias_label != expected_bias:
            return False
    return True


def check_adversarial_no_verdict(response: Dict[str, Any]) -> bool:
    confidence = response.get("confidence") or {}
    verdict_found = confidence.get("verdict_found", False)
    level = confidence.get("level", "")
    if verdict_found is not False or level == "high":
        return False
    return True


def audit_trace(
    response: Dict[str, Any],
    claim_type: str = "verifiable",
    *,
    deadline_ms: Optional[float] = None,
) -> TraceAuditReport:
    checks: Dict[str, bool] = {}
    failures: List[str] = []

    # 1. all_nodes_present
    nodes_ok = check_all_nodes_present(response)
    checks["all_nodes_present"] = nodes_ok
    if not nodes_ok:
        failures.append("all_nodes_present")

    # 2. deadline
    deadline_ok = check_deadline(response, deadline_ms=deadline_ms)
    checks["deadline"] = deadline_ok
    if not deadline_ok:
        failures.append("deadline")

    # 3. degradation_flagged
    degraded_ok = check_degradation_flagged(response)
    checks["degradation_flagged"] = degraded_ok
    if not degraded_ok:
        failures.append("degradation_flagged")

    # 4. bias_from_allsides
    bias_ok = check_bias_from_allsides(response)
    checks["bias_from_allsides"] = bias_ok
    if not bias_ok:
        failures.append("bias_from_allsides")

    # 5. adversarial_no_verdict (adversarial only)
    if claim_type == "adversarial":
        adv_ok = check_adversarial_no_verdict(response)
        checks["adversarial_no_verdict"] = adv_ok
        if not adv_ok:
            failures.append("adversarial_no_verdict")

    return TraceAuditReport(
        passed=len(failures) == 0,
        failures=failures,
        checks=checks,
    )


def make_eval_case(claim_id: str, claim_text: str, claim_type: str, response: Dict[str, Any]) -> Dict[str, Any]:
    import json
    events = []
    for agent in response.get("agents", []):
        events.append({
            "author": agent.get("name", "agent"),
            "content": {
                "role": "model",
                "parts": [{"text": json.dumps(agent)}]
            }
        })

    return {
        "eval_case_id": claim_id,
        "prompt": {
            "role": "user",
            "parts": [{"text": claim_text}]
        },
        "responses": [
            {
                "response": {
                    "role": "model",
                    "parts": [{"text": json.dumps(response)}]
                }
            }
        ],
        "agent_data": {
            "turns": [
                {
                    "turn_index": 0,
                    "events": events
                }
            ]
        },
        "claim_type": claim_type,
        "verify_response": response,
    }
