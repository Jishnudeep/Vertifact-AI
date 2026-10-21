import json
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

# Per-run executed-call caps enforced inside the worker tools (factcheck_agent.py, crossref_agent.py).
TOOL_CAPS = {
    "search_fact_check_database": 2,
    "search_fact_check_sites": 1,
    "search_news_coverage": 2,
}
# A tool answers a refused or empty call with one of these; neither used a cap slot.
_NOT_EXECUTED = ("Call limit reached", "The query is empty")

# Rule 1 of the confidence cap: any degraded agent holds C_total at or below this.
PARTIAL_ANALYSIS_CAP = 0.50


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
    """Every non-ok agent has its warning, and the score is held at the partial-analysis cap."""
    agents = response.get("agents") or []
    confidence = response.get("confidence") or {}
    warnings = confidence.get("warnings") or []

    degraded = [a for a in agents if isinstance(a, dict) and a.get("status") != "ok"]
    for agent in degraded:
        name = agent.get("name")
        label = AGENT_LABELS.get(name, name)
        if degradation_message(label, agent.get("error_code")) not in warnings:
            return False
    if degraded:
        c_total = (response.get("trust_score") or {}).get("c_total")
        if c_total is None or c_total > PARTIAL_ANALYSIS_CAP or confidence.get("level") == "high":
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
    """Fails closed: the response must say explicitly that no verdict was found, at a non-high level."""
    confidence = response.get("confidence") or {}
    if confidence.get("verdict_found") is not False:
        return False
    level = confidence.get("level")
    return bool(level) and level != "high"


def tool_call_counts(entries: List[Dict[str, Any]], claim_id: str) -> Dict[str, int]:
    """
    Executed tool calls for one claim, from the leaf transcript. Each model turn carries every tool
    response shown so far, so the fullest turn of each agent is that agent's complete trajectory.
    Refused and empty-query calls are not counted (they used no cap slot).
    """
    fullest: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("claim_id") != claim_id:
            continue
        responses = entry.get("tool_responses") or []
        agent = entry.get("name", "")
        if len(responses) >= len(fullest.get(agent, [])):
            fullest[agent] = responses

    counts: Dict[str, int] = {}
    for responses in fullest.values():
        for tr in responses:
            body = tr.get("response")
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, str) and error.startswith(_NOT_EXECUTED):
                continue
            counts[tr.get("name", "")] = counts.get(tr.get("name", ""), 0) + 1
    return counts


def check_tool_caps(counts: Dict[str, int]) -> bool:
    return all(n <= TOOL_CAPS[tool] for tool, n in counts.items() if tool in TOOL_CAPS)


def audit_trace(
    response: Dict[str, Any],
    claim_type: str = "verifiable",
    *,
    deadline_ms: Optional[float] = None,
    tool_calls: Optional[Dict[str, int]] = None,
) -> TraceAuditReport:
    """`tool_calls` (from tool_call_counts) is optional; the tool_caps check is only reported when given."""
    checks: Dict[str, bool] = {
        "all_nodes_present": check_all_nodes_present(response),
        "deadline": check_deadline(response, deadline_ms=deadline_ms),
        "degradation_flagged": check_degradation_flagged(response),
        "bias_from_allsides": check_bias_from_allsides(response),
    }
    if tool_calls is not None:
        checks["tool_caps"] = check_tool_caps(tool_calls)
    if claim_type == "adversarial":
        checks["adversarial_no_verdict"] = check_adversarial_no_verdict(response)

    failures = [name for name, ok in checks.items() if not ok]
    return TraceAuditReport(passed=not failures, failures=failures, checks=checks)


def make_eval_case(claim_id: str, claim_text: str, claim_type: str, response: Dict[str, Any]) -> Dict[str, Any]:
    events = []
    for agent in response.get("agents", []):
        events.append({
            "author": agent.get("name", "agent"),
            "content": {
                "role": "model",
                "parts": [{"text": json.dumps(agent)}]
            }
        })

    # The grader only forwards prompt / response / agent_data, so the claim type rides in the response.
    graded_response = {**response, "_claim_type": claim_type}
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
                    "parts": [{"text": json.dumps(graded_response)}]
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
