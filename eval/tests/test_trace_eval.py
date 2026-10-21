import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from app.agents.crossref_agent import CrossRefResult
from app.agents.factcheck_agent import make_neutral_factcheck
from app.agents.linguistic_agent import LinguisticResult
from app.agents.orchestrator import verify_claim
from app.config import settings
from app.utils.errors import degradation_message
from eval.trace_eval import audit_trace
from tests.agent_patches import patched_agents


def _resp(agents=None, warnings=None, sources=None, ms=1200, verdict_found=False, level="medium", c_total=0.4):
    agents = agents or [
        {"name": n, "status": "ok", "duration_ms": 300, "error_code": None}
        for n in ("factcheck", "crossref", "linguistic")
    ]
    return {
        "agents": agents,
        "processing_time_ms": ms,
        "sources": sources or [],
        "trust_score": {"c_total": c_total},
        "confidence": {
            "warnings": warnings or [],
            "verdict_found": verdict_found,
            "level": level,
        },
    }


def test_clean_trace_passes():
    assert audit_trace(_resp(), claim_type="verifiable").passed


def test_missing_node_fails():
    r = _resp(agents=[{"name": "factcheck", "status": "ok", "duration_ms": 1, "error_code": None}])
    assert "all_nodes_present" in audit_trace(r, "verifiable").failures


def test_degraded_agent_without_its_warning_fails():
    agents = [
        {"name": "factcheck", "status": "ok", "duration_ms": 1, "error_code": None},
        {"name": "crossref", "status": "degraded", "duration_ms": 8000, "error_code": "TIMEOUT"},
        {"name": "linguistic", "status": "ok", "duration_ms": 1, "error_code": None},
    ]
    assert "degradation_flagged" in audit_trace(_resp(agents=agents), "verifiable").failures
    ok = _resp(agents=agents, warnings=[degradation_message("Cross-reference", "TIMEOUT")])
    assert audit_trace(ok, "verifiable").passed


def test_bias_label_must_equal_allsides_lookup():
    src = {"url": "https://example-unrated.org/x", "domain": "example-unrated.org", "bias_label": "Center"}
    assert "bias_from_allsides" in audit_trace(_resp(sources=[src]), "verifiable").failures

    # Unrated should pass for unknown domain
    src_valid = {"url": "https://example-unrated.org/x", "domain": "example-unrated.org", "bias_label": "Unrated"}
    assert audit_trace(_resp(sources=[src_valid]), "verifiable").passed


def test_source_without_domain_is_unverifiable_not_passing():
    src = {"url": "https://vertexaisearch.cloud.google.com/redirect/abc", "bias_label": "Unrated"}
    assert "bias_from_allsides" in audit_trace(_resp(sources=[src]), "verifiable").failures


def test_deadline_exceeded_fails():
    # settings.request_deadline_s defaults to 35s in code (or custom in settings)
    exceeded_ms = (settings.request_deadline_s * 1000) + 100
    assert "deadline" in audit_trace(_resp(ms=exceeded_ms), "verifiable").failures


def test_adversarial_with_verdict_fails():
    assert "adversarial_no_verdict" in audit_trace(_resp(verdict_found=True), "adversarial").failures
    assert "adversarial_no_verdict" in audit_trace(_resp(verdict_found=False, level="high"), "adversarial").failures
    assert audit_trace(_resp(verdict_found=False, level="low"), "adversarial").passed


@pytest.mark.anyio
async def test_fault_injection_crossref_search_raises(monkeypatch):
    """When crossref raises, verify_claim completes, crossref is degraded, and audit_trace passes."""
    fc = make_neutral_factcheck(degraded=False)
    ling = LinguisticResult(
        clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
        informational_density=1.0, b_ling_score=0.0, degraded=False
    )

    async def cr_fail(claim):
        raise RuntimeError("simulated search provider error")

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patched_agents(fc=AsyncMock(return_value=fc), cr=cr_fail, ling=AsyncMock(return_value=ling)):
        result = await verify_claim("The Earth orbits the Sun", bypass_cache=True)

    cr_agent = next((a for a in result["agents"] if a["name"] == "crossref"), None)
    assert cr_agent is not None
    assert cr_agent["status"] == "degraded"
    assert cr_agent["error_code"] is not None

    fc_agent = next((a for a in result["agents"] if a["name"] == "factcheck"), None)
    ling_agent = next((a for a in result["agents"] if a["name"] == "linguistic"), None)
    assert fc_agent["status"] == "ok"
    assert ling_agent["status"] == "ok"

    report = audit_trace(result, claim_type="verifiable")
    assert report.passed, f"Trace audit failed: {report.failures}"


@pytest.mark.anyio
async def test_fault_injection_crossref_timeout(monkeypatch):
    """When crossref sleeps past its budget, verify_claim completes within deadline and audit_trace passes."""
    from app.agents import crossref_agent
    monkeypatch.setattr(settings, "agent_timeout_s", 0.05)
    monkeypatch.setattr(settings, "request_deadline_s", 0.5)
    monkeypatch.setattr(crossref_agent.crossref_node, "timeout", 0.05)

    fc = make_neutral_factcheck(degraded=False)
    ling = LinguisticResult(
        clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
        informational_density=1.0, b_ling_score=0.0, degraded=False
    )

    async def cr_slow(claim):
        await asyncio.sleep(2.0)
        return CrossRefResult(
            entities_extracted=[], sources_found=[],
            coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
            v_consensus_score=0.0, degraded=False
        )

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patched_agents(fc=AsyncMock(return_value=fc), cr=cr_slow, ling=AsyncMock(return_value=ling)):
        result = await verify_claim("The Earth orbits the Sun", bypass_cache=True)

    cr_agent = next((a for a in result["agents"] if a["name"] == "crossref"), None)
    assert cr_agent is not None
    assert cr_agent["status"] == "degraded"
    assert cr_agent["error_code"] == "TIMEOUT"

    report = audit_trace(result, claim_type="verifiable")
    assert report.passed, f"Trace audit failed: {report.failures}"
    assert result["processing_time_ms"] <= 500


def test_make_eval_case_schema():
    from eval.trace_eval import make_eval_case
    case = make_eval_case("claim-1", "Test claim", "verifiable", _resp())
    assert case["eval_case_id"] == "claim-1"
    assert case["prompt"]["role"] == "user"
    assert case["prompt"]["parts"][0]["text"] == "Test claim"
    assert len(case["responses"]) == 1
    assert case["responses"][0]["response"]["role"] == "model"
    assert "agent_data" in case
    assert len(case["agent_data"]["turns"]) == 1
    turn = case["agent_data"]["turns"][0]
    assert len(turn["events"]) == 3
    authors = {ev["author"] for ev in turn["events"]}
    assert authors == {"factcheck", "crossref", "linguistic"}


def test_adk_eval_metrics_functions():
    from eval.trace_eval import make_eval_case
    from eval.adk_eval.metrics import (
        metric_all_nodes_present,
        metric_deadline_respected,
        metric_degradation_flagged,
        metric_bias_from_allsides,
        metric_adversarial_no_verdict,
    )
    case = make_eval_case("claim-1", "Test claim", "verifiable", _resp())
    assert metric_all_nodes_present(case)["score"] == 1.0
    assert metric_deadline_respected(case)["score"] == 1.0
    assert metric_degradation_flagged(case)["score"] == 1.0
    assert metric_bias_from_allsides(case)["score"] == 1.0
    assert metric_adversarial_no_verdict(case)["score"] == 1.0

    # Negative case for nodes
    bad_resp = dict(_resp())
    bad_resp["agents"] = [a for a in bad_resp["agents"] if a["name"] != "crossref"]
    bad_case = make_eval_case("claim-2", "Test claim", "verifiable", bad_resp)
    assert metric_all_nodes_present(bad_case)["score"] == 0.0


def test_adk_eval_export():
    import json
    from pathlib import Path
    from eval.trace_eval import audit_trace
    traces_dir = Path("eval/results/traces")
    if not traces_dir.exists():
        pytest.skip("No traces generated yet")
    trace_files = list(traces_dir.glob("*.json"))
    if not trace_files:
        pytest.skip("No trace files in traces dir")
    for f in trace_files:
        data = json.loads(f.read_text("utf-8"))
        assert "eval_cases" in data
        for case in data["eval_cases"]:
            assert "eval_case_id" in case
            assert "prompt" in case
            assert "responses" in case
            assert "verify_response" in case
            report = audit_trace(case["verify_response"], claim_type=case.get("claim_type", "verifiable"))
            assert report.passed, f"Trace audit failed for {case['eval_case_id']}: {report.failures}"



def _real_crossref_result():
    """A CrossRefResult built by the real finalize(): one read page and one unread rated outlet."""
    from app.agents import crossref_agent as cr
    from app.services.bias_lookup import get_bias_label, load_bias_ratings

    load_bias_ratings()
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"

    def record(domain, url, stance):
        return {"domain": domain, "url": url, "title": "T", "publisher": domain,
                "bias_label": get_bias_label(domain), "stance": stance,
                "stance_confidence": 0.9 if stance else None, "is_claimreview": False}

    cov = cr.Coverage(batches=[[record("reuters.com", "https://www.reuters.com/a", "supports"),
                                record("foxnews.com", redirect, None)]], calls=1, answered=1)
    return cr.finalize([], cov)


async def _verify_with_real_crossref_sources():
    cr_result = _real_crossref_result()
    fc = make_neutral_factcheck(degraded=False)
    ling = LinguisticResult(
        clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
        informational_density=1.0, b_ling_score=0.0, degraded=False
    )
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patched_agents(fc=AsyncMock(return_value=fc), cr=AsyncMock(return_value=cr_result),
                        ling=AsyncMock(return_value=ling)):
        return await verify_claim("NASA announces Mars landing", bypass_cache=True)


@pytest.mark.anyio
async def test_real_crossref_sources_pass_the_bias_audit_even_when_unread():
    result = await _verify_with_real_crossref_sources()
    domains = {s["domain"] for s in result["sources"]}
    assert domains == {"reuters.com", "foxnews.com"}, "an unread source must keep its grounding domain"
    report = audit_trace(result, claim_type="verifiable")
    assert report.checks["bias_from_allsides"], report.failures


@pytest.mark.anyio
async def test_real_response_with_a_tampered_bias_label_fails_the_audit():
    result = await _verify_with_real_crossref_sources()
    fox = next(s for s in result["sources"] if s["domain"] == "foxnews.com")
    fox["bias_label"] = "Left" if fox["bias_label"] != "Left" else "Right"
    assert "bias_from_allsides" in audit_trace(result, claim_type="verifiable").failures
