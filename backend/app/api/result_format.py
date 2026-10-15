from typing import Dict, Any, List, Optional


def format_stored_result(row: Dict[str, Any], claim_text: str, *, cached: bool) -> Dict[str, Any]:
    """
    Single canonical mapper from a stored `results` row (joined with claim text) to a VerifyResponse dict.
    Used by both GET /api/results/{id} (cached=False) and cache hits (cached=True).
    """
    agent_outputs = row.get("agent_outputs") or {}
    scoring_dict = agent_outputs.get("scoring") or {}

    # Extract agents list from agent_outputs if present
    agents: List[Dict[str, Any]] = []
    if agent_outputs and any(k in agent_outputs for k in ("factcheck", "crossref", "linguistic")):
        for name in ("factcheck", "crossref", "linguistic"):
            agent_data = agent_outputs.get(name) or {}
            degraded = agent_data.get("degraded", False)
            status = "degraded" if degraded else "ok"
            duration_ms = agent_data.get("duration_ms")
            err_details = agent_data.get("error_details") or {}
            error_code = err_details.get("code") if isinstance(err_details, dict) else None
            agents.append({
                "name": name,
                "status": status,
                "duration_ms": duration_ms,
                "error_code": error_code,
            })

    # Search suggestions can be in row or inside agent_outputs
    search_suggestions_html = (
        row.get("search_suggestions_html")
        or agent_outputs.get("search_suggestions_html")
    )

    # Legacy warnings: scored before stance-aware consensus (evidence_status is None)
    legacy_warning = (
        [] if row.get("evidence_status")
        else ["Legacy result: scored before stance-aware consensus; treat corroboration as unverified"]
    )
    warnings = (row.get("warnings") or []) + legacy_warning

    # Linguistic profile default fallback
    linguistic_profile = row.get("linguistic_profile") or {
        "clickbait_syntax": 0.0,
        "emotional_framing": 0.0,
        "sensationalism": 0.0,
        "informational_density": 0.5,
    }

    # Sources pass through with found_by left as stored
    sources = row.get("source_citations") or []

    # Claim text fallback
    claim = claim_text or row.get("extracted_claim") or "Historical Claim"

    result_id = str(row.get("id") or row.get("claim_id") or "")

    return {
        "id": result_id,
        "claim": claim,
        "trust_score": {
            "c_total": row.get("c_total"),
            "s_fact": row.get("s_fact"),
            "b_ling": row.get("b_ling"),
            "v_consensus": row.get("v_consensus"),
            "v_refute": row.get("v_refute") or 0.0,
            "weights": row.get("weights") or {"w1": 0.5, "w2": 0.2, "w3": 0.3},
            "raw_c_total": scoring_dict.get("raw_c_total"),
            "cap_rule": scoring_dict.get("cap_rule"),
        },
        "confidence": {
            "level": row.get("confidence_level"),
            "warnings": warnings,
            "verdict_found": row.get("confidence_level") == "high",
            "evidence_status": row.get("evidence_status") or "unverified",
        },
        "linguistic_profile": linguistic_profile,
        "sources": sources,
        "processing_time_ms": row.get("processing_time_ms") or 0,
        "cached": cached,
        "search_suggestions_html": search_suggestions_html,
        "agents": agents,
        "created_at": row.get("created_at"),
    }
