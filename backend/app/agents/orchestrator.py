import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from app.utils.validators import sanitize_text, validate_input_length, is_verifiable_claim
from app.services.embedding import generate_embedding
from app.services.cache import lookup_cache, insert_cache
from app.scoring.engine import ScoringEngine
from app.agents import factcheck_agent, crossref_agent, linguistic_agent
from app.services.bias_lookup import get_bias_label
from app.utils.errors import classify_provider_error
from app.utils.text_processing import extract_domain

logger = logging.getLogger(__name__)

class InvalidInputError(ValueError):
    """Raised when input text is empty or exceeds the character limit (HTTP 400)."""
    pass

class NonVerifiableClaimError(ValueError):
    """Raised when input text is classified as an opinion or question (HTTP 422)."""
    pass

class AllAgentsFailedError(RuntimeError):
    """Raised when all verification agents fail or time out (HTTP 500)."""
    pass

async def verify_claim(raw_input: str, input_type: str = "text", bypass_cache: bool = False) -> Dict[str, Any]:
    """
    Main orchestration pipeline for claim verification:
    1. Sanitize input (strip HTML/scripts, URL tracking parameters, whitespace).
    2. Validate input constraints (length <= 2000 chars).
    3. Validate claim verifiability (reject subjective opinions & questions -> 422).
    4. Generate embedding and query semantic cache (short-circuit on cosine similarity >= 0.95, unless bypass_cache=True).
    5. On cache miss, dispatch factcheck, crossref, and linguistic agents in parallel (8s timeouts).
    6. Guardrail: If all agents fail, raise AllAgentsFailedError (never fabricate a score).
    7. Compute deterministic composite trust score and apply confidence capping.
    8. Consolidate sources (fact_check and corroborating).
    9. Persist result and claim to Supabase cache (with confidence warnings; skip on low/insufficient confidence or degraded workers).
    10. Return full structured verification response.
    """
    start_time = time.time()
    
    # 1. Sanitize input
    cleaned_text = sanitize_text(raw_input)
    
    # 2. Validate length
    valid_len, len_error = validate_input_length(cleaned_text)
    if not valid_len:
        raise InvalidInputError(len_error)
        
    # 3. Validate claim verifiability
    verifiable, ver_error = is_verifiable_claim(cleaned_text)
    if not verifiable:
        raise NonVerifiableClaimError(ver_error)

    # 4. Generate embedding
    embedding = await asyncio.to_thread(generate_embedding, cleaned_text)

    # 5. Semantic cache lookup (skip if bypass_cache=True)
    if not bypass_cache:
        cached = await lookup_cache(embedding, similarity_threshold=0.95)
        if cached:
            logger.info(f"Returning cached verification result for: {cleaned_text}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "id": str(cached.get("claim_id") or cached.get("id")),
                "claim": cached.get("extracted_claim") or cleaned_text,
                "trust_score": {
                    "c_total": cached.get("c_total"),
                    "s_fact": cached.get("s_fact"),
                    "b_ling": cached.get("b_ling"),
                    "v_consensus": cached.get("v_consensus"),
                    "v_refute": cached.get("v_refute") or 0.0,
                    "weights": cached.get("weights") or {"w1": 0.50, "w2": 0.20, "w3": 0.30}
                },
                "confidence": {
                    "level": cached.get("confidence_level"),
                    "warnings": cached.get("warnings") or [],
                    "verdict_found": cached.get("confidence_level") == "high",
                    "evidence_status": cached.get("evidence_status"),
                },
                "linguistic_profile": cached.get("linguistic_profile") or {
                    "clickbait_syntax": 0.0,
                    "emotional_framing": 0.0,
                    "sensationalism": 0.0,
                    "informational_density": 0.5
                },
                "sources": cached.get("source_citations") or [],
                "processing_time_ms": elapsed_ms,
                "cached": True
            }

    # 6. Parallel agent dispatch
    logger.info(f"Cache miss/bypass. Dispatching verification workers in parallel for: {cleaned_text}")
    fc_task = factcheck_agent.run(cleaned_text)
    cr_task = crossref_agent.run(cleaned_text)
    ling_task = linguistic_agent.run(cleaned_text)
    
    results = await asyncio.gather(
        fc_task, cr_task, ling_task, return_exceptions=True
    )
    fc_res = (
        results[0]
        if isinstance(results[0], factcheck_agent.FactCheckResult)
        else factcheck_agent.make_neutral_factcheck(
            degraded=True,
            error=str(results[0]),
            error_details=classify_provider_error(results[0], provider="factcheck_agent")
        )
    )
    cr_res = (
        results[1]
        if isinstance(results[1], crossref_agent.CrossRefResult)
        else crossref_agent.make_neutral_crossref(
            degraded=True,
            error=str(results[1]),
            error_details=classify_provider_error(results[1], provider="crossref_agent")
        )
    )
    ling_res = (
        results[2]
        if isinstance(results[2], linguistic_agent.LinguisticResult)
        else linguistic_agent.make_neutral_linguistic(
            degraded=True,
            error=str(results[2]),
            error_details=classify_provider_error(results[2], provider="linguistic_agent")
        )
    )

    # Compute success signal from typed dataclasses
    agents_succeeded = sum([
        1 if not fc_res.degraded else 0,
        1 if not cr_res.degraded else 0,
        1 if not ling_res.degraded else 0,
    ])
    
    # 7. CRITICAL GUARDRAIL: All agents fail -> return an error, NEVER fabricate a score!
    if agents_succeeded == 0:
        logger.error(f"All 3 verification agents failed or degraded on claim: {cleaned_text}")
        raise AllAgentsFailedError("All verification agents failed to complete analysis. Unable to verify claim.")

    # 8. Compute scoring
    fact_check_found = fc_res.matched_review
    verdict_found = fc_res.verdict_found
    engine = ScoringEngine()
    scoring_res = engine.compute(
        s_fact=fc_res.s_fact_score,
        b_ling=ling_res.b_ling_score,
        v_consensus=cr_res.v_consensus_score,
        agents_succeeded=agents_succeeded,
        total_agents=3,
        fact_check_found=fact_check_found,
        verdict_found=verdict_found,
        v_refute=cr_res.v_refute_score,
    )

    # Enrich warnings with transparent provider degradation reasons
    enriched_warnings = list(scoring_res.warnings)
    for agent_label, res in [("Fact-check", fc_res), ("Cross-reference", cr_res), ("Linguistic", ling_res)]:
        if res.degraded:
            ed = getattr(res, "error_details", None)
            if ed is not None:
                enriched_warnings.append(f"{agent_label} agent degraded: {ed.message} ({ed.provider})")
            elif res.error:
                enriched_warnings.append(f"{agent_label} agent degraded: {res.error}")

    # 9. Consolidate sources
    sources: List[Dict[str, Any]] = []
    
    # Add fact-check sources
    for m in fc_res.all_matches:
        url_val = m.get("url") or ""
        domain = extract_domain(url_val)
        bias = get_bias_label(domain)
        src_type = "corroborating" if m.get("rating") == "Web Search Result" else "fact_check"
        sources.append({
            "url": url_val,
            "title": m.get("claim_reviewed") or "Fact-Check Match",
            "publisher": m.get("publisher") or "Fact-Checker",
            "rating": m.get("rating"),
            "bias_label": bias,
            "type": src_type,
            "stance": None
        })
        
    # Add corroborating sources
    for s in cr_res.sources_found:
        sources.append({
            "url": s.get("url"),
            "title": s.get("title") or "Corroborating Article",
            "publisher": s.get("publisher") or "News Outlet",
            "rating": None,
            "bias_label": s.get("bias_label") or "Unrated",
            "type": "corroborating",
            "stance": s.get("stance")
        })

    # Linguistic profile
    linguistic_profile = {
        "clickbait_syntax": ling_res.clickbait_syntax,
        "emotional_framing": ling_res.emotional_framing,
        "sensationalism": ling_res.sensationalism,
        "informational_density": ling_res.informational_density
    }

    elapsed_ms = int((time.time() - start_time) * 1000)

    # 10. Persist result and claim to Supabase (skip on bypass_cache; always persist otherwise so GET /api/results/{id} works; lookup_cache skips low/insufficient)
    claim_id: Optional[str] = None
    if not bypass_cache:
        try:
            claim_id = await insert_cache(
                input_type=input_type,
                raw_input=raw_input,
                extracted_claim=cleaned_text,
                embedding=embedding,
                c_total=scoring_res.c_total,
                s_fact=scoring_res.components.s_fact,
                b_ling=scoring_res.components.b_ling,
                v_consensus=scoring_res.components.v_consensus,
                weights=scoring_res.weights,
                confidence_level=scoring_res.confidence_level,
                agent_outputs={
                    "factcheck": {
                        "matches_found": fc_res.matches_found,
                        "s_fact_score": fc_res.s_fact_score,
                        "degraded": fc_res.degraded,
                        "error": fc_res.error,
                        "error_details": fc_res.error_details.to_dict() if getattr(fc_res, "error_details", None) else None,
                    },
                    "crossref": {
                        "entities_extracted": cr_res.entities_extracted,
                        "v_consensus_score": cr_res.v_consensus_score,
                        "v_refute_score": cr_res.v_refute_score,
                        "degraded": cr_res.degraded,
                        "error": cr_res.error,
                        "error_details": cr_res.error_details.to_dict() if getattr(cr_res, "error_details", None) else None,
                    },
                    "linguistic": {
                        "b_ling_score": ling_res.b_ling_score,
                        "flagged_phrases": ling_res.flagged_phrases,
                        "degraded": ling_res.degraded,
                        "error": ling_res.error,
                        "error_details": ling_res.error_details.to_dict() if getattr(ling_res, "error_details", None) else None,
                    }
                },
                source_citations=sources,
                linguistic_profile=linguistic_profile,
                processing_time_ms=elapsed_ms,
                warnings=enriched_warnings,
                v_refute=scoring_res.components.v_refute,
                evidence_status=scoring_res.evidence_status,
            )
        except Exception as exc:
            logger.warning(f"Failed to persist result to database: {exc}")

    return {
        "id": claim_id,
        "claim": cleaned_text,
        "trust_score": {
            "c_total": scoring_res.c_total,
            "s_fact": scoring_res.components.s_fact,
            "b_ling": scoring_res.components.b_ling,
            "v_consensus": scoring_res.components.v_consensus,
            "v_refute": scoring_res.components.v_refute,
            "weights": scoring_res.weights
        },
        "confidence": {
            "level": scoring_res.confidence_level,
            "warnings": enriched_warnings,
            "verdict_found": verdict_found,
            "evidence_status": scoring_res.evidence_status,
        },
        "linguistic_profile": linguistic_profile,
        "sources": sources,
        "processing_time_ms": elapsed_ms,
        "cached": False
    }
