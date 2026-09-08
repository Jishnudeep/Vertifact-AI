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

async def verify_claim(raw_input: str, input_type: str = "text") -> Dict[str, Any]:
    """
    Main orchestration pipeline for claim verification:
    1. Sanitize input (strip HTML/scripts, URL tracking parameters, whitespace).
    2. Validate input constraints (length <= 2000 chars).
    3. Validate claim verifiability (reject subjective opinions & questions -> 422).
    4. Generate embedding and query semantic cache (short-circuit on cosine similarity >= 0.95).
    5. On cache miss, dispatch factcheck, crossref, and linguistic agents in parallel (8s timeouts).
    6. Guardrail: If all agents fail, raise AllAgentsFailedError (never fabricate a score).
    7. Compute deterministic composite trust score and apply confidence capping.
    8. Consolidate sources (fact_check and corroborating).
    9. Persist result and claim to Supabase cache (with confidence warnings).
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
    embedding = generate_embedding(cleaned_text)

    # 5. Semantic cache lookup
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
                "weights": cached.get("weights") or {"w1": 0.50, "w2": 0.20, "w3": 0.30}
            },
            "confidence": {
                "level": cached.get("confidence_level"),
                "warnings": cached.get("warnings") or []
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
    logger.info(f"Cache miss. Dispatching verification workers in parallel for: {cleaned_text}")
    fc_task = factcheck_agent.run(cleaned_text)
    cr_task = crossref_agent.run(cleaned_text)
    ling_task = linguistic_agent.run(cleaned_text)
    
    fc_res, cr_res, ling_res = await asyncio.gather(
        fc_task, cr_task, ling_task, return_exceptions=False
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
    fact_check_found = (fc_res.matches_found > 0)
    engine = ScoringEngine()
    scoring_res = engine.compute(
        s_fact=fc_res.s_fact_score,
        b_ling=ling_res.b_ling_score,
        v_consensus=cr_res.v_consensus_score,
        agents_succeeded=agents_succeeded,
        total_agents=3,
        fact_check_found=fact_check_found
    )

    # 9. Consolidate sources
    sources: List[Dict[str, Any]] = []
    
    # Add fact-check sources
    for m in fc_res.all_matches:
        url_val = m.get("url") or ""
        domain = extract_domain(url_val)
        bias = get_bias_label(domain)
        sources.append({
            "url": url_val,
            "title": m.get("claim_reviewed") or "Fact-Check Match",
            "publisher": m.get("publisher") or "Fact-Checker",
            "rating": m.get("rating"),
            "bias_label": bias,
            "type": "fact_check"
        })
        
    # Add corroborating sources
    for s in cr_res.sources_found:
        sources.append({
            "url": s.get("url"),
            "title": s.get("title") or "Corroborating Article",
            "publisher": s.get("publisher") or "News Outlet",
            "rating": None,
            "bias_label": s.get("bias_label") or "Unrated",
            "type": "corroborating"
        })

    # Linguistic profile
    linguistic_profile = {
        "clickbait_syntax": ling_res.clickbait_syntax,
        "emotional_framing": ling_res.emotional_framing,
        "sensationalism": ling_res.sensationalism,
        "informational_density": ling_res.informational_density
    }

    elapsed_ms = int((time.time() - start_time) * 1000)

    # 10. Persist result and claim to Supabase cache
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
                "error": fc_res.error
            },
            "crossref": {
                "entities_extracted": cr_res.entities_extracted,
                "v_consensus_score": cr_res.v_consensus_score,
                "degraded": cr_res.degraded,
                "error": cr_res.error
            },
            "linguistic": {
                "b_ling_score": ling_res.b_ling_score,
                "flagged_phrases": ling_res.flagged_phrases,
                "degraded": ling_res.degraded,
                "error": ling_res.error
            }
        },
        source_citations=sources,
        linguistic_profile=linguistic_profile,
        processing_time_ms=elapsed_ms,
        warnings=scoring_res.warnings
    )

    return {
        "id": claim_id,
        "claim": cleaned_text,
        "trust_score": {
            "c_total": scoring_res.c_total,
            "s_fact": scoring_res.components.s_fact,
            "b_ling": scoring_res.components.b_ling,
            "v_consensus": scoring_res.components.v_consensus,
            "weights": scoring_res.weights
        },
        "confidence": {
            "level": scoring_res.confidence_level,
            "warnings": scoring_res.warnings
        },
        "linguistic_profile": linguistic_profile,
        "sources": sources,
        "processing_time_ms": elapsed_ms,
        "cached": False
    }
