import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.db.connection import get_supabase
from app.services.cache_policy import unservable_reason
from app.services.claim_equivalence import are_equivalent

logger = logging.getLogger(__name__)

async def lookup_cache(
    embedding: List[float],
    claim_text: str,
    similarity_threshold: float = 0.95,
    match_count: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Serve a cached result only when it is safe to (C2):
    1. pgvector returns up to `match_count` candidates with cosine >= similarity_threshold.
    2. Each candidate must pass the row policy: confidence, evidence_status, complete run, TTL.
    3. The first remaining candidate whose claim is equivalent to `claim_text` (same numbers
       and bidirectional NLI entailment) is served. Embeddings alone cannot tell a claim from
       its negation.
    Fails closed: any error is a cache miss.
    """
    if not embedding or all(v == 0.0 for v in embedding):
        logger.debug("Cannot lookup cache with empty or all-zero embedding.")
        return None

    try:
        supabase = get_supabase()
        # supabase-py is synchronous; run the blocking round-trip off the event loop
        response = await asyncio.to_thread(
            supabase.rpc(
                "match_claims",
                {
                    "query_embedding": embedding,
                    "similarity_threshold": similarity_threshold,
                    "match_count": match_count,
                }
            ).execute
        )

        now = datetime.now(timezone.utc)
        for match_data in response.data or []:
            reason = unservable_reason(match_data, now)
            if reason:
                logger.info(f"Semantic match skipped: {reason}.")
                continue

            cached_claim = match_data.get("extracted_claim") or ""
            if not await asyncio.to_thread(are_equivalent, claim_text, cached_claim):
                logger.info("Semantic match skipped: cached claim is not equivalent (negation, number or scope).")
                continue

            # Ensure warnings is always a list
            if match_data.get("warnings") is None:
                match_data["warnings"] = []
            logger.info(f"Semantic cache HIT. Match similarity: {match_data.get('similarity')}")
            return match_data

        logger.debug("Semantic cache MISS.")
        return None
    except Exception as exc:
        logger.error(f"Error during semantic cache lookup: {exc}")
        return None

async def insert_cache(
    input_type: str,
    raw_input: str,
    extracted_claim: str,
    embedding: List[float],
    c_total: float,
    s_fact: float,
    b_ling: float,
    v_consensus: float,
    weights: Dict[str, float],
    confidence_level: str,
    agent_outputs: Dict[str, Any],
    source_citations: List[Dict[str, Any]],
    linguistic_profile: Dict[str, Any],
    processing_time_ms: int,
    warnings: Optional[List[str]] = None,
    v_refute: float = 0.0,
    evidence_status: Optional[str] = None,
) -> Optional[str]:
    """
    Insert a verified claim and its results into Supabase.
    Rejects all-zero vectors to prevent cache poisoning.
    Persists confidence warnings.
    Returns the created claim ID on success, or None on failure.
    """
    if not embedding or all(v == 0.0 for v in embedding):
        logger.warning("Attempted to cache claim with empty or zero-vector embedding. Skipping cache insertion.")
        return None

    supabase = get_supabase()
    claim_id = None

    # supabase-py is synchronous; run each blocking round-trip off the event loop
    def _execute(query):
        return query.execute()

    try:
        # 1. Insert into claims table
        claim_data = {
            "input_type": input_type,
            "raw_input": raw_input,
            "extracted_claim": extracted_claim,
            "embedding": embedding
        }
        claim_resp = await asyncio.to_thread(_execute, supabase.table("claims").insert(claim_data))
        if not claim_resp.data:
            logger.error("Failed to insert claim: No data returned from Supabase.")
            return None
            
        claim_id = claim_resp.data[0]["id"]
        
        # 2. Insert into results table with warnings
        result_data = {
            "claim_id": claim_id,
            "c_total": c_total,
            "s_fact": s_fact,
            "b_ling": b_ling,
            "v_consensus": v_consensus,
            "v_refute": v_refute,
            "evidence_status": evidence_status,
            "weights": weights,
            "confidence_level": confidence_level,
            "agent_outputs": agent_outputs,
            "source_citations": source_citations,
            "linguistic_profile": linguistic_profile,
            "warnings": warnings or [],
            "processing_time_ms": processing_time_ms
        }
        result_resp = await asyncio.to_thread(_execute, supabase.table("results").insert(result_data))
        if not result_resp.data:
            logger.error(f"Failed to insert result for claim {claim_id}: No data returned.")
            # Rollback claim insertion to keep DB clean
            await asyncio.to_thread(_execute, supabase.table("claims").delete().eq("id", claim_id))
            return None
            
        logger.info(f"Successfully cached claim and results with warnings. Claim ID: {claim_id}")
        return claim_id
        
    except Exception as exc:
        logger.error(f"Error during cache insertion: {exc}")
        if claim_id:
            try:
                await asyncio.to_thread(_execute, supabase.table("claims").delete().eq("id", claim_id))
            except Exception as rollback_exc:
                logger.error(f"Failed to rollback claim {claim_id} insertion: {rollback_exc}")
        return None
