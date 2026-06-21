import logging
from typing import List, Dict, Any, Optional
from app.db.connection import get_supabase

logger = logging.getLogger(__name__)

async def lookup_cache(
    embedding: List[float],
    similarity_threshold: float = 0.95
) -> Optional[Dict[str, Any]]:
    """
    Search the semantic cache for a claim with a cosine similarity >= similarity_threshold.
    Returns the joined claim and result data if a match is found, otherwise None.
    """
    try:
        supabase = get_supabase()
        response = supabase.rpc(
            "match_claims",
            {
                "query_embedding": embedding,
                "similarity_threshold": similarity_threshold
            }
        ).execute()
        
        if response.data and len(response.data) > 0:
            match_data = response.data[0]
            similarity = match_data.get("similarity")
            if isinstance(similarity, (int, float)):
                logger.info(f"Semantic cache HIT. Match similarity: {similarity:.4f}")
            else:
                logger.info(f"Semantic cache HIT. Match similarity: {similarity}")
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
    processing_time_ms: int
) -> Optional[str]:
    """
    Insert a verified claim and its results into Supabase.
    Returns the created claim ID on success, or None on failure.
    """
    supabase = get_supabase()
    claim_id = None
    
    try:
        # NOTE: The insert operation here is non-atomic as we are performing two sequential inserts
        # (claims then results) from the application layer. If a failure/crash occurs in between,
        # a claim row may be orphaned. For v2, this best-effort application-level rollback/cleanup
        # is acceptable. In a future production iteration, a single Postgres transaction or a database
        # RPC/function executing both inserts atomically should be preferred.
        
        # 1. Insert into claims table
        claim_data = {
            "input_type": input_type,
            "raw_input": raw_input,
            "extracted_claim": extracted_claim,
            "embedding": embedding
        }
        claim_resp = supabase.table("claims").insert(claim_data).execute()
        if not claim_resp.data:
            logger.error("Failed to insert claim: No data returned from Supabase.")
            return None
            
        claim_id = claim_resp.data[0]["id"]
        
        # 2. Insert into results table
        result_data = {
            "claim_id": claim_id,
            "c_total": c_total,
            "s_fact": s_fact,
            "b_ling": b_ling,
            "v_consensus": v_consensus,
            "weights": weights,
            "confidence_level": confidence_level,
            "agent_outputs": agent_outputs,
            "source_citations": source_citations,
            "linguistic_profile": linguistic_profile,
            "processing_time_ms": processing_time_ms
        }
        result_resp = supabase.table("results").insert(result_data).execute()
        if not result_resp.data:
            logger.error(f"Failed to insert result for claim {claim_id}: No data returned.")
            # Rollback claim insertion to keep DB clean
            supabase.table("claims").delete().eq("id", claim_id).execute()
            return None
            
        logger.info(f"Successfully cached claim and results. Claim ID: {claim_id}")
        return claim_id
        
    except Exception as exc:
        logger.error(f"Error during cache insertion: {exc}")
        if claim_id:
            # Attempt rollback of claim on exception
            try:
                supabase.table("claims").delete().eq("id", claim_id).execute()
            except Exception as rollback_exc:
                logger.error(f"Failed to rollback claim {claim_id} insertion: {rollback_exc}")
        return None
