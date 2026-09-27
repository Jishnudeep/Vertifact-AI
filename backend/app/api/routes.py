import asyncio
import logging
import uuid
from fastapi import APIRouter, Request, HTTPException, status
from slowapi import Limiter
from app.api.client_ip import client_ip

from app.api.schemas import VerifyRequest, VerifyResponse, ErrorResponse
from app.agents.orchestrator import verify_claim
from app.db.connection import get_supabase

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=client_ip)
router = APIRouter(prefix="/api")

@router.post(
    "/verify",
    response_model=VerifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input claim"},
        422: {"model": ErrorResponse, "description": "Non-verifiable claim (opinion/question)"},
        429: {"description": "Rate limit exceeded (10 requests/minute)"},
        500: {"model": ErrorResponse, "description": "All agents failed"},
    }
)
@limiter.limit("10/minute")
async def verify_endpoint(request: Request, body: VerifyRequest):
    """
    Main verification endpoint.
    Accepts text claim and returns composite trust score, confidence level,
    linguistic profile, and citations.
    """
    result = await verify_claim(body.input, body.input_type)
    return result

@router.get(
    "/results/{result_id}",
    response_model=VerifyResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Result not found"},
    }
)
@limiter.limit("60/minute")
async def get_result_endpoint(request: Request, result_id: str):
    """
    Retrieve a previously verified claim result by result ID or claim ID.
    """
    try:
        uuid.UUID(result_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid result ID format: '{result_id}'. Expected valid UUID."
        )

    try:
        supabase = get_supabase()
        
        # Try lookup by result id first (run blocking query off event loop)
        def _fetch_by_id(r_id: str):
            res = supabase.table("results").select("*, claims(extracted_claim)").eq("id", r_id).execute()
            if not res.data:
                res = supabase.table("results").select("*, claims(extracted_claim)").eq("claim_id", r_id).execute()
            return res.data

        data = await asyncio.to_thread(_fetch_by_id, result_id)
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Verification result for ID '{result_id}' not found."
            )
            
        row = data[0]
        claims_data = row.get("claims") or {}
        claim_text = claims_data.get("extracted_claim") or "Historical Claim"
        
        return {
            "id": row.get("id"),
            "claim": claim_text,
            "trust_score": {
                "c_total": row.get("c_total"),
                "s_fact": row.get("s_fact"),
                "b_ling": row.get("b_ling"),
                "v_consensus": row.get("v_consensus"),
                "v_refute": row.get("v_refute") or 0.0,
                "weights": row.get("weights") or {"w1": 0.5, "w2": 0.2, "w3": 0.3}
            },
            "confidence": {
                "level": row.get("confidence_level"),
                "warnings": (row.get("warnings") or []) + (
                    [] if row.get("evidence_status")
                    else ["Legacy result: scored before stance-aware consensus; treat corroboration as unverified"]
                ),
                "verdict_found": row.get("confidence_level") == "high",
                "evidence_status": row.get("evidence_status") or "unverified",
            },
            "linguistic_profile": row.get("linguistic_profile") or {
                "clickbait_syntax": 0.0,
                "emotional_framing": 0.0,
                "sensationalism": 0.0,
                "informational_density": 0.5
            },
            "sources": row.get("source_citations") or [],
            "processing_time_ms": row.get("processing_time_ms") or 0,
            "cached": True
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error retrieving result {result_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve result."
        )
