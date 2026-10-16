import asyncio
import logging
import uuid
from typing import List
from fastapi import APIRouter, Request, HTTPException, status
from slowapi import Limiter
from app.api.client_ip import client_ip

from app.api.schemas import VerifyRequest, VerifyResponse, ErrorResponse, RecentResultModel
from app.api.result_format import format_stored_result
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
    "/results/recent",
    response_model=List[RecentResultModel],
)
@limiter.limit("60/minute")
async def get_recent_results_endpoint(request: Request, limit: int = 10):
    """
    Retrieve recent featured verification results for the landing feed.
    """
    clamped_limit = max(1, min(20, limit))
    try:
        supabase = get_supabase()

        def _fetch_recent(lim: int):
            return (
                supabase.table("results")
                .select("id, c_total, evidence_status, confidence_level, created_at, claims(extracted_claim)")
                .eq("featured", True)
                .order("created_at", desc=True)
                .limit(lim)
                .execute()
            ).data

        data = await asyncio.to_thread(_fetch_recent, clamped_limit)
        if not data:
            return []

        items = []
        for r in data:
            try:
                c_total = r.get("c_total")
                confidence_level = r.get("confidence_level")
                if c_total is None or confidence_level is None:
                    continue
                claims_data = r.get("claims") or {}
                claim_text = claims_data.get("extracted_claim") or ""
                items.append(RecentResultModel(
                    id=str(r.get("id")),
                    claim=claim_text,
                    c_total=c_total,
                    evidence_status=r.get("evidence_status") or "unverified",
                    confidence_level=confidence_level,
                    created_at=str(r.get("created_at")),
                ))
            except Exception as row_exc:
                logger.warning(f"Skipping invalid recent result row {r.get('id')}: {row_exc}")
                continue
        return items
    except Exception as exc:
        err_msg = str(exc)
        if "42703" in err_msg or "undefined_column" in err_msg.lower():
            logger.warning(f"Featured column not available in results table (migration 07 pending): {exc}")
            return []
        logger.error(f"Error retrieving recent results: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent results."
        )


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
        
        return format_stored_result(row, claim_text, cached=False)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error retrieving result {result_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve result."
        )
