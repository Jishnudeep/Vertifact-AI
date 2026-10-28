import asyncio
import logging
import uuid
from typing import List
from fastapi import APIRouter, Request, HTTPException, status
from starlette.formparsers import MultiPartParser
from pydantic import ValidationError
from slowapi import Limiter
from app.api.client_ip import client_ip

from app.api.schemas import (
    VerifyRequest, VerifyResponse, UrlVerifyResponse, ImageVerifyResponse, ErrorResponse, RecentResultModel,
)
from app.api.result_format import format_stored_result
from app.agents.ocr_agent import MAX_IMAGE_BYTES, OCRError, extract_text
from app.agents.orchestrator import InvalidInputError, verify_claim
from app.agents.scraper_agent import scrape_url
from app.utils.validators import sanitize_text
from app.db.connection import get_supabase

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=client_ip)
router = APIRouter(prefix="/api")

_UNDEFINED_COLUMN = "42703"


def _is_undefined_column(exc: Exception) -> bool:
    """Postgres undefined_column (e.g. migration 07 `featured` not applied yet)."""
    if getattr(exc, "code", None) == _UNDEFINED_COLUMN:
        return True
    err_msg = str(exc)
    return _UNDEFINED_COLUMN in err_msg or "undefined_column" in err_msg.lower()


@router.post(
    "/verify",
    response_model=UrlVerifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input claim, or a URL that cannot be read"},
        422: {"model": ErrorResponse, "description": "Non-verifiable claim (opinion/question)"},
        429: {"description": "Rate limit exceeded (10 requests/minute)"},
        500: {"model": ErrorResponse, "description": "All agents failed"},
    }
)
@limiter.limit("10/minute")
async def verify_endpoint(request: Request, body: VerifyRequest):
    """
    Main verification endpoint.
    Accepts a text claim or an article URL and returns composite trust score, confidence level,
    linguistic profile, and citations. For a URL the claim is the article headline (TDD §4.6).
    """
    if body.input_type == "url":
        # sanitize_text strips tracking parameters from the link before it is fetched.
        scraped = await scrape_url(sanitize_text(body.input))
        if scraped.error:
            raise InvalidInputError(scraped.error)
        result = await verify_claim(scraped.headline, "url")
        return {**result, "source_url": scraped.final_url, "scrape_degraded": scraped.degraded}
    if body.input_type == "image":
        raise InvalidInputError("Upload screenshots to /api/verify/image.")
    return await verify_claim(body.input, body.input_type)


_IMAGE_CONTENT_TYPES = frozenset({"image/png", "image/jpeg"})
# One file part plus multipart framing; anything larger is refused without being buffered.
_MAX_UPLOAD_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024


class _InMemoryMultiPartParser(MultiPartParser):
    # Starlette spools file parts bigger than 1 MB to a temp file on disk; the body is already
    # capped, so keep it all in memory (TDD §11: no disk writes for uploads).
    spool_max_size = _MAX_UPLOAD_BODY_BYTES


async def _read_image_upload(request: Request) -> bytes:
    """Read the single `image` part with the body capped while streaming (before any parsing/spooling)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _MAX_UPLOAD_BODY_BYTES:
        raise InvalidInputError("The image is larger than 5 MB.")
    if "multipart/form-data" not in request.headers.get("content-type", "").lower():
        raise InvalidInputError("Send the screenshot as multipart/form-data in the 'image' field.")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > _MAX_UPLOAD_BODY_BYTES:
            raise InvalidInputError("The image is larger than 5 MB.")

    async def replay():
        yield bytes(body)
        yield b""

    form = await _InMemoryMultiPartParser(request.headers, replay()).parse()
    try:
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise InvalidInputError("Attach the screenshot in the 'image' field.")
        if (upload.content_type or "").lower() not in _IMAGE_CONTENT_TYPES:
            raise InvalidInputError("Only PNG and JPEG images are supported.")
        return await upload.read()
    finally:
        await form.close()


@router.post(
    "/verify/image",
    response_model=ImageVerifyResponse,
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["image"],
        "properties": {"image": {"type": "string", "format": "binary"}}}}}}},
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported, oversized or unreadable image; no text found"},
        422: {"model": ErrorResponse, "description": "Non-verifiable text (opinion/question)"},
        429: {"description": "Rate limit exceeded (10 requests/minute)"},
        500: {"model": ErrorResponse, "description": "All agents failed"},
    }
)
@limiter.limit("10/minute")
async def verify_image_endpoint(request: Request):
    """
    Screenshot verification: OCR the image (in memory, never stored), then run the standard pipeline.
    The body is read by hand (not as an UploadFile) so an oversized upload is refused while streaming.
    """
    data = await _read_image_upload(request)
    try:
        ocr = await extract_text(data)
    except OCRError as exc:
        raise InvalidInputError(str(exc)) from exc
    result = await verify_claim(ocr.extracted_text, "image")
    return {**result, "ocr_extracted_text": result["claim"], "ocr_confidence": ocr.confidence}


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
            row_id = r.get("id")
            # created_at is required by RecentResultModel: a NULL is skipped rather than served
            # as a made-up timestamp (str(None) would render as fake recency in the feed).
            missing = [f for f in ("c_total", "confidence_level", "created_at") if r.get(f) is None]
            if missing:
                logger.warning("Skipping recent result row %s: missing %s", row_id, ", ".join(missing))
                continue
            try:
                claims_data = r.get("claims") or {}
                claim_text = claims_data.get("extracted_claim") or ""
                items.append(RecentResultModel(
                    id=str(row_id),
                    claim=claim_text,
                    c_total=r.get("c_total"),
                    evidence_status=r.get("evidence_status") or "unverified",
                    confidence_level=r.get("confidence_level"),
                    created_at=str(r.get("created_at")),
                ))
            except ValidationError as row_exc:
                # Log field names only: pydantic messages embed input values (row contents).
                fields = sorted({".".join(str(p) for p in e["loc"]) for e in row_exc.errors()})
                logger.warning("Skipping invalid recent result row %s: invalid %s", row_id, ", ".join(fields))
            except Exception as row_exc:
                logger.warning("Skipping recent result row %s: %s", row_id, type(row_exc).__name__)
        return items
    except Exception as exc:
        if _is_undefined_column(exc):
            logger.warning("Featured column not available in results table (migration 07 pending)")
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
