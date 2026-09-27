import logging
import httpx
from typing import List, Dict, Any

from app.services.errors import EvidenceAPIError

logger = logging.getLogger(__name__)

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

async def search_factchecks(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Asynchronously query the Google Fact Check Tools API with the given claim query.
    
    Args:
        query: The claim text or search query.
        api_key: The Google Cloud API key for Fact Check Tools API.
        
    Returns:
        A list of claims objects containing reviews. An empty list means "the API answered
        with no matches" — nothing else.

    Raises:
        EvidenceAPIError: the service is unconfigured, could not be reached, or returned a
        non-200 status. Callers must treat this as an outage (degraded=True), not as
        "no fact-checks exist".
    """
    if not api_key:
        # R8.4: "not configured" == "could not run" == degraded (CLAUDE.md §4.3).
        logger.error("Google Fact Check API key is not configured; cannot execute lookup.")
        raise EvidenceAPIError("Google Fact Check API key not configured", provider="google_factcheck")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                FACTCHECK_API_URL,
                params={"query": query, "key": api_key, "languageCode": "en"},
                timeout=5.0,
            )
            if resp.status_code != 200:
                logger.error(
                    f"Google Fact Check API returned error status {resp.status_code}"
                )
                raise EvidenceAPIError(
                    f"Google Fact Check returned HTTP {resp.status_code}",
                    status_code=resp.status_code,
                    provider="google_factcheck",
                )

            data = resp.json()
            return data.get("claims", [])
    except EvidenceAPIError:
        raise
    except httpx.HTTPError as http_err:
        logger.error(f"Google Fact Check request failed: {type(http_err).__name__}")
        raise EvidenceAPIError(
            f"Google Fact Check request failed: {type(http_err).__name__}",
            provider="google_factcheck",
        ) from http_err
    except Exception as exc:
        logger.error(f"Unexpected error occurred during Google Fact Check search: {type(exc).__name__}")
        raise EvidenceAPIError(
            f"Google Fact Check unexpected error: {type(exc).__name__}",
            provider="google_factcheck",
        ) from exc

