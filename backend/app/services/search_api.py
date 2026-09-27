import logging
import httpx
from typing import List, Dict, Any

from app.services.errors import EvidenceAPIError

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Broad domain scope for web evidence retrieval via Tavily
FACTCHECK_DOMAINS = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "healthfeedback.org",
    "apnews.com",
    "reuters.com",
    "usatoday.com",
    "factcheck.afp.com",
    "leadstories.com",
    "checkyourfact.com",
    "climatefeedback.org",
]

# Strict fact-checking authorities for Layer-2 verdict authority (excluding general news)
CLAIMREVIEW_DOMAINS = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "healthfeedback.org",
    "factcheck.afp.com",
    "afp.com",
    "leadstories.com",
    "checkyourfact.com",
    "climatefeedback.org",
]

async def search_web(
    query: str,
    api_key: str,
    max_results: int = 5,
    include_domains: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Asynchronously query the Tavily Search API.
    
    Args:
        query: The search query string.
        api_key: The Tavily API key.
        max_results: The maximum number of search results to return.
        include_domains: An optional list of domains to restrict the search to.
        
    Returns:
        A list of search result dicts: [{"title": ..., "url": ..., "content": ..., "score": ...}]
        An empty list means "the API answered with no results" — nothing else.

    Raises:
        EvidenceAPIError: the service is unconfigured, could not be reached, or returned a
        non-200 status. Callers must treat this as an outage (degraded=True), not as
        "no evidence exists".
    """
    if not api_key:
        # R8.4: "not configured" == "could not run" == degraded. Returning [] here made the
        # agent report degraded=False and tell the user "no sources found" for a search
        # that was never executed (CLAUDE.md §4.3).
        logger.error("Tavily API key is not configured; cannot execute evidence search.")
        raise EvidenceAPIError("Tavily API key not configured", provider="tavily")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
    }
    
    if include_domains:
        payload["include_domains"] = include_domains

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
            if resp.status_code != 200:
                if resp.status_code == 432:
                    logger.error(
                        "Tavily Search API returned HTTP 432: monthly search quota exhausted (1,000 requests cap reached)"
                    )
                else:
                    logger.error(
                        f"Tavily Search API returned error status {resp.status_code}"
                    )
                raise EvidenceAPIError(
                    f"Tavily search returned HTTP {resp.status_code}",
                    status_code=resp.status_code,
                    provider="tavily",
                )

            data = resp.json()
            return data.get("results", [])
    except EvidenceAPIError:
        raise
    except httpx.HTTPError as http_err:
        logger.error(f"Tavily Search request failed: {type(http_err).__name__}")
        raise EvidenceAPIError(
            f"Tavily search request failed: {type(http_err).__name__}",
            provider="tavily",
        ) from http_err
    except Exception as exc:
        logger.error(f"Unexpected error occurred during Tavily search: {type(exc).__name__}")
        raise EvidenceAPIError(
            f"Tavily search unexpected error: {type(exc).__name__}",
            provider="tavily",
        ) from exc

