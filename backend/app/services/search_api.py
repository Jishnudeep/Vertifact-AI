import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

FACTCHECK_DOMAINS = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "apnews.com",
    "reuters.com",
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
    """
    if not api_key:
        logger.warning("Tavily API key is missing. Skipping search query.")
        return []

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
                logger.error(
                    f"Tavily Search API returned error status {resp.status_code}"
                )
                return []
                
            data = resp.json()
            return data.get("results", [])
    except httpx.HTTPError as http_err:
        logger.error(f"Tavily Search request failed: {type(http_err).__name__}")
    except Exception as exc:
        logger.error(f"Unexpected error occurred during Tavily search: {type(exc).__name__}")
        
    return []

