import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

async def search_factchecks(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Asynchronously query the Google Fact Check Tools API with the given claim query.
    
    Args:
        query: The claim text or search query.
        api_key: The Google Cloud API key for Fact Check Tools API.
        
    Returns:
        A list of claims objects containing reviews, or empty list on failure or no matches.
    """
    if not api_key:
        logger.warning("Google Fact Check API key is missing. Skipping Fact Check Tools API query.")
        return []

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                FACTCHECK_API_URL,
                params={"query": query, "key": api_key, "languageCode": "en"},
                timeout=5.0,
            )
            if resp.status_code != 200:
                logger.error(
                    f"Google Fact Check API returned error status {resp.status_code}: {resp.text}"
                )
                return []
            
            data = resp.json()
            return data.get("claims", [])
    except httpx.HTTPError as http_err:
        logger.error(f"HTTP error occurred during Google Fact Check search: {http_err}")
    except Exception as exc:
        logger.error(f"Unexpected error occurred during Google Fact Check search: {exc}")
        
    return []
