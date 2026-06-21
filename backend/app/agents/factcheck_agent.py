import asyncio
import logging
import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.config import settings
from app.services.factcheck_api import search_factchecks
from app.services.search_api import search_web, FACTCHECK_DOMAINS
from app.utils.adk_helpers import run_adk_agent, parse_adk_response

logger = logging.getLogger(__name__)
TIMEOUT = 8.0

# Ensure GROQ_API_KEY is set in environment for LiteLLM
if settings.groq_api_key:
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)

RATING_MAP = {
    "true": 1.0,
    "correct": 1.0,
    "accurate": 1.0,
    "mostly true": 0.75,
    "mostly correct": 0.75,
    "half true": 0.5,
    "mixture": 0.5,
    "mixed": 0.5,
    "mostly false": 0.25,
    "mostly incorrect": 0.25,
    "false": 0.0,
    "incorrect": 0.0,
    "pants on fire": 0.0,
    "unproven": 0.5,
    "unverified": 0.5,
    "outdated": 0.4,
}

@dataclass
class FactCheckResult:
    matches_found: int
    best_match: Optional[Dict[str, Any]] = None  # {url, publisher, claim_reviewed, rating, review_date}
    all_matches: List[Dict[str, Any]] = field(default_factory=list)
    s_fact_score: float = 0.5         # 0.0 to 1.0 (0.5 is neutral/unknown)
    search_queries_used: List[str] = field(default_factory=list)

NEUTRAL = FactCheckResult(
    matches_found=0,
    best_match=None,
    all_matches=[],
    s_fact_score=0.5,
    search_queries_used=[]
)

def token_similarity(s1: str, s2: str) -> float:
    """
    Compute a simple Jaccard similarity of word tokens between two strings.
    """
    w1 = set(re.findall(r'\w+', s1.lower()))
    w2 = set(re.findall(r'\w+', s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

def map_rating_to_score(rating: str) -> float:
    """
    Map textual ratings from fact-checking sources to a 0.0 - 1.0 numeric score.
    """
    if not rating:
        return 0.5
        
    clean_rating = rating.strip().lower()
    
    # Exact lookup
    if clean_rating in RATING_MAP:
        return RATING_MAP[clean_rating]
        
    # Substring fallback matching - ORDER IS IMPORTANT: Mixed/Half first
    # 1. Mixed/Half True/Mixture/Misleading
    if any(word in clean_rating for word in ["mixed", "mixture", "half", "somewhat", "misleading", "distorted"]):
        return 0.5
        
    # 2. False/Incorrect/Debunked
    if any(word in clean_rating for word in ["false", "incorrect", "pants on fire", "fake", "debunk", "lying", "untrue", "lie"]):
        if any(w in clean_rating for w in ["mostly", "partially"]):
            return 0.25
        return 0.0
        
    # 3. True/Correct/Accurate
    if any(word in clean_rating for word in ["true", "correct", "accurate", "truth"]):
        if any(w in clean_rating for w in ["mostly", "partially"]):
            return 0.75
        return 1.0
        
    # 4. Outdated
    if "outdated" in clean_rating:
        return 0.4
        
    return 0.5

# Duplicated JSON parsing and ADK runner functions removed. Imported from app.utils.adk_helpers.

async def _classify_rating_with_llm(claim: str, rating_text: str) -> str:
    """
    Use an ADK Agent to classify a verbose/conversational fact-checker rating text
    into a standard rating string.
    """
    instruction = (
        "You are a fact-checking assistant. Your task is to map a verbose fact-checker rating text "
        "to a standard rating category for a claim. Return your output in raw JSON format with "
        "'rating' and 'explanation' keys. The 'rating' key must be one of: true, mostly true, "
        "half true, mostly false, false, unproven. Use double quotes for all JSON properties and values. "
        "Do not use unescaped double quotes inside the explanation text."
    )
    prompt = f"Claim: {claim}\nFact-checker rating text: {rating_text}"
    try:
        content = await run_adk_agent("rating_classifier", instruction, prompt, "verifact-factcheck")
        data = parse_adk_response(content, expected_keys=["rating", "explanation"])
        return data.get("rating", "unproven")
    except Exception as exc:
        logger.error(f"Error classifying verbose rating with ADK: {exc}")
        return "unproven"

async def _align_claim_match(query: str, claim_reviewed: str, rating_str: str) -> Optional[float]:
    """
    Use an ADK Agent to align a search match's rating to the user's query.
    Handles cases where the matched claim is the opposite of the query (e.g. 'Sun orbits Earth' vs 'Earth orbits Sun').
    Returns the aligned score (0.0 to 1.0) or None if the match is irrelevant.
    """
    instruction = (
        "You are a fact-checking assistant. Your task is to analyze if a fact-checked claim from a database "
        "is directly relevant to a user's query, and if so, map the rating. Return your response in raw JSON "
        "format with keys: 'relevant' (boolean), 'opposite' (boolean), 'implied_rating' (one of: true, "
        "mostly true, half true, mostly false, false, unproven), and 'explanation' (string). Use double quotes. "
        "Do not use unescaped double quotes inside the explanation text."
    )
    # SECURITY: Prompt injection risk via untrusted query/claim_reviewed.
    # Mitigation: Handled at eval/guardrail phase (faithfulness test + scoring confidence-cap).
    # TODO: Verify safety in eval/guardrail calibration.
    prompt = f"""User's Query: "{query}"
Fact-Checked Claim: "{claim_reviewed}"
Original Rating: "{rating_str}"

Relevance Instructions:
1. The fact-checked claim is directly relevant ONLY if it directly supports, debunks, or addresses the core assertion of the User's Query.
2. If the fact-checked claim is about a specific photo, video, or viral post that is related but does not directly verify the user's query itself, it is NOT directly relevant. Mark 'relevant' as false.
3. If the User's Query is "The Earth orbits the Sun" and the Fact-Checked Claim is about clouds behind the sun suggesting it orbits the earth, they are NOT directly relevant because the fact-check is about specific photos, not a general verification of the Earth's orbit. Mark 'relevant' as false.

Alignment Instructions:
If the claim is relevant, determine the implied rating for the user's query:
- If the User's Query is the opposite of the Fact-Checked Claim, and the claim is rated false, the User's Query is true.
- Otherwise, map the original rating directly."""

    try:
        content = await run_adk_agent("claim_aligner", instruction, prompt, "verifact-factcheck")
        data = parse_adk_response(
            content, 
            expected_keys=["relevant", "opposite", "implied_rating", "explanation"], 
            bool_keys=["relevant", "opposite"]
        )
        if not data.get("relevant", False):
            return None
        
        rating = data.get("implied_rating", "unproven")
        return map_rating_to_score(rating)
    except Exception as exc:
        logger.error(f"Error in claim match alignment: {exc}")
        return None

async def _analyze(claim: str) -> FactCheckResult:
    """
    Internal analysis function doing the actual work of fact-checking.
    """
    search_queries_used = [claim]
    
    # Layer 1: Google Fact Check Tools API
    logger.info(f"Querying Google Fact Check API for: {claim}")
    google_results = await search_factchecks(claim, settings.google_fact_check_api_key)
    
    all_matches = []
    if google_results:
        for claim_obj in google_results:
            claim_reviewed = claim_obj.get("text") or claim_obj.get("claim") or ""
            for review in claim_obj.get("claimReview", []):
                url = review.get("url")
                publisher_info = review.get("publisher", {})
                publisher = publisher_info.get("name", "Unknown")
                rating = review.get("textualRating", "")
                review_date = review.get("reviewDate")
                
                all_matches.append({
                    "url": url,
                    "publisher": publisher,
                    "claim_reviewed": claim_reviewed,
                    "rating": rating,
                    "review_date": review_date
                })
                
        # Align and verify matches in parallel (process top 3 only to keep latency low)
        async def process_match(match):
            claim_reviewed = match["claim_reviewed"]
            rating_str = match["rating"]
            
            sim = token_similarity(claim, claim_reviewed)
            if sim >= 0.75:
                score = map_rating_to_score(rating_str)
                if score == 0.5 and len(rating_str) > 15:
                    logger.info(f"Rating '{rating_str}' is verbose. Classifying with ADK...")
                    classified_rating = await _classify_rating_with_llm(claim, rating_str)
                    score = map_rating_to_score(classified_rating)
                return match, score
            else:
                logger.info(f"Loose match: '{claim}' vs '{claim_reviewed}'. Aligning using ADK...")
                score = await _align_claim_match(claim, claim_reviewed, rating_str)
                if score is not None:
                    match_copy = match.copy()
                    match_copy["rating"] = f"Aligned: {rating_str} -> Implied score {score}"
                    return match_copy, score
            return None

        tasks = [process_match(m) for m in all_matches[:3]]
        results = await asyncio.gather(*tasks)
        aligned_matches = [r for r in results if r is not None]
                    
        if aligned_matches:
            best_match_obj, s_fact_score = aligned_matches[0]
            logger.info(f"Google Fact Check API match aligned successfully: s_fact={s_fact_score}")
            return FactCheckResult(
                matches_found=len(aligned_matches),
                best_match=best_match_obj,
                all_matches=[m for m, _ in aligned_matches],
                s_fact_score=s_fact_score,
                search_queries_used=search_queries_used
            )
            
    # Layer 2: Tavily search scoped to fact-checking
    logger.info(f"Google Fact Check API missed or yielded no relevant matches. Querying Tavily fallback with parallel queries for: {claim}")
    queries = [
        f"{claim} fact check",
        f"{claim} debunked",
        f"is {claim} true"
    ]
    for q in queries:
        search_queries_used.append(q)
        
    tasks = [
        search_web(
            query=q,
            api_key=settings.tavily_api_key,
            max_results=5,
            include_domains=FACTCHECK_DOMAINS
        )
        for q in queries
    ]
    results_list = await asyncio.gather(*tasks)
    
    # Deduplicate results by URL and limit to top 5 unique results
    seen_urls = set()
    tavily_results = []
    for results in results_list:
        for r in results:
            url = r.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                tavily_results.append(r)
                if len(tavily_results) >= 5:
                    break
        if len(tavily_results) >= 5:
            break
    
    if not tavily_results:
        logger.info("Tavily fallback returned no results.")
        return FactCheckResult(
            matches_found=0,
            best_match=None,
            all_matches=[],
            s_fact_score=0.5,
            search_queries_used=search_queries_used
        )
        
    # Format snippets for LLM
    snippets_text = ""
    for idx, r in enumerate(tavily_results):
        snippets_text += (
            f"[{idx + 1}] Source: {r.get('url')} | Title: {r.get('title')}\n"
            f"Snippet: {r.get('content')}\n\n"
        )
        
    instruction = (
        "You are a fact-checking assistant. Your task is to extract a single fact-checker verdict/rating "
        "for the claim based ONLY on the provided search snippets. Return your output in raw JSON format "
        "with keys: 'rating' (one of: true, mostly true, half true, mostly false, false, unproven), "
        "'explanation' (string), and 'publisher' (string). Use double quotes. Do not use unescaped double quotes inside JSON string values."
    )
    # SECURITY: Prompt injection risk via untrusted claim/snippets_text.
    # Mitigation: Handled at eval/guardrail phase (faithfulness test + scoring confidence-cap).
    # TODO: Verify safety in eval/guardrail calibration.
    prompt = f"Claim: {claim}\n\nSearch Snippets:\n{snippets_text}"

    try:
        content = await run_adk_agent("verdict_extractor", instruction, prompt, "verifact-factcheck")
        data = parse_adk_response(content, expected_keys=["rating", "explanation", "publisher"])
        extracted_rating = data.get("rating", "unproven")
        explanation = data.get("explanation", "")
        publisher = data.get("publisher", "Unknown")
        logger.info(f"ADK rating extraction successful: rating={extracted_rating}, publisher={publisher}")
    except Exception as exc:
        logger.error(f"Error extracting verdict from snippets using ADK Agent: {exc}")
        extracted_rating = "unproven"
        explanation = "Error extracting verdict from snippets."
        publisher = "Unknown"
        
    s_fact_score = map_rating_to_score(extracted_rating)
    
    # Build matches list
    all_matches = []
    for r in tavily_results:
        all_matches.append({
            "url": r.get("url"),
            "publisher": r.get("title") or "Search Result",
            "claim_reviewed": claim,
            "rating": "Web Search Result",
            "review_date": None
        })
        
    best_match = None
    if tavily_results:
        best_result = tavily_results[0]
        # Try to match the publisher in the domain
        for r in tavily_results:
            url_lower = r.get("url", "").lower()
            if publisher != "Unknown" and publisher.lower() in url_lower:
                best_result = r
                break
        best_match = {
            "url": best_result.get("url"),
            "publisher": publisher if publisher != "Unknown" else "Web Search Result",
            "claim_reviewed": claim,
            "rating": extracted_rating,
            "review_date": None
        }
        
    return FactCheckResult(
        matches_found=len(tavily_results),
        best_match=best_match,
        all_matches=all_matches,
        s_fact_score=s_fact_score,
        search_queries_used=search_queries_used
    )

async def run(claim: str) -> FactCheckResult:
    """
    Exposes the FactCheckLookupAgent interface.
    Enforces a strict 8-second timeout, falling back to NEUTRAL.
    """
    try:
        return await asyncio.wait_for(_analyze(claim), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"Fact-Check Agent timed out on claim: {claim}")
        return NEUTRAL
    except Exception as exc:
        logger.error(f"Fact-Check Agent failed on claim: {claim}. Error: {exc}")
        return NEUTRAL
