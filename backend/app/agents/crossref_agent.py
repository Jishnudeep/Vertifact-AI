import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.config import settings
from app.agents.linguistic_agent import get_spacy_nlp
from app.services.search_api import search_web
from app.services.bias_lookup import get_bias_label, get_outlet_name
from app.utils.text_processing import extract_domain

logger = logging.getLogger(__name__)
TIMEOUT = 8.0

@dataclass
class CrossRefResult:
    entities_extracted: List[str]             # People, organizations, locations
    sources_found: List[Dict[str, Any]]       # {url, title, publisher, bias_label, relevance_score}
    coverage_distribution: Dict[str, int]     # {"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0}
    v_consensus_score: float                  # 0.0 to 1.0 (0.0 when no sources or degraded)
    degraded: bool = False                    # True if agent timed out or failed
    error: Optional[str] = None               # Failure message if degraded

def make_neutral_crossref(degraded: bool = False, error: Optional[str] = None) -> CrossRefResult:
    """
    Factory creating a fresh neutral CrossRefResult.
    v_consensus_score is strictly 0.0 on degradation or empty search.
    """
    return CrossRefResult(
        entities_extracted=[],
        sources_found=[],
        coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0,
        degraded=degraded,
        error=error
    )

# Backward compatibility reference (fresh instance)
NEUTRAL = make_neutral_crossref()

def _clean_publisher_name(domain: str) -> str:
    """
    Generate a clean publisher fallback name from a domain.
    E.g. 'edition.cnn.com' -> 'CNN', 'bbc.co.uk' -> 'BBC', 'reuters.com' -> 'Reuters'.
    """
    parts = domain.split('.')
    if len(parts) > 1:
        # Skip common generic subdomains
        if parts[0] in {"www", "news", "edition", "opinion", "blogs"}:
            pub_part = parts[1]
        else:
            pub_part = parts[0]
    else:
        pub_part = domain
        
    return pub_part.upper() if len(pub_part) <= 4 else pub_part.capitalize()

async def _analyze(claim: str) -> CrossRefResult:
    """
    Internal implementation of cross-reference analysis.
    """
    if not claim or not claim.strip():
        return make_neutral_crossref(degraded=False)

    # 1. spaCy NER extraction
    nlp = get_spacy_nlp()
    doc = nlp(claim)
    
    # We want entities of types: PERSON (people), ORG (organizations), GPE (locations), NORP (nationalities/religious/political groups)
    # also FAC (facilities), LOC (geographic features), PRODUCT, EVENT
    allowed_labels = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "PRODUCT", "EVENT"}
    entities = []
    seen_entities = set()
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            clean_ent = ent.text.strip()
            if len(clean_ent) > 1 and clean_ent.lower() not in seen_entities:
                seen_entities.add(clean_ent.lower())
                entities.append(clean_ent)

    # 2. Construct search queries
    # Query 1: The original claim text (truncated to avoid excessively long search queries)
    queries = [claim[:150]]

    # Query 2 & 3: Combine top entities if available
    if len(entities) >= 2:
        queries.append(f"{entities[0]} {entities[1]}")
    elif len(entities) == 1:
        queries.append(f"{entities[0]} news")
        
    # Limit queries to at most 3
    queries = queries[:3]

    # 3. Parallel search execution via Tavily
    api_key = settings.tavily_api_key
    search_tasks = [
        search_web(query=q, api_key=api_key, max_results=5)
        for q in queries
    ]
    
    results_nested = await asyncio.gather(*search_tasks, return_exceptions=True)

    # 4. Flatten and deduplicate results by domain
    all_results = []
    for res in results_nested:
        if isinstance(res, list):
            all_results.extend(res)
        elif isinstance(res, Exception):
            logger.warning(f"A search query failed during cross-reference: {res}")

    domain_to_best_article: Dict[str, Dict[str, Any]] = {}
    for item in all_results:
        url = item.get("url", "")
        domain = extract_domain(url)
        if not domain:
            continue
            
        score = item.get("score", 0.0)
        # Keep the highest scoring article for each unique domain
        if domain not in domain_to_best_article or score > domain_to_best_article[domain].get("score", 0.0):
            domain_to_best_article[domain] = item

    # 5. Bias lookup and coverage distribution calculation
    coverage_distribution = {
        "Left": 0,
        "Lean Left": 0,
        "Center": 0,
        "Lean Right": 0,
        "Right": 0
    }
    
    sources_found = []
    reputable_sources = []

    for domain, article in domain_to_best_article.items():
        bias_label = get_bias_label(domain)
        known_outlet = get_outlet_name(domain)
        publisher = known_outlet if known_outlet else _clean_publisher_name(domain)

        # Map search relevance score to 0.0 - 1.0 (Tavily score is typically 0.0 - 1.0)
        relevance_score = float(article.get("score", 0.5))

        source_dict = {
            "url": article.get("url"),
            "title": article.get("title", ""),
            "publisher": publisher,
            "bias_label": bias_label,
            "relevance_score": round(relevance_score, 2)
        }
        sources_found.append(source_dict)

        # Reputable check: AllSides rated outlets or institutional/primary domains (.gov, .edu, who.int, nasa.gov) per B4/D3
        is_reputable = (
            bias_label in coverage_distribution
            or domain.endswith(".gov")
            or domain.endswith(".edu")
            or domain in {"who.int", "nasa.gov", "reuters.com", "apnews.com", "bbc.com"}
        )
        
        if bias_label in coverage_distribution:
            coverage_distribution[bias_label] += 1
            
        if is_reputable:
            reputable_sources.append(source_dict)

    # Sort sources by relevance score descending
    sources_found.sort(key=lambda s: s["relevance_score"], reverse=True)

    # 6. Compute consensus score
    reputable_count = len(reputable_sources)
    if reputable_count == 0:
        v_consensus_score = 0.0
    else:
        avg_relevance = sum(s["relevance_score"] for s in reputable_sources) / reputable_count
        v_consensus_score = min(reputable_count / 5.0, 1.0) * avg_relevance
        v_consensus_score = round(v_consensus_score, 3)

    return CrossRefResult(
        entities_extracted=entities,
        sources_found=sources_found,
        coverage_distribution=coverage_distribution,
        v_consensus_score=v_consensus_score,
        degraded=False,
        error=None
    )

async def run(claim: str) -> CrossRefResult:
    """
    Exposes the CrossReferenceAgent interface.
    Enforces a strict 8-second timeout, falling back to degraded neutral (v_consensus=0.0).
    """
    try:
        return await asyncio.wait_for(_analyze(claim), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"Cross-Reference Agent timed out on claim: {claim}")
        return make_neutral_crossref(degraded=True, error="Timeout: Cross-Reference Agent exceeded 8.0s")
    except Exception as exc:
        logger.error(f"Cross-Reference Agent failed on claim: {claim}. Error: {exc}")
        return make_neutral_crossref(degraded=True, error=str(exc))
