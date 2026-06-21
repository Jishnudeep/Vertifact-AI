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
    entities_extracted: List[str]   # People, organizations, locations
    sources_found: List[Dict[str, Any]]       # {url, title, publisher, bias_label, relevance_score}
    coverage_distribution: Dict[str, int]     # {"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0}
    v_consensus_score: float        # 0.0 to 1.0

NEUTRAL = CrossRefResult(
    entities_extracted=[],
    sources_found=[],
    coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
    v_consensus_score=0.5
)

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
        return NEUTRAL

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
    queries = [claim[:200]]
    if entities:
        # Query 2: Combination of all extracted unique entities
        entity_query = " ".join(entities)
        if entity_query.lower() != claim.lower().strip() and len(entity_query.strip()) > 0:
            queries.append(entity_query)

    # 3. Call Tavily in parallel async
    tasks = [
        search_web(q, settings.tavily_api_key, max_results=5)
        for q in queries
    ]
    search_results_list = await asyncio.gather(*tasks)

    # Combine results
    all_results = []
    for results in search_results_list:
        all_results.extend(results)

    # 4. Deduplicate results by domain, keeping the highest score (relevance)
    deduped_sources = {}
    for item in all_results:
        url = item.get("url")
        if not url:
            continue
        domain = extract_domain(url)
        # Handle case where score isn't a float or missing
        try:
            score = float(item.get("score", 0.0))
        except (ValueError, TypeError):
            score = 0.5
            
        title = item.get("title", "")

        if domain in deduped_sources:
            if score > deduped_sources[domain]["relevance_score"]:
                deduped_sources[domain] = {
                    "url": url,
                    "title": title,
                    "relevance_score": score,
                }
        else:
            deduped_sources[domain] = {
                "url": url,
                "title": title,
                "relevance_score": score,
            }

    # 5. For each unique source, resolve bias rating and publisher name
    sources_found = []
    coverage_distribution = {"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0}
    reputable_sources = []

    for domain, source in deduped_sources.items():
        url = source["url"]
        bias_label = get_bias_label(url) # Returns "Left", "Lean Left", "Center", "Lean Right", "Right", or "Unrated"
        
        # Get publisher name from database or fall back to domain heuristic
        publisher = get_outlet_name(url)
        if not publisher:
            publisher = _clean_publisher_name(domain)

        source_dict = {
            "url": url,
            "title": source["title"],
            "publisher": publisher,
            "bias_label": bias_label,
            "relevance_score": round(source["relevance_score"], 3)
        }
        sources_found.append(source_dict)

        # Reputable check: if the domain is present in AllSides (i.e., not Unrated)
        if bias_label in coverage_distribution:
            coverage_distribution[bias_label] += 1
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
        v_consensus_score=v_consensus_score
    )

async def run(claim: str) -> CrossRefResult:
    """
    Exposes the CrossReferenceAgent interface.
    Enforces a strict 8-second timeout, falling back to NEUTRAL.
    """
    try:
        return await asyncio.wait_for(_analyze(claim), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"Cross-Reference Agent timed out on claim: {claim}")
        return NEUTRAL
    except Exception as exc:
        logger.error(f"Cross-Reference Agent failed on claim: {claim}. Error: {exc}")
        return NEUTRAL
