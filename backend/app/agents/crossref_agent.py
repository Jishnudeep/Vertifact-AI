import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from app.config import settings
from app.agents.linguistic_agent import get_spacy_nlp
from app.services.search_api import search_web, CLAIMREVIEW_DOMAINS
from app.services.errors import EvidenceAPIError
from app.services.bias_lookup import get_bias_label, get_outlet_name
from app.services.nli import NLIScores, classify_pairs
from app.utils.errors import AgentErrorDetails, classify_provider_error
from app.utils.text_processing import extract_domain

logger = logging.getLogger(__name__)
TIMEOUT = 8.0

# C1: a source counts as supporting/refuting only when the NLI model is at least this sure;
# anything less is "neutral" and moves neither score.
# Track 3.1: calibrate STANCE_MIN_PROB against the eval stance fixtures (Plan 3).
STANCE_MIN_PROB = 0.6

# Premise = title + snippet, truncated before tokenization (the model also truncates at
# settings.nli_max_length tokens).
PREMISE_MAX_CHARS = 1000

# Fact-check sites belong to the Fact-Check agent. Their snippets restate the claim they
# debunk ("Claim: X. Rating: False"), which NLI reads as entailment, so counting them here
# would both double count and invert their stance.
_CLAIMREVIEW_DOMAIN_SET = frozenset(CLAIMREVIEW_DOMAINS)

@dataclass
class CrossRefResult:
    entities_extracted: List[str]             # People, organizations, locations
    sources_found: List[Dict[str, Any]]       # {url, title, publisher, bias_label, relevance_score}
    coverage_distribution: Dict[str, int]     # {"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0}
    v_consensus_score: float                  # 0.0 to 1.0 (0.0 when no sources or degraded)
    v_refute_score: float = 0.0               # 0.0 to 1.0: relevance-weighted refuting reputable coverage
    degraded: bool = False                    # True if agent timed out or failed
    error: Optional[str] = None               # Failure message if degraded
    error_details: Optional[AgentErrorDetails] = None # Structured error classification if degraded

def make_neutral_crossref(
    degraded: bool = False,
    error: Optional[str] = None,
    error_details: Optional[AgentErrorDetails] = None
) -> CrossRefResult:
    """
    Factory creating a fresh neutral CrossRefResult.
    v_consensus_score and v_refute_score are strictly 0.0 on degradation or empty search.
    """
    return CrossRefResult(
        entities_extracted=[],
        sources_found=[],
        coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0,
        v_refute_score=0.0,
        degraded=degraded,
        error=error,
        error_details=error_details
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

def label_stance(scores: NLIScores) -> Tuple[str, float]:
    """
    Map NLI probabilities (premise = source text, hypothesis = claim) to a stance label
    and the probability behind it.
    """
    if scores.entailment >= STANCE_MIN_PROB and scores.entailment > scores.contradiction:
        return "supports", round(scores.entailment, 3)
    if scores.contradiction >= STANCE_MIN_PROB and scores.contradiction > scores.entailment:
        return "refutes", round(scores.contradiction, 3)
    return "neutral", round(scores.neutral, 3)


def compute_consensus(reputable_sources: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Stance-aware consensus (C1). Returns (v_consensus, v_refute), each in [0, 1].
    Each supporting (refuting) reputable source adds its relevance score; five fully relevant
    sources saturate the scale, matching the PRD's "5+ reputable sources = 1.0".
    Neutral sources add to neither side.
    """
    support = sum(s["relevance_score"] for s in reputable_sources if s.get("stance") == "supports")
    refute = sum(s["relevance_score"] for s in reputable_sources if s.get("stance") == "refutes")
    return round(min(support / 5.0, 1.0), 3), round(min(refute / 5.0, 1.0), 3)


def _premise(article: Dict[str, Any]) -> str:
    title = (article.get("title") or "").strip()
    content = (article.get("content") or "").strip()
    return f"{title}. {content}".strip(". ")[:PREMISE_MAX_CHARS]

async def _analyze(claim: str) -> CrossRefResult:
    """
    Internal implementation of cross-reference analysis.
    """
    if not claim or not claim.strip():
        return make_neutral_crossref(degraded=False)

    # 1. spaCy NER extraction
    nlp = get_spacy_nlp()
    doc = await asyncio.to_thread(nlp, claim)
    
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

    # R7.B4: if EVERY search failed with an evidence-API outage the agent reached no
    # evidence source at all. That is degradation (→ "Partial Analysis"), not "no sources
    # exist". A partial failure (at least one list returned) proceeds normally.
    if results_nested and all(isinstance(res, EvidenceAPIError) for res in results_nested):
        logger.error(f"Cross-Reference evidence retrieval unavailable: {results_nested[0]}")
        err_det = classify_provider_error(results_nested[0], provider="tavily")
        return make_neutral_crossref(
            degraded=True,
            error=f"Evidence retrieval API unavailable: {results_nested[0]}",
            error_details=err_det
        )

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

    # 5. Stance per source (C1): premise = article text, hypothesis = claim.
    articles = list(domain_to_best_article.items())
    pairs = [(_premise(article), claim) for _, article in articles]
    try:
        nli_scores = await asyncio.to_thread(classify_pairs, pairs)
    except Exception as exc:
        # Without stance, coverage cannot be read as agreement: degrade, never fall back to
        # the coverage-only count (that is the bug C1 fixes).
        logger.error(f"Stance model failed during cross-reference: {exc}")
        return make_neutral_crossref(
            degraded=True,
            error=f"Stance model unavailable: {exc}",
            error_details=classify_provider_error(exc, provider="nli"),
        )

    # 6. Bias lookup and coverage distribution calculation
    coverage_distribution = {
        "Left": 0,
        "Lean Left": 0,
        "Center": 0,
        "Lean Right": 0,
        "Right": 0
    }

    sources_found = []
    reputable_sources = []

    for (domain, article), scores in zip(articles, nli_scores):
        bias_label = get_bias_label(domain)
        known_outlet = get_outlet_name(domain)
        publisher = known_outlet if known_outlet else _clean_publisher_name(domain)

        # Map search relevance score to 0.0 - 1.0 (Tavily score is typically 0.0 - 1.0)
        relevance_score = float(article.get("score", 0.5))
        stance, stance_confidence = label_stance(scores)

        source_dict = {
            "url": article.get("url"),
            "title": article.get("title", ""),
            "publisher": publisher,
            "bias_label": bias_label,
            "relevance_score": round(relevance_score, 2),
            "stance": stance,
            "stance_confidence": stance_confidence,
        }
        sources_found.append(source_dict)

        # Reputable check: AllSides rated outlets or institutional/primary domains per B4/D3.
        # Wire services (reuters/apnews/bbc) are intentionally NOT listed: they are AllSides-rated,
        # so the first clause already covers them.
        # Track 3.1: tighten institutional-domain trust (a .edu personal page is not a source)
        is_reputable = (
            bias_label in coverage_distribution
            or domain.endswith(".gov")
            or domain.endswith(".edu")
            # nasa.gov is intentionally NOT listed: `.gov` above already covers it.
            or domain in {"who.int"}
        )

        if bias_label in coverage_distribution:
            coverage_distribution[bias_label] += 1

        # Claim-review sites are shown but never counted (see _CLAIMREVIEW_DOMAIN_SET).
        if is_reputable and domain not in _CLAIMREVIEW_DOMAIN_SET:
            reputable_sources.append(source_dict)

    # Sort sources by relevance score descending
    sources_found.sort(key=lambda s: s["relevance_score"], reverse=True)

    # 7. Stance-aware consensus
    v_consensus_score, v_refute_score = compute_consensus(reputable_sources)

    return CrossRefResult(
        entities_extracted=entities,
        sources_found=sources_found,
        coverage_distribution=coverage_distribution,
        v_consensus_score=v_consensus_score,
        v_refute_score=v_refute_score,
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
    except (asyncio.TimeoutError, TimeoutError) as te:
        logger.warning(f"Cross-Reference Agent timed out on claim: {claim}")
        err_det = classify_provider_error(te, provider="crossref_agent")
        return make_neutral_crossref(
            degraded=True,
            error="Timeout: Cross-Reference Agent exceeded 8.0s",
            error_details=err_det
        )
    except Exception as exc:
        logger.error(f"Cross-Reference Agent failed on claim: {claim}. Error: {exc}")
        err_det = classify_provider_error(exc, provider="crossref_agent")
        return make_neutral_crossref(
            degraded=True,
            error=str(exc),
            error_details=err_det
        )
