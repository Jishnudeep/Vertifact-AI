import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from app.agents.linguistic_agent import get_spacy_nlp
from app.services.source_domains import CLAIMREVIEW_DOMAINS
from app.services.grounded_search import GroundedSource, grounded_search
from app.services.page_fetch import PageText, fetch_page_text
from app.services.errors import EvidenceAPIError
from app.services.bias_lookup import get_bias_label, get_outlet_name
from app.services.nli import NLIScores, classify_pairs
from app.utils.errors import AgentErrorDetails, classify_provider_error


logger = logging.getLogger(__name__)
TIMEOUT = 8.0

# C1: a source counts as supporting/refuting only when the NLI model is at least this sure;
# anything less is "neutral" and moves neither score.
# Track 3.1: calibrate STANCE_MIN_PROB against the eval stance fixtures (Plan 3).
STANCE_MIN_PROB = 0.6

# Premise = fetched page title + text, truncated before tokenization (the model also
# truncates at settings.nli_max_length tokens).
PREMISE_MAX_CHARS = 1000

# Plan 6: grounding has no per-result relevance, so every cited source carries 1.0 and each
# supporting (refuting) reputable source adds 1 to its side; SATURATION_COUNT of them = 1.0.
CITED_RELEVANCE = 1.0
SATURATION_COUNT = 5

# Fact-check sites belong to the Fact-Check agent. Their pages restate the claim they
# debunk ("Claim: X. Rating: False"), which NLI reads as entailment, so counting them here
# would both double count and invert their stance.
_CLAIMREVIEW_DOMAIN_SET = frozenset(CLAIMREVIEW_DOMAINS)

@dataclass
class CrossRefResult:
    entities_extracted: List[str]             # People, organizations, locations
    sources_found: List[Dict[str, Any]]       # {url, title, publisher, bias_label, relevance_score, stance, stance_confidence}
    coverage_distribution: Dict[str, int]     # {"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0}
    v_consensus_score: float                  # 0.0 to 1.0 (0.0 when no sources or degraded)
    v_refute_score: float = 0.0               # 0.0 to 1.0: refuting reputable coverage (count / 5)
    degraded: bool = False                    # True if agent timed out or failed
    error: Optional[str] = None               # Failure message if degraded
    error_details: Optional[AgentErrorDetails] = None # Structured error classification if degraded
    search_suggestions_html: Optional[str] = None     # Google search-suggestion HTML (Plan 6 D5); set when the search ran

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
    Stance-aware consensus (C1, counting rule from Plan 6). Returns (v_consensus, v_refute),
    each in [0, 1]. Each supporting (refuting) reputable source adds 1; SATURATION_COUNT (5)
    of them saturate the scale (PRD §6). Neutral and unfetched (stance None) sources add to
    neither side.
    """
    support = sum(1 for s in reputable_sources if s.get("stance") == "supports")
    refute = sum(1 for s in reputable_sources if s.get("stance") == "refutes")
    return (
        round(min(support / SATURATION_COUNT, 1.0), 3),
        round(min(refute / SATURATION_COUNT, 1.0), 3),
    )


def _premise(page: PageText) -> str:
    title = (page.title or "").strip()
    text = (page.text or "").strip()
    return f"{title}. {text}".strip(". ")[:PREMISE_MAX_CHARS]


_BIAS_BUCKETS = ("Left", "Lean Left", "Center", "Lean Right", "Right")


def _is_reputable(domain: str, bias_label: str) -> bool:
    """
    AllSides-rated outlets, or the institutional/primary allowlist (B4/D3). Wire services
    (reuters/apnews/bbc) are intentionally NOT listed: they are AllSides-rated, so the first
    clause already covers them.
    Track 3.1: tighten institutional-domain trust (a .edu personal page is not a source).
    """
    return (
        bias_label in _BIAS_BUCKETS
        or domain.endswith(".gov")
        or domain.endswith(".edu")
        # nasa.gov is intentionally NOT listed: `.gov` above already covers it.
        or domain in {"who.int"}
    )


async def _fetch_cited(sources: List[GroundedSource]) -> Dict[str, Optional[PageText]]:
    """
    Fetch every cited page concurrently, keyed by source domain. `fetch_page_text` never
    raises and bounds itself (settings.page_fetch_timeout_s); a None page is "not fetched"
    and is never counted. Cancellation (the agent budget) propagates.
    """
    if not sources:
        return {}
    pages = await asyncio.gather(
        *(fetch_page_text(s.url, expected_domain=s.domain) for s in sources),
        return_exceptions=True,
    )
    fetched: Dict[str, Optional[PageText]] = {}
    for source, page in zip(sources, pages):
        if isinstance(page, BaseException):
            if not isinstance(page, Exception):
                raise page
            # Contract breach by the fetcher: treat as unfetched, but never silently.
            logger.warning(f"Cited page fetch raised {type(page).__name__} for {source.domain}")
            page = None
        fetched[source.domain] = page
    return fetched

async def _analyze(claim: str) -> CrossRefResult:
    """
    Internal implementation of cross-reference analysis.
    """
    if not claim or not claim.strip():
        return make_neutral_crossref(degraded=False)

    # 1. spaCy NER extraction. It feeds `entities_extracted` only; since Plan 6 the search
    # agent builds its own queries. Load and parse both run in a worker thread: a cold
    # `spacy.load` takes seconds and, on the loop, would stall every request and `wait_for`.
    doc = await asyncio.to_thread(lambda: get_spacy_nlp()(claim))
    
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

    # 2. One grounded coverage search (Plan 6 D2). An outage (key missing, budget exhausted,
    # provider error) is degradation (R7.B4); a search that cites nothing is "no sources".
    try:
        search = await grounded_search("coverage", claim)
    except EvidenceAPIError as exc:
        logger.error(f"Cross-Reference evidence retrieval unavailable: {exc}")
        return make_neutral_crossref(
            degraded=True,
            error=f"Evidence retrieval API unavailable: {exc}",
            error_details=classify_provider_error(exc, provider="gemini_search"),
        )

    # 3. Fetch every cited non-claim-review page concurrently (D3: stance is read from the
    # publisher's own page, never from Gemini's answer). Claim-review pages are listed but
    # never scored (see _CLAIMREVIEW_DOMAIN_SET), so they are not fetched.
    to_fetch = [s for s in search.sources if s.domain not in _CLAIMREVIEW_DOMAIN_SET]
    pages = await _fetch_cited(to_fetch)

    # 4. Stance per fetched page (C1): premise = page title + text, hypothesis = claim.
    scored = [(domain, page) for domain, page in pages.items() if page is not None]
    pairs = [(_premise(page), claim) for _, page in scored]
    try:
        nli_scores = await asyncio.to_thread(classify_pairs, pairs) if pairs else []
    except Exception as exc:
        # Without stance, coverage cannot be read as agreement: degrade, never fall back to
        # the coverage-only count (that is the bug C1 fixes).
        logger.error(f"Stance model failed during cross-reference: {exc}")
        return make_neutral_crossref(
            degraded=True,
            error=f"Stance model unavailable: {exc}",
            error_details=classify_provider_error(exc, provider="nli"),
        )
    scores_by_domain = {domain: scores for (domain, _), scores in zip(scored, nli_scores)}

    # 5. Bias lookup, coverage distribution and the reputable set. Coverage distribution
    # counts every LISTED AllSides-rated source (fetched or not, claim-review or not), as
    # before Plan 6; only the consensus scores require a fetched page.
    coverage_distribution = {bucket: 0 for bucket in _BIAS_BUCKETS}
    sources_found: List[Dict[str, Any]] = []
    reputable_sources: List[Dict[str, Any]] = []

    for source in search.sources:  # first-cited order (grounding has no relevance ranking)
        domain = source.domain
        # Guardrail: the label comes from AllSides for the grounding domain only, never from
        # the page or an LLM.
        bias_label = get_bias_label(domain)
        known_outlet = get_outlet_name(domain)
        publisher = known_outlet if known_outlet else _clean_publisher_name(domain)

        is_claimreview = domain in _CLAIMREVIEW_DOMAIN_SET
        page = pages.get(domain)
        if is_claimreview or page is None:
            # Claim-review: stance unreliable, no chip. Unfetched: nothing was read, so no
            # stance. Either way the source is shown but never counted.
            stance, stance_confidence = None, None
        else:
            stance, stance_confidence = label_stance(scores_by_domain[domain])

        source_dict = {
            # A fetched page has the publisher's own URL; otherwise keep the cited (possibly
            # Google-redirect) URL so the citation can still be displayed.
            "url": page.final_url if page else source.url,
            "title": (page.title if page and page.title else source.title),
            "publisher": publisher,
            "bias_label": bias_label,
            "relevance_score": CITED_RELEVANCE,
            "stance": stance,
            "stance_confidence": stance_confidence,
        }
        sources_found.append(source_dict)

        if bias_label in coverage_distribution:
            coverage_distribution[bias_label] += 1

        if _is_reputable(domain, bias_label) and not is_claimreview and stance is not None:
            reputable_sources.append(source_dict)

    # 6. Stance-aware consensus
    v_consensus_score, v_refute_score = compute_consensus(reputable_sources)

    return CrossRefResult(
        entities_extracted=entities,
        sources_found=sources_found,
        coverage_distribution=coverage_distribution,
        v_consensus_score=v_consensus_score,
        v_refute_score=v_refute_score,
        degraded=False,
        error=None,
        search_suggestions_html=search.search_suggestions_html,
    )

async def run(claim: str, timeout: Optional[float] = None) -> CrossRefResult:
    """
    Exposes the CrossReferenceAgent interface.
    Enforces a timeout (default TIMEOUT = 8 s; the orchestrator may pass less to fit its
    request deadline), falling back to degraded neutral (v_consensus=0.0).
    """
    budget = TIMEOUT if timeout is None else timeout
    try:
        return await asyncio.wait_for(_analyze(claim), timeout=budget)
    except (asyncio.TimeoutError, TimeoutError) as te:
        logger.warning(f"Cross-Reference Agent timed out on claim: {claim}")
        err_det = classify_provider_error(te, provider="crossref_agent")
        return make_neutral_crossref(
            degraded=True,
            error=f"Timeout: Cross-Reference Agent exceeded {budget:.1f}s",
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
