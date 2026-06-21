import asyncio
import logging
import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import settings
from app.utils.adk_helpers import run_adk_agent, parse_adk_response

logger = logging.getLogger(__name__)
TIMEOUT = 8.0

# Ensure GROQ_API_KEY is set in environment for LiteLLM
if settings.groq_api_key:
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)

# Lazy load spaCy NLP model
nlp = None
def get_spacy_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    return nlp

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Common factual acronyms to exclude from clickbait ALL CAPS heuristics
EXCLUDED_ACRONYMS = {
    "GDP", "CPI", "USD", "EUR", "GBP", "JPY", "INR", "CNY",
    "USA", "NYC", "DC", "UN", "EU", "UK", "US", "WHO", "FDA", "CDC",
    "NASA", "BLS", "FBI", "CIA", "IRS", "EPA", "SEC", "FTC", "FCC",
    "AI", "ML", "NLP", "CPU", "GPU", "RAM", "ROM", "OS", "IT",
    "COVID", "SARS", "HIV", "AIDS", "DNA", "RNA", "IMF", "WB",
    "NATO", "UNICEF", "UNESCO", "GOP", "DNC", "PAC", "SCOTUS", "POTUS"
}

@dataclass
class LinguisticResult:
    clickbait_syntax: float       # 0.0 (factual) to 1.0 (heavy clickbait)
    emotional_framing: float      # 0.0 (neutral) to 1.0 (emotionally charged)
    sensationalism: float         # 0.0 (measured) to 1.0 (sensational)
    informational_density: float  # 0.0 (sparse) to 1.0 (information-rich)
    b_ling_score: float           # Composite: mean of clickbait, emotional, sensationalism, and (1 - density)
    flagged_phrases: List[str] = field(default_factory=list)

NEUTRAL = LinguisticResult(
    clickbait_syntax=0.5,
    emotional_framing=0.5,
    sensationalism=0.5,
    informational_density=0.5,
    b_ling_score=0.5,
    flagged_phrases=[]
)

# Duplicated JSON parsing and ADK runner functions removed. Imported from app.utils.adk_helpers.

async def _rate_sensationalism_adk(claim: str) -> float:
    """
    Use an ADK agent to rate the sensationalism of the text on a 0.0 to 1.0 scale.
    """
    instruction = (
        "You are a linguistic analysis assistant. Rate the sensationalism of the given text on a 0.0 to 1.0 scale "
        "(where 0.0 is completely neutral and objective, and 1.0 is extremely sensational, exaggerated, or clickbaity). "
        "Consider word choice, tone, and whether claims are presented with appropriate nuance. Return your output "
        "in raw JSON format with a single key: 'sensationalism' (a float between 0.0 and 1.0). Use double quotes."
    )
    # SECURITY: Prompt injection risk via untrusted claim text.
    # Mitigation: Handled at eval/guardrail phase (faithfulness test + scoring confidence-cap).
    # TODO: Verify safety in eval/guardrail calibration.
    prompt = f"Analyze this text: \"{claim}\""
    
    try:
        content = await run_adk_agent("sensationalism_analyzer", instruction, prompt, "verifact-linguistic")
        data = parse_adk_response(content, expected_keys=["sensationalism"])
        score_val = data.get("sensationalism")
        if score_val is not None:
            return min(max(float(score_val), 0.0), 1.0)
    except Exception as exc:
        logger.error(f"Error calling ADK sensationalism agent: {exc}")
        
    return _rate_sensationalism_heuristics(claim)

def _rate_sensationalism_heuristics(claim: str) -> float:
    """
    Fallback word-list heuristic for sensationalism scoring.
    """
    sensational_words = [
        "shocking", "insane", "miracle", "secret", "exposed", "disaster",
        "panic", "crisis", "chaos", "apocalypse", "doom", "catastrophe",
        "scandal", "unbelievable", "impossible", "amazing", "magic", "wonder"
    ]
    words = re.findall(r'\w+', claim.lower())
    match_count = sum(1 for w in words if w in sensational_words)
    return min(match_count * 0.25, 1.0)

def _analyze_clickbait(claim: str) -> float:
    """
    Heuristics to calculate clickbait syntax.
    """
    score = 0.0
    if "?" in claim:
        score += 0.3
        
    words = re.findall(r'\b[A-Z]{3,}\b', claim)
    caps_words = [w for w in words if w not in EXCLUDED_ACRONYMS]
    if caps_words:
        score += 0.3
        
    clickbait_patterns = [
        r"\byou won't believe\b",
        r"\bshocking\b",
        r"\bunbelievable\b",
        r"\bmagic formula\b",
        r"\bsecret to\b",
        r"\bthe truth about\b",
        r"\bwhat happens next\b",
        r"\bthis is why\b",
        r"\bscientists found\b",
        r"\bbreakthrough\b",
        r"\bbreaking\b",
        r"\bwarning\b",
        r"\bcheck this out\b",
        r"\bblow your mind\b",
        r"\bshocked\b"
    ]
    claim_lower = claim.lower()
    for pattern in clickbait_patterns:
        if re.search(pattern, claim_lower):
            score += 0.2
            
    return min(score, 1.0)

def _analyze_density(nlp_doc) -> float:
    """
    Calculate informational density based on spaCy POS/NER tags.
    """
    total_tokens = len(nlp_doc)
    if total_tokens == 0:
        return 0.5
        
    factual_tokens = 0
    for token in nlp_doc:
        if token.pos_ in ["PROPN", "NUM"] or token.ent_type_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORG", "GPE"]:
            factual_tokens += 1
            
    density = factual_tokens / total_tokens
    return min(max(density, 0.0), 1.0)

async def _analyze(claim: str) -> LinguisticResult:
    """
    Run the full 4-axis analysis on the claim text.
    """
    if not claim or not claim.strip():
        return NEUTRAL
        
    # spaCy parsing
    nlp_model = get_spacy_nlp()
    doc = nlp_model(claim)
    
    # Axis 1: Clickbait syntax
    clickbait = _analyze_clickbait(claim)
    
    # Axis 2: Emotional framing
    vader_scores = vader_analyzer.polarity_scores(claim)
    compound = vader_scores.get("compound", 0.0)
    emotional = abs(compound)
    
    # Axis 3: Sensationalism (ADK agent with heuristics fallback)
    sensational = await _rate_sensationalism_adk(claim)
    
    # Axis 4: Informational density
    density = _analyze_density(doc)
    
    # Flag phrases
    flagged_phrases = []
    
    # CAPS flags (excluding common acronyms)
    words = re.findall(r'\b[A-Z]{3,}\b', claim)
    caps_words = [w for w in words if w not in EXCLUDED_ACRONYMS]
    for w in caps_words:
        flagged_phrases.append(f"ALL CAPS word: '{w}'")
        
    # Trigger words
    clickbait_words = ["shocking", "unbelievable", "you won't believe", "breaking", "warning", "must watch", "insane"]
    claim_lower = claim.lower()
    for w in clickbait_words:
        if w in claim_lower:
            flagged_phrases.append(f"Clickbait/sensational phrase: '{w}'")
            
    # Emotional flags
    if emotional >= 0.5:
        flagged_phrases.append(f"Strong emotional sentiment (VADER compound: {compound:.2f})")
        
    # b_ling_score calculation (density is inverted: 1 - density)
    b_ling_score = (clickbait + emotional + sensational + (1.0 - density)) / 4.0
    b_ling_score = round(b_ling_score, 3)
    
    return LinguisticResult(
        clickbait_syntax=round(clickbait, 3),
        emotional_framing=round(emotional, 3),
        sensationalism=round(sensational, 3),
        informational_density=round(density, 3),
        b_ling_score=b_ling_score,
        flagged_phrases=flagged_phrases
    )

async def run(claim: str) -> LinguisticResult:
    """
    Exposes the LinguisticAnalysisAgent interface.
    Enforces a strict 8-second timeout, falling back to NEUTRAL.
    """
    try:
        return await asyncio.wait_for(_analyze(claim), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"Linguistic Agent timed out on claim: {claim}")
        return NEUTRAL
    except Exception as exc:
        logger.error(f"Linguistic Agent failed on claim: {claim}. Error: {exc}")
        return NEUTRAL
