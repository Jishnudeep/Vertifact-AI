import re
from typing import Tuple, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

MAX_INPUT_LENGTH = 2000

# Tracking parameters commonly found in shared links
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_eid", "igshid", "_hsenc", "_hsmi"
}

OPINION_PATTERNS = [
    r"^(i think|i believe|in my opinion|in my view|i feel)\b",
    r"\b((is|was|are|were) the (best|worst|greatest|most beautiful|ugliest|most boring|tastiest|funniest))\b",
    r"\b(should be banned|should be illegal|(is|was|are|were)\s+(totally|completely|utterly)\s+(overrated|underrated|pointless|meaningless))\b",
    r"\b(my favorite|i personally prefer|to be honest, i like)\b"
]

# Patterns indicating questions rather than factual claims
QUESTION_PATTERNS = [
    r"^(why|how|what|when|where|who|which|is it true that|can someone explain|could you tell me)\b",
    r"\?\s*$"
]

def sanitize_text(text: str) -> str:
    """
    Sanitize raw input:
    - Strip HTML / XML / script tags
    - Strip tracking parameters from embedded URLs
    - Normalize whitespace
    """
    if not text:
        return ""
        
    # 1. Remove script tags and HTML tags
    clean = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)
    clean = re.sub(r'<[^>]+>', ' ', clean)
    
    # 2. Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # 3. Strip tracking parameters if input looks like a URL
    if clean.startswith("http://") or clean.startswith("https://"):
        try:
            parsed = urlparse(clean)
            if parsed.query:
                query_params = parse_qs(parsed.query, keep_blank_values=False)
                filtered_query = {
                    k: v for k, v in query_params.items()
                    if k.lower() not in TRACKING_PARAMS
                }
                new_query = urlencode(filtered_query, doseq=True)
                clean = urlunparse(parsed._replace(query=new_query))
        except Exception:
            pass
            
    return clean

def validate_input_length(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that input is not empty and within the 2000-character cap.
    """
    if not text or not text.strip():
        return False, "Input claim cannot be empty."
    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input claim exceeds maximum length of {MAX_INPUT_LENGTH} characters."
    return True, None

def is_verifiable_claim(text: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if the input text represents a verifiable factual claim.
    Rejects subjective opinions and questions (HTTP 422 per TDD Section 3).
    """
    clean_lower = text.strip().lower()
    
    # Check question patterns
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, clean_lower):
            return False, "Input is a question or inquiry, not a verifiable factual assertion."
            
    # Check opinion patterns
    for pattern in OPINION_PATTERNS:
        if re.search(pattern, clean_lower):
            return False, "Input is classified as a subjective opinion or value judgement, not a verifiable factual claim."
            
    return True, None
