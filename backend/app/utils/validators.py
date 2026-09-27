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

# JSON keys the agents parse out of LLM output. If untrusted text can carry these as
# `key:` syntax (or braces), an echoed claim becomes a syntactically valid answer object
# that the parser may adopt (R8.1).
PROMPT_JSON_KEYS = (
    "rating",
    "relevant",
    "opposite",
    "implied_rating",
    "s_fact",
    "explanation",
    "confidence",
)

_JSON_KEY_COLON_RE = re.compile(
    r"\b(" + "|".join(PROMPT_JSON_KEYS) + r")\s*:",
    flags=re.IGNORECASE,
)


def neutralize_for_prompt(text: str) -> str:
    """
    Make untrusted text structurally incapable of impersonating the model's JSON answer.

    R8.1 layer 1. Every agent prompt built from user- or third-party-controlled text runs
    through this before it reaches an LLM, because the model echoes that text back in its
    reasoning/preamble and the response parser then has to tell the echo apart from the
    model's own object. Removing the syntax at the source is the only reliable cut:

      - `{` / `}` become `(` / `)`  — an echoed claim can no longer form a JSON object.
      - `rating:` / `relevant:` / ... become `rating -` / `relevant -` — an echoed claim can
        no longer form a JSON (or bare) key/value pair for any key the agents read.

    Factual claims do not meaningfully need literal braces or `key:` JSON syntax, so this is
    lossless for the verification task. Layers 2 and 3 (the balanced-span parser and the
    regex fallback in `adk_helpers`) are defence in depth behind this.
    """
    if not text:
        return ""

    neutral = text.replace("{", "(").replace("}", ")")
    neutral = _JSON_KEY_COLON_RE.sub(r"\1 -", neutral)
    return neutral


def sanitize_text(text: str) -> str:
    """
    Sanitize raw input:
    - Strip HTML / XML / script tags
    - Strip tracking parameters from embedded URLs
    - Normalize whitespace
    - Neutralize JSON-answer syntax (braces / `key:`) so the claim cannot impersonate
      an LLM answer object when it is echoed back (R8.1)
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

    # 4. Neutralize JSON-answer syntax before the text can reach any prompt (R8.1)
    clean = neutralize_for_prompt(clean)

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
