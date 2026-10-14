"""
Evidence-source domain lists (TDD §8.2).

Moved out of `search_api.py` unchanged (Plan 6 Task 1) so they outlive the Tavily client.
Members are compared against `extract_domain(...)` output.
"""
from typing import List

# Broad domain scope for fact-check / news evidence retrieval
FACTCHECK_DOMAINS: List[str] = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "healthfeedback.org",
    "apnews.com",
    "reuters.com",
    "usatoday.com",
    "factcheck.afp.com",
    "leadstories.com",
    "checkyourfact.com",
    "climatefeedback.org",
]

# Strict fact-checking authorities for Layer-2 verdict authority (excluding general news)
CLAIMREVIEW_DOMAINS: List[str] = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "healthfeedback.org",
    "factcheck.afp.com",
    "afp.com",
    "leadstories.com",
    "checkyourfact.com",
    "climatefeedback.org",
]
