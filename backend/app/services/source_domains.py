"""
Evidence-source domain lists (TDD §8.2).

Moved out of `search_api.py` (Plan 6 Task 1) to decouple domain lists from the search client.

Members are compared against `extract_domain(...)` output, which keeps only the registrable
domain for standard TLDs: `extract_domain("factcheck.afp.com") == "afp.com"`. The
`factcheck.afp.com` entries can therefore never match on their own; `afp.com` is the member
that does (added to FACTCHECK_DOMAINS in the Task 1 review, matching TDD §8.2). The old
entries are kept so the lists stay a superset of the documented members.
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
    "afp.com",
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
