import re
from typing import Set

from app.services.nli import classify_pairs

# Both directions must entail at least this strongly for two claims to count as the same (C2).
# Track 3.1: calibrate against eval paraphrase/negation pairs (Plan 3).
EQUIVALENCE_MIN_ENTAILMENT = 0.8

# Integers, decimals, thousands separators and percentages: "3", "3.5", "1,000,000", "8%".
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _numbers(text: str) -> Set[str]:
    # "1,000" == "1000" and "3%" == "3" (so "3%" matches "3 percent"); "3" != "8" always.
    return {match.replace(",", "").rstrip("%") for match in _NUMBER_RE.findall(text)}


def numbers_match(a: str, b: str) -> bool:
    """NLI models are weak on numbers and dates, so any numeric difference makes claims distinct."""
    return _numbers(a) == _numbers(b)


def are_equivalent(claim: str, cached_claim: str) -> bool:
    """
    True only when the two claims assert the same thing: identical after normalization, or
    same numbers AND each entails the other. One-way entailment is a narrower or broader claim
    ("visited Paris" vs "visited France"), and contradiction is a negation. Neither is the same.
    Synchronous and CPU-bound: call via asyncio.to_thread.
    """
    if _normalize(claim) == _normalize(cached_claim):
        return True
    if not numbers_match(claim, cached_claim):
        return False
    forward, backward = classify_pairs([(cached_claim, claim), (claim, cached_claim)])
    return (
        forward.entailment >= EQUIVALENCE_MIN_ENTAILMENT
        and backward.entailment >= EQUIVALENCE_MIN_ENTAILMENT
    )
