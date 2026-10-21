"""
Curated dataset integrity (Task C review B10, N1): mixed labels must be genuinely mixed, the plan's
bucket minimums hold, and the provenance fields are filled in for every claim added after the seed.
"""
import json
import re
from pathlib import Path

import pytest

from eval.datasets.claim_dataset_schema import EvalClaim

SEED = Path(__file__).resolve().parent.parent / "datasets" / "curated_claims_seed.json"

# A claim that asserts an absolute cannot honestly be "mixed": the nuance in its label_note makes it false.
ABSOLUTES = re.compile(
    r"\b(zero|sole|entirely|conclusively|all|every|directly|confirmed|definitely|binding|always|never|proven)\b",
    re.IGNORECASE,
)
# Unsourced placeholders about invented 2026 events: 0.5 means "cannot be corroborated", not "mixed".
PLACEHOLDER_IDS = {"claim-020", "claim-021"}


def _claims():
    return [EvalClaim.model_validate(c) for c in json.loads(SEED.read_text(encoding="utf-8"))]


def _verifiable():
    return [c for c in _claims() if c.claim_type == "verifiable"]


def test_mixed_labelled_claims_do_not_assert_absolutes():
    offenders = {
        c.id: ABSOLUTES.findall(c.claim_text)
        for c in _verifiable()
        if c.ground_truth_score == 0.5 and c.id not in PLACEHOLDER_IDS and ABSOLUTES.search(c.claim_text)
    }
    assert not offenders, f"a categorical claim cannot be labelled mixed (0.5): {offenders}"


def test_bucket_minimums_from_the_plan():
    verifiable = _verifiable()
    true_ = [c for c in verifiable if c.ground_truth_score >= 0.67]
    false_ = [c for c in verifiable if c.ground_truth_score < 0.34]
    mixed = [c for c in verifiable if 0.34 <= c.ground_truth_score < 0.67]
    assert len(true_) >= 15 and len(false_) >= 15 and len(mixed) >= 12


@pytest.mark.parametrize("field", ["label_source", "label_note"])
def test_every_claim_added_after_the_seed_has_provenance(field):
    missing = [c.id for c in _verifiable()
               if int(c.id.split("-")[1]) >= 26 and not getattr(c, field)]
    assert not missing, f"{field} missing on {missing}"


def test_a_recent_event_must_not_cite_a_source_older_than_the_event():
    # claim tagged recent_2025_2026 whose only provenance is a URL dated before 2025
    stale = []
    for c in _verifiable():
        if "recent_2025_2026" in (c.tags or []):
            years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", c.label_source or "")]
            if years and max(years) < 2025:
                stale.append((c.id, c.label_source))
    assert not stale, f"recent claims citing pre-2025 sources: {stale}"
