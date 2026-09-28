import json
from pathlib import Path

from eval.datasets.claim_dataset_schema import EvalClaim

SEED = Path(__file__).resolve().parent.parent / "datasets" / "curated_claims_seed.json"


def test_existing_seed_defaults_to_curated_with_no_origin():
    for raw in json.loads(SEED.read_text(encoding="utf-8")):
        claim = EvalClaim(**raw)
        assert claim.source_dataset == "curated"
        assert claim.origin_domain is None


def test_origin_domain_round_trips():
    claim = EvalClaim(id="liar-1", claim_text="x", ground_truth_score=0.0, category="politics",
                      source_dataset="liar", origin_domain="politifact.com")
    assert claim.origin_domain == "politifact.com"
