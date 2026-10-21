"""The curated labels are frozen (tag plan9-labels-frozen, plan9-evals-handoff.md section 9).

Editing a label must fail here; it needs a new record run and a scorecard note, not a quiet change.
"""
from pathlib import Path

from eval.run_eval import dataset_sha256

SEED = Path(__file__).resolve().parents[1] / "datasets" / "curated_claims_seed.json"
# Canonical-JSON hash, identical on CRLF and LF checkouts. It is what summary.dataset_sha256 reports.
FROZEN_DATASET_SHA256 = "d20146d3a722fa3df922a364d8bed2e664590fc675313c597bbd7190a667f05c"


def test_curated_dataset_matches_the_frozen_labels():
    assert dataset_sha256([SEED]) == FROZEN_DATASET_SHA256, (
        "curated_claims_seed.json changed after the label freeze. Labels may only change with a new "
        "record run and a note in the scorecard; update this hash in the same commit and say why."
    )
