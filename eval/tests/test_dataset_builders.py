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


from eval.datasets.build_fever import build_fever
from eval.datasets.build_liar import build_liar

LIAR_ROWS = "\n".join([
    "1.json\tfalse\tSays the moon is cheese.\tscience\tjoe\t\t\t\t0\t0\t0\t0\t0\tspeech",
    "2.json\ttrue\tUnemployment fell to 3.5 percent in 2019.\teconomy\tann\t\t\t\t0\t0\t0\t0\t0\ttweet",
    "3.json\tpants-fire\tVaccines contain microchips.\thealth\tbob\t\t\t\t0\t0\t0\t0\t0\tad",
    "4.json\thalf-true\tThe bill cut taxes for most families.\ttaxes\tsue\t\t\t\t0\t0\t0\t0\t0\tdebate",
])

FEVER_ROWS = "\n".join([
    '{"id": 1, "label": "SUPPORTS", "claim": "Paris is the capital of France."}',
    '{"id": 2, "label": "REFUTES", "claim": "The Pacific is the smallest ocean."}',
    '{"id": 3, "label": "NOT ENOUGH INFO", "claim": "Some band released an album."}',
])


def test_build_liar_maps_labels_and_sets_origin(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(LIAR_ROWS, encoding="utf-8")
    rows = build_liar(path, n=4, seed=1)
    by_text = {r["claim_text"]: r for r in rows}
    assert by_text["Says the moon is cheese."]["ground_truth_score"] == 0.15
    assert by_text["Vaccines contain microchips."]["ground_truth_score"] == 0.0
    for r in rows:
        claim = EvalClaim(**r)
        assert claim.source_dataset == "liar"
        assert claim.origin_domain == "politifact.com"   # leave-source-out target


def test_build_liar_is_seeded_and_stratified(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(LIAR_ROWS, encoding="utf-8")
    assert build_liar(path, n=2, seed=7) == build_liar(path, n=2, seed=7)
    labels = {r["tags"][0] for r in build_liar(path, n=4, seed=7)}
    assert labels == {"false", "true", "pants-fire", "half-true"}


def test_build_fever_maps_labels_without_origin(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(FEVER_ROWS, encoding="utf-8")
    rows = build_fever(path, n=3, seed=1)
    scores = sorted(r["ground_truth_score"] for r in rows)
    assert scores == [0.0, 0.5, 1.0]
    for r in rows:
        claim = EvalClaim(**r)
        assert claim.source_dataset == "fever" and claim.origin_domain is None

