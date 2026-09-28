"""
Build an EvalClaim dataset from a local LIAR split (Wang 2017; TSV, 14 columns:
id, label, statement, subject, speaker, ...). The user downloads the dataset; this script
never fetches anything. Every claim gets origin_domain="politifact.com": LIAR labels ARE
PolitiFact verdicts, so evidence from politifact.com is excluded during eval (C3).

Usage: uv run python -m eval.datasets.build_liar eval/datasets/raw/liar/test.tsv --n 40 \
         --out eval/datasets/liar_mapped.json
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

LIAR_SCORES = {
    "pants-fire": 0.0, "false": 0.15, "barely-true": 0.3,
    "half-true": 0.5, "mostly-true": 0.75, "true": 1.0,
}


def _stratified_sample(groups: Dict[str, List[dict]], n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    labels = sorted(groups)
    for label in labels:
        rng.shuffle(groups[label])
    picked: List[dict] = []
    i = 0
    while len(picked) < n and any(groups[l] for l in labels):
        label = labels[i % len(labels)]
        if groups[label]:
            picked.append(groups[label].pop())
        i += 1
    return picked


def build_liar(tsv_path: Path, n: int, seed: int = 42) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4 or row[1] not in LIAR_SCORES:
                continue
            groups[row[1]].append({
                "id": f"liar-{row[0].removesuffix('.json')}",
                "claim_text": row[2].strip(),
                "ground_truth_score": LIAR_SCORES[row[1]],
                "category": row[3].split(",")[0] or "politics",
                "claim_type": "verifiable",
                "expected_confidence": "medium",
                "tags": [row[1]],
                "source_dataset": "liar",
                "origin_domain": "politifact.com",
            })
    return _stratified_sample(groups, n, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/datasets/liar_mapped.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(build_liar(args.tsv, args.n, args.seed), indent=2), encoding="utf-8")
