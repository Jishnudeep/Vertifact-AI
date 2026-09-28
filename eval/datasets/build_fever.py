"""
Build an EvalClaim dataset from a local FEVER jsonl split (e.g. shared_task_dev.jsonl;
fields: id, label, claim). FEVER claims are Wikipedia-derived and rarely have ClaimReview
coverage: report them as their own segment, never blended into the headline (C3).

Usage: uv run python -m eval.datasets.build_fever eval/datasets/raw/fever/shared_task_dev.jsonl \
         --n 30 --out eval/datasets/fever_subset.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from eval.datasets.build_liar import _stratified_sample

FEVER_SCORES = {"SUPPORTS": 1.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 0.5}


def build_fever(jsonl_path: Path, n: int, seed: int = 42) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in FEVER_SCORES:
                continue
            groups[row["label"]].append({
                "id": f"fever-{row['id']}",
                "claim_text": row["claim"].strip(),
                "ground_truth_score": FEVER_SCORES[row["label"]],
                "category": "encyclopedic",
                "claim_type": "verifiable",
                "expected_confidence": "low",
                "tags": [row["label"].lower().replace(" ", "_")],
                "source_dataset": "fever",
                "origin_domain": None,
            })
    return _stratified_sample(groups, n, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/datasets/fever_subset.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(build_fever(args.jsonl, args.n, args.seed), indent=2), encoding="utf-8")
