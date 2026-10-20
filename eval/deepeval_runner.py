"""
DeepEval faithfulness guardrail and leaf groundedness runner (Plan 9, Task 5).

Evaluates claim verification explanations against retrieved source snippets using DeepEval's
FaithfulnessMetric (primary, threshold=0.8) and HallucinationMetric (reported).

Modes:
- guardrail: tests the explanation hallucination dataset (clean vs poisoned pairs).
- leaf: evaluates live agent transcripts from eval/results/leaf_transcript.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from app.config import settings

logger = logging.getLogger("deepeval_runner")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "eval" / "datasets" / "explanation_hallucination_set.json"
DEFAULT_TRANSCRIPT = BASE_DIR / "eval" / "results" / "leaf_transcript.jsonl"
DEFAULT_GUARDRAIL_OUT = BASE_DIR / "eval" / "results" / "deepeval_guardrail.json"
DEFAULT_LEAF_OUT = BASE_DIR / "eval" / "results" / "leaf_groundedness.json"


def get_judge_model():
    """
    Instantiate the judge model from settings.eval_judge_model (Decision D6).
    Refuses to run if eval_judge_model is None.
    """
    model_name = settings.eval_judge_model
    if not model_name:
        raise ValueError(
            "settings.eval_judge_model is None (Decision D6). "
            "Please configure EVAL_JUDGE_MODEL to run DeepEval evaluations."
        )

    from deepeval.models import LiteLLMModel
    return LiteLLMModel(model=model_name, temperature=0.0)


def items_from_leaf_transcript(transcript_path: Path | str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Parse verdict_extractor / factcheck_judge entries from a leaf transcript file.
    Extracts search snippets from <search_snippets> tags into retrieval_context,
    and explanation from the response JSON into actual_output.
    """
    path = Path(transcript_path)
    if not path.exists():
        return [], {"file_not_found": 1}

    items: List[Dict[str, Any]] = []
    skipped_reasons: Dict[str, int] = {
        "non_target_agent": 0,
        "missing_search_snippets": 0,
        "missing_explanation": 0,
        "invalid_json": 0,
    }

    target_agents = {"verdict_extractor", "factcheck_judge"}

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                skipped_reasons["invalid_json"] += 1
                continue

            agent_name = entry.get("name", "")
            if agent_name not in target_agents:
                skipped_reasons["non_target_agent"] += 1
                continue

            prompt = entry.get("prompt", "")
            snippets_match = re.search(r"<search_snippets>(.*?)</search_snippets>", prompt, re.DOTALL | re.IGNORECASE)
            if not snippets_match:
                # Fallback: check for <snippets> tags
                snippets_match = re.search(r"<snippets>(.*?)</snippets>", prompt, re.DOTALL | re.IGNORECASE)

            if not snippets_match:
                skipped_reasons["missing_search_snippets"] += 1
                continue

            raw_snippets = snippets_match.group(1).strip()
            # Split snippets if nested in <snippet> tags
            snippet_items = re.findall(r"<snippet>(.*?)</snippet>", raw_snippets, re.DOTALL | re.IGNORECASE)
            if snippet_items:
                snippets = [s.strip() for s in snippet_items if s.strip()]
            else:
                snippets = [s.strip() for s in raw_snippets.split("\n\n") if s.strip()]

            if not snippets:
                snippets = [raw_snippets]

            resp_payload = entry.get("response")
            explanation = ""
            rating = ""
            if isinstance(resp_payload, dict):
                explanation = resp_payload.get("explanation", "")
                rating = resp_payload.get("rating", "")
            elif isinstance(resp_payload, str):
                try:
                    parsed = json.loads(resp_payload)
                    if isinstance(parsed, dict):
                        explanation = parsed.get("explanation", "")
                        rating = parsed.get("rating", "")
                    else:
                        explanation = str(parsed)
                except Exception:
                    m = re.search(r'"explanation"\s*:\s*"([^"]+)"', resp_payload)
                    if m:
                        explanation = m.group(1)

            if not explanation:
                skipped_reasons["missing_explanation"] += 1
                continue

            claim_match = re.search(r"<claim>(.*?)</claim>", prompt, re.DOTALL | re.IGNORECASE)
            claim = claim_match.group(1).strip() if claim_match else f"Claim {idx+1}"

            items.append({
                "id": f"leaf-{idx+1:03d}",
                "claim": claim,
                "input": claim,
                "source_snippets": snippets,
                "retrieval_context": snippets,
                "context": snippets,
                "actual_output": explanation,
                "output": {"rating": rating, "explanation": explanation},
            })

    return items, skipped_reasons


def score_guardrail(
    items: List[Dict[str, Any]],
    judge: Any = None,
    delay_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Score the hallucination guardrail dataset.
    Flagging rule: faithfulness < 0.8.
    HallucinationMetric is reported and does not decide the flag.
    """
    scored_items: List[Dict[str, Any]] = []

    # If judge is not provided and not a callable, resolve it
    if judge is None:
        judge = get_judge_model()

    from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase

    for item in items:
        claim = item.get("claim") or item.get("input", "")
        output_data = item.get("output", {})
        if isinstance(output_data, dict):
            actual_output = output_data.get("explanation", "")
            rating = output_data.get("rating", "")
        else:
            actual_output = str(output_data)
            rating = ""

        snippets = item.get("source_snippets") or item.get("retrieval_context", [])
        is_hallu = item.get("is_hallucinated", False)
        hallu_type = item.get("hallucination_type")
        twin_of = item.get("twin_of")
        item_id = item.get("id", "")

        if callable(judge):
            # Test fixture / fake judge
            eval_res = judge(item)
            faith_score = float(eval_res.get("faithfulness", 1.0))
            hallu_score = float(eval_res.get("hallucination", 1.0))
            reason = eval_res.get("reason", "")
        else:
            # DeepEval LLM evaluation
            test_case = LLMTestCase(
                input=claim,
                actual_output=actual_output,
                retrieval_context=snippets,
                context=snippets,
            )
            fm = FaithfulnessMetric(threshold=0.8, model=judge, async_mode=False)
            fm.measure(test_case)
            faith_score = float(fm.score)
            reason = getattr(fm, "reason", "") or ""

            hm = HallucinationMetric(threshold=0.5, model=judge, async_mode=False)
            hm.measure(test_case)
            hallu_score = float(hm.score)

        flagged = (faith_score < 0.8)

        scored_items.append({
            "id": item_id,
            "claim": claim,
            "actual_output": actual_output,
            "rating": rating,
            "is_hallucinated": is_hallu,
            "hallucination_type": hallu_type,
            "twin_of": twin_of,
            "faithfulness": faith_score,
            "hallucination": hallu_score,
            "flagged": flagged,
            "reason": reason,
        })

        if delay_s > 0:
            time.sleep(delay_s)

    total_poisoned = sum(1 for x in scored_items if x["is_hallucinated"])
    detected_poisoned = sum(1 for x in scored_items if x["is_hallucinated"] and x["flagged"])
    hallucination_catch_rate = (detected_poisoned / total_poisoned) if total_poisoned > 0 else 1.0

    total_clean = sum(1 for x in scored_items if not x["is_hallucinated"])
    flagged_clean = sum(1 for x in scored_items if not x["is_hallucinated"] and x["flagged"])
    false_positive_rate = (flagged_clean / total_clean) if total_clean > 0 else 0.0

    model_label = getattr(judge, "model_name", None) or getattr(judge, "model", None) or str(judge)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": model_label,
        "total_items": len(scored_items),
        "clean_items": total_clean,
        "poisoned_items": total_poisoned,
        "detected_poisoned": detected_poisoned,
        "hallucination_catch_rate": round(hallucination_catch_rate, 4),
        "flagged_clean": flagged_clean,
        "false_positive_rate": round(false_positive_rate, 4),
        "items": scored_items,
    }


def score_leaf_groundedness(
    items: List[Dict[str, Any]],
    judge: Any = None,
    delay_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Score agent explanations from live transcripts for groundedness.
    """
    scored_items: List[Dict[str, Any]] = []

    if judge is None:
        judge = get_judge_model()

    from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase

    for item in items:
        claim = item.get("claim") or item.get("input", "")
        actual_output = item.get("actual_output") or item.get("output", {}).get("explanation", "")
        snippets = item.get("source_snippets") or item.get("retrieval_context", [])
        item_id = item.get("id", "")

        if callable(judge):
            eval_res = judge(item)
            faith_score = float(eval_res.get("faithfulness", 1.0))
            hallu_score = float(eval_res.get("hallucination", 1.0))
            reason = eval_res.get("reason", "")
        else:
            test_case = LLMTestCase(
                input=claim,
                actual_output=actual_output,
                retrieval_context=snippets,
                context=snippets,
            )
            fm = FaithfulnessMetric(threshold=0.8, model=judge, async_mode=False)
            fm.measure(test_case)
            faith_score = float(fm.score)
            reason = getattr(fm, "reason", "") or ""

            hm = HallucinationMetric(threshold=0.5, model=judge, async_mode=False)
            hm.measure(test_case)
            hallu_score = float(hm.score)

        flagged = (faith_score < 0.8)

        scored_items.append({
            "id": item_id,
            "claim": claim,
            "actual_output": actual_output,
            "faithfulness": faith_score,
            "hallucination": hallu_score,
            "flagged": flagged,
            "reason": reason,
        })

        if delay_s > 0:
            time.sleep(delay_s)

    total_items = len(scored_items)
    unsupported_items = sum(1 for x in scored_items if x["flagged"])
    grounded_items = total_items - unsupported_items
    groundedness_rate = (grounded_items / total_items) if total_items > 0 else 1.0
    mean_faithfulness = (sum(x["faithfulness"] for x in scored_items) / total_items) if total_items > 0 else 1.0

    model_label = getattr(judge, "model_name", None) or getattr(judge, "model", None) or str(judge)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": model_label,
        "total_items": total_items,
        "grounded_items": grounded_items,
        "unsupported_items": unsupported_items,
        "groundedness_rate": round(groundedness_rate, 4),
        "mean_faithfulness": round(mean_faithfulness, 4),
        "items": scored_items,
    }


def main():
    parser = argparse.ArgumentParser(description="DeepEval faithfulness guardrail runner")
    parser.add_argument("--mode", choices=["guardrail", "leaf"], default="guardrail",
                        help="guardrail (evaluates hallucination set) or leaf (evaluates leaf transcript)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="path to explanation_hallucination_set.json (guardrail mode)")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT,
                        help="path to leaf_transcript.jsonl (leaf mode)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="delay in seconds between sequential evaluations")
    parser.add_argument("--output", type=Path, default=None,
                        help="output path for the results JSON")
    args = parser.parse_args()

    if args.mode == "guardrail":
        out_path = args.output or DEFAULT_GUARDRAIL_OUT
        print(f"Loading guardrail dataset from {args.dataset}...")
        with open(args.dataset, "r", encoding="utf-8") as f:
            items = json.loads(f.read())
        print(f"Running guardrail evaluation on {len(items)} items (delay={args.delay}s)...")
        results = score_guardrail(items, delay_s=args.delay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Guardrail results saved to {out_path}:")
        print(f"  • Catch Rate (poisoned detected): {results['detected_poisoned']}/{results['poisoned_items']} ({results['hallucination_catch_rate']:.1%})")
        print(f"  • False Positive (clean flagged): {results['flagged_clean']}/{results['clean_items']} ({results['false_positive_rate']:.1%})")
    else:
        out_path = args.output or DEFAULT_LEAF_OUT
        print(f"Loading leaf transcript from {args.transcript}...")
        items, skipped = items_from_leaf_transcript(args.transcript)
        print(f"Parsed {len(items)} eligible leaf entries (skipped: {skipped})...")
        if not items:
            print("No items to evaluate.")
            sys.exit(1)
        results = score_leaf_groundedness(items, delay_s=args.delay)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Leaf groundedness results saved to {out_path}:")
        print(f"  • Groundedness rate: {results['grounded_items']}/{results['total_items']} ({results['groundedness_rate']:.1%})")
        print(f"  • Mean Faithfulness: {results['mean_faithfulness']:.4f}")


if __name__ == "__main__":
    main()
