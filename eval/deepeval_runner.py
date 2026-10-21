"""
DeepEval faithfulness guardrail and leaf groundedness runner (Plan 9, Task 5).

Evaluates claim verification explanations against retrieved source snippets using DeepEval's
FaithfulnessMetric (primary, threshold=0.8) and HallucinationMetric (reported).

Modes:
- guardrail: tests the explanation hallucination dataset (clean vs poisoned pairs).
- leaf: evaluates the Fact-Check agent's real decisions from eval/results/leaf_transcript.jsonl.

Every judge call spends quota, so the CLI needs an explicit --live. A missing, NaN or errored judge
score counts as FLAGGED (fail closed), and any judge error makes the run INVALID.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repo root and backend are on sys.path (the script is run as `python eval/deepeval_runner.py`)
BASE_DIR = Path(__file__).resolve().parent.parent
for _p in (BASE_DIR, BASE_DIR / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from app.config import settings  # noqa: E402

logger = logging.getLogger("deepeval_runner")

DEFAULT_DATASET = BASE_DIR / "eval" / "datasets" / "explanation_hallucination_set.json"
DEFAULT_TRANSCRIPT = BASE_DIR / "eval" / "results" / "record_run.leaf_transcript.jsonl"
DEFAULT_GUARDRAIL_OUT = BASE_DIR / "eval" / "results" / "deepeval_guardrail.json"
DEFAULT_LEAF_OUT = BASE_DIR / "eval" / "results" / "leaf_groundedness.json"

FAITHFULNESS_THRESHOLD = 0.8
# Free-tier models with a 20 requests/day cap: 30 items need well over 100 judge calls (handoff section 7).
_QUOTA_STARVED_JUDGE = re.compile(r"gemini[-/]2\.5-flash(-lite)?\b", re.IGNORECASE)
# FaithfulnessMetric (about 4 calls) + HallucinationMetric without reasons (about 1)
JUDGE_CALLS_PER_ITEM = 5

LEAF_AGENT = "factcheck_judge"


def is_flagged(faithfulness: Optional[float]) -> bool:
    """Fail closed: only a finite score at or above the threshold counts as faithful."""
    return not (faithfulness is not None and math.isfinite(faithfulness) and faithfulness >= FAITHFULNESS_THRESHOLD)


def get_judge_model():
    """
    Instantiate the judge model from settings.eval_judge_model (Decision D6).
    Refuses to run if eval_judge_model is None or a 20-requests-per-day free-tier model.
    """
    model_name = settings.eval_judge_model
    if not model_name:
        raise ValueError(
            "settings.eval_judge_model is None (Decision D6). "
            "Please configure EVAL_JUDGE_MODEL to run DeepEval evaluations."
        )
    if _QUOTA_STARVED_JUDGE.search(model_name):
        raise ValueError(
            f"EVAL_JUDGE_MODEL={model_name!r} has a 20 RPD free-tier cap; a run needs about "
            f"{JUDGE_CALLS_PER_ITEM * 30} calls. Use gemini/gemini-3.1-flash-lite and disclose self-judging."
        )

    from deepeval.models import LiteLLMModel
    # LiteLLM's gemini provider reads GEMINI_API_KEY; app.config only exports GOOGLE_API_KEY.
    return LiteLLMModel(model=model_name, api_key=settings.gemini_api_key or None, temperature=0.0)


def _judge_label(judge: Any) -> str:
    """The judge's model id. deepeval models keep it in `.name`; never fall back to an object repr."""
    for attr in ("name", "model_name", "model"):
        value = getattr(judge, attr, None)
        if isinstance(value, str) and value:
            return value
    if callable(judge):                                   # a test fixture
        return getattr(judge, "__name__", "callable-judge")
    raise ValueError(f"cannot tell which judge model {judge!r} is; it must expose its id as .name")


def _is_self_judging(judge: Any) -> bool:
    return bool(settings.model_id) and settings.model_id in _judge_label(judge)


# --- leaf transcript -> groundedness items ------------------------------------------------------

def _context_from_tool_responses(tool_responses: List[Dict[str, Any]]) -> List[str]:
    """What the Fact-Check agent was actually shown: fetched fact-check page excerpts and database reviews."""
    context: List[str] = []
    for tr in tool_responses or []:
        resp = tr.get("response")
        if not isinstance(resp, dict):
            continue
        if tr.get("name") == "search_fact_check_sites":
            context.extend(s["excerpt"] for s in resp.get("sources") or [] if s.get("excerpt"))
        elif tr.get("name") == "search_fact_check_database":
            for r in resp.get("reviews") or []:
                if r.get("claim_reviewed") and r.get("rating"):
                    context.append(f"{r.get('publisher') or 'A fact-checker'} rated the claim "
                                   f"\"{r['claim_reviewed']}\" as {r['rating']}.")
    return context


def items_from_leaf_transcript(transcript_path: Path | str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Turn the Fact-Check agent's final decisions into groundedness items.

    A transcript entry is one model turn (see EvidenceReplay.write_leaf_transcript). The final turn
    carries the structured `decision` (a set_model_response call or the JSON text answer) and, in
    `tool_responses`, every tool result the agent had been shown: the retrieval context. The
    decision's `explanation` is the output under test.
    """
    path = Path(transcript_path)
    if not path.exists():
        return [], {"file_not_found": 1}

    items: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {
        "non_target_agent": 0,
        "not_a_final_turn": 0,
        "missing_context": 0,
        "missing_explanation": 0,
        "invalid_json": 0,
    }

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                skipped["invalid_json"] += 1
                continue

            if entry.get("name") != LEAF_AGENT:
                skipped["non_target_agent"] += 1
                continue
            decision = entry.get("decision")
            if not isinstance(decision, dict):
                skipped["not_a_final_turn"] += 1
                continue
            explanation = decision.get("explanation") or ""
            if not explanation:
                skipped["missing_explanation"] += 1
                continue
            context = _context_from_tool_responses(entry.get("tool_responses") or [])
            if not context:
                skipped["missing_context"] += 1
                continue

            claim_match = re.search(r"<claim>(.*?)</claim>", entry.get("prompt", ""), re.DOTALL | re.IGNORECASE)
            claim = claim_match.group(1).strip() if claim_match else f"Claim {idx + 1}"
            items.append({
                "id": f"{entry.get('claim_id') or 'leaf'}-{idx + 1:03d}",
                "claim": claim,
                "input": claim,
                "source_snippets": context,
                "retrieval_context": context,
                "context": context,
                "actual_output": explanation,
                "output": {"rating": decision.get("rating", ""), "explanation": explanation},
            })

    return items, skipped


# --- scoring ------------------------------------------------------------------------------------

def _judge_item(item: Dict[str, Any], judge: Any, claim: str, actual_output: str,
                snippets: List[str]) -> Tuple[Optional[float], Optional[float], str]:
    """(faithfulness, hallucination, reason) for one item; a callable judge is a test fixture."""
    if callable(judge):
        res = judge(item)
        faith, hallu = res.get("faithfulness"), res.get("hallucination")
        return (None if faith is None else float(faith),
                None if hallu is None else float(hallu), res.get("reason", ""))

    from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(input=claim, actual_output=actual_output,
                            retrieval_context=snippets, context=snippets)
    # penalize_ambiguous_claims: an "idk" verdict (unsupported, not contradicted) must not count as faithful.
    fm = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, async_mode=False,
                            penalize_ambiguous_claims=True)
    fm.measure(test_case)
    hm = HallucinationMetric(threshold=0.5, model=judge, async_mode=False, include_reason=False)
    hm.measure(test_case)
    return fm.score, hm.score, getattr(fm, "reason", "") or ""


def _score_items(
    items: List[Dict[str, Any]],
    judge: Any,
    delay_s: float,
    on_item: Optional[Callable[[List[Dict[str, Any]]], None]],
    describe: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for item in items:
        fields = describe(item)
        row = {**fields["row"]}
        try:
            faith, hallu, reason = _judge_item(item, judge, fields["claim"], fields["actual_output"],
                                               fields["snippets"])
            row.update(faithfulness=faith, hallucination=hallu, reason=reason)
        except Exception as exc:  # noqa: BLE001 - a judge failure is a recorded, flagged item
            logger.error("judge failed on %s: %s", row.get("id"), exc)
            row.update(faithfulness=None, hallucination=None, reason="",
                       judge_error=f"{type(exc).__name__}: {str(exc)[:200]}")
        if "judge_error" not in row and (row["faithfulness"] is None or not math.isfinite(row["faithfulness"])):
            row["judge_error"] = "judge returned no finite faithfulness score"
        row["flagged"] = is_flagged(row["faithfulness"])
        scored.append(row)
        if on_item:
            on_item(scored)
        if delay_s > 0:
            time.sleep(delay_s)
    return scored


def _disclosure(judge: Any, scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = sum(1 for x in scored if x.get("judge_error"))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": _judge_label(judge),
        "pipeline_model": settings.model_id,
        "self_judging": _is_self_judging(judge),
        "judge_errors": errors,
        "status": "OK" if errors == 0 else "INVALID",
    }


def _describe_guardrail(item: Dict[str, Any]) -> Dict[str, Any]:
    claim = item.get("claim") or item.get("input", "")
    output_data = item.get("output", {})
    if isinstance(output_data, dict):
        actual_output, rating = output_data.get("explanation", ""), output_data.get("rating", "")
    else:
        actual_output, rating = str(output_data), ""
    return {
        "claim": claim, "actual_output": actual_output,
        "snippets": item.get("source_snippets") or item.get("retrieval_context", []),
        "row": {"id": item.get("id", ""), "claim": claim, "actual_output": actual_output, "rating": rating,
                "is_hallucinated": item.get("is_hallucinated", False),
                "hallucination_type": item.get("hallucination_type"), "twin_of": item.get("twin_of")},
    }


def _describe_leaf(item: Dict[str, Any]) -> Dict[str, Any]:
    claim = item.get("claim") or item.get("input", "")
    actual_output = item.get("actual_output") or item.get("output", {}).get("explanation", "")
    return {
        "claim": claim, "actual_output": actual_output,
        "snippets": item.get("source_snippets") or item.get("retrieval_context", []),
        "row": {"id": item.get("id", ""), "claim": claim, "actual_output": actual_output},
    }


def score_guardrail(
    items: List[Dict[str, Any]],
    judge: Any = None,
    delay_s: float = 0.0,
    on_item: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """
    Score the hallucination guardrail dataset. Flagging rule: faithfulness < 0.8, fail closed
    (see is_flagged). HallucinationMetric is reported and does not decide the flag.
    """
    if judge is None:
        judge = get_judge_model()
    scored = _score_items(items, judge, delay_s, on_item, _describe_guardrail)

    poisoned = [x for x in scored if x["is_hallucinated"]]
    clean = [x for x in scored if not x["is_hallucinated"]]
    detected = sum(1 for x in poisoned if x["flagged"])
    flagged_clean = sum(1 for x in clean if x["flagged"])
    return {
        **_disclosure(judge, scored),
        "total_items": len(scored),
        "clean_items": len(clean),
        "poisoned_items": len(poisoned),
        "detected_poisoned": detected,
        "hallucination_catch_rate": round(detected / len(poisoned), 4) if poisoned else None,
        "flagged_clean": flagged_clean,
        "false_positive_rate": round(flagged_clean / len(clean), 4) if clean else None,
        "items": scored,
    }


def score_leaf_groundedness(
    items: List[Dict[str, Any]],
    judge: Any = None,
    delay_s: float = 0.0,
    on_item: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """Score agent explanations from live transcripts for groundedness (fail closed)."""
    if judge is None:
        judge = get_judge_model()
    scored = _score_items(items, judge, delay_s, on_item, _describe_leaf)

    total = len(scored)
    unsupported = sum(1 for x in scored if x["flagged"])
    finite = [x["faithfulness"] for x in scored if x["faithfulness"] is not None and math.isfinite(x["faithfulness"])]
    return {
        **_disclosure(judge, scored),
        "total_items": total,
        "grounded_items": total - unsupported,
        "unsupported_items": unsupported,
        "groundedness_rate": round((total - unsupported) / total, 4) if total else None,
        "mean_faithfulness": round(sum(finite) / len(finite), 4) if finite and len(finite) == total else None,
        "items": scored,
    }


def _writer(path: Path) -> Callable[[List[Dict[str, Any]]], None]:
    """Persist partial results after every item so a crash or a 429 keeps what quota already paid for."""
    def write(scored: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.stem + ".partial.json")
        partial.write_text(json.dumps({"status": "PARTIAL", "items": scored}, indent=2), encoding="utf-8")
    return write


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DeepEval faithfulness guardrail runner")
    parser.add_argument("--mode", choices=["guardrail", "leaf"], default="guardrail",
                        help="guardrail (evaluates hallucination set) or leaf (evaluates leaf transcript)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="path to explanation_hallucination_set.json (guardrail mode)")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT,
                        help="path to leaf_transcript.jsonl (leaf mode)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="delay in seconds between sequential evaluations (15 RPM free tier)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output path for the results JSON")
    parser.add_argument("--live", action="store_true",
                        help="actually call the judge model (spends quota; needs lead approval)")
    args = parser.parse_args(argv)

    if args.mode == "guardrail":
        out_path = args.output or DEFAULT_GUARDRAIL_OUT
        items = json.loads(args.dataset.read_text(encoding="utf-8"))
    else:
        out_path = args.output or DEFAULT_LEAF_OUT
        items, skipped = items_from_leaf_transcript(args.transcript)
        print(f"Parsed {len(items)} eligible leaf entries (skipped: {skipped})...")
        if not items:
            print("No items to evaluate.")
            sys.exit(1)

    estimate = len(items) * JUDGE_CALLS_PER_ITEM
    if not args.live:
        print(f"Refusing to run without --live: {len(items)} items need about {estimate} judge calls "
              f"(judge: {settings.eval_judge_model}). Quota is gated; get the lead's approval first.")
        sys.exit(2)

    judge = get_judge_model()
    print(f"Running {args.mode} on {len(items)} items, about {estimate} judge calls, delay={args.delay}s...")
    scorer = score_guardrail if args.mode == "guardrail" else score_leaf_groundedness
    results = scorer(items, judge=judge, delay_s=args.delay, on_item=_writer(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_path.with_name(out_path.stem + ".partial.json").unlink(missing_ok=True)
    print(f"Results saved to {out_path} (status {results['status']}, judge {results['judge_model']}, "
          f"self-judging: {results['self_judging']})")
    if args.mode == "guardrail":
        print(f"  Catch rate: {results['detected_poisoned']}/{results['poisoned_items']}; "
              f"false positives: {results['flagged_clean']}/{results['clean_items']}")
    else:
        print(f"  Groundedness: {results['grounded_items']}/{results['total_items']}")
    if results["status"] != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()
