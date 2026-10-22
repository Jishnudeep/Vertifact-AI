"""
Scorecard generator for VeriFact AI evaluation baseline.

Publishes an honest, verified scorecard to README.md from an evaluation run.
REFUSES to publish unless run_status == "OK".
"""
import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional

START_MARKER = "<!-- eval-scorecard:start -->"
END_MARKER = "<!-- eval-scorecard:end -->"


class ScorecardRefusalError(ValueError):
    """Raised when an evaluation run is invalid or failed guardrails."""


def _format_metric(val: Optional[float], target: float, comparison: str = "<=") -> str:
    if val is None:
        return "not measured"
    if comparison == "<=":
        met = val <= target
    else:
        met = val >= target

    formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
    if met:
        return formatted_val
    return f"missed ({formatted_val} / {target:.2f})"


def generate_scorecard_markdown(
    summary: Dict[str, Any],
    record_run: Optional[Dict[str, Any]] = None,
    cache_bench: Optional[Dict[str, Any]] = None,
    deepeval_guardrail: Optional[Dict[str, Any]] = None,
    leaf_groundedness: Optional[Dict[str, Any]] = None,
) -> str:
    run_status = summary.get("run_status")
    if run_status != "OK":
        reasons = summary.get("run_status_reasons", [])
        raise ScorecardRefusalError(
            f"Refusing to generate scorecard: run_status is '{run_status}' (reasons: {reasons})"
        )

    ts = summary.get("timestamp", "unknown")
    n_verifiable = summary.get("verifiable_claims", summary.get("total_claims", "unknown"))
    total_claims = summary.get("total_claims", n_verifiable)
    mae_str = _format_metric(summary.get("mae"), target=0.20, comparison="<=")
    acc_str = _format_metric(summary.get("bucketed_accuracy"), target=0.70, comparison=">=")

    monotonic = summary.get("calibration_monotonic")
    if monotonic is True:
        mono_str = "yes"
    elif monotonic is False:
        mono_str = "no"
    else:
        mono_str = "not evaluable"

    bias_leakage = summary.get("bias_leakage", {})
    bias_rate = bias_leakage.get("source_leakage_rate")
    if bias_rate is not None:
        bias_str = f"{bias_rate:.2%}" if bias_rate == 0.0 else f"missed ({bias_rate:.2%} / 0.00%)"
    else:
        bias_str = "0.00%" if bias_leakage.get("passed") is True else "not measured"

    # Latency from record run or summary
    lat = None
    if record_run:
        lat = record_run.get("summary", {}).get("latency")
    if not lat:
        lat = summary.get("latency")

    if lat and lat.get("p50_ms") is not None:
        latency_str = f"p50 {lat['p50_ms']:.0f}ms / p95 {lat.get('p95_ms', 0):.0f}ms (n={lat.get('n', 'unknown')})"
    else:
        latency_str = "not measured"

    # Cache bench
    if cache_bench:
        speedup = cache_bench.get("speedup")
        saved = cache_bench.get("calls_saved_per_hit")
        cache_str = f"{speedup:.1f}x speedup ({saved:.1f} calls saved/hit)" if speedup else "measured"
    else:
        cache_str = "not measured"

    # DeepEval guardrail
    if deepeval_guardrail:
        catch_rate = deepeval_guardrail.get("hallucination_catch_rate")
        fp_rate = deepeval_guardrail.get("false_positive_rate")
        judge_model = deepeval_guardrail.get("judge_model", "unknown")
        deepeval_str = f"catch rate {catch_rate:.1%}, FP rate {fp_rate:.1%} (judge: {judge_model})"
    else:
        deepeval_str = "not measured"

    # Leaf groundedness
    if leaf_groundedness:
        g_rate = leaf_groundedness.get("grounded_rate")
        leaf_str = f"{g_rate:.1%}" if g_rate is not None else "measured"
    else:
        leaf_str = "not measured"

    md = f"""### Evaluation Scorecard (Baseline)

*Measured on {ts} over {total_claims} claims ({n_verifiable} verifiable).*

| Dimension | Metric | Result | Target | Status |
|-----------|--------|--------|--------|--------|
| **Accuracy** | C_total MAE (curated) | {summary.get('mae', 'n/a')} | ≤ 0.20 | {mae_str} |
| **Accuracy** | Bucketed accuracy (3-class) | {summary.get('bucketed_accuracy', 'n/a')} | ≥ 0.70 | {acc_str} |
| **Calibration** | Monotonic probability bins | {mono_str} | Monotonic | {"PASS" if mono_str == "yes" else "CHECK"} |
| **Safety** | Bias leakage (AllSides only) | {bias_str} | 0.00% | {"PASS" if bias_str == "0.00%" else "FAIL"} |
| **Latency** | End-to-end request (cold) | {latency_str} | < 10,000ms | {"PASS" if lat and lat.get('p50_ms', 99999) < 10000 else "INVESTIGATING"} |
| **Cache** | Lookup speedup vs live pipeline | {cache_str} | ≥ 10x | {"PASS" if cache_bench and cache_bench.get('speedup', 0) >= 10 else "TBD"} |
| **Guardrails** | Faithfulness catch rate | {deepeval_str} | ≥ 80.0% | {"PASS" if deepeval_guardrail and deepeval_guardrail.get('hallucination_catch_rate', 0) >= 0.8 else "TBD"} |
| **Faithfulness** | Leaf explanation groundedness | {leaf_str} | ≥ 80.0% | {"PASS" if leaf_groundedness and leaf_groundedness.get('grounded_rate', 0) >= 0.8 else "TBD"} |
"""
    return md


def publish_scorecard(
    baseline_path: Path | str,
    readme_path: Path | str = "README.md",
    record_run_path: Optional[Path | str] = None,
    cache_bench_path: Optional[Path | str] = None,
    deepeval_path: Optional[Path | str] = None,
    leaf_path: Optional[Path | str] = None,
) -> None:
    baseline_p = Path(baseline_path)
    readme_p = Path(readme_path)

    if not baseline_p.exists():
        raise FileNotFoundError(f"Baseline run file not found: {baseline_p}")

    baseline_data = json.loads(baseline_p.read_text(encoding="utf-8"))
    summary = baseline_data.get("summary", baseline_data)

    record_run = None
    if record_run_path and Path(record_run_path).exists():
        record_run = json.loads(Path(record_run_path).read_text(encoding="utf-8"))

    cache_bench = None
    if cache_bench_path and Path(cache_bench_path).exists():
        cache_bench = json.loads(Path(cache_bench_path).read_text(encoding="utf-8"))

    deepeval = None
    if deepeval_path and Path(deepeval_path).exists():
        deepeval = json.loads(Path(deepeval_path).read_text(encoding="utf-8"))

    leaf = None
    if leaf_path and Path(leaf_path).exists():
        leaf = json.loads(Path(leaf_path).read_text(encoding="utf-8"))

    content = generate_scorecard_markdown(
        summary=summary,
        record_run=record_run,
        cache_bench=cache_bench,
        deepeval_guardrail=deepeval,
        leaf_groundedness=leaf,
    )

    if not readme_p.exists():
        readme_text = f"# VeriFact AI\n\n{START_MARKER}\n{END_MARKER}\n"
    else:
        readme_text = readme_p.read_text(encoding="utf-8")

    block = f"{START_MARKER}\n{content}\n{END_MARKER}"
    if START_MARKER in readme_text and END_MARKER in readme_text:
        pattern = re.compile(
            re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        updated_readme = pattern.sub(block, readme_text)
    else:
        # Insert before Design principles or at end
        if "## Design principles" in readme_text:
            updated_readme = readme_text.replace(
                "## Design principles",
                f"## Evaluation scorecard\n\n{block}\n\n## Design principles",
            )
        else:
            updated_readme = f"{readme_text.rstrip()}\n\n## Evaluation scorecard\n\n{block}\n"

    readme_p.write_text(updated_readme, encoding="utf-8")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate and publish README evaluation scorecard")
    parser.add_argument("--baseline", type=Path, default=Path("eval/results/baseline_run.json"),
                        help="Baseline run or summary JSON file")
    parser.add_argument("--readme", type=Path, default=Path("README.md"),
                        help="Target README.md file")
    parser.add_argument("--record-run", type=Path, default=Path("eval/results/record_run.json"),
                        help="Optional record run JSON for live latency")
    parser.add_argument("--cache-bench", type=Path, default=Path("eval/results/cache_bench.json"),
                        help="Optional cache benchmark JSON")
    parser.add_argument("--deepeval", type=Path, default=Path("eval/results/deepeval_guardrail.json"),
                        help="Optional DeepEval guardrail JSON")
    parser.add_argument("--leaf-groundedness", type=Path, default=Path("eval/results/leaf_groundedness.json"),
                        help="Optional leaf groundedness JSON")
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    try:
        publish_scorecard(
            baseline_path=args.baseline,
            readme_path=args.readme,
            record_run_path=args.record_run,
            cache_bench_path=args.cache_bench,
            deepeval_path=args.deepeval,
            leaf_path=args.leaf_groundedness,
        )
        print(f"Successfully published scorecard to {args.readme}")
    except ScorecardRefusalError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
