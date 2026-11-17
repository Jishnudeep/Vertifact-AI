# VeriFact AI — evaluation

Harness, metrics and guardrails that measure the pipeline. Design: [`../docs/superpowers/specs/2026-10-16-evals-design.md`](../docs/superpowers/specs/2026-10-16-evals-design.md). Plan and task status: [`../docs/superpowers/plans/2026-10-16-evals.md`](../docs/superpowers/plans/2026-10-16-evals.md).

**Status:** the tooling is built and unit-tested. No valid baseline run exists yet, so no accuracy, latency or cache figures are published. The record run waits on free-tier quota.

All commands run from the repo root.

## What is here

| Path | Purpose |
|---|---|
| `datasets/curated_claims_seed.json` | 64 curated claims (50 verifiable) with label provenance; labels frozen under git tag `plan9-labels-frozen`. Only the nine reworded 0.5 claims were reviewed one by one. |
| `run_eval.py` | Runs `verify_claim` over the dataset, computes MAE, bucketed accuracy, calibration, bias leakage and a run-validity verdict |
| `replay.py` | Records every search, fetch, fact-check and LLM call to disk, then replays them for free and reproducible runs |
| `metrics/` | MAE, bucketed accuracy, calibration, retrieval recall, bias leakage, segments |
| `trace_eval.py`, `adk_eval/` | Audit of `verify_claim` responses (all nodes ran, degradation flagged, bias labels from AllSides, no verdict on adversarial claims), exportable to `agents-cli eval grade` |
| `deepeval_runner.py` | DeepEval faithfulness guardrail on seeded faithful/poisoned explanations, plus groundedness of real leaf outputs |
| `latency_bench.py` | Live p50/p95 per agent, LLM turn and tool call. Refuses to run without `--live`. |
| `cache_bench.py` | Cache hit latency and counted external-call savings. Needs `--confirm-db`. |
| `publish_scorecard.py` | Writes the README scorecard from valid runs only; refuses an invalid one |
| `calibrate_nli.py` | Calibrates NLI thresholds on stance and equivalence pairs |

## Run statuses

`run_eval.py` ends every run with one of three statuses:

- `OK`: the numbers can be read.
- `INVALID_RUN`: the environment failed (empty replay cache, broken search, no sources). The numbers measure the environment, not the scorer. Never publish.
- `GUARDRAIL_FAIL`: the measurement worked and a zero-tolerance rule broke (for example bias leakage).

## Commands

```bash
# Offline test suite
uv run --project backend pytest eval/tests

# Reproducible run from recorded evidence (free)
uv run --project backend python -m eval.run_eval --replay replay

# SPENDS QUOTA (a cold claim costs about 8 Gemini requests; free tier is 15 RPM, about 500/day)
uv run --project backend python -m eval.run_eval --replay record --claim-delay 6
uv run --project backend python eval/latency_bench.py --live --n 4 --delay 6
```

Recorded evidence lives in `eval/results/evidence_cache/` and reports in `eval/results/*.json`. Both are gitignored: a published baseline is a deliberate, reviewed deliverable made with live keys.

Prompt changes change the replay key, so existing LLM recordings must be re-recorded after editing an agent instruction.
