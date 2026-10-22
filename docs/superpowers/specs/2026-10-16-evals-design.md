# VeriFact AI — Evals & Agent Ops Design Brief (Plan 9 Spec)

**Status:** Draft, revised 2026-10-16 after stress test; amended 2026-10-20 for ADK-native agents (`cd69a7f`, `e518dd7`). Six open decisions (§6) need the project lead before execution.
**Amendments (2026-10-20):** Reconciled for ADK-native agents (`cd69a7f`, `e518dd7`, merged 2026-10-20). Groq/LiteLLM leaf prompts replaced by native Gemini `LlmAgent`s (`gemini-3.1-flash-lite`). Timeouts are config backstops. Replay hooks `backend/app/agents/common.py::get_model`. See `docs/superpowers/plans/2026-10-16-evals.md` §Amendments.
**Roadmap:** Phase D of `docs/superpowers/specs/2026-10-15-realigned-roadmap-design.md` (decision R7; Phase D scope list)
**Companion:** `documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md` §2.3 and §2.5, `documents/VeriFact_AI_Build_Order.md` Phase 3
**Implementation Plan:** `docs/superpowers/plans/2026-10-16-evals.md`

---

## 1. Why Plan 9

The backend runs an ADK 2.12 `Workflow` graph with three verification agents, a pure scoring engine and an NLI-guarded semantic cache. The frontend MVP is done (Plan 7). What is missing is evidence that any of it works:

1. **The only baseline is invalid.** `eval/results/baseline_run.json` has `summary.run_status == "INVALID_RUN"`: the replay cache was empty, so all 56 external calls missed and every verifiable claim had zero sources.
2. **The curated set is too small to measure.** It has 25 claims, but only 16 are `verifiable`, and only those feed MAE, bucketed accuracy and calibration. Mixed claims (ground truth 0.5) number 2.
3. **The DeepEval guardrail (Stack & Eval §2.5) is not built.**
4. **ADK eval over traces (roadmap R7) is not built.** Nothing checks that every agent node ran, the deadline held, degradation was flagged honestly, and the Groq leaf outputs are grounded in the page text they read.
5. **The latency and cache claims are unmeasured.** p50/p95, cache hit latency and saved calls have never been measured.

---

## 2. Facts the design depends on (checked against code, 2026-10-16)

| Fact | Where |
|---|---|
| Run status values are the strings `"OK"`, `"INVALID_RUN"`, `"GUARDRAIL_FAIL"`. `STATUS_OK` is only the Python constant name. | `eval/run_eval.py` |
| Report shape: `{"summary": {...}, "results": [...]}`. Metrics live under `summary`: `run_status`, `mae`, `bucketed_accuracy`, `verifiable_claims`, `bias_leakage.passed`, `replay.misses`. Nothing sits at the top level. | `eval/run_eval.py` |
| `eval/results/*.json` and `eval/results/evidence_cache/` are **gitignored**. Subdirectories (`eval/results/traces/`) and `.jsonl` files are **not**, so new outputs need their own rules. | `.gitignore:75`, `.gitignore:78` |
| The harness always calls `verify_claim(..., bypass_cache=True)`. The cache is never exercised, and nothing is persisted. | `eval/run_eval.py` |
| Agent node names are `factcheck`, `crossref`, `linguistic`. `verify_claim` returns `agents: [{name, status: "ok"/"degraded", duration_ms, error_code}]`, plus `confidence.warnings` built by `degradation_message()` (for example `"Cross-reference check failed"`), plus per-source `sources[].bias_label`. | `backend/app/agents/orchestrator.py:253`, `backend/app/utils/errors.py:55` |
| Request deadline is `settings.request_deadline_s = 9.5`. | `backend/app/config.py:72` |
| No user-facing free-text explanation exists. The only LLM explanations are the Groq leaf outputs: `verdict_extractor` (`{rating, explanation}` over fetched page text in `<search_snippets>`), `claim_aligner`, `rating_classifier`, `sensationalism_analyzer`. | `backend/app/agents/factcheck_agent.py:186,232,534`, `linguistic_agent.py:117` |
| `EvidenceReplay` intercepts every leaf LLM call (`run_adk_agent`) and every search/fetch call. The recorded file stores only the response, keyed by a hash of the payload. | `eval/replay.py:59-147` |
| `bias_leakage` already checks labels against `VALID_BIAS_RATINGS` and bridging domains. | `eval/metrics/bias_leakage.py` |
| `eval/tests` is collected with the repo root as rootdir, so `backend/pyproject.toml`'s `addopts = "-m 'not live'"` does **not** apply. Live-marked eval tests run by default. | `eval/conftest.py:26-29` |
| `EvalClaim.claim_type` is one of `verifiable / opinion / question / adversarial`. The schema has no `load_claims()` helper and no label-provenance field. | `eval/datasets/claim_dataset_schema.py` |
| `deepeval` is not a dependency. `agents-cli` is not installed (`uv tool install google-agents-cli`). | `backend/pyproject.toml` |
| The README has no evaluation section. The status line at `README.md:149` still says "🔜 Evaluation". | `README.md` |

---

## 3. Decisions

| ID | Decision | Rationale |
|---|---|---|
| **E1** | **Size the dataset by verifiable claims: ≥ 50 verifiable.** Add ≥ 34 verifiable claims (≈10 true, ≈12 false, ≈12 mixed), 3 refusal claims (opinion/question) and 2 adversarial claims. That gives about 64 claims in total and 50 verifiable, with roughly 18 true / 18 false / 14 mixed. | Only verifiable claims are scored. Today's set has 2 mixed claims, so the mixed bucket can't be measured. |
| **E2** | **Every new claim carries label provenance.** Add an optional `label_source` (URL or citation) and a `label_note` to `EvalClaim`. Labels for 2025–2026 claims must cite a primary source. The project lead reviews all new labels, and the dataset is committed (labels frozen) **before** any record run. | An LLM-drafted label for a recent event is the least trustworthy number in the eval. Freezing labels before seeing scores prevents tuning labels to fit the model. |
| **E3** | **DeepEval guardrail per Stack & Eval §2.5, shaped like the real leaf.** Use 30 seeded items (15 faithful, 15 with one injected unsupported fact). Each item mirrors `verdict_extractor`: a claim, fetched-page snippets as `retrieval_context`, and a `{rating, explanation}` output where the `explanation` is graded. Run `FaithfulnessMetric(threshold=0.8)` as the flagging rule and report `HallucinationMetric(threshold=0.5)` alongside it. Pass bar: flag ≥ 12/15 poisoned and ≤ 3/15 clean. | This is §2.5 as written. Mirroring the leaf's shape means the guardrail tests the same kind of text the pipeline produces. |
| **E4** | **Grade real leaf outputs too.** `EvidenceReplay` gains an in-memory transcript of leaf calls (`name`, `prompt`, `response`) in both record and replay mode. Each run writes them to `eval/results/leaf_transcript.jsonl`. The same Faithfulness judge then scores every `verdict_extractor` explanation against the `<search_snippets>` in its own prompt. The result is reported as `leaf_groundedness` (mean score, share below 0.8). It is informational and does not gate this plan. | This adds a real-pipeline faithfulness number at no extra search quota, since replay serves the inputs. It covers the roadmap's "groundedness judges on the Groq leaf outputs". |
| **E5** | **ADK eval (`agents-cli eval grade`) over exported pipeline traces.** The harness exports one trace per claim in agents-cli's grading-input format to `eval/results/traces/`. `eval/adk_eval/eval_config.yaml` declares local `custom_function` code metrics: all three nodes present, `processing_time_ms ≤ 9500`, every degraded agent has its `degradation_message` warning, every `sources[].bias_label` equals `get_bias_label(domain)`, and adversarial claims have no verdict. The same metric functions live in `eval/trace_eval.py` and are unit-tested with pytest, so the checks still run in CI where `agents-cli` is absent. | Roadmap R7 names `agents-cli eval`. `eval generate` cannot drive the pipeline (it is not a `/run_sse` chat agent), but `eval grade --traces` grades hand-exported traces with local code metrics. A spike step (plan Task 3) confirms the trace format first, with a recorded fallback if it does not fit. |
| **E6** | **The trace evaluator reads the real `verify_claim` response**, not an invented schema. Fault injection is tested end to end: monkeypatch `crossref_agent`'s search to raise, and in a second test to sleep past the agent budget, then assert that `verify_claim` completes, `crossref` is `degraded` with an `error_code`, the warning is present, and the other two agents are `ok`. Bias correctness compares each label to `get_bias_label(domain)`. Membership in the AllSides label set alone can't show where a label came from. | Tests that feed a validator dicts written to suit it prove nothing. |
| **E7** | **Live latency is measured during the record run, not in replay.** Replay serves responses from disk, so replay timings are meaningless. The record run calls everything live, so per-claim `processing_time_ms` from that run gives live p50/p95. `claim_delay` is excluded, and 429 retries are counted and reported. | One live pass serves two purposes, which saves quota. |
| **E8** | **Measure cache cost first, then set a target.** Time `lookup_cache` directly for a cache hit (embed, pgvector round trip, row policy, NLI equivalence) with `time.perf_counter`. Report hit p50/p95, speedup over live p50, and saved external calls. Saved calls are *counted* from the record run's `EvidenceReplay` tallies per claim, not assumed. No fixed `<50 ms` target: a Supabase cloud round trip plus a CPU cross-encoder probably can't meet it. The one stated target is "cache-hit p95 under 10% of live p50", reported as met or missed. | The plan had three conflicting targets (<50 ms, >50×, >90%) and none had been measured. |
| **E9** | **Validity gates the baseline; quality targets are reported.** See open decision D5. | — |
| **E10** | **Eval-only dependencies stay out of the runtime image.** Use `uv add --project backend --group eval deepeval`. DeepEval telemetry is disabled (`DEEPEVAL_TELEMETRY_OPT_OUT=YES`, set in `eval/conftest.py` and the runner). | DeepEval is large, and the production image does not need it. |
| **E11** | **Tests that call external APIs are opt-in in `eval/`.** Add a new marker, `live_api`, registered in `eval/conftest.py`. Those tests are skipped unless `--run-live-api` (or `VERIFACT_RUN_LIVE_API=1`) is given. The existing `live` marker (`test_cache_safety.py:42`, `test_calibrate_nli.py:45`, local NLI model only) keeps its behaviour. | Otherwise a plain `pytest eval/tests` spends Groq quota. |

---

## 4. Subsystems

```
eval/datasets/curated_claims_seed.json (≈64, ≥50 verifiable, label_source)      eval/datasets/explanation_hallucination_set.json (30)
                │                                                                                   │
                ▼                                                                                   ▼
eval/run_eval.py ── EvidenceReplay (record|replay; + leaf transcript, + per-claim call tallies)   eval/deepeval_runner.py
   │                                                                                                │  FaithfulnessMetric + HallucinationMetric
   ├─ core metrics (MAE, bucketed, calibration, segments, bias leakage)                            │  judge = D6
   ├─ trace export → eval/results/traces/*.json ──► agents-cli eval grade (eval/adk_eval/) ◄── eval/trace_eval.py (same checks, pytest)
   ├─ leaf_transcript.jsonl ─────────────────────► deepeval_runner --leaf  → leaf_groundedness
   ├─ latency (record run only) → p50/p95, 429 count
   └─ summary → eval/results/baseline_run.json → README "Evaluation" section (only if run_status == "OK")

eval/cache_bench.py → lookup_cache timing on a target chosen in D4 → eval/results/cache_bench.json
```

### 4.1 Dataset
- New claims start at `claim-026`. Keep the existing 25 unchanged.
- Mixed claims use `ground_truth_score` in `[0.4, 0.6]` (the bucket is 0.34–0.66). True claims use 1.0 and false claims use 0.0.
- Each new verifiable claim has `label_source`. Each recent claim (2025–2026) has a `recent_2025_2026` tag.
- Refusal claims use `claim_type: "opinion"` or `"question"`. `"unverifiable"` is not a valid type.
- A test enforces the counts and, for new claims, that every `label_source` is non-empty.

### 4.2 DeepEval guardrail
- Runner: `eval/deepeval_runner.py`. It is not under `tests/`, because it is a module the harness also calls.
- Judge wiring: DeepEval's LiteLLM model class (or a `DeepEvalBaseLLM` subclass if that class is unavailable in the pinned version), `temperature=0`. The model comes from config through a new `EVAL_JUDGE_MODEL` setting (D6).
- Throttling: run items sequentially with a configurable delay. Faithfulness makes about 3–4 judge calls per item, so 30 items need about 120 calls, against Groq's free-tier TPM limit.
- Output: `eval/results/deepeval_guardrail.json` with per-item scores, `hallucination_catch_rate`, `false_positive_rate`, judge model, and timestamp.
- Statistical note: with n = 15 per class, one item is 6.7 points. The report states the counts, not only the rates.

### 4.3 Trace evaluator and ADK eval
- `eval/trace_eval.py`: pure functions over a `verify_claim` response plus claim metadata, returning `TraceAuditReport(checks: dict[str, bool], failures: list[str])`.
- `run_eval.py` runs the evaluator on every verifiable and adversarial result. It adds `summary.trace_audit = {cases, failures_by_check}` and exports traces.
- The `agents-cli` config reuses `trace_eval.py` functions through `custom_function_file`.

### 4.4 Latency and cache
- Latency comes from the record run (E7).
- Cache bench: `eval/cache_bench.py` inserts N results, then looks up the same claim texts (exact and paraphrased) and times `lookup_cache`. Where it runs depends on D4.

### 4.5 Baseline and README
- The README "Evaluation" section is generated from `baseline_run.json`, `deepeval_guardrail.json` and `cache_bench.json` by `eval/publish_scorecard.py`. It refuses to write unless `summary.run_status == "OK"` (roadmap: "only from a valid run").
- Every number in the README carries its run timestamp and n.

---

## 5. Exit criteria

1. The dataset has ≥ 50 verifiable claims, each new one with a `label_source`, all reviewed by the project lead, and is committed before the record run.
2. `pytest eval/tests` passes offline with no network and no Groq calls.
3. DeepEval guardrail result is recorded: flags ≥ 12/15 poisoned and ≤ 3/15 clean (judge per D6).
4. The trace evaluator passes its unit tests and both fault-injection tests (raise, timeout). `agents-cli eval grade` runs over the exported traces, or the fallback is recorded (E5).
5. A record run and a replay run complete. The replay run reports `summary.replay.misses == 0` and `summary.run_status == "OK"`.
6. Live p50/p95, cache-hit p50/p95, speedup and counted saved calls are reported.
7. Quality targets (MAE ≤ 0.20, bucketed accuracy ≥ 0.70, bias leakage = 0) are reported, with consequences as set by D5.
8. The README evaluation section is generated from the valid run.
9. Review gate (CLAUDE.md §5): `code-reviewer` and `eval-guardian` subagents report 0 blocking findings.

---

## 6. Open decisions (project lead)

| ID | Question | Options | Recommendation |
|---|---|---|---|
| **D1** | Commit `eval/results/baseline_run.json`? | (a) Un-ignore and commit it. (b) Keep it ignored and commit only `eval/results/baseline_summary.json` (the `summary` block). | (b). The full `results` list carries claim-level detail that changes on every run, while the summary is what the README and `test_baseline_validity` need. The test reads the committed file and skips with a clear message when it is absent. |
| **D2** | Commit `eval/results/evidence_cache/`? | (a) Commit it, so replay is reproducible from a clone. (b) Keep it local, so only the lead's machine can replay. (c) Keep it local and archive a tarball outside git. | Undecided. (a) makes the baseline reproducible, but the cache holds fetched third-party page text (copyright, size). Measure the size after recording, then decide. |
| **D3** | Approve the record run (spends quota). | One `--replay record --claim-delay 2.0` run over the frozen dataset. Estimate it beforehand from the replay-mode miss count (one miss is one live call). | Approve once the dataset is frozen and the estimate is shown. This single run also yields the live latency figures (E7). |
| **D4** | Where does the cache bench write? `lookup_cache` needs rows, and `verify_claim` without `bypass_cache` persists to Supabase. | (a) A Supabase branch for the bench. (b) Production tables with rows tagged by a bench run id and deleted afterwards. (c) Skip the bench and report the cache as unmeasured. | (a) if branching is available on the plan, else (b) with cleanup asserted in the script. |
| **D5** | What do quality targets gate? | (a) `run_status == "OK"` gates the baseline; MAE, bucketed accuracy and calibration are reported as met or missed, and a miss becomes a findings section and the next plan's input. (b) Targets block Plan 9 until met. | (a). (b) invites tuning labels or thresholds against the eval set. Bias leakage = 0 stays a hard gate either way (it already yields `GUARDRAIL_FAIL`). |
| **D6** | DeepEval judge model. | (a) Groq `gpt-oss-120b`, the same model that writes the leaf outputs, so it may favour its own text. (b) A different Groq model. (c) Gemini via `GEMINI_API_KEY`. | (b) or (c). Record the judge in every result file either way. |
