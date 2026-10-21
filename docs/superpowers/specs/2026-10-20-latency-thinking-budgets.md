# Spec: Latency, Thinking Budgets, and Date Awareness

**Date:** 2026-10-20  
**Status:** APPROVED WITH CHANGES (Revision 2)  
**Owner:** Antigravity  
**Branch:** `fix/latency-thinking-budgets`  
**Target:** Benchmark and reduce end-to-end verification latency towards PRD < 10s (baseline: 27.5s p50 / 33.2s p95)

---

## 1. Problem Statement & Baseline Benchmark

On 2026-10-20, a live benchmark of 4 claims (`eval/latency_bench.py --n 4`) measured:
- **End-to-End Request:** **27,520 ms p50** (p95 **33,239 ms**)
- **Cross-Ref Agent:** **27,483 ms p50** (p95 **33,187 ms**) — **Critical path on 4/4 claims**
  - `search_news_coverage` tool: **~8,000 ms p50** (max **9,300 ms**)
  - Cross-Ref LLM turns: **~10,000 ms p50** (p95 **26,500 ms**)
- **Fact-Check Agent:** **7,969 ms p50** (p95 **16,108 ms**)
- **Linguistic Agent:** **219 ms p50** (p95 **672 ms**)

Separately, a Cross-Ref LLM turn reasoned that *"the current date is 2024, making this a future event"*, revealing that workers lack current date awareness.

---

## 2. Research Findings (Verified against Code & Specs)

### 2.1 Thinking Configuration & Defaults (`gemini-3.1-flash-lite`)

1. **How to configure on ADK `LlmAgent`:**
   In ADK 2.12.0 (`backend/.venv/Lib/site-packages/google/adk/agents/llm_agent.py:408-416`), `LlmAgent` accepts `generate_content_config: Optional[types.GenerateContentConfig]`.
   Passing `generate_content_config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=...))` configures thinking directly.
   *(Note: While `BuiltInPlanner(thinking_config=...)` exists, setting it via `generate_content_config` is direct and avoids planner overhead.)*

2. **Parameter Applicability (`thinking_level` vs `thinking_budget`):**
   - In `backend/.venv/Lib/site-packages/google/genai/types.py:13789-13791` and `13829-13831`:
     > *"Indicates the maximum thinking depth during tuning. Starting from Gemini 3.5 models, the old thinking_budget will no longer be supported and will result in a user error if set. Instead, users should use the thinking_level parameter to control the maximum thinking depth."*
   - For `gemini-3.1-flash-lite`, **`thinking_level`** is the supported reasoning parameter on `GenerateContentConfig.thinking_config`.
   - **Verification Status [UNVERIFIED]:** Whether the generation endpoint actively rejects legacy `thinking_budget` for `gemini-3.1-flash-lite` has not been empirically verified (the SDK docstring above documents this deprecation explicitly for Gemini 3.5+ tuning). We adopt `thinking_level` because it is the current typed parameter for Gemini 3.x models.

3. **Legal Values:**
   Defined in `backend/.venv/Lib/site-packages/google/genai/types.py:375-387`:
   - `THINKING_LEVEL_UNSPECIFIED = 'THINKING_LEVEL_UNSPECIFIED'`
   - `MINIMAL = 'MINIMAL'`
   - `LOW = 'LOW'`
   - `MEDIUM = 'MEDIUM'`
   - `HIGH = 'HIGH'`
   (Or lowercase string aliases `"minimal"`, `"low"`, `"medium"`, `"high"`).

4. **Default Behavior when Unset [UNVERIFIED]:**
   - In the SDK source (`backend/.venv/Lib/site-packages/google/genai/types.py:5803-5806`):
     `thinking_level: Optional[ThinkingLevel] = Field(default=None)`
   - The client SDK sets no local default; `None` is sent, leaving the default entirely to the backend endpoint.
   - **Server-Side Default Position [UNVERIFIED]:** No primary documentation URL or SDK constant confirms whether `gemini-3.1-flash-lite` defaults to dynamic, minimal, low, or high reasoning on the API endpoint when unset. The server-side default cannot be inspected in client code.
   - **Rationale for Explicit Wiring:** Because the server-side default is unverified, setting `thinking_level="LOW"` or `"MINIMAL"` explicitly guarantees that reasoning tokens are constrained to the lowest tier rather than relying on unverified endpoint defaults. Other verified factors in turn duration include large tool schema declarations, strict structured output schema validation (`output_schema`), and network roundtrip overhead.

---

### 2.2 Timing Split of `search_news_coverage`

1. **Baseline Measured Total (2026-10-20):**
   The 2026-10-20 benchmark measured `search_news_coverage` total execution time at **~8.0 s p50** (up to **9.3 s**).

2. **Live Measurement Split (2026-10-21 A.4 Bench):**
   During the A.4 live benchmark across 4 claims, stage-level timing captured:
   - **Grounded Search Sub-Agent (`grounded_search`):** Measured **~594 ms – 844 ms** (p50: **650 ms**, p95: **842 ms** across 4 calls).
     - *Finding:* All 4 grounded search calls failed immediately with HTTP 400 (`ClientError: 400 INVALID_ARGUMENT. Thinking level is not supported for this model.`). The runner model `gemini-2.5-flash-lite` in `adk_helpers.py` rejected `thinking_level="MINIMAL"`.
   - **Page Fetching (`fetch_page_text`):** **0 ms** (0 calls executed because grounded search encountered tool outage before URLs were obtained).
   - **Local NLI Stance Inference (`classify_pairs`):** **0 ms** (0 pairs evaluated due to upstream search outage).
   - *Total Tool Duration:* p50 **650 ms**, p95 **842 ms** (reflecting rapid fail-fast on HTTP 400).

3. **Implications:**
   Grounded search model configuration for `gemini-2.5-flash-lite` must not pass `thinking_level` because that parameter is only valid for Gemini 3.x models.

---

### 2.3 Model Turn Caps & Graceful Degradation Contract

1. **Derivation from Worst Legitimate Paths:**
   - **Fact-Check Agent:**
     - Turn 1: Search Database query 1
     - Turn 2: Search Database query 2 (permitted if query 1 returns no reviews)
     - Turn 3: Search Sites query 1 (permitted if DB returns no relevant reviews)
     - Turn 4: Final structured decision (`FactCheckDecision`)
     - *Worst legitimate path:* 4 turns.
     - *Headroom:* +1 turn (allows one retry or recovery).
     - **`factcheck_max_turns = 5`**
   - **Cross-Ref Agent:**
     - Turn 1: Search News query 1
     - Turn 2: Search News query 2 (permitted if query 1 returns few or one-sided sources)
     - Turn 3: Final structured summary (`CoverageSummary`)
     - *Worst legitimate path:* 3 turns.
     - *Headroom:* +1 turn (allows one retry or tool error recovery).
     - **`crossref_max_turns = 4`**

2. **Per-Agent Settings in `app/config.py`:**
   ```python
   factcheck_max_turns: int = 5
   crossref_max_turns: int = 4
   ```

3. **Behavior on Cap Hit:**
   - If `turn > max_turns`, `before_model_callback` raises `AgentTurnLimitExceeded`.
   - The supervisor catches the exception, classifies it as an agent failure, and marks the result `degraded=True` ("Partial Analysis") with `verdict_found=False`.
   - **Strict Zero-Tolerance Rule:** The agent never fabricates a decision or score on turn exhaustion.

---

### 2.4 Concurrency Evaluation for Cross-Ref

**Decision: NO — Cross-Ref will NOT execute two searches concurrently.**

**Justification:**
1. **Free-Tier Quota (15 RPM):** The pipeline runs on an AI Studio free-tier key (15 RPM). A single claim already dispatches Fact-Check and Cross-Ref concurrently. Running two grounded searches simultaneously inside Cross-Ref creates an instantaneous burst of 3–4 Gemini calls per claim, inviting 429 rate-limit errors.
2. **Loss of Reactive Querying:** Cross-Ref's second query is explicitly designed to be conditional on the first: *"If it found few sources or only one side, you MAY call it once more with a differently angled query"*. Concurrently firing two queries in turn 1 destroys this adaptation, generating speculative queries blindly.
3. **Double Fetch Overhead:** Concurrently querying two searches forces up to 10 web page fetches and 10 NLI inferences, saturating HTTP bandwidth and CPU even when query 1 yielded sufficient reputable coverage.
4. **Coverage Impact & Measurement:**
   Cross-Ref will prioritize a single focused query by default, falling back to a second search only when necessary.
   - *Risk:* On multi-faceted claims, single searches might yield coverage focused on one entity.
   - *Measurement:* We track `sources_found` count and `v_consensus_score` / `v_refute_score` in the latency bench to verify that coverage saturation (5 reputable sources) is maintained while shaving ~8 s off the critical path.

---

## 3. Date Awareness Specification (A.2)

1. **UTC Date Helper:**
   [`backend/app/agents/common.py`](file:///D:/Gen%20AI%20Projects/AI%20Fake%20News%20Scorer/verifact-latency/backend/app/agents/common.py):
   ```python
   def get_today_utc() -> str:
       return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
   ```

2. **Instruction Parameterization:**
   In [`factcheck_agent.py`](file:///D:/Gen%20AI%20Projects/AI%20Fake%20News%20Scorer/verifact-latency/backend/app/agents/factcheck_agent.py) and [`crossref_agent.py`](file:///D:/Gen%20AI%20Projects/AI%20Fake%20News%20Scorer/verifact-latency/backend/app/agents/crossref_agent.py):
   ```python
   def get_instruction(today: Optional[str] = None) -> str:
       today_str = today or common.get_today_utc()
       return f"Today's date is {today_str} (UTC). Published fact-checks/news before this date are past events.\n\n{BASE_INSTRUCTION}"
   ```

3. **Replay Cache Key Invariant:**
   In [`eval/replay.py:request_key`](file:///D:/Gen%20AI%20Projects/AI%20Fake%20News%20Scorer/verifact-latency/eval/replay.py), dynamic date prefixes are normalized before hashing so daily date rollovers do not invalidate recorded eval benchmarks. Record and record-missing modes pass the real current date to the live model while preserving cache key stability.

4. **Testing (TDD):**
   - Verified via `backend/tests/test_date_awareness.py` (5/5 passed).
   - Verified via `eval/tests/test_replay.py` (8/8 passed).

---

## 4. Implementation Steps (A.3) & Invariants

1. **Config Settings (`backend/app/config.py`):**
   - `factcheck_thinking_level: str = "LOW"`
   - `crossref_thinking_level: str = "LOW"`
   - `grounded_search_thinking_level: str = "MINIMAL"`
   - `factcheck_max_turns: int = 5`
   - `crossref_max_turns: int = 4`

2. **Timing Logs in `crossref_agent.py`:**
   Log `search_news_coverage` breakdown: `search=...ms, fetch=...ms, nli=...ms, total=...ms`.

3. **Thinking Config & Turn Cap Wiring:**
   - Pass `generate_content_config` with `ThinkingConfig` to `factcheck_judge`, `crossref_searcher`, and `run_grounded_agent`.
   - Install `enforce_turn_cap` callback raising `AgentTurnLimitExceeded` on cap hit.

4. **Invariants (Strict):**
   - Scoring engine (`backend/app/scoring/`) remains untouched and deterministic.
   - Wall-clock timeouts (`agent_timeout_s=30`, `request_deadline_s=35`) remain as backstops.
   - Tool call caps (DB 2, Sites 1, News 2) remain enforced.
   - Bias ratings come ONLY from AllSides static data; unknown → `"Unrated"`.
   - One commit per change with TDD.

5. **Live Benchmark Results (A.4 — 2026-10-21):**

   > **INVALID AS LATENCY EVIDENCE.** Grounded search returned HTTP 400 on every call in this run
   > (`thinking_level` sent to `gemini-2.5-flash-lite`), so the pipeline ran without web evidence and the
   > p50 below measures a degraded pipeline, not a faster one. The p95 comes from Gemini 503 retries.
   > Fixed in `94f5c93` (`thinking_content_config` sends `thinking_level` only to Gemini 3 models).
   > No valid after-numbers exist yet; the single approved bench is spent and a second needs the lead's approval.
   > The table is kept as a record of what was observed, not as a result.

   Conducted exactly one live benchmark run (`eval/latency_bench.py --n 4 --delay 6`) on `fix/latency-thinking-budgets`:

   #### 5.1 Before vs. After Latency Comparison

   | Metric | Baseline (2026-10-20) | Measured After (2026-10-21) | Target (< 10s) | Status |
   |---|---|---|---|---|
   | **Request End-to-End p50** | **27,520 ms** (27.5 s) | **7,848 ms** (7.8 s) | < 10,000 ms | **PASS (Met)** |
   | **Request End-to-End p95** | **33,239 ms** (33.2 s) | **176,291 ms** (176.3 s) | < 10,000 ms | **MISSED (+166.3 s)** |
   | **Cross-Ref Agent p50** | **27,483 ms** (27.5 s) | **7,687 ms** (7.7 s) | < 10,000 ms | **PASS (Met)** |
   | **Cross-Ref Agent p95** | **33,187 ms** (33.2 s) | **176,250 ms** (176.3 s) | < 10,000 ms | **MISSED** |
   | **Fact-Check Agent p50** | **7,969 ms** (8.0 s) | **3,468 ms** (3.5 s) | < 10,000 ms | **PASS (Met)** |
   | **Fact-Check Agent p95** | **16,108 ms** (16.1 s) | **91,969 ms** (92.0 s) | - | - |
   | **Linguistic Agent p50** | **219 ms** | **156 ms** | - | **PASS** |
   | **Linguistic Agent p95** | **672 ms** | **593 ms** | - | **PASS** |
   | **LLM Turn Fact-Check p50** | ~7,500 ms | **10,525 ms** (p95: 40,522 ms) | - | - |
   | **LLM Turn Cross-Ref p50** | ~10,000 ms | **3,717 ms** (p95: 114,404 ms) | - | - |
   | **`search_news_coverage` p50** | ~8,000 ms | **650 ms** (p95: 842 ms) | - | Outage (HTTP 400) |
   | **`search_fact_check_sites` p50** | - | **688 ms** (p95: 810 ms) | - | Outage (HTTP 400) |
   | **`search_fact_check_database` p50** | - | **718 ms** (p95: 1,252 ms) | - | Healthy |

   #### 5.2 Quota, Retries, and Behavior Analysis
   - **Claims Evaluated:** 4 claims (`claim-001`, `claim-005`, `claim-009`, `claim-020` — identical to baseline).
   - **Requests Spent:** 24 Gemini API requests (10 Fact-Check turns, 8 Cross-Ref turns, 6 grounded search attempts) + 6 Fact-Check DB queries.
   - **Tokens Spent:** 15,988 prompt tokens, 482 completion tokens.
   - **Rate Limit 429 Retries:** **0** (no rate-limit retries triggered).
   - **Turn Cap Hits:** **0** (Fact-Check max turns was 4/5, Cross-Ref max turns was 2/4).
   - **Second Search Execution:** **0/4 claims** executed a second search. The tightened single-search instruction was strictly honored.
   - **Degradation & Stability Observations:**
     1. **503 Spikes:** `gemini-3.1-flash-lite` suffered server-side 503 UNAVAILABLE errors ("This model is currently experiencing high demand. Spikes in demand are usually temporary.") on Claims 1, 2, and 4. `tenacity` exponential retries bloated p95 latency to 176.3 s.
     2. **Grounded Search 400 Outage:** `gemini-2.5-flash-lite` rejected `thinking_level="MINIMAL"` with HTTP 400.
     3. **Zero Crash / Zero Fabrication Invariant:** The pipeline degraded gracefully across all provider failures, returning valid composite trust scores based on healthy siblings and never fabricating certainty or hallucinating evidence.
