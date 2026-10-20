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
   - In `backend/.venv/Lib/site-packages/google/genai/types.py:13828-13832`:
     > *"Starting from Gemini 3.5 models, the old thinking_budget will no longer be supported and will result in a user error if set. Instead, users should use the thinking_level parameter to control the maximum thinking depth."*
   - For `gemini-3.1-flash-lite`, **`thinking_level`** is the supported reasoning parameter. Passing legacy `thinking_budget` or combining both triggers an API error.

3. **Legal Values:**
   Defined in `backend/.venv/Lib/site-packages/google/genai/types.py:375-387`:
   - `THINKING_LEVEL_UNSPECIFIED = 'THINKING_LEVEL_UNSPECIFIED'`
   - `MINIMAL = 'MINIMAL'`
   - `LOW = 'LOW'`
   - `MEDIUM = 'MEDIUM'`
   - `HIGH = 'HIGH'`
   (Or lowercase string aliases `"minimal"`, `"low"`, `"medium"`, `"high"`).

4. **Default Behavior when Unset:**
   - In the SDK source (`backend/.venv/Lib/site-packages/google/genai/types.py:5803-5806`):
     `thinking_level: Optional[ThinkingLevel] = Field(default=None)`
   - The client SDK sets no local default; `None` is sent, leaving the default entirely to the backend endpoint.
   - Official Gemini API documentation states that while `gemini-3.1-pro` defaults to high reasoning, `gemini-3.1-flash-lite` defaults to dynamic/minimal reasoning depending on endpoint routing.
   - **Verification Note:** Because the server-side default cannot be inspected in client code, **we do NOT cite an unverified default as the sole explanation for the ~10 s LLM turns**. Other verified factors in turn duration include large tool schema declarations, strict structured output schema validation (`output_schema`), and roundtrip overhead. Setting `thinking_level="LOW"` or `"MINIMAL"` explicitly guarantees that reasoning tokens are constrained to the lowest tier.

---

### 2.2 Timing Split of `search_news_coverage`

1. **Measured Total:**
   The 2026-10-20 benchmark measured `search_news_coverage` total execution time at **~8.0 s p50** (up to **9.3 s**).

2. **Sub-Stage Breakdown (Analytical Estimates):**
   The initial benchmark instrumented tool execution as a whole and did not log internal sub-stages. Based on component timeouts and architecture:
   - **Grounded Search Sub-Agent (`grounded_search`):** *[Estimate: ~4.0 s – 5.0 s]*  
     Executes an inner `InMemoryRunner` with Gemini `gemini-3.1-flash-lite` calling the built-in `google_search` tool (`run_grounded_agent` in `app/utils/adk_helpers.py`).
   - **Page Fetching (`fetch_page_text`):** *[Estimate: ~1.5 s – 2.0 s]*  
     Fetches up to 5 cited domain URLs concurrently with `page_fetch_timeout_s=2.0`. Slow or blocking news outlets consume up to 2.0 s.
   - **Local NLI Stance Inference (`classify_pairs`):** *[Estimate: ~0.5 s – 1.0 s]*  
     Runs DeBERTa-v3-small cross-encoder inference on CPU for up to 5 premise-claim pairs in a worker thread (`asyncio.to_thread`).

3. **Instrumentation Action:**
   In A.3, we add explicit stage-level timing logs in `crossref_agent.py` around `grounded_search`, `fetch_page_text`, and `classify_pairs`. The A.4 bench will report the measured split.

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
   In [`eval/replay.py:request_key`](file:///D:/Gen%20AI%20Projects/AI%20Fake%20News%20Scorer/verifact-latency/eval/replay.py), dynamic date prefixes are normalized before hashing so daily date rollovers do not invalidate recorded eval benchmarks. Furthermore, `EvidenceReplay.install()` pins `get_today_utc` to `"2026-10-20"`.

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

5. **Measurement (A.4):**
   Run exactly one live bench (`eval/latency_bench.py --n 4 --delay 6`). Report before and after p50/p95 faithfully without claiming 10 s unless measured.
