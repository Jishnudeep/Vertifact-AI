# Plan 7 — Frontend MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Antigravity: start with `/execute-plan` (`.agents/workflows/execute-plan.md`).

**Goal:** A phone-first, fully responsive web app where a user pastes a claim and gets the Midnight Transit Diagram verdict, its breakdown and its sources, plus the small backend additions that make the screen honest.

**Architecture:**
- **Backend (Tasks 1–4):** the API gains:
  - per-agent status;
  - cap transparency;
  - `found_by` on sources;
  - stored-result parity;
  - `Retry-After`;
  - a featured-only `GET /api/results/recent`.

  All additive; `c_total` is unchanged.
- **Frontend (Tasks 5–9):** a new Vite + React + Tailwind + Recharts app in `frontend/`. It has three routes: Home `/`, Result `/r/:id`, and Not found. Every state is driven by JSON fixtures. The signature component is `TransitDiagram`, an SVG in which three agent lines meet at the verdict interchange.

**Tech Stack:**
- **Backend:** FastAPI, Pydantic v2, supabase-py, pytest/anyio.
- **Frontend:**
  - Node ≥ 20 with Vite;
  - React 18, react-router-dom 6;
  - Tailwind CSS 3, Recharts 2;
  - Vitest with @testing-library/react and jsdom;
  - ESLint.

**Spec:** `docs/superpowers/specs/2026-10-15-frontend-mvp-design.md` (the brief; binding). Also read:
- `docs/superpowers/specs/2026-10-15-realigned-roadmap-design.md` (Phase B);
- `PRODUCT.md`;
- `.impeccable/mocks/decision/r2-transit.html`: composition only. Its colours, demo numbers and "partial service" copy are wrong; brief §3 wins.

## Global Constraints

- The direction is pinned: **Midnight Transit Diagram**. Do not run impeccable's concept roll. Load the `impeccable` skill for the build and finish review, and read its `reference/mode-operate.md`. The build is code-led.
- **Colours** (brief §3), exact values:
  - **Ground:** midnight `#0F1B3D`, surface `#16244D`.
  - **Text:** porcelain `#F4F1EA`.
  - **Agent lines:** Fact-Check lilac `#B9A2F2`, Cross-Ref teal `#2BB3A3`, Linguistic porcelain `#F4F1EA` with a navy casing.
  - **Verdict bands:** `#DC2626` "Likely false" (0–33), `#F59E0B` "Mixed" (34–66), `#10B981` "Likely true" (67–100), `#8A94B0` "Not enough evidence" (unverified).
  - Agent lines are never red, blue, amber or green.
- **Verdict colour** appears only on the interchange ring and the verdict word, always with the word.
- **Score display:** `Math.round(c_total * 100)`. Band thresholds apply to that integer: ≤33, 34–66, ≥67.
- **`evidence_status === "unverified"`:** no composite score anywhere (circle, recent row, formula result, points/total column).
- **Copy is plain:** no transit or game words in verdicts, warnings or errors ("Partial analysis", never "partial service").
- **Bias chips** render `bias_label` verbatim. The client never computes, maps or recolours a label beyond the fixed per-label chip style. `Unrated` is a real state.
- **No secrets in the client.** The API base comes from `VITE_API_BASE_URL` (default `http://localhost:8000`).
- **Search suggestions** render only in `<iframe sandbox="allow-popups allow-popups-to-escape-sandbox" srcdoc=…>`, unmodified.
- **AllSides attribution line** under the sources: "Media bias ratings: AllSides.com (CC BY-NC 4.0)", linking to `https://www.allsides.com/media-bias/ratings`.
- **Layout:** verify at 375, 428, 768, 1024, 1280 and 1440px; touch targets ≥44px; no horizontal scroll; `prefers-reduced-motion` respected.
- **Backend:** additive only. `ScoringEngine` math and `apply_confidence_capping` behavior are unchanged (`eval-guardian` confirms). Use `uv add`, never pip.
- **Migrations** are numbered files the project lead runs by hand in the Supabase SQL editor. Never re-run 01–03.

## Review Focus

1. **`GET /api/results/recent` shadowed by `GET /api/results/{result_id}`** ("recent" parsed as an id → 404). The recent route must be registered first. → Task 4 `test_recent_route_not_shadowed`.
2. **An unverified claim leaking a number** through `c_total` in a recent row or in the breakdown's total. → Task 7 `ScoreBreakdown hides total when unverified`, Task 8 `RecentList shows no score for unverified`.
3. **Legacy stored rows:**
   - no `duration_ms`, `found_by`, cap fields or `created_at` in `agent_outputs`;
   - `evidence_status` null.

   These must render without crashing, with durations hidden. → Task 3 `test_stored_legacy_row_formats`, Task 7 `renders legacy fixture`.
4. **Search-suggestion HTML trying to script the page.** The iframe must not get `allow-scripts` or `allow-same-origin`. → Task 7 `SearchSuggestions uses exact sandbox`.
5. **Submitting twice while a check runs,** or a 429 without `Retry-After`. Submit is disabled while checking, and after a 429 for `Retry-After` seconds (default 60). → Task 8 `ClaimInput disables during check` and `rate limit falls back to 60s`.

---

### Task 0: Pre-flight

- [ ] **Step 1: Check that Plan 6 is merged.**
  - `docs/superpowers/plans/README.md` shows Plan 6 `completed`.
  - `git grep -n search_suggestions_html -- backend/app/api/schemas.py` has a hit.

  If either check fails, stop and ask the project lead.
- [ ] **Step 2: Check the working tree.** `git status --porcelain` must print nothing. Then create the branch: `git switch -c feat/frontend-mvp`.
- [ ] **Step 3: Check Node.** `node --version` must be ≥ v20. If it isn't, stop and ask the project lead.
- [ ] **Step 4: Baseline.**
  - From `backend/`: `uv run pytest -q`.
  - From the root: `uv run --project backend pytest eval/tests -m "not live" -q`.

  Both must be all green. Record the counts in the task artifact.

---

### Task 1: Cap transparency in scoring

**Files:**
- Modify: `backend/app/scoring/confidence.py` (add `binding_cap_rule`)
- Modify: `backend/app/scoring/engine.py` (`ScoringResult.raw_c_total`, `ScoringResult.cap_rule`)
- Test: `backend/tests/test_scoring_engine.py`

**Interfaces:**
- Produces:
  - `CAP_RULES = ("false_verdict", "disputed", "partial_analysis", "low_evidence")`, which is also the tie-break order.
  - `binding_cap_rule(c_total_raw: float, s_fact: float, v_consensus: float, agents_succeeded: int, total_agents: int = 3, fact_check_found: bool = False, verdict_found: bool = False, v_refute: float = 0.0) -> Optional[str]`: returns the rule whose cap is the lowest value below `c_total_raw`, or `None`.
  - The caps are the ones `apply_confidence_capping` already applies:
    - `partial_analysis` 0.50;
    - `low_evidence` 0.55, only when not disputed;
    - `false_verdict` `min(s_fact + 0.15, 0.25)`;
    - `disputed` 0.33.
  - `ScoringResult` gains `raw_c_total: Optional[float] = None` and `cap_rule: Optional[str] = None` (the last fields, with defaults).

- [ ] **Step 1: Write failing tests.**
  - `test_no_cap_rule_when_uncapped`: all agents ok, verdict found, s_fact 0.9 → `cap_rule is None` and `raw_c_total == c_total`.
  - `test_partial_analysis_cap_rule`: 2/3 agents, raw > 0.50 → `cap_rule == "partial_analysis"`, `c_total == 0.50`.
  - `test_disputed_beats_partial`: 2/3 agents and disputed → `"disputed"`.
  - `test_low_evidence_rule`: no fact-check, v_consensus 0.1, raw 0.62 → `"low_evidence"`.
  - `test_false_verdict_rule`: verdict_found, s_fact 0.0 → `"false_verdict"`.
  - `test_cap_rule_none_when_cap_not_binding`: Rule 1 applies but raw 0.40 < 0.50 → `None` (the warning is still present).
  - `test_c_total_unchanged_by_transparency`: for the 6 inputs above, `c_total` equals `apply_confidence_capping(...)[0]` called directly.
- [ ] **Step 2: Run** `uv run pytest tests/test_scoring_engine.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `binding_cap_rule` as a pure function in `confidence.py`, and set both new fields in `ScoringEngine.compute` from the raw value it already computes. Do not touch `apply_confidence_capping`.
- [ ] **Step 4: Run** `uv run pytest -q` (backend/). Expected: all pass. Then run `uv run --project backend pytest eval/tests/test_guardrails.py -q` from the root. Expected: pass.
- [ ] **Step 5: Commit** `feat(scoring): expose raw score and binding cap rule`.

---

### Task 2: Per-agent status, `found_by`, and sanitized degradation warnings

**Files:**
- Modify: `backend/app/agents/orchestrator.py`:
  - `dispatch_agents_node` timing;
  - `PipelineOutcome.agents`;
  - `score_node` warnings;
  - `_consolidate_sources`;
  - the `verify_claim` response and `agent_outputs`.
- Modify: `backend/app/api/schemas.py` (`AgentStatusModel`, `VerifyResponse.agents`, `SourceCitationModel.found_by`, `TrustScoreModel.raw_c_total`/`cap_rule`, `VerifyResponse.created_at`)
- Modify: `backend/app/utils/errors.py` (`degradation_message`)
- Test: `backend/tests/test_orchestrator_workflow.py`, `backend/tests/test_error_codes.py`

**Interfaces:**
- Consumes: Task 1 `ScoringResult.raw_c_total`, `ScoringResult.cap_rule`.
- Produces:
  - `AgentStatusModel { name: Literal["factcheck","crossref","linguistic"]; status: Literal["ok","degraded"]; duration_ms: Optional[int]; error_code: Optional[str] }`
  - `VerifyResponse.agents: List[AgentStatusModel] = []`
  - `VerifyResponse.created_at: Optional[str] = None` (ISO-8601)
  - `TrustScoreModel.raw_c_total: Optional[float] = None`, `TrustScoreModel.cap_rule: Optional[Literal["false_verdict","disputed","partial_analysis","low_evidence"]] = None`
  - `SourceCitationModel.found_by: Optional[Literal["factcheck","crossref"]] = None`
  - `degradation_message(agent_label: str, error_code: Optional[str]) -> str` in `utils/errors.py`. It returns a fixed sentence per code; `UNKNOWN` or `None` gives `f"{agent_label} check failed"`. It never includes exception text.
  - Persisted shape: `agent_outputs[<name>]["duration_ms"]: int` and `agent_outputs["scoring"] = {"raw_c_total": float, "cap_rule": str|None}`.

- [ ] **Step 1: Write failing tests.**
  - `test_response_has_agent_status_and_durations`: all mocks healthy → 3 entries in order factcheck, crossref, linguistic; `status == "ok"`; every `duration_ms` is a non-negative int.
  - `test_degraded_agent_status_and_code`: crossref raises `TimeoutError` → `status == "degraded"`, `error_code == "TIMEOUT"`, and `duration_ms` is set.
  - `test_unknown_error_text_never_in_warnings`: linguistic raises `RuntimeError("secret-path /etc/x")` → no warning contains `"secret-path"`, and one warning equals `"Linguistic check failed"`.
  - `test_found_by_marks_layer2_results_as_factcheck`: a Fact-Check match with `rating == "Web Search Result"` → `type == "corroborating"` and `found_by == "factcheck"`; a Cross-Ref source → `found_by == "crossref"`.
  - `test_cap_fields_in_response`: a partial run → `trust_score.cap_rule == "partial_analysis"` and `raw_c_total > c_total`.
  - `test_agent_outputs_persist_durations_and_scoring`: the `insert_cache` mock receives `agent_outputs["crossref"]["duration_ms"]` and `agent_outputs["scoring"]["cap_rule"]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_orchestrator_workflow.py tests/test_error_codes.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - In `dispatch_agents_node`, wrap each `ctx.run_node(...)` in a coroutine that records `time.monotonic()` before and after, so a failed agent still has a duration. Return the timings under `outputs["timings"] = {name: ms}`.
  - `score_node` builds `PipelineOutcome.agents`, a list of dicts matching `AgentStatusModel`.
  - Replace the two degradation-warning f-strings with `degradation_message(agent_label, ed.code.value if ed else None)`, using the agent labels "Fact-check", "Cross-reference" and "Linguistic".
  - The raw error stays in `agent_outputs` and the logs only.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(api): per-agent status, found_by, cap fields; no raw errors in warnings`.

---

### Task 3: Stored-result parity and `Retry-After`

**Files:**
- Create: `backend/app/api/result_format.py`
- Modify: `backend/app/agents/orchestrator.py` (`_format_cached_response` delegates to it)
- Modify: `backend/app/api/routes.py` (`get_result_endpoint` delegates to it)
- Modify: `backend/app/main.py` (`rate_limited_handler` adds `headers={"Retry-After": "60"}`)
- Test: `backend/tests/test_api_routes.py`, `backend/tests/test_cache.py`

**Interfaces:**
- Consumes: Task 2 persisted `agent_outputs` shape.
- Produces: `format_stored_result(row: Dict[str, Any], claim_text: str, *, cached: bool) -> Dict[str, Any]`.
  - It is the single mapper from a `results` row (joined claim text) to a `VerifyResponse` dict.
  - It fills `agents` from `agent_outputs` (`duration_ms: None` when absent, `status` from `degraded`, `error_code` from `error_details["code"]`).
  - It also fills `search_suggestions_html` from `agent_outputs`, `raw_c_total`/`cap_rule` from `agent_outputs["scoring"]` (else `None`), and `created_at` from `row["created_at"]`.
  - Sources pass through, with `found_by` left as stored.
  - It keeps the existing legacy-warning and default-profile behavior from `routes.py`.

- [ ] **Step 1: Write failing tests.**
  - `test_get_result_not_marked_cached`: GET returns `cached is False` and a `created_at`.
  - `test_get_result_has_agents_and_suggestions`.
  - `test_stored_legacy_row_formats`: a row with no `agent_outputs` and `evidence_status=None` → `agents == []`, `cap_rule is None`, `evidence_status == "unverified"`, and the legacy warning is present.
  - `test_cache_hit_uses_shared_formatter`: the cache-hit response equals `format_stored_result(row, claim, cached=True)` apart from `processing_time_ms`.
  - `test_rate_limit_sets_retry_after`: the 11th POST in a minute → 429 with header `Retry-After: 60`.
- [ ] **Step 2: Run** `uv run pytest tests/test_api_routes.py tests/test_cache.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Move the dict-building out of `routes.py` and `_format_cached_response` into `format_stored_result`. Both callers become one-liners. The 404 behavior is unchanged.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `refactor(api): one stored-result formatter; GET parity; Retry-After on 429`.

---

### Task 4: Featured results and `GET /api/results/recent`

**Files:**
- Create: `backend/app/db/migrations/07_featured_results.sql`
- Modify: `backend/app/api/schemas.py` (`RecentResultModel`)
- Modify: `backend/app/api/routes.py` (new route, registered **above** `/results/{result_id}`)
- Test: `backend/tests/test_api_routes.py`

**Interfaces:**
- Produces:
  - **Migration:** `ALTER TABLE results ADD COLUMN IF NOT EXISTS featured BOOLEAN NOT NULL DEFAULT false;` plus a partial index `ON results (created_at DESC) WHERE featured`.
  - `RecentResultModel { id: str; claim: str; c_total: float; evidence_status: Literal[...]; confidence_level: Literal[...]; created_at: str }`
  - `GET /api/results/recent?limit=10`:
    - returns `List[RecentResultModel]`, featured rows only, newest first;
    - `limit` is clamped to 1..20; rate limit `60/minute`;
    - selects only `id, c_total, evidence_status, confidence_level, created_at, claims(extracted_claim)`;
    - rows with `evidence_status` null come back as `"unverified"`.

- [ ] **Step 1: Write failing tests.**
  - `test_recent_route_not_shadowed`: GET `/api/results/recent` → 200, never the invalid-UUID 404.
  - `test_recent_only_featured_newest_first`: the Supabase mock asserts `.eq("featured", True)` and `.order("created_at", desc=True)`.
  - `test_recent_limit_clamped`: `limit=500` → the mock receives `.limit(20)`; `limit=0` → 1.
  - `test_recent_field_allow_list`: response items have exactly the six model keys.
  - `test_recent_empty_list`: no featured rows → `[]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_api_routes.py -k recent -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the route and the migration. The migration is **not** applied by the agent: add it to the "Before starting (user)" list in `docs/superpowers/plans/README.md` as "07 (Plan 7)".
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(api): featured-only recent results endpoint (+ migration 07)`.

---

### Task 5: Frontend scaffold, tokens, API client, fixtures

**Files:**
- Create the Vite React app in `frontend/` (`npm create vite@latest . -- --template react` inside the existing folder, keeping `src/{components,hooks,styles}`), then add tailwindcss@3, postcss, autoprefixer, recharts, react-router-dom, vitest, @testing-library/react, @testing-library/jest-dom and jsdom.
- Create: `frontend/tailwind.config.js` (tokens below), `frontend/src/styles/globals.css`, `frontend/.env.example` (`VITE_API_BASE_URL=http://localhost:8000`)
- Create: `frontend/src/lib/api.js`, `frontend/src/lib/verdict.js`, `frontend/src/lib/copy.js`
- Create: `frontend/src/fixtures/`, holding `fact_checked.json`, `corroborated.json`, `disputed_capped.json`, `unverified.json`, `partial.json`, `cached.json`, `legacy.json`, `long_claim.json`, `recent.json`, `errors.json`. All are valid against the Task 2–4 schemas. Pinned values the tests use:
  - `disputed_capped`: `raw_c_total` 0.52, `c_total` 0.33, `cap_rule` "disputed", one Fact-Check source with `type: "corroborating"`, `found_by: "factcheck"`.
  - `partial`: Linguistic `status: "degraded"`, `error_code: "TIMEOUT"`.
  - `legacy`: `agents: []`, no cap fields, no `found_by`, no `created_at`.
  - `unverified`: `c_total` 0.48.
  - `long_claim`: a 2000-char claim.
- Test: `frontend/src/lib/*.test.js`

**Interfaces:**
- Produces:
  - **Tailwind colour tokens:** `midnight`, `surface`, `porcelain`, `line-factcheck`, `line-crossref`, `line-linguistic`, `verdict-false`, `verdict-mixed`, `verdict-true`, `verdict-none`, `bias-left` `#3B6FD8`, `bias-leanleft` `#8FB0F0`, `bias-center` `#9AA3B5`, `bias-leanright` `#F0A39B`, `bias-right` `#D9544A`.
  - **Font:** Hanken Grotesk, self-hosted via `@fontsource/hanken-grotesk` (weights 500/600/700/800), with `font-variant-numeric: tabular-nums` on numbers.
  - `api.js`:
    - `verifyClaim(input: string, inputType = "text") -> Promise<VerifyResponse>`
    - `getResult(id: string) -> Promise<VerifyResponse>`
    - `getRecent(limit = 10) -> Promise<RecentItem[]>`
    - All three throw `ApiError { status: number, code: string|null, detail: string|null, retryAfterS: number|null }`. `code` comes from the body when it is an `ErrorResponse`, else `null`; `status === 0` means a network error.
  - `verdict.js`:
    - `score100(cTotal: number) -> number`
    - `bandFor(score: number) -> "false"|"mixed"|"true"`
    - `verdictView(resp) -> { label: string, colorToken: string, showScore: boolean, score: number|null, disputed: boolean }`
  - `copy.js` exports:
    - `ERROR_COPY: Record<code|"network"|"not_found"|"server", string>`, with the exact strings from brief §6.
    - `CAP_COPY: Record<cap_rule, string>` ("Partial analysis: not every check finished", "No fact-check and little coverage", "A fact-check rated this false", "Reputable sources contradict this claim").
    - `CONFIDENCE_COPY: Record<level, string>` ("High confidence" …).
    - `AGENT_LABELS = { factcheck: "Fact-Check", crossref: "Cross-Ref", linguistic: "Linguistic" }`.

- [ ] **Step 1: Write failing tests.**
  - `bandFor`: 0/33 → false, 34/66 → mixed, 67/100 → true.
  - `verdictView(unverified)`: `showScore === false`, `score === null`, `label === "Not enough evidence"`.
  - `verdictView(disputed_capped)`: `disputed === true`.
  - `ApiError` parsing:
    - a 422 `ErrorResponse` body → `code === "non_verifiable_claim"`;
    - a 404 `{detail}` body → `code === null`, `status === 404`;
    - a 429 with no header → `retryAfterS === 60`;
    - fetch rejecting → `status === 0`.
- [ ] **Step 2: Run** `npm run test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add the scripts `dev`, `build`, `preview`, `lint`, `test`.
- [ ] **Step 4: Run** `npm run test -- --run && npm run lint && npm run build`. Expected: pass.
- [ ] **Step 5: Commit** `feat(frontend): scaffold, tokens, API client, verdict logic, fixtures`.

---

### Task 6: TransitDiagram (the signature component)

**Files:**
- Create: `frontend/src/components/TransitDiagram.jsx`
- Test: `frontend/src/components/TransitDiagram.test.jsx`

**Interfaces:**
- Consumes: `verdictView`, `AGENT_LABELS`, `CAP_COPY`.
- Produces: `<TransitDiagram mode="checking"|"result" agents={AgentStatus[]} verdict={verdictView(...)} capRule={string|null} elapsedS={number} />`.
  - **Geometry:** an SVG with `viewBox="0 0 390 300"` scaled to width (max 640px). There are three 8px lines bending only at 0/45/90°: Fact-Check enters top-left, Cross-Ref middle-left, Linguistic bottom-left. All three meet at an interchange circle (radius ≈58) on the right.
  - **Labels:** fixed-size HTML/SVG text ≥11px, never scaled below 11px; at <400px, labels sit above their lines.
  - **`result` mode:**
    - each line is solid, or dashed at 50% opacity when `status === "degraded"`;
    - each label reads `"<Agent> · <s> s"`, with the duration omitted when `duration_ms` is null, and a status word for degraded agents ("Partial");
    - the ring stroke uses `verdict.colorToken`;
    - the circle shows `score` and the label, or only "Not enough evidence" when `!showScore`;
    - the cap note (`CAP_COPY[capRule]`) sits under the circle when `capRule` is set.
  - **`checking` mode:** all three lines are dashed with a moving dash offset, the circle reads "Checking…", and `elapsedS` is shown. There is no per-agent completion.
  - **Motion:** in `result` mode each line's `stroke-dashoffset` draws once (600 ms ease-out), then the ring fills. Under `prefers-reduced-motion: reduce` there is no animation in either mode.
  - **Accessibility:** `role="img"` with `aria-describedby` pointing at a visually hidden `<p>` that summarises each agent and the verdict in words.

- [ ] **Step 1: Write failing tests** (render the fixtures):
  - `fact_checked` shows "Likely true" or the band label and the score;
  - `unverified` shows "Not enough evidence" and no digits inside the circle;
  - `partial` marks the degraded line `data-status="degraded"` and the summary contains "Partial";
  - `checking` mode shows "Checking…" and no score;
  - `legacy` (null durations) renders no "· s".
- [ ] **Step 2: Run** `npm run test -- --run TransitDiagram`. Expected: FAIL.
- [ ] **Step 3: Implement.** Lift the composition from `.impeccable/mocks/decision/r2-transit.html`: the path shapes, the interchange and the label positions. Apply the Global Constraints colours instead of the sketch's.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(frontend): TransitDiagram verdict interchange`.

---

### Task 7: Result page

**Files:**
- Create: `frontend/src/components/{ResultView,VerdictLine,ScoreBreakdown,SourceList,SourceRow,BiasChip,StanceChip,LanguageRadar,SearchSuggestions,ResultFooter,ErrorNotice}.jsx`
- Create: `frontend/src/pages/ResultPage.jsx`, `frontend/src/hooks/useResult.js`
- Test: `frontend/src/components/*.test.jsx`, `frontend/src/pages/ResultPage.test.jsx`

**Interfaces:**
- Consumes: Tasks 5–6.
- Produces:
  - `<ResultView result={VerifyResponse} fresh={boolean} />`: the full result layout, reused by Home after a submit.
  - `ResultPage`: `/r/:id` via `useResult(id) -> {status:"loading"|"done"|"error", result, error}`.
  - Section order and content follow brief §5.1. On ≥1024px the layout is two columns: left (3/5) is diagram, VerdictLine and ScoreBreakdown; right (2/5) is a sticky SourceList, then LanguageRadar, SearchSuggestions and ResultFooter.

- [ ] **Step 1: Write failing tests.**
  - `ScoreBreakdown shows formula with weights`: renders `0.50`, `0.20`, `0.30` from `weights` and the per-agent table.
  - `ScoreBreakdown hides total when unverified`: the unverified fixture → no element with `data-testid="total"`, and the text "Inputs measured — no score given" is present.
  - `ScoreBreakdown shows cap`: `disputed_capped` → "Capped from 52 to 33" plus `CAP_COPY.disputed`.
  - `SourceList groups by found_by`: a `found_by: "factcheck"` source with `type: "corroborating"` is listed under Fact-Check.
  - `BiasChip renders label verbatim`: `"Unrated"` → text "Unrated"; an unknown label string → shown verbatim with the neutral style.
  - `StanceChip null renders nothing`.
  - `LanguageRadar scales and degrades`: a value of 0.28 → "28"; with the Linguistic agent degraded → "Language analysis unavailable for this check" and no chart.
  - `SearchSuggestions uses exact sandbox`: `iframe.getAttribute("sandbox") === "allow-popups allow-popups-to-escape-sandbox"`; nothing renders when `search_suggestions_html` is null.
  - `AllSides attribution present` with the exact text and link.
  - `ResultFooter`: `fresh && cached` → "Cached result"; `!fresh` → "Checked on" plus the date; `id === null` → no share button.
  - `renders legacy fixture` without throwing.
  - `ResultPage 404 → not-found message`.
- [ ] **Step 2: Run** `npm run test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** Load Recharts only on this route (`React.lazy` for `LanguageRadar`). Source titles link out with `target="_blank" rel="noopener noreferrer"`. Share uses `navigator.share` when present, else `navigator.clipboard.writeText(location.href)`, showing a "Link copied" status.
- [ ] **Step 4: Run** `npm run test -- --run && npm run lint`. Expected: pass.
- [ ] **Step 5: Commit** `feat(frontend): result page sections and states`.

---

### Task 8: Home, submit flow, checking state, errors

**Files:**
- Create: `frontend/src/components/{ClaimInput,ExampleClaims,RecentList,AppHeader}.jsx`
- Create: `frontend/src/pages/{HomePage,NotFoundPage}.jsx`, `frontend/src/hooks/{useVerify,useRecent}.js`, `frontend/src/App.jsx` (routes)
- Test: `frontend/src/components/*.test.jsx`, `frontend/src/hooks/useVerify.test.js`

**Interfaces:**
- Consumes: `verifyClaim`, `getRecent`, `ResultView`, `TransitDiagram`, `ERROR_COPY`.
- Produces:
  - `useVerify() -> { status: "idle"|"checking"|"done"|"error", result, error: ApiError|null, elapsedS: number, submit(text: string): void, blockedUntil: number|null }`.
  - On success with an `id`, navigate to `/r/:id` with `state.result`, so `ResultPage` renders without refetching (`fresh = true`). With `id === null`, render `ResultView` in place.
  - `ClaimInput`:
    - a textarea with a counter `n/2000` and a hard limit of 2000;
    - Submit is disabled when the text is empty or over the limit, during `checking`, and until `blockedUntil`;
    - the user's text stays after an error.
  - `ExampleClaims` holds three authored factual claims; tapping one fills the input. It does not auto-submit.
  - `RecentList`:
    - up to 10 rows, each with a verdict dot, the claim clamped to 2 lines, the verdict word and the relative time;
    - unverified rows show "Not enough evidence" and no number;
    - when the list is empty or the fetch fails, the section is hidden and the examples stay.
  - `AppHeader`: wordmark (links `/`) and a "New check" button (to `/`, focusing the input).

- [ ] **Step 1: Write failing tests.**
  - `ClaimInput disables during check`.
  - `ClaimInput keeps text after error`.
  - `counter blocks 2001 chars`.
  - `non_verifiable_claim shows ERROR_COPY text`.
  - `all_agents_failed shows no score and a retry button`.
  - `rate limit falls back to 60s`: a 429 without `retryAfterS` → `blockedUntil ≈ now + 60000`.
  - `network error copy`.
  - `RecentList shows no score for unverified`.
  - `RecentList hidden when empty`.
  - `checking state renders TransitDiagram in checking mode`.
- [ ] **Step 2: Run** `npm run test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** On ≥1024px Home is two columns: input and examples on the left, Recently checked on the right.
- [ ] **Step 4: Run** `npm run test -- --run && npm run lint && npm run build`. Expected: pass.
- [ ] **Step 5: Commit** `feat(frontend): home, submit flow, checking state, error states`.

---

### Task 9: Responsive, accessibility, performance, and the impeccable finish

**Files:** fixes only, in the files from Tasks 5–8. Create `DESIGN.md` and `.impeccable/design.json` (via the documenter).

- [ ] **Step 1: Run the stack.**
  - `uv run uvicorn app.main:app` (backend/)
  - `npm run dev` (frontend/)

  Capture each fixture state with the dev-only fixture switcher (`?fixture=<name>`, compiled out of production by an `import.meta.env.DEV` guard) at 375, 428, 768, 1024, 1280 and 1440px. Save the screenshots to `.impeccable/review/` (`mobile.png` 390-wide and `desktop.png` 1440-wide for the finish reviewer, plus `w<width>-<state>.png`).
- [ ] **Step 2: Check each capture:**
  - no horizontal scroll;
  - touch targets ≥44px;
  - diagram labels ≥11px;
  - the verdict word is always present;
  - the 2000-char claim (`long_claim`) wraps or clamps cleanly.

  Fix everything found in one batch, then recapture once.
- [ ] **Step 3: Run the checks.**
  - **Detector:** `C:\Users\MSI\.claude\skills\impeccable\scripts\impeccable.cmd detect --json frontend/src`; on Antigravity, use the impeccable skill's detect command. Fix the mechanical findings.
  - **Lighthouse:** run it on `npm run build && npm run preview`, mobile preset. Expected: Performance ≥85, Accessibility ≥95.
  - **axe:** keyboard-only pass (Tab through Home and Result; the focus ring is visible).
- [ ] **Step 4: Finish review.** Spawn impeccable's finish reviewer with:
  - the brief;
  - the direction contract (brief §3);
  - the screenshots;
  - `r2-transit.html` as the critique reference;
  - `.impeccable/quality-bar/*.webp`.

  Act on its disposition (two fix rounds at most). Then run the documenter to write `DESIGN.md` and `.impeccable/design.json`.
- [ ] **Step 5: Commit** `feat(frontend): responsive and a11y polish; DESIGN.md`.

---

### Task 10: Docs

**Files:**
- `documents/VeriFact_AI_TDD.md`:
  - §3.1: the new response fields and the `GET /api/results/recent` subsection;
  - §3.3: parity and `cached: false`;
  - §5: the `featured` column;
  - §9.1: the TransitDiagram replaces the semicircle gauge, with the same thresholds and colours;
  - §9.2: radar values ×100, plus the degraded state;
  - §9.3: grouping by `found_by`.
- `documents/VeriFact_AI_Build_Order.md` Phase 2 steps 2.4–2.7: point to Plan 7.
- `.agents/skills/frontend-viz/SKILL.md` and `.claude/skills/frontend-viz/SKILL.md`: the component list (`TransitDiagram` replaces `TrustGauge`; `AgentPipeline` is the diagram's checking mode), tokens, and the brief path.
- `README.md`: frontend run instructions (`cd frontend && npm install && npm run dev`, `.env` from `.env.example`).
- `docs/superpowers/plans/README.md`: Plan 7 status and migration 07.

- [ ] **Step 1: Make the edits.** Then run `git grep -n "TrustGauge" -- documents .agents .claude README.md`. Every hit must be historical or renamed.
- [ ] **Step 2: Commit** `docs: frontend MVP (Plan 7)`.

---

### Task 11: Gate

- [ ] **Step 1: Tests.**
  - `uv run pytest -q` (backend)
  - `uv run --project backend pytest eval/tests -m "not live" -q` (root)
  - `npm run test -- --run && npm run lint && npm run build` (frontend)

  All must be green. Paste the counts.
- [ ] **Step 2: Live check (project lead's OK first; it spends quota).** Three live claims end to end through the UI on `localhost:5173`, including one opinion. Screenshot each at 375 and 1440px.
- [ ] **Step 3: Review gate** (CLAUDE.md §5 / AGENTS.md §5):
  - `code-reviewer` (Claude Code) or `/review` (Antigravity) on the branch diff;
  - `eval-guardian` on Tasks 1–3: `c_total` byte-identical, guardrail tests green.

  Fix and re-review until approved.
- [ ] **Step 4: Log the outcome.** Update `docs/superpowers/plans/README.md` (status `completed`, migration 07 pending the user) and write the outcome to `review-fixes-handoff.md` ("Plan execution log").
