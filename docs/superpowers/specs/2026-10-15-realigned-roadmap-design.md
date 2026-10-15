# VeriFact AI — Realigned Roadmap (2026-10-15)

**Status:** draft for the project lead's approval
**Replaces:** the Phase 2 / Phase 3 ordering in `documents/VeriFact_AI_Build_Order.md` and PRD §4 (see "Doc updates" below)
**Companion:** `docs/superpowers/specs/2026-10-15-frontend-mvp-design.md` (the frontend brief)

## 1. Why this realignment

VeriFact exists for two reasons, both written in the spec:

- **The product:** a phone user gets a source-backed credibility verdict in under 10 seconds (PRD §1).
- **The portfolio:** hiring managers judge the architecture, eval rigor and production patterns as much as the UX (PRD §2, secondary user). The Resume Impact Checklist (Stack & Eval doc, Part 3) lists the claims the project must support.

The project lead wants the resume story to cover what it lacks today: **agent ops** — robust evals of agentic behavior, agent governance, monitoring, logging and traceability of a deployed system. Building agents is not the gap.

The order changes so that a working MVP exists first (something real to evaluate, trace and deploy), and the agent-ops work then lands on a product that runs end to end.

## 2. Where the project stands (checked 2026-10-15)

| Phase (Build Order) | State |
|---|---|
| 0 — Foundation | Done. |
| 1 — Core pipeline | All seven steps built; Plans 1–5 (stance, cache safety, hardening, eval integrity, doc sync) done; the orchestrator is an ADK 2.12 `Workflow` graph (commit d169fff). **Plan 6** (Google Search grounding replaces Tavily): Tasks 0–2 committed, Tasks 3–4 uncommitted in the working tree, Tasks 5–9 open. The Phase 1 gate "eval harness runs against 50+ labeled claims" is **not met**: 25 curated claims; the only stored baseline (`eval/results/baseline_run.json`, 2026-09-28) is `INVALID_RUN` (empty replay cache, zero sources). |
| 2 — Multi-modal + frontend | Not started. `frontend/src/{components,hooks,styles}` are empty. No `/api/verify/image`, no OCR, no URL scraper; a URL input is currently verified as if it were claim text. |
| 3 — Eval + polish | Harness pieces exist (replay, leave-source-out, segments, LIAR/FEVER builders, MAE, calibration, bias leakage). No DeepEval, no ADK eval, no tracing, no deployment. |

Resume checklist drift: hybrid retrieval (BGE-M3) is not built (MiniLM dense only, by design in dev); the cache cost saving and latency claims are unmeasured; DeepEval guardrails are not started.

## 3. Decisions (project lead, 2026-10-15)

| # | Decision |
|---|---|
| R1 | **Orchestration stays deterministic.** Python decides which tool runs; no LLM chooses tools. Trajectory evals grade the fixed path from traces (did each node run, in budget, with which outcome), not an LLM's tool choices. |
| R2 | **Finish Plan 6 first**, before any frontend work. The frontend builds on the final API, including `search_suggestions_html`. |
| R3 | **Frontend MVP next**, designed with the impeccable skill, built by Antigravity from a written brief and plan. |
| R4 | **Phone user wins** when the two audiences conflict; the reviewer's depth (agents, weights, timings) is one deliberate step away. |
| R5 | **No existing branding**; the visual world is created fresh. Chosen 2026-10-15: **Midnight Transit Diagram** (the agents as lines meeting at a verdict interchange); responsive across phone, tablet, laptop and desktop. Full brief: `2026-10-15-frontend-mvp-design.md`. |
| R6 | **Homepage = claim input + "Recently checked" claims** from VeriFact's own database. No daily news feed in the MVP. |
| R7 | **Keep DeepEval and add ADK eval.** DeepEval stays for the faithfulness/hallucination guardrail as specified (Stack & Eval §2.5); ADK eval (`agents-cli eval`) adds agent-behavior and trace-level evaluation. |
| R8 | **URL and screenshot input (Plan 8) runs last**, after evals and governance (revised 2026-10-15). The MVP is text input end to end (Phases A–B); evals, tracing and deployment work on that. |
| R9 | **Deployment target is still open.** TDD §10's constraint stands: the backend needs about 2 GB of RAM; Render free is too small, Railway has no free tier, Hugging Face Docker Spaces need PRO for new Spaces since June 2026. Cloud Run (free tier, card on file) is a further candidate. Decided in Phase E. |

## 4. Realigned phases

Run order: **A → B → D → E → C**. Each phase becomes one plan file under `docs/superpowers/plans/`, executable by Antigravity's `/execute-plan` or Claude Code's subagent-driven-development, with the usual Task 0 pre-flight and review-gate task.

### Phase A — Finish Plan 6 (Google Search grounding)

Plan file exists: `2026-10-14-google-search-grounding.md`, Tasks 3–9. Commit or finish the in-progress Tasks 3–4 first (the working tree must be clean for `/execute-plan`).

**Gate:** Plan 6's own Task 9 gate; `VerifyResponse` carries `search_suggestions_html`; Tavily removed.

### Phase B — Frontend MVP (Plan 7)

Text input end to end on a phone. Scope and look: the frontend brief.

- Home: claim input, example claims, "Recently checked" list.
- Waiting state: the agent pipeline shown as it runs (simulated from timings until the WebSocket exists; see brief).
- Results: verdict, score and confidence, warnings, score breakdown, linguistic radar, source list with AllSides chips and attribution, Google search suggestions in a sandboxed iframe, share link.
- Shared result page: `/r/{id}` via `GET /api/results/{id}`.
- All states: loading, every error code in `ErrorResponse`, `unverified` ("Not enough evidence"), `disputed`, partial analysis, cached result.
- **Backend addition:** a read-only `GET /api/results/recent` endpoint for the homepage (shape and privacy rule in the brief).

**Gate:** frontend-viz Done-when (375/428/768/1024px, touch targets ≥44px, all states, no client-side bias, no secrets, `npm run lint && npm run build`), Lighthouse Performance ≥85 on mobile, impeccable finish review, `/review`.

### Phase C — Multi-modal input (Plan 8) — runs last, after Phase E

- **URL input (P2-02):** detect `input_type: "url"`, fetch with the existing SSRF-guarded `app/services/page_fetch.py`, extract the article's primary claim (headline + lede; an ADK leaf prompt only if extraction needs it), then run the normal pipeline. Never verify the URL string itself.
- **Screenshot input (P2-01, P2-03):** Surya OCR loaded once at startup; `POST /api/verify/image` (PNG/JPEG, ≤5 MB, in memory, MIME validated) returning `ocr_extracted_text` and `ocr_confidence` (TDD §3.2); low OCR confidence surfaces as a warning, and the user can edit the extracted text before verifying.
- Frontend: InputBox auto-detects URLs and accepts an image (picker + drag-drop); shows extracted text for confirmation.
- `eval/tests/test_ocr_accuracy.py` per Stack & Eval §2.6.

**Gate (Build Order Phase 2 checkpoint):** a phone user can submit text, a URL, or a screenshot and get a rendered result; OCR accuracy test green; Build Order Phase 2 checkpoint met.

### Phase D — Evals (Plan 9, designed separately before it starts)

To be shaped in its own brainstorm once the MVP is done. Fixed scope boundaries now:

- A **valid** baseline first: grow the curated set to ≥50 claims, record once with `--replay record` (spends quota: needs the lead's OK), then every later run replays.
- ADK eval over the pipeline's traces: custom code metrics for the fixed path (every agent node ran, deadline respected, degradation flagged honestly, bias labels only from AllSides) plus groundedness/response-quality judges on the Groq leaf outputs (`verdict_extractor`, `claim_aligner`) against the fetched page text.
- DeepEval faithfulness guardrail as specified: ≥30 seeded explanations, flags ≥80% of injected unsupported facts, ≤20% false flags.
- Measure the unmeasured resume claims: cache hit rate and saved calls, p50/p95 latency.
- Publish an eval report in the README (only from a valid run).

### Phase E — Governance, observability, deployment (Plan 10, designed separately)

To be shaped in its own brainstorm. Fixed scope boundaries now:

- **Tracing:** OpenTelemetry spans for every workflow node, agent, leaf prompt, search and fetch call, exported to a free-tier backend (Cloud Trace if on Cloud Run; otherwise Phoenix or Langfuse), with prompt/response content off by default.
- **Logging:** structured JSON logs with a request id that also appears in the trace and the stored result.
- **Governance:** a written agent registry (each agent's purpose, tools, model, budget, timeout, data access, failure mode), and the existing controls (spend budgets, timeouts, SSRF guard, AllSides-only bias, input rejection, retention purge) documented as enforced policies, each tied to the test that proves it. Per-run audit record: model ids, prompt versions, agent outcomes.
- **Deployment:** decide R9, deploy, run the eval suite against the deployed URL (`agents-cli eval generate --url` or the harness), keep Supabase from pausing.

### Stretch (after Phase E, not in the MVP)

- WebSocket agent status (P2-07) replacing the simulated pipeline.
- "Today's coverage" panel built on GDELT (free, multi-outlet; titles + links only, no article text).
- Hybrid retrieval (BGE-M3) if the resume claim is kept.

## 5. Doc updates this realignment requires

Done in the same change that adds Plans 7 and 8 (CLAUDE.md §1: the doc wins or is updated in the same change):

- `documents/VeriFact_AI_Build_Order.md`: phase order A–E above; Step 3.5 keeps DeepEval and adds ADK eval.
- `documents/VeriFact_AI_PRD_v2.md` §4: phase table reordered; P3-02 adds ADK eval.
- `CLAUDE.md` §3 and `AGENTS.md`: Eval line becomes "DeepEval + ADK eval (`agents-cli eval`) + custom harness".
- `README.md` "Project status": current state.
- `docs/superpowers/plans/README.md`: rows for Plans 6 (status), 7, 8, and placeholders for 9, 10.

## 6. What is left to the builder

Task breakdown, file layout inside `frontend/src/` beyond the named components, test tooling for the frontend (Vitest + Testing Library is the default), and the choice of a lightweight router.
