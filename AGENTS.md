# AGENTS.md — VeriFact AI

Project rules for **Google Antigravity** (and any agent reading the open AGENTS.md standard) working in this repository. Treat these as binding instructions, not suggestions. Antigravity reads this file at the project root for standing rules, and the `.agents/` directory for skills, workflows, and subagents.

---

## 1. Project

**VeriFact AI** turns a suspicious claim (text, URL, or screenshot) into a visual, source-backed credibility assessment in under 10 seconds — a multi-agent verification pipeline with a transparent composite trust score and citations. **Not** a chatbot.

Authoritative specs live in `documents/` — read them before non-trivial work; they outrank model priors:

| Doc | Use for |
|-----|---------|
| `documents/VeriFact_AI_PRD_v2.md` | Scope, user stories, scoring framework, success metrics, out-of-scope |
| `documents/VeriFact_AI_TDD.md` | Architecture, structure, API spec, agent contracts, DB schema, scoring code |
| `documents/VeriFact_AI_Build_Order.md` | Phased build order, per-step gates, dependency graph, pitfalls |
| `documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md` | Stack rationale, evaluation framework, metric targets |

If your change contradicts a doc, the doc wins — or update the doc in the same change and say so.

---

## 2. Architecture

```
Frontend (Vite + React + Tailwind + Recharts)
   │  REST /api/verify · /api/verify/image · WS /ws/status/{id}
   ▼
FastAPI (async)
   ├─ Semantic cache (pgvector, cosine ≥ 0.95) → short-circuit on hit
   ├─ Orchestrator (ADK hub, LiteLLM → Groq Llama 3.1 8B in dev)
   │     ├─ Fact-Check  → Google Fact Check API + Tavily   → S_fact
   │     ├─ Cross-Ref    → spaCy NER + Tavily + AllSides     → V_consensus
   │     └─ Linguistic   → VADER + heuristics/HF classifier  → B_ling
   ├─ Scoring → C_total = w1·S_fact + w2·(1−B_ling) + w3·V_consensus
   └─ Supabase (Postgres + pgvector): claims · results · bias_ratings · eval_runs
```

Worker agents run in parallel under the orchestrator, each with an 8-second timeout, degrading gracefully on failure (neutral value + "Partial Analysis") — never crashing the pipeline or fabricating a score.

Layout: `backend/app/{api,agents,scoring,services,db,utils}`, `frontend/src/{components,hooks,styles}`, `eval/{datasets,tests,metrics}`, `data/`.

---

## 3. Stack & tooling

- **Backend:** Python 3.12+, FastAPI (async), managed with **`uv`** (`uv add`, `uv run …`; never bare `pip` or hand-edited deps — there is a `uv.lock`).
- **Agents:** Google ADK + LiteLLM. Model id comes from `app/config.py`, never hardcoded. Dev `groq/llama-3.1-8b-instant`; prod Qwen3-30B-A3B / Llama 3.1 70B via vLLM.
- **DB:** Supabase cloud (Postgres + pgvector). No local Postgres.
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim) dev / `BAAI/bge-m3` (1024-dim) prod. Dimension is a **config var** — swap = config + migration, never code.
- **Frontend:** Vite + React + Tailwind + Recharts, mobile-first (375px).
- **Eval:** DeepEval + custom harness in `eval/`.

```bash
uv run uvicorn app.main:app --reload      # backend dev (from backend/)
uv run pytest                              # backend tests
uv add <package>                           # add dependency
uv run python eval/run_eval.py             # eval harness (repo root)
npm run dev | npm run build | npm run lint # frontend (from frontend/)
```

---

## 4. Engineering principles

1. Read spec → read code → write. Don't reinvent contracts the TDD already defines.
2. Build leaves before the hub: implement and test each worker agent standalone before the orchestrator.
3. Honor the agent contract: typed dataclass output, self-enforced 8s timeout, documented neutral value on failure, no silent exceptions.
4. The scoring engine is pure and deterministic: no I/O, no LLM, no randomness; assert `Σw=1.0`, `w1≥0.4`, clamp output to `[0,1]`.
5. Never fabricate certainty: enforce confidence capping; refuse opinions/questions with a clear message; "not enough information" is a valid answer.
6. **Bias ratings come ONLY from AllSides static data.** The LLM never generates a bias label. Unknown domain → `"Unrated"`. Zero tolerance.
7. Async everywhere; load heavy models (embeddings, OCR) once at startup.
8. Secrets stay server-side via `app/config.py`; never expose to frontend, never log, never commit `.env`.
9. Sanitize input (strip scripts/tracking params, 2000-char cap); validate images (MIME, ≤5MB, in-memory); rate-limit 10 req/min/IP.
10. Tests and Build-Order gates are part of "done."
11. Match surrounding code; type-hint Python; small single-purpose functions.
12. Small, reviewable, single-concern changes.

---

## 5. Mandatory review gate — ALWAYS review work

**No code change is done until it has been reviewed.** This is a hard rule.

After implementing or modifying any code, review before reporting completion:

- Apply the **`code-review`** skill (`.agents/skills/code-review/`) — or run the **`/review`** workflow, or delegate to the **`code-reviewer`** subagent — on the working diff, and address every finding (or justify why it's out of scope).
- For changes to scoring, agents, guardrails, or eval, also engage **`eval-guardian`** to confirm no metric/guardrail regression.

The review checks, at minimum: correctness against the relevant `documents/` spec; the agent/scoring contracts in §4; security (input sanitization, secret handling, injection); the confidence-capping and bias-leakage guardrails; error/degradation paths; and test coverage.

Report the outcome honestly — what was checked, found, fixed, deferred. If tests fail, say so with the output. Never call unreviewed or unverified work complete.

---

## 6. Skills, workflows & subagents (`.agents/`)

**Skills** (`.agents/skills/<name>/SKILL.md`) — directory-based, each with `name` + `description` frontmatter; the `description` is the semantic trigger Antigravity matches against your prompt. They activate automatically when relevant; you can also invoke the matching `/`-command explicitly.

| Skill | Triggers on |
|-------|-------------|
| `verifact-agent` | building/modifying an ADK worker or orchestrator agent |
| `scoring-engine` | C_total composite + confidence capping |
| `eval-harness` | eval tests, guardrails, calibration, weight tuning |
| `api-endpoint` | FastAPI routes, schemas, middleware |
| `frontend-viz` | React dashboard components & hooks |
| `code-review` | reviewing any change before it's called done |

**Workflows / slash commands** (`.agents/workflows/*.md`): `/verifact-agent`, `/scoring-engine`, `/eval-harness`, `/api-endpoint`, `/frontend-viz`, `/review` — explicit entry points that apply the matching skill end-to-end.

**Subagents** (`.agents/agents.md`): Antigravity spawns subagents **dynamically** on demand; that file is the roster of roles to delegate to — `backend-engineer`, `frontend-engineer`, `eval-guardian`, `code-reviewer`. The loop is always **plan → implement → test → review → fix → report.**

---

## 7. Gotchas

- Don't load BGE-M3 in dev on 8GB RAM — use MiniLM, `vector(384)`.
- Build the WebSocket status feed last; it's demo-flashy, architecturally least important.
- Never let the LLM guess bias ratings. AllSides only.
- Don't skip the hand-curated eval set — academic datasets are too old for 2026 claims.
- Handle "no fact-checks found" via confidence capping; most real claims have none.
- English only for v2.
- All agents fail → return an error, never a fabricated score.

---

## 8. Definition of done

Matches the spec; the Build Order gate passes; unit/eval tests for the change are green; guardrails (capping, bias leakage, input rejection) hold; the §5 review gate has passed; and the outcome is reported honestly with deferrals named.
