# CLAUDE.md — VeriFact AI

Guidance for Claude Code (and any agent) working in this repository. **These instructions override default behavior. Follow them exactly.**

---

## 1. What this project is

**VeriFact AI** turns a suspicious claim (text, URL, or screenshot) into a visual, source-backed credibility assessment in under 10 seconds. It is **not** a chatbot. It is a multi-agent verification pipeline that produces a transparent composite trust score with citations.

The canonical spec lives in `documents/`. Read these before non-trivial work — they are the source of truth, not your training data:

| Doc | Use it for |
|-----|-----------|
| `documents/VeriFact_AI_PRD_v2.md` | Product scope, user stories, scoring framework, success metrics, what's out of scope |
| `documents/VeriFact_AI_TDD.md` | Architecture, project structure, API spec, agent contracts, DB schema, scoring code |
| `documents/VeriFact_AI_Build_Order.md` | Phased build order, per-step gates, dependency graph, pitfalls |
| `documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md` | Stack rationale, evaluation framework, metrics & targets |

If your change contradicts a doc, the doc wins — or you update the doc in the same change and say so.

---

## 2. Architecture in one screen

```
Frontend (Vite + React + Tailwind + Recharts)
   │  REST /api/verify · /api/verify/image · WS /ws/status/{id}
   ▼
FastAPI (async)
   ├─ Semantic cache (pgvector, cosine ≥ 0.95) → short-circuit on hit
   ├─ Orchestrator agent (ADK hub, LiteLLM → Groq Llama 3.1 8B in dev)
   │     ├─ Fact-Check agent   → Google Fact Check API + Tavily   → S_fact
   │     ├─ Cross-Ref agent     → spaCy NER + Tavily + AllSides     → V_consensus
   │     └─ Linguistic agent    → VADER + heuristics/HF classifier  → B_ling
   ├─ Scoring engine → C_total = w1·S_fact + w2·(1−B_ling) + w3·V_consensus
   └─ Supabase (Postgres + pgvector): claims · results · bias_ratings · eval_runs
```

Worker agents run in **parallel** under the orchestrator, each with an **8-second timeout**. A failed/timed-out agent degrades gracefully (neutral value + "Partial Analysis" flag) — it never crashes the pipeline or fabricates a score.

**Layout** (see TDD §2 for the full tree): `backend/app/{api,agents,scoring,services,db,utils}`, `frontend/src/{components,hooks,styles}`, `eval/{datasets,tests,metrics}`, `data/`.

---

## 3. Stack & tooling (non-negotiable choices)

- **Backend:** Python **3.12+**, FastAPI (async), managed with **`uv`** (there is a `uv.lock` — use `uv add` / `uv run`, never edit `pyproject.toml` deps by hand or use bare `pip`).
- **Agents:** Google ADK + LiteLLM. Models are swapped via config, never hardcoded in agent logic. Dev: `groq/llama-3.1-8b-instant`. Prod: Qwen3-30B-A3B / Llama 3.1 70B via vLLM.
- **DB:** Supabase cloud (Postgres + pgvector). Do not stand up local Postgres.
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim) in dev, `BAAI/bge-m3` (1024-dim) in prod. **Dimension is a config var** — swapping is a config + migration, never a code change.
- **Frontend:** Vite + React + Tailwind + Recharts. Mobile-first (375px baseline).
- **Eval:** DeepEval + custom harness in `eval/`.

Run commands:
```bash
# backend (from backend/)
uv run uvicorn app.main:app --reload      # dev server
uv run pytest                              # unit tests
uv add <package>                           # add a dependency

# eval (from repo root)
uv run python eval/run_eval.py             # full eval harness

# frontend (from frontend/)
npm run dev                                # vite dev server
npm run build && npm run lint
```

---

## 4. Engineering principles (how a senior engineer works here)

1. **Read the spec, then the code, then write.** Don't invent contracts that already exist in the TDD.
2. **Build leaves before the hub.** Implement and test each worker agent standalone before wiring the orchestrator (Build Order §1). A working ugly leaf beats a half-built orchestrator.
3. **Respect the agent contract.** Every worker agent returns its typed dataclass (`FactCheckResult`, `CrossRefResult`, `LinguisticResult`, …), enforces its own timeout, and degrades to a documented neutral value on failure. No silent exceptions.
4. **The scoring engine is pure and deterministic.** No I/O, no LLM calls, no randomness. Invariants are asserted: `Σw = 1.0`, `w1 ≥ 0.4`, output clamped to `[0,1]`. Confidence-capping rules are enforced, not optional.
5. **Never fabricate certainty.** If `S_fact = 0.5` (no fact-check) and `V_consensus < 0.3`, cap and warn. If agents fail, cap and warn. If input is an opinion/question, refuse with a clear message. "I don't have enough information" is a correct answer.
6. **Bias ratings come ONLY from AllSides static data.** The LLM must never generate a political bias label. Unknown domain → `"Unrated"`. This is a zero-tolerance guardrail (tested in `eval/tests/test_guardrails.py`).
7. **Async all the way.** External calls (Fact Check API, Tavily) are `async` and dispatched concurrently. Load heavy models (embeddings, OCR) **once at startup**, never per request.
8. **Secrets stay server-side.** API keys load via `app/config.py` (Pydantic Settings) from `.env`. Never expose keys to the frontend, never commit `.env`, never log secret values.
9. **Sanitize all input.** Strip scripts/tracking params, cap at 2000 chars, validate image MIME + size (≤5MB, in-memory). Rate-limit 10 req/min/IP.
10. **Tests and gates are part of "done."** Each Build Order step has an explicit gate. A step isn't done until its gate passes and its tests are green.
11. **Match the surrounding code.** Mirror existing naming, typing, async style, and module boundaries. Type-hint everything in Python; keep functions small and single-purpose.
12. **Small, reviewable changes.** Prefer focused diffs tied to one Build Order step over sprawling multi-concern edits.

---

## 5. Mandatory review gate — agents ALWAYS review work

**No code change is "done" until it has been reviewed.** This is a hard rule, not a suggestion.

After implementing or modifying any code, you MUST run a review before reporting completion:

- For substantive changes, **delegate to the `code-reviewer` subagent** (`.claude/agents/code-reviewer.md`) and address every finding it returns, or explicitly justify why a finding is out of scope.
- For changes touching scoring, agents, guardrails, or eval, **also engage `eval-guardian`** to confirm no metric/guardrail regression.
- You may also invoke the bundled `/code-review` skill on the working diff.

The review must check, at minimum: correctness against the relevant `documents/` spec, the agent/scoring contracts in §4, security (input sanitization, secret handling, injection), the confidence-capping and bias-leakage guardrails, error/degradation paths, and test coverage for the change.

Report the review outcome honestly: what was checked, what was found, what was fixed, and anything deferred. If tests fail, say so with the output. Never describe unreviewed or unverified work as complete.

---

## 6. Subagents available

Delegate proactively; keep the main thread focused on orchestration and integration.

| Subagent | When to use it |
|----------|----------------|
| `backend-engineer` | Implement/modify FastAPI routes, ADK agents, scoring, services, DB code |
| `frontend-engineer` | Build/modify React components, hooks, Tailwind/Recharts visualizations |
| `eval-guardian` | Build eval tests, run the harness, check metric/guardrail regressions |
| `code-reviewer` | **Always** — final review gate on any code change before it's called done |

Every implementation subagent ends its work by handing off to review (§5). The loop is: **plan → implement → test → review → fix → report.**

---

## 7. Guardrails & gotchas (learned the hard way — see Build Order §"Common Pitfalls")

- Don't load BGE-M3 in dev on 8GB RAM — use MiniLM and set `vector(384)`.
- Don't build the WebSocket status feed until the rest works (it's demo-flashy, architecturally least important).
- Don't let the LLM guess bias ratings. AllSides only.
- Don't skip the manually-curated eval set — academic datasets (LIAR/FEVER) are too old for 2026 claims.
- Handle "no fact-checks found" gracefully via confidence capping — most real claims won't have a fact-check.
- English only for v2. No multilingual scope creep.
- Never return a fabricated score when all agents fail — return an error.

---

## 8. Definition of done

A change is done when: it matches the relevant spec; the Build Order gate for the step passes; unit/eval tests for the change are green; guardrails (capping, bias leakage, input rejection) hold; it has passed the §5 review gate; and the outcome is reported honestly with any deferrals named.
