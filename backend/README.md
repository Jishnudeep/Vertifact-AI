# VeriFact AI — backend

FastAPI service that runs the verification pipeline: an ADK 2.x `Workflow` of three worker agents, a deterministic scoring engine and an NLI-guarded semantic cache on Supabase (Postgres + pgvector). Project overview: [`../README.md`](../README.md).

## Run

```bash
cp ../.env.example ../.env       # fill in Supabase, Gemini and Fact Check keys
uv sync
uv run uvicorn app.main:app --reload
curl localhost:8000/health        # → {"status":"ok"}
```

On startup the app preloads the AllSides ratings, spaCy, the embedding model and the NLI model, so the first request does not pay for model loading. The first boot downloads the Hugging Face models.

Apply `app/db/migrations/*.sql` by hand in the Supabase SQL editor, in numeric order. Never re-run 01–03 after 04: they recreate an old `match_claims` overload. Migration 07 adds `results.featured`; set it to `true` on a row to show that result in the "Recently checked" feed.

### ADK dev UI

```bash
uv run adk web                     # then pick "verifact"; input is {"text": "<claim>"}
```

`verifact/agent.py` is the entry point. It prepares the request (sanitize, validate, embed) and runs the same workflow as the API. The dev UI keeps its session store in `.adk/` (gitignored); that store holds claims and agent outputs.

## API

| Method and path | Purpose | Limit |
|---|---|---|
| `POST /api/verify` | Verify a claim: `{"input": "...", "input_type": "text"}` | 10/min per client IP |
| `GET /api/results/{id}` | A stored result, by result id or claim id | 60/min |
| `GET /api/results/recent` | Featured recent results for the landing feed | 60/min |
| `GET /health` | Liveness | — |

Errors: `400` invalid input, `422` not a verifiable claim (opinion or question), `429` rate limit (with `Retry-After`), `500` all agents failed. The schema is in `app/api/schemas.py`; interactive docs at `/docs`.

## Layout

```
app/
  agents/      orchestrator (Workflow graph), factcheck / crossref / linguistic workers, shared ADK helpers
  scoring/     engine, confidence caps, weights: pure functions, no I/O
  services/    Fact Check API, grounded search, page fetch (SSRF-guarded), NLI, embeddings, cache, budgets
  api/         routes, schemas, stored-result formatting, client IP
  db/          Supabase connection, SQL migrations, retention purge
  utils/       validators, error classification, text processing
verifact/      `adk web` entry point
tests/         unit and contract tests (providers are scripted or mocked)
```

## Pipeline contract

1. `cache_lookup`: embed the claim, query pgvector (cosine ≥ 0.95), then require a fresh row, a complete run and NLI equivalence. Any error is a miss.
2. `dispatch_agents`: run Fact-Check, Cross-Ref and Linguistic concurrently. Each is a child node with its own timeout; a raise or timeout becomes that agent's neutral result with `degraded=True` and an error code.
3. `score`: refuse to score when all three degraded; otherwise `ScoringEngine.compute` returns `c_total`, confidence level, evidence status, warnings and the binding cap rule.
4. Persistence runs as a background task after the response and is drained on shutdown.

## Config that matters

All settings live in `app/config.py` and are documented in [`../.env.example`](../.env.example). The ones to know:

- `MODEL_ID`, `SEARCH_MODEL_ID`: models are config, never hardcoded in agent logic.
- `*_THINKING_LEVEL`, `*_MAX_TURNS`: per-agent reasoning level (Gemini 3 models only) and model-turn cap.
- `AGENT_TIMEOUT_S`, `REQUEST_DEADLINE_S`: backstops, not tuned values.
- `SEARCH_DAILY_CALL_BUDGET`, `LLM_DAILY_CALL_BUDGET`: per-process daily ceilings. They do not yet sum to the provider's real daily cap or survive a restart; Plan 10 fixes that.
- `TRUSTED_PROXY_HOPS`: must be set behind a reverse proxy, or every user shares one rate-limit bucket.

## Tests

```bash
uv run pytest                     # offline; tests marked `live` are deselected
uv run pytest -m live             # hits real APIs and spends quota: ask first
```

The evaluation suite lives in `../eval/` and runs from the repo root with `uv run --project backend pytest eval/tests`.
