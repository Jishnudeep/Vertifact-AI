# Hardening Implementation Plan (latency, spend, abuse, retention)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Antigravity) or superpowers:subagent-driven-development (Claude Code) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `docs/superpowers/plans/README.md` first.

**Goal:** Close findings H1–H4. This plan adds:
- a single request deadline, so p95 stays under 10 s
- daily spend budgets, so one user can't drain the Tavily or Groq free tiers
- rate limiting that works behind a proxy
- a reusable SSRF guard for the future URL scraper
- a data retention policy

**Architecture:** Changes are small and separate:
1. The orchestrator computes a deadline and hands each agent a `timeout=` that fits inside it. Persistence runs as a shielded background task, so a slow DB write never blocks the response.
2. `app/services/budget.py` holds per-day counters, charged at the two cost choke points: `search_web` and `run_adk_agent`. When a budget is exhausted, the call raises an outage error, and the existing degradation path reports "Partial analysis".
3. `app/api/client_ip.py` reads `X-Forwarded-For` only across a configured number of trusted proxy hops.
4. `app/utils/url_safety.py` validates a URL resolves only to public IPs.
5. Migration 05 adds a purge function for rows older than the retention window.

**Tech Stack:** Python 3.12, FastAPI/Starlette, slowapi, asyncio, `ipaddress`/`socket`, Supabase Postgres (optional pg_cron).

**Spec:** `docs/superpowers/specs/2026-09-23-prd-tdd-review.md` (H1–H4). Also read `documents/VeriFact_AI_TDD.md` §3 and §11, and `documents/VeriFact_AI_PRD_v2.md` §8–§10.

## Global Constraints

- **Run after Plans 1 and 2** (it edits the same orchestrator lines) and **before Plan 3's recording step**, because crossref search depth changes the replay cache keys. See the order in `README.md`.
- Python **3.12+**, `uv` only. Backend commands run from `backend/`.
- Agent contract unchanged: typed result, self-enforced timeout, neutral value plus `degraded=True` on failure. `timeout` becomes an **optional** argument; the default stays 8.0 s.
- Budget exhaustion is an **outage**, never "no evidence". It must surface as "Partial analysis", never as a confident score (CLAUDE.md §4.5).
- Never log secrets or full client IPs at INFO level. Timing logs carry the path, status and milliseconds only.
- Every commit message ends with the executor's attribution line (Claude Code: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app/config.py` | Modify | Deadline, budgets, proxy hops, retention settings |
| `backend/app/agents/orchestrator.py` | Modify | Deadline, per-agent budget, bounded cache lookup, shielded persistence, sanitized `raw_input` |
| `backend/app/agents/{factcheck,crossref,linguistic}_agent.py` | Modify | `run(claim, timeout=None)` |
| `backend/app/main.py` | Modify | `X-Process-Time-Ms` middleware |
| `backend/app/services/budget.py` | Create | `DailyBudget`, `get_tavily_budget()`, `get_llm_budget()`, `LLMBudgetExceeded` |
| `backend/app/services/search_api.py` | Modify | `search_depth` parameter; charge the Tavily budget |
| `backend/app/utils/adk_helpers.py` | Modify | Charge the LLM budget per attempt |
| `backend/app/api/client_ip.py` | Create | Proxy-aware client IP for slowapi |
| `backend/app/api/routes.py` | Modify | `Limiter(key_func=client_ip)` |
| `backend/app/utils/url_safety.py` | Create | `validate_public_url()` (SSRF guard) |
| `backend/app/db/migrations/05_retention.sql`, `backend/app/db/purge.py` | Create | Retention purge |
| `backend/tests/test_deadline.py`, `test_budget.py`, `test_client_ip.py`, `test_url_safety.py`, `test_purge.py` | Create | Tests |
| `documents/VeriFact_AI_TDD.md` §3.1/§11, `documents/VeriFact_AI_PRD_v2.md` §8/§9 | Modify | Specs |

---

### Task 0: Pre-flight

- [ ] Run `git status --short`. Expected: empty. Confirm Plans 1 and 2 are merged.
- [ ] Run `git checkout -b fix/hardening`.
- [ ] Run `uv run pytest -q` from `backend/`. Expected: all pass. Record the count.

---

### Task 1: Request deadline and per-agent budgets (H1)

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/agents/orchestrator.py`
- Modify: the three agents' `run()`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_deadline.py`

**Interfaces:**
- Produces:
  - `settings.request_deadline_s: float = 9.5`
  - `orchestrator.agent_budget(remaining_s: float) -> float`
  - constants `AGENT_TIMEOUT_CAP = 8.0`, `MIN_AGENT_TIMEOUT = 1.0`, `PERSIST_RESERVE_S = 1.0`, `CACHE_LOOKUP_TIMEOUT_S = 1.5`, `PERSIST_MIN_WAIT_S = 0.25`
  - each agent exposes `run(claim: str, timeout: Optional[float] = None)`
  - response header `X-Process-Time-Ms`

- [ ] **Step 1: Write the failing tests.** Create `backend/tests/test_deadline.py`:

```python
import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents import orchestrator
from app.agents.orchestrator import agent_budget, verify_claim
from app.agents.crossref_agent import CrossRefResult, run as crossref_run
from app.agents.factcheck_agent import make_neutral_factcheck
from app.agents.linguistic_agent import LinguisticResult
from app.config import settings


def test_agent_budget_bounds():
    assert agent_budget(9.5) == 8.0          # capped at the 8 s agent contract
    assert agent_budget(5.0) == 4.0          # leaves PERSIST_RESERVE_S for the DB write
    assert agent_budget(0.3) == 1.0          # never below MIN_AGENT_TIMEOUT


def _ok_results():
    fc = make_neutral_factcheck(degraded=False)
    cr = CrossRefResult(entities_extracted=[], sources_found=[], coverage_distribution={
        "Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0, degraded=False)
    ling = LinguisticResult(clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
                            informational_density=1.0, b_ling_score=0.0, degraded=False)
    return fc, cr, ling


@pytest.mark.anyio
async def test_agents_get_the_remaining_budget(monkeypatch):
    monkeypatch.setattr(settings, "request_deadline_s", 4.0)
    fc, cr, ling = _ok_results()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)) as fc_run, \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")):
        await verify_claim("The Earth orbits the Sun")
    timeout = fc_run.call_args.kwargs["timeout"]
    assert 1.0 <= timeout <= 3.0


@pytest.mark.anyio
async def test_slow_cache_lookup_counts_as_miss(monkeypatch):
    async def slow_lookup(*args, **kwargs):
        await asyncio.sleep(5)
    monkeypatch.setattr(orchestrator, "CACHE_LOOKUP_TIMEOUT_S", 0.05)
    fc, cr, ling = _ok_results()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", side_effect=slow_lookup), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)) as fc_run, \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")):
        result = await verify_claim("The Earth orbits the Sun")
    assert fc_run.called
    assert result["cached"] is False


@pytest.mark.anyio
async def test_slow_persistence_does_not_block_the_response(monkeypatch):
    monkeypatch.setattr(settings, "request_deadline_s", 1.0)
    async def slow_insert(**kwargs):
        await asyncio.sleep(3)
        return "late-id"
    fc, cr, ling = _ok_results()
    start = time.perf_counter()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)), \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", side_effect=slow_insert):
        result = await verify_claim("The Earth orbits the Sun")
    assert result["id"] is None
    assert time.perf_counter() - start < 2.0


@pytest.mark.anyio
async def test_raw_input_is_persisted_sanitized():
    """H4: never store the unsanitized input (scripts, tracking params)."""
    fc, cr, ling = _ok_results()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)), \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")) as insert:
        await verify_claim("NASA confirms <script>alert(1)</script>water on Mars")
    kwargs = insert.call_args.kwargs
    assert "<script>" not in kwargs["raw_input"]
    assert kwargs["raw_input"] == kwargs["extracted_claim"]


@pytest.mark.anyio
async def test_agent_honours_explicit_timeout():
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(1.0)
        return []
    with patch("app.agents.crossref_agent.search_web", side_effect=slow_search):
        result = await crossref_run("NASA announces Mars landing", timeout=0.01)
    assert result.degraded is True and "Timeout" in result.error


def test_process_time_header():
    from app.main import app
    response = TestClient(app).get("/health")
    assert int(response.headers["X-Process-Time-Ms"]) >= 0
```

- [ ] **Step 2:** Run `uv run pytest tests/test_deadline.py -v`. Expected: FAIL with `ImportError: cannot import name 'agent_budget'`.

- [ ] **Step 3: Settings.** In `backend/app/config.py`, after the cache TTL settings from Plan 2, add:

```python

    # H1: total wall-clock budget per /api/verify request (PRD §10: < 10 s end-to-end).
    request_deadline_s: float = 9.5
```

- [ ] **Step 4: Agents accept `timeout`.** In each of `factcheck_agent.py`, `crossref_agent.py` and `linguistic_agent.py`, change `run` like this (shown for crossref; the others differ only in names):

```python
async def run(claim: str, timeout: Optional[float] = None) -> CrossRefResult:
    """
    Exposes the CrossReferenceAgent interface.
    Enforces a timeout (default TIMEOUT = 8 s; the orchestrator may pass less to fit its
    request deadline), falling back to degraded neutral (v_consensus=0.0).
    """
    budget = TIMEOUT if timeout is None else timeout
    try:
        return await asyncio.wait_for(_analyze(claim), timeout=budget)
    except (asyncio.TimeoutError, TimeoutError) as te:
        logger.warning(f"Cross-Reference Agent timed out on claim: {claim}")
        err_det = classify_provider_error(te, provider="crossref_agent")
        return make_neutral_crossref(
            degraded=True,
            error=f"Timeout: Cross-Reference Agent exceeded {budget:.1f}s",
            error_details=err_det
        )
```

Keep each file's existing `except Exception` branch unchanged. `Optional` is already imported
in all three files. `budget` is read at call time, so tests that patch the module `TIMEOUT`
keep working.

- [ ] **Step 5: Orchestrator.** In `backend/app/agents/orchestrator.py`:

(a) Add `from app.config import settings` to the imports. Below the exception classes, add:

```python
# H1: request deadline plumbing
AGENT_TIMEOUT_CAP = 8.0        # the per-agent contract (CLAUDE.md §2)
MIN_AGENT_TIMEOUT = 1.0        # below this an agent cannot do useful work
PERSIST_RESERVE_S = 1.0        # time kept back for the DB write
CACHE_LOOKUP_TIMEOUT_S = 1.5   # pgvector round-trip + up to 3 NLI equivalence checks
PERSIST_MIN_WAIT_S = 0.25

# Persistence tasks that outlived their request; referenced so they are not GC'd mid-write.
_background_tasks: set = set()


def agent_budget(remaining_s: float) -> float:
    """Per-agent timeout that fits the remaining request budget."""
    return max(MIN_AGENT_TIMEOUT, min(AGENT_TIMEOUT_CAP, remaining_s - PERSIST_RESERVE_S))
```

(b) At the top of `verify_claim`, directly after `start_time = time.time()`, add:

```python
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.request_deadline_s
```

(c) Replace the cache lookup line (`cached = await lookup_cache(embedding, claim_text=cleaned_text, similarity_threshold=0.95)`) with:

```python
        try:
            cached = await asyncio.wait_for(
                lookup_cache(embedding, claim_text=cleaned_text, similarity_threshold=0.95),
                timeout=CACHE_LOOKUP_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Semantic cache lookup exceeded its budget; treating as a miss.")
            cached = None
```

(d) Replace the three task lines `fc_task = ...`, `cr_task = ...` and `ling_task = ...` with:

```python
    per_agent_timeout = agent_budget(deadline - loop.time())
    fc_task = factcheck_agent.run(cleaned_text, timeout=per_agent_timeout)
    cr_task = crossref_agent.run(cleaned_text, timeout=per_agent_timeout)
    ling_task = linguistic_agent.run(cleaned_text, timeout=per_agent_timeout)
```

(e) In the persistence block, change `raw_input=raw_input,` to
`raw_input=cleaned_text,  # H4: never persist unsanitized input`. Then replace
`claim_id = await insert_cache(` and its closing `)` with a shielded task, like this:

```python
            persist_task = asyncio.create_task(insert_cache(
                # ... every existing keyword argument, unchanged except raw_input above ...
            ))
            try:
                claim_id = await asyncio.wait_for(
                    asyncio.shield(persist_task),
                    timeout=max(PERSIST_MIN_WAIT_S, deadline - loop.time()),
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("Persistence exceeded the request deadline; returning without an id (write continues).")
                _background_tasks.add(persist_task)
                persist_task.add_done_callback(_background_tasks.discard)
                claim_id = None
```

The existing outer `try:/except Exception as exc:` around persistence stays.

- [ ] **Step 6: Timing middleware.** In `backend/app/main.py`, add `import time` at the top. After the `app.add_middleware(CORSMiddleware, ...)` block, add:

```python
@app.middleware("http")
async def process_time_header(request: Request, call_next):
    """H1 / PRD §10: per-request latency, logged for p95 and returned to the client."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    logger.info("request_timing path=%s status=%s ms=%d", request.url.path, response.status_code, elapsed_ms)
    return response
```

- [ ] **Step 7:** Run `uv run pytest tests/test_deadline.py -v`, then `uv run pytest -q`. Expected: all pass. Existing orchestrator tests patch `run` with a plain `return_value`; the mocks accept the new `timeout` kwarg.
- [ ] **Step 8: Commit**

```bash
git add backend/app/config.py backend/app/agents/ backend/app/main.py backend/tests/test_deadline.py
git commit -m "feat(api): request deadline, per-agent budgets, shielded persistence, timing header (H1, H4)"
```

---

### Task 2: Daily spend budgets and cheaper cross-ref search (H2, H3)

**Files:**
- Create: `backend/app/services/budget.py`
- Modify: `backend/app/services/search_api.py`
- Modify: `backend/app/utils/adk_helpers.py`
- Modify: `backend/app/agents/crossref_agent.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_budget.py`

**Interfaces:**
- Produces:
  - `DailyBudget(name: str, limit: int, clock=...)` with `.try_charge(units: int = 1) -> bool` and `.used -> int` (a limit of 0 means unlimited)
  - `get_tavily_budget() -> DailyBudget`
  - `get_llm_budget() -> DailyBudget`
  - `LLMBudgetExceeded(RuntimeError)`
  - `search_web(..., search_depth: Literal["basic", "advanced"] = "advanced")`
  - `settings.tavily_daily_credit_budget: int = 30`
  - `settings.llm_daily_call_budget: int = 900`

- [ ] **Step 1: Write the failing tests.** Create `backend/tests/test_budget.py`:

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services import budget as budget_mod
from app.services.budget import DailyBudget, LLMBudgetExceeded
from app.services.errors import EvidenceAPIError
from app.services.search_api import search_web


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_budget_charges_until_limit_then_refuses():
    b = DailyBudget("t", limit=3, clock=Clock())
    assert b.try_charge(2) and b.used == 2
    assert not b.try_charge(2)          # would exceed; nothing charged
    assert b.used == 2
    assert b.try_charge(1) and b.used == 3


def test_budget_resets_on_new_utc_day():
    clock = Clock()
    b = DailyBudget("t", limit=1, clock=clock)
    assert b.try_charge()
    assert not b.try_charge()
    clock.now += timedelta(days=1)
    assert b.try_charge()


def test_zero_limit_disables_guard():
    b = DailyBudget("t", limit=0, clock=Clock())
    assert all(b.try_charge(100) for _ in range(5))


@pytest.mark.anyio
async def test_exhausted_tavily_budget_is_an_outage_without_http(monkeypatch):
    monkeypatch.setattr(budget_mod, "get_tavily_budget", lambda: DailyBudget("tavily", limit=1, clock=Clock()))
    with patch("app.services.search_api.httpx.AsyncClient") as client:
        with pytest.raises(EvidenceAPIError) as err:
            await search_web("q", api_key="k", search_depth="advanced")  # costs 2 > 1
    assert err.value.provider == "budget"
    client.assert_not_called()


@pytest.mark.anyio
async def test_basic_depth_costs_one_credit(monkeypatch):
    b = DailyBudget("tavily", limit=5, clock=Clock())
    monkeypatch.setattr(budget_mod, "get_tavily_budget", lambda: b)
    fake_resp = AsyncMock()
    fake_resp.status_code = 200
    fake_resp.json = lambda: {"results": []}
    with patch("app.services.search_api.httpx.AsyncClient") as client:
        client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake_resp)
        await search_web("q", api_key="k", search_depth="basic")
        payload = client.return_value.__aenter__.return_value.post.call_args.kwargs["json"]
    assert payload["search_depth"] == "basic"
    assert b.used == 1


@pytest.mark.anyio
async def test_exhausted_llm_budget_raises_before_calling_the_model(monkeypatch):
    from app.utils import adk_helpers
    exhausted = DailyBudget("llm", limit=1, clock=Clock())
    exhausted.try_charge()
    monkeypatch.setattr(budget_mod, "get_llm_budget", lambda: exhausted)
    with patch.object(adk_helpers, "get_adk_runner") as runner:
        with pytest.raises(LLMBudgetExceeded):
            await adk_helpers.run_adk_agent("n", "i", "p", "app")
    runner.return_value.run_async.assert_not_called()


@pytest.mark.anyio
async def test_crossref_uses_basic_depth():
    from app.agents.crossref_agent import run
    with patch("app.agents.crossref_agent.search_web", new=AsyncMock(return_value=[])) as sw:
        await run("NASA announces Mars landing")
    assert all(c.kwargs.get("search_depth") == "basic" for c in sw.call_args_list)
```

(`search_web` calls `httpx.AsyncClient().post(TAVILY_SEARCH_URL, json=payload, ...)`, which is
what `test_basic_depth_costs_one_credit` mocks. The budget check in `run_adk_agent` sits
before `create_session`, so `run_async` is never reached.)

- [ ] **Step 2:** Run `uv run pytest tests/test_budget.py -v`. Expected: collection ERROR, `ModuleNotFoundError: app.services.budget`.

- [ ] **Step 3: Settings.** In `config.py`, after `request_deadline_s`, add:

```python

    # H2/H3: daily spend ceilings (UTC day, per process). 0 disables a guard.
    # Tavily free tier = 1,000 credits/month (~33/day); advanced search = 2 credits, basic = 1.
    tavily_daily_credit_budget: int = 30
    # Groq free tier for gpt-oss-120b = 1,000 requests/day; keep headroom for retries.
    llm_daily_call_budget: int = 900
```

- [ ] **Step 4: Budget module.** Create `backend/app/services/budget.py`:

```python
import threading
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Callable, Optional

from app.config import settings


class LLMBudgetExceeded(RuntimeError):
    """The daily LLM call budget is spent. Agents degrade; they never guess."""


class DailyBudget:
    """
    A per-UTC-day counter. try_charge() is atomic and charges nothing when it refuses.
    In-process only: it resets on restart and is not shared across instances, which is
    acceptable for the single-instance free-tier deployment (TDD §11).
    """

    def __init__(self, name: str, limit: int, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.name = name
        self.limit = limit
        self._clock = clock
        self._day: Optional[date] = None
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = self._clock().date()
        if today != self._day:
            self._day, self._used = today, 0

    def try_charge(self, units: int = 1) -> bool:
        with self._lock:
            self._roll()
            if self.limit > 0 and self._used + units > self.limit:
                return False
            self._used += units
            return True

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used


@lru_cache()
def get_tavily_budget() -> DailyBudget:
    return DailyBudget("tavily_credits", settings.tavily_daily_credit_budget)


@lru_cache()
def get_llm_budget() -> DailyBudget:
    return DailyBudget("llm_calls", settings.llm_daily_call_budget)
```

- [ ] **Step 5: Charge Tavily.** In `backend/app/services/search_api.py`:
  - Add `from typing import Literal`.
  - Add `from app.services import budget`. Import the **module**, not the function, so tests can monkeypatch `budget_mod.get_tavily_budget`.
  - Add `search_depth: Literal["basic", "advanced"] = "advanced",` as the last parameter of `search_web`.
  - Set the payload's `"search_depth": search_depth,` in place of the hardcoded `"advanced"`.
  - Directly after the `if not api_key:` block, add:

```python
    # H2/H3: refuse before spending. Exhaustion is an outage, so the agent reports degraded.
    credits = 2 if search_depth == "advanced" else 1
    if not budget.get_tavily_budget().try_charge(credits):
        logger.error("Daily Tavily credit budget exhausted; refusing search.")
        raise EvidenceAPIError("Daily evidence-search budget exhausted", provider="budget")
```

- [ ] **Step 6: Charge the LLM.** In `backend/app/utils/adk_helpers.py`, add `from app.services import budget`. In `run_adk_agent`, as the first statement inside `for attempt in range(max_retries + 1):`, add:

```python
        if not budget.get_llm_budget().try_charge():
            raise budget.LLMBudgetExceeded("Daily LLM call budget exhausted")
```

- [ ] **Step 7: Cheaper cross-ref.** In `crossref_agent.py`, change the search call to
`search_web(query=q, api_key=api_key, max_results=5, search_depth="basic")`. Add this comment
above it: `# H2: basic depth (1 credit) is enough for corroboration discovery; the Fact-Check agent keeps advanced.`

- [ ] **Step 8:** Add this autouse fixture to `backend/tests/conftest.py`. It stops one test spending another test's process-wide budget:

```python
@pytest.fixture(autouse=True)
def _fresh_budgets():
    from app.services import budget
    budget.get_tavily_budget.cache_clear()
    budget.get_llm_budget.cache_clear()
    yield
```

Then run `uv run pytest tests/test_budget.py -v` and `uv run pytest -q`. Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add backend/app/services/budget.py backend/app/services/search_api.py backend/app/utils/adk_helpers.py backend/app/agents/crossref_agent.py backend/app/config.py backend/tests/
git commit -m "feat(cost): daily Tavily/LLM budgets that degrade honestly; basic-depth cross-ref search (H2, H3)"
```

---

### Task 3: Proxy-aware rate-limit key (H3)

**Files:**
- Create: `backend/app/api/client_ip.py`
- Modify: `backend/app/api/routes.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_client_ip.py`

**Interfaces:**
- Produces:
  - `client_ip(request: Request) -> str`
  - `settings.trusted_proxy_hops: int = 0` (0 in dev; 1 behind Render, Railway or Fly)

- [ ] **Step 1: Write the failing tests.** Create `backend/tests/test_client_ip.py`:

```python
import pytest
from starlette.requests import Request

from app.api.client_ip import client_ip
from app.config import settings


def _req(xff=None, peer="10.0.0.1"):
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_no_trusted_proxy_ignores_forwarded_header(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 0)
    assert client_ip(_req("1.1.1.1")) == "10.0.0.1"


def test_one_hop_uses_rightmost_entry(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    assert client_ip(_req("203.0.113.9")) == "203.0.113.9"


def test_spoofed_left_entries_are_ignored(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 1)
    # client sent "X-Forwarded-For: 6.6.6.6"; the proxy appended the real peer
    assert client_ip(_req("6.6.6.6, 203.0.113.9")) == "203.0.113.9"


def test_short_header_falls_back_to_peer(monkeypatch):
    monkeypatch.setattr(settings, "trusted_proxy_hops", 2)
    assert client_ip(_req("203.0.113.9")) == "10.0.0.1"
```

- [ ] **Step 2:** Run `uv run pytest tests/test_client_ip.py -v`. Expected: collection ERROR.

- [ ] **Step 3: Implement.** Add this to `config.py` after the budgets:

```python

    # H3: number of reverse proxies in front of the API that append to X-Forwarded-For.
    # 0 = trust nothing (local dev); 1 = one platform proxy (Render/Railway/Fly).
    trusted_proxy_hops: int = 0
```

Create `backend/app/api/client_ip.py`:

```python
from starlette.requests import Request
from slowapi.util import get_remote_address

from app.config import settings


def client_ip(request: Request) -> str:
    """
    Rate-limit key (H3). Behind N trusted proxies the real client is the N-th entry from the
    RIGHT of X-Forwarded-For. Entries further left are client-supplied and spoofable.
    With no trusted proxy, use the socket peer.
    """
    hops = settings.trusted_proxy_hops
    if hops > 0:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return get_remote_address(request)
```

In `routes.py`, replace `from slowapi.util import get_remote_address` with
`from app.api.client_ip import client_ip`. Replace `limiter = Limiter(key_func=get_remote_address)`
with `limiter = Limiter(key_func=client_ip)`.

- [ ] **Step 4:** Run `uv run pytest tests/test_client_ip.py tests/test_api_routes.py -v`. Expected: all pass.
- [ ] **Step 5: Commit**

```bash
git add backend/app/api/client_ip.py backend/app/api/routes.py backend/app/config.py backend/tests/test_client_ip.py
git commit -m "fix(api): proxy-aware rate-limit key with configurable trusted hops (H3)"
```

---

### Task 4: SSRF guard for URL inputs (H3)

**Files:**
- Create: `backend/app/utils/url_safety.py`
- Test: `backend/tests/test_url_safety.py`

**Interfaces:**
- Produces:
  - `validate_public_url(url: str, resolver=socket.getaddrinfo) -> str`
  - `UnsafeURLError(ValueError)`

  The P2-02 scraper must call it on the initial URL **and on every redirect hop**. The TDD
  requirement is added in Task 6.

- [ ] **Step 1: Write the failing tests.** Create `backend/tests/test_url_safety.py`:

```python
import socket

import pytest

from app.utils.url_safety import UnsafeURLError, validate_public_url


def resolver_for(*ips):
    def _resolve(host, port, proto=0, **kwargs):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port))
                for ip in ips]
    return _resolve


def test_public_https_url_is_allowed():
    url = "https://example.com/article"
    assert validate_public_url(url, resolver=resolver_for("93.184.216.34")) == url


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.5", "192.168.1.10", "172.16.0.1", "169.254.169.254",
    "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "0.0.0.0",
])
def test_non_public_addresses_are_rejected(ip):
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://evil.test/", resolver=resolver_for(ip))


def test_any_private_address_among_many_rejects():
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://mixed.test/", resolver=resolver_for("93.184.216.34", "10.0.0.1"))


@pytest.mark.parametrize("url", [
    "ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)",
    "http://user:pass@example.com/", "http://example.com:8080/", "https:///nohost",
    "http://example.com:99999/", "https://example.com/" + "a" * 2100,
])
def test_bad_shapes_are_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_public_url(url, resolver=resolver_for("93.184.216.34"))


def test_unresolvable_host_is_rejected():
    def boom(*a, **k):
        raise socket.gaierror("nope")
    with pytest.raises(UnsafeURLError):
        validate_public_url("https://no-such-host.test/", resolver=boom)
```

- [ ] **Step 2:** Run `uv run pytest tests/test_url_safety.py -v`. Expected: collection ERROR.

- [ ] **Step 3: Implement.** Create `backend/app/utils/url_safety.py`:

```python
import ipaddress
import socket
from typing import Callable
from urllib.parse import urlsplit

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}


class UnsafeURLError(ValueError):
    """The URL must not be fetched server-side (SSRF guard, H3)."""


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])  # drop IPv6 zone id
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_global


def validate_public_url(url: str, resolver: Callable = socket.getaddrinfo) -> str:
    """
    Allow only http(s) URLs on default ports, without credentials, whose host resolves
    exclusively to public addresses. Callers that fetch must re-validate every redirect hop
    and connect to the validated address (DNS-rebinding defence). See TDD §11.
    """
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURLError("URL too long")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme '{parts.scheme}' not allowed")
    if parts.username or parts.password:
        raise UnsafeURLError("credentials in URL not allowed")
    try:
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError("invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError(f"port {port} not allowed")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("missing host")
    try:
        infos = resolver(host, port or (443 if parts.scheme.lower() == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as exc:
        raise UnsafeURLError("host does not resolve") from exc
    ips = {info[4][0] for info in infos}
    if not ips or not all(_is_public(ip) for ip in ips):
        raise UnsafeURLError("host resolves to a non-public address")
    return url
```

- [ ] **Step 4:** Run `uv run pytest tests/test_url_safety.py -v`. Expected: all pass (21 cases).
- [ ] **Step 5: Commit**

```bash
git add backend/app/utils/url_safety.py backend/tests/test_url_safety.py
git commit -m "feat(security): SSRF guard for server-side URL fetching (H3)"
```

---

### Task 5: Retention purge (H4)

**Files:**
- Create: `backend/app/db/migrations/05_retention.sql`
- Create: `backend/app/db/purge.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_purge.py`

**Interfaces:**
- Produces:
  - RPC `purge_old_claims(retention_days int) -> int`
  - `async def purge_old_claims(retention_days: Optional[int] = None) -> int`
  - `settings.retention_days: int = 90`

- [ ] **Step 1: Write the failing test.** Create `backend/tests/test_purge.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from app.db.purge import purge_old_claims


@pytest.mark.anyio
async def test_purge_calls_rpc_with_configured_days():
    supabase = MagicMock()
    supabase.rpc.return_value.execute.return_value = MagicMock(data=7)
    with patch("app.db.purge.get_supabase", return_value=supabase):
        removed = await purge_old_claims()
    supabase.rpc.assert_called_once_with("purge_old_claims", {"retention_days": 90})
    assert removed == 7


@pytest.mark.anyio
async def test_purge_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        await purge_old_claims(0)
```

- [ ] **Step 2:** Run `uv run pytest tests/test_purge.py -v`. Expected: collection ERROR.

- [ ] **Step 3: Implement.** In `config.py`, add:

```python

    # H4: claims/results older than this are deleted by purge_old_claims (migration 05).
    retention_days: int = 90
```

Create `backend/app/db/migrations/05_retention.sql`:

```sql
-- VeriFact AI - Migration 05: data retention (H4)
-- No accounts exist; inputs (including OCR text from screenshots) are kept at most
-- retention_days, then deleted with their results. GET /api/results/{id} links expire with them.

CREATE OR REPLACE FUNCTION purge_old_claims(retention_days int DEFAULT 90)
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    removed int;
BEGIN
    IF retention_days IS NULL OR retention_days < 1 THEN
        RAISE EXCEPTION 'retention_days must be >= 1';
    END IF;
    DELETE FROM results
    WHERE claim_id IN (SELECT id FROM claims WHERE created_at < NOW() - make_interval(days => retention_days));
    DELETE FROM claims WHERE created_at < NOW() - make_interval(days => retention_days);
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$;

-- Optional: daily schedule with pg_cron (Supabase: Database -> Extensions -> enable pg_cron).
-- SELECT cron.schedule('verifact-retention', '15 3 * * *', $cron$SELECT purge_old_claims(90)$cron$);
```

Create `backend/app/db/purge.py`:

```python
"""
Retention purge (H4). Run manually or from a scheduler:
    uv run python -m app.db.purge
Prefer the pg_cron schedule in migration 05 when available.
"""
import asyncio
import logging
from typing import Optional

from app.config import settings
from app.db.connection import get_supabase

logger = logging.getLogger(__name__)


async def purge_old_claims(retention_days: Optional[int] = None) -> int:
    days = settings.retention_days if retention_days is None else retention_days
    if days < 1:
        raise ValueError("retention_days must be >= 1")
    supabase = get_supabase()
    response = await asyncio.to_thread(
        supabase.rpc("purge_old_claims", {"retention_days": days}).execute
    )
    removed = int(response.data or 0)
    logger.info(f"Retention purge removed {removed} claims older than {days} days.")
    return removed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(purge_old_claims()))
```

- [ ] **Step 4:** Run `uv run pytest tests/test_purge.py -v`. Expected: 2 passed.
- [ ] **Step 5 (user action):** Ask the user to apply `05_retention.sql` in Supabase. Optionally they can enable pg_cron and run the commented `cron.schedule` line.
- [ ] **Step 6: Commit**

```bash
git add backend/app/db/migrations/05_retention.sql backend/app/db/purge.py backend/app/config.py backend/tests/test_purge.py
git commit -m "feat(privacy): 90-day retention purge for claims and results (H4)"
```

---

### Task 6: Spec updates

**Files:** `documents/VeriFact_AI_TDD.md` §3.1 and §11; `documents/VeriFact_AI_PRD_v2.md` §8 and §9.

- [ ] **Step 1: TDD §11.** Replace the section body with:

```markdown
- **Input sanitization:** strip `<script>`/HTML, tracking parameters (utm_*, fbclid, …) and JSON-answer syntax; limit input to 2000 characters. Only the sanitized text is persisted.
- **Rate limiting:** 10 requests/minute per client IP (slowapi). The key is `app/api/client_ip.py`: behind `TRUSTED_PROXY_HOPS` proxies it takes the N-th `X-Forwarded-For` entry from the right; with 0 hops it uses the socket peer. Set `TRUSTED_PROXY_HOPS=1` on Render/Railway/Fly.
- **Spend budgets:** per-UTC-day ceilings on Tavily credits (`TAVILY_DAILY_CREDIT_BUDGET`, default 30) and LLM calls (`LLM_DAILY_CALL_BUDGET`, default 900). When a budget is exhausted, the call raises an outage error, the agent degrades and the response carries "Partial analysis". It never returns a confident score. The counters are in-process (single instance).
- **Prompt injection:** every LLM prompt fences untrusted text in XML tags, states an instruction hierarchy, strips JSON-answer syntax (`neutralize_for_prompt`), length-caps inputs, and enum-validates outputs. Adversarial claims in the eval set must never yield `verdict_found=true` or `high` confidence.
- **SSRF (URL inputs, P2-02):** the scraper must call `validate_public_url` (`app/utils/url_safety.py`) on the initial URL and on every redirect hop (at most 3, `follow_redirects=False`). It must connect to the validated address, cap the response at 2 MB and 5 s, and accept only `text/html`.
- **API keys:** never exposed to the frontend. All external calls happen server-side.
- **Image uploads:** validate the MIME type, reject files over 5 MB, process in memory (no disk writes), and never store image bytes.
- **Retention:** claims and results older than `RETENTION_DAYS` (default 90) are deleted by `purge_old_claims` (migration 05; pg_cron recommended). Shared result links expire with them.
- **CORS:** restricted to the frontend domain in production.
```

- [ ] **Step 2: TDD §3.1.** Add under "Error Responses": "Every response carries `X-Process-Time-Ms`. The pipeline runs under a 9.5 s request deadline (`REQUEST_DEADLINE_S`). Agents get `min(8 s, remaining − 1 s)`. A slow DB write returns the result with `id: null` while the write completes in the background."

- [ ] **Step 3: PRD §8.** Replace the "LLM API rate limited" row with:
`| LLM rate limited / daily LLM or search budget exhausted | Retry with backoff inside the agent's time budget; if it still fails, degrade that agent and show "Partial Analysis" |`.
Replace the "Fact-Check API timeout (>5s)" row with:
`| Fact-Check API outage or timeout | Layer 2 (Tavily) is tried; if the agent still cannot reach evidence it degrades (S_fact = 0.5, "Partial Analysis") |`.

- [ ] **Step 4: PRD §9.** Replace the Groq and Tavily rows with verified limits:
`| Groq (gpt-oss-120b, free) | 30 req/min · 1,000 req/day · 8K tokens/min | $0 |` and
`| Tavily (web search) | 1,000 credits/month; advanced = 2 credits, basic = 1 | $0 (≈ 100–150 verifications/month) |`.
Add a line: "Daily budgets (TDD §11) keep the service inside these limits. Eval runs use recorded evidence (`--replay`) so they do not consume quota."

- [ ] **Step 5: Commit**

```bash
git add documents/VeriFact_AI_TDD.md documents/VeriFact_AI_PRD_v2.md
git commit -m "docs: specify deadline, budgets, proxy-aware limits, SSRF and retention (H1-H4)"
```

---

### Task 7: Review gate (mandatory)

- [ ] Run `uv run pytest -q` (backend) and `uv run pytest eval/tests/ -q` (repo root). Record the counts.
- [ ] Run the review gate from `README.md` (`/review` + `security-review` focus, plus `eval-guardian` because agents changed). Ask specifically:
  - Can budget exhaustion ever produce a non-degraded score?
  - Is the persistence task ever cancelled mid-write?
  - Can `X-Forwarded-For` be spoofed with `trusted_proxy_hops = 1`?
  - Does `validate_public_url` cover IPv4-mapped IPv6 and zone ids?
- [ ] Fix blocking findings, writing a test first. Report:
  - migration 05 needs applying
  - `TRUSTED_PROXY_HOPS` must be set per host at deploy time
  - the budgets are per-process
