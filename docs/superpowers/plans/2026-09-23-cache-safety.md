# Semantic Cache Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Antigravity) or superpowers:subagent-driven-development (Claude Code) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `docs/superpowers/plans/README.md` first (run order, review-gate mapping).

**Goal:** The semantic cache must never serve a result for a *different* claim. That
includes a negation, a changed number, or a narrower or broader claim. It must also never
serve stale results or results from a partial run.

**Architecture:** pgvector stays the cheap first filter (cosine ≥ 0.95), but it now returns
the top 3 candidates instead of 1. Each candidate then passes two checks:
1. A pure row policy (`app/services/cache_policy.py`) checks confidence, `evidence_status`,
   that every agent completed, and a TTL keyed by evidence status.
2. An equivalence check (`app/services/claim_equivalence.py`) requires identical numbers and
   **bidirectional** NLI entailment. It reuses Plan 1's `classify_pairs`.

The first candidate that passes both is served. Any error counts as a miss.

**Tech Stack:** Python 3.12, `app/services/nli.py` (from Plan 1), supabase-py RPC, Postgres/pgvector (migration applied by hand), pytest + anyio.

**Spec:** `docs/superpowers/specs/2026-09-23-prd-tdd-review.md` (finding C2, decision D1).
Also read `documents/VeriFact_AI_TDD.md` §6.

## Global Constraints

- **Depends on Plan 1** (`2026-09-23-stance-aware-consensus.md`) being merged. It needs `app/services/nli.py` and the `evidence_status` column. Branch from `main` after Plan 1 lands.
- Python **3.12+**, `uv` only. Run backend commands from `backend/` and eval from the repo root.
- Heavy models load once at startup. This plan loads **no new model**.
- The cache **fails closed**. Any doubt, error or missing field counts as a miss, never a hit.
- Keep the PRD latency target on a cache hit: **< 500 ms**. The equivalence check costs at most 2 NLI pairs per candidate and runs only after the cheap policy checks pass.
- Every "Run:" step must actually be executed and its output recorded. Antigravity runs them directly. In Claude Code, use a dispatched subagent, because the main-thread shell cannot run `uv`.
- Antigravity may edit in parallel. Re-check `git status` and re-read files before editing.
- Every commit message ends with the executor's attribution line. The commit blocks below show Claude Code's (`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`); Antigravity replaces it with its own or omits it.

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app/services/cache_policy.py` | Create | Pure row rules: `unservable_reason()`, `is_complete_run()`, `is_fresh()`, `ttl_for()` |
| `backend/app/services/claim_equivalence.py` | Create | `are_equivalent(claim, cached_claim)` via a number guard and bidirectional NLI |
| `backend/app/services/cache.py` | Modify | `lookup_cache` takes `claim_text`, fetches the top-k, applies policy then equivalence |
| `backend/app/config.py` | Modify | TTL settings |
| `backend/app/agents/orchestrator.py` | Modify | Pass `claim_text` to `lookup_cache` |
| `backend/app/db/migrations/04_cache_topk_and_ttl.sql` | Create | `match_claims(..., match_count)` returning `created_at` |
| `backend/tests/test_cache_policy.py` | Create | Row policy unit tests |
| `backend/tests/test_claim_equivalence.py` | Create | Equivalence unit tests plus live tests |
| `backend/tests/test_cache.py`, `backend/tests/test_orchestrator.py` | Modify | New `lookup_cache` contract |
| `eval/tests/test_cache_safety.py` | Modify | New contract, plus negation and number guardrails |
| `documents/VeriFact_AI_TDD.md` §6, `documents/VeriFact_AI_PRD_v2.md` §10, `CLAUDE.md` §2 | Modify | Spec updates |

---

### Task 0: Pre-flight

- [ ] **Step 1:** Run `git status --short`. Expected: empty. If it isn't, stop and ask the user.
- [ ] **Step 2:** Confirm Plan 1 has landed. Run `git log --oneline -5` and look for the "stance-aware" commits. `backend/app/services/nli.py` must exist.
- [ ] **Step 3:** Create the branch with `git checkout -b fix/cache-safety`.
- [ ] **Step 4:** Run (from `backend/`) `uv run pytest -q`. Expected: all pass. Record the count.

---

### Task 1: Row policy (pure)

**Files:**
- Create: `backend/app/services/cache_policy.py`
- Modify: `backend/app/config.py` (after the `nli_max_length` setting added by Plan 1)
- Test: `backend/tests/test_cache_policy.py`

**Interfaces:**
- Produces:
  - `unservable_reason(row: dict, now: datetime) -> Optional[str]` (`None` means the row may be served)
  - `is_complete_run(agent_outputs: Any) -> bool`
  - `is_fresh(evidence_status: str, created_at: str | datetime | None, now: datetime) -> bool`
  - `ttl_for(evidence_status: str) -> timedelta`
  - `settings.cache_ttl_fact_checked_hours: int = 720`
  - `settings.cache_ttl_default_hours: int = 24`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_cache_policy.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from app.services.cache_policy import is_complete_run, is_fresh, unservable_reason

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
COMPLETE = {
    "factcheck": {"degraded": False},
    "crossref": {"degraded": False},
    "linguistic": {"degraded": False},
}


def _row(**overrides):
    row = {
        "confidence_level": "medium",
        "evidence_status": "corroborated",
        "agent_outputs": COMPLETE,
        "created_at": (NOW - timedelta(hours=1)).isoformat(),
    }
    row.update(overrides)
    return row


def test_servable_row_has_no_reason():
    assert unservable_reason(_row(), NOW) is None


@pytest.mark.parametrize("level", ["low", "insufficient", "LOW"])
def test_low_confidence_rows_are_not_served(level):
    assert "confidence" in unservable_reason(_row(confidence_level=level), NOW)


def test_legacy_rows_without_evidence_status_are_not_served():
    assert "evidence_status" in unservable_reason(_row(evidence_status=None), NOW)


def test_partial_runs_are_not_served():
    partial = {**COMPLETE, "crossref": {"degraded": True}}
    assert "partial" in unservable_reason(_row(agent_outputs=partial), NOW)


@pytest.mark.parametrize("agent_outputs", [None, {}, {"factcheck": {"degraded": False}}, "not-a-dict"])
def test_missing_agent_outputs_fail_closed(agent_outputs):
    assert is_complete_run(agent_outputs) is False


def test_missing_degraded_flag_fails_closed():
    outputs = {**COMPLETE, "linguistic": {}}
    assert is_complete_run(outputs) is False


@pytest.mark.parametrize("status,age,fresh", [
    ("fact_checked", timedelta(days=29), True),
    ("fact_checked", timedelta(days=31), False),
    ("corroborated", timedelta(hours=23), True),
    ("corroborated", timedelta(hours=25), False),
    ("disputed", timedelta(hours=25), False),
    ("unverified", timedelta(hours=25), False),
])
def test_ttl_depends_on_evidence_status(status, age, fresh):
    assert is_fresh(status, (NOW - age).isoformat(), NOW) is fresh


def test_supabase_z_suffix_and_naive_timestamps_parse():
    assert is_fresh("corroborated", "2026-09-23T11:00:00Z", NOW) is True
    assert is_fresh("corroborated", datetime(2026, 9, 23, 11, 0), NOW) is True  # naive -> UTC


@pytest.mark.parametrize("created_at", [None, "", "yesterday", 12345])
def test_unparseable_created_at_fails_closed(created_at):
    assert is_fresh("fact_checked", created_at, NOW) is False


def test_stale_rows_are_not_served():
    old = (NOW - timedelta(days=2)).isoformat()
    assert "stale" in unservable_reason(_row(created_at=old), NOW)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cache_policy.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'app.services.cache_policy'`.

- [ ] **Step 3: Add the settings**

In `backend/app/config.py`, after `nli_max_length: int = 256`, add:

```python

    # Semantic cache TTLs (C2). A matched fact-check changes slowly; everything else is
    # news-driven and may be overtaken by new reporting within a day.
    cache_ttl_fact_checked_hours: int = 720
    cache_ttl_default_hours: int = 24
```

- [ ] **Step 4: Implement**

Create `backend/app/services/cache_policy.py`:

```python
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from app.config import settings

# Every worker agent whose outputs orchestrator.verify_claim persists in results.agent_outputs.
_AGENTS = ("factcheck", "crossref", "linguistic")


def ttl_for(evidence_status: str) -> timedelta:
    """Fact-checked results live longer; news-driven statuses expire within a day."""
    if evidence_status == "fact_checked":
        return timedelta(hours=settings.cache_ttl_fact_checked_hours)
    return timedelta(hours=settings.cache_ttl_default_hours)


def is_complete_run(agent_outputs: Any) -> bool:
    """True only when every agent's persisted output says degraded=False. Missing data fails closed."""
    if not isinstance(agent_outputs, dict):
        return False
    return all(
        isinstance(agent_outputs.get(agent), dict) and agent_outputs[agent].get("degraded") is False
        for agent in _AGENTS
    )


def is_fresh(evidence_status: str, created_at: Union[str, datetime, None], now: datetime) -> bool:
    """True when the cached result is younger than its TTL. Unparseable timestamps fail closed."""
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return False
    if not isinstance(created_at, datetime):
        return False
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return now - created_at <= ttl_for(evidence_status)


def unservable_reason(row: Dict[str, Any], now: datetime) -> Optional[str]:
    """
    Cheap, pure checks run before the (costlier) equivalence check.
    Returns None when the row may be served, otherwise a short reason for the log.
    """
    conf = str(row.get("confidence_level", "")).lower()
    if conf in {"low", "insufficient"}:
        return f"confidence is '{conf}'"
    status = row.get("evidence_status")
    if not status:
        return "predates stance-aware scoring (no evidence_status)"
    if not is_complete_run(row.get("agent_outputs")):
        return "partial analysis (an agent degraded)"
    if not is_fresh(status, row.get("created_at"), now):
        return "stale (past TTL)"
    return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cache_policy.py -v`
Expected: all pass (23 cases, counting the parametrized ones).

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/cache_policy.py backend/app/config.py backend/tests/test_cache_policy.py
git commit -m "feat(cache): add pure cache row policy with evidence-keyed TTL (C2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Claim equivalence (number guard + bidirectional NLI)

**Files:**
- Create: `backend/app/services/claim_equivalence.py`
- Test: `backend/tests/test_claim_equivalence.py`

**Interfaces:**
- Consumes: `classify_pairs`, `NLIScores` from `app.services.nli` (Plan 1).
- Produces:
  - `are_equivalent(claim: str, cached_claim: str) -> bool` (synchronous and CPU-bound; call it via `asyncio.to_thread`)
  - `numbers_match(a: str, b: str) -> bool`
  - constant `EQUIVALENCE_MIN_ENTAILMENT = 0.8`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_claim_equivalence.py`:

```python
from unittest.mock import patch

import pytest

from app.services.claim_equivalence import are_equivalent, numbers_match
from app.services.nli import NLIScores

HIGH = NLIScores(entailment=0.95, contradiction=0.02, neutral=0.03)
LOW = NLIScores(entailment=0.10, contradiction=0.10, neutral=0.80)
CONTRA = NLIScores(entailment=0.02, contradiction=0.95, neutral=0.03)


def test_identical_claims_skip_the_model():
    with patch("app.services.claim_equivalence.classify_pairs") as mock_nli:
        assert are_equivalent("NASA found water on Mars", "  nasa found WATER on mars ") is True
        mock_nli.assert_not_called()


@pytest.mark.parametrize("a,b,expected", [
    ("Inflation hit 3% in May", "Inflation hit 8% in May", False),
    ("Unemployment was 3.5 percent", "Unemployment was 3.5 percent last month", True),
    ("Unemployment was 3.5%", "Unemployment was 3.5 percent", True),
    ("The bill cost $1,000,000", "The bill cost $1000000", True),
    ("Built in 1999", "Built in 2001", False),
    ("No numbers here", "Still no numbers", True),
])
def test_numbers_match(a, b, expected):
    assert numbers_match(a, b) is expected


def test_number_mismatch_is_not_equivalent_and_skips_the_model():
    with patch("app.services.claim_equivalence.classify_pairs") as mock_nli:
        assert are_equivalent("Inflation hit 3% in May", "Inflation hit 8% in May") is False
        mock_nli.assert_not_called()


def test_bidirectional_entailment_is_equivalent():
    with patch("app.services.claim_equivalence.classify_pairs", return_value=[HIGH, HIGH]) as mock_nli:
        assert are_equivalent("NASA confirms water on Mars", "NASA has confirmed water exists on Mars") is True
    pairs = mock_nli.call_args[0][0]
    assert pairs == [
        ("NASA has confirmed water exists on Mars", "NASA confirms water on Mars"),
        ("NASA confirms water on Mars", "NASA has confirmed water exists on Mars"),
    ]


def test_one_way_entailment_is_not_equivalent():
    """'visited Paris' entails 'visited France' but not the reverse: different claims."""
    with patch("app.services.claim_equivalence.classify_pairs", return_value=[HIGH, LOW]):
        assert are_equivalent("The president visited France", "The president visited Paris") is False


def test_contradiction_is_not_equivalent():
    with patch("app.services.claim_equivalence.classify_pairs", return_value=[CONTRA, CONTRA]):
        assert are_equivalent("Vaccines cause autism", "Vaccines do not cause autism") is False


@pytest.mark.live
@pytest.mark.parametrize("a,b,expected", [
    ("Vaccines cause autism", "Vaccines do not cause autism", False),
    ("The Eiffel Tower is in Paris", "The Eiffel Tower is not in Paris", False),
    ("NASA confirms water found on Mars", "NASA has confirmed that water was found on Mars", True),
])
def test_real_model_equivalence(a, b, expected):
    assert are_equivalent(a, b) is expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_claim_equivalence.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'app.services.claim_equivalence'`.

- [ ] **Step 3: Implement**

Create `backend/app/services/claim_equivalence.py`:

```python
import re
from typing import Set

from app.services.nli import classify_pairs

# Both directions must entail at least this strongly for two claims to count as the same (C2).
# Track 3.1: calibrate against eval paraphrase/negation pairs (Plan 3).
EQUIVALENCE_MIN_ENTAILMENT = 0.8

# Integers, decimals, thousands separators and percentages: "3", "3.5", "1,000,000", "8%".
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _numbers(text: str) -> Set[str]:
    # "1,000" == "1000" and "3%" == "3" (so "3%" matches "3 percent"); "3" != "8" always.
    return {match.replace(",", "").rstrip("%") for match in _NUMBER_RE.findall(text)}


def numbers_match(a: str, b: str) -> bool:
    """NLI models are weak on numbers and dates, so any numeric difference makes claims distinct."""
    return _numbers(a) == _numbers(b)


def are_equivalent(claim: str, cached_claim: str) -> bool:
    """
    True only when the two claims assert the same thing: identical after normalization, or
    same numbers AND each entails the other. One-way entailment is a narrower or broader claim
    ("visited Paris" vs "visited France"), and contradiction is a negation. Neither is the same.
    Synchronous and CPU-bound: call via asyncio.to_thread.
    """
    if _normalize(claim) == _normalize(cached_claim):
        return True
    if not numbers_match(claim, cached_claim):
        return False
    forward, backward = classify_pairs([(cached_claim, claim), (claim, cached_claim)])
    return (
        forward.entailment >= EQUIVALENCE_MIN_ENTAILMENT
        and backward.entailment >= EQUIVALENCE_MIN_ENTAILMENT
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_claim_equivalence.py -v`
Expected: all offline tests pass; the live tests are deselected.

Run: `uv run pytest tests/test_claim_equivalence.py -m live -v`
Expected: 3 passed. If the paraphrase case fails because entailment is below 0.8, do **not**
lower the threshold here. Record the observed scores in the task report. Plan 3 calibrates
`EQUIVALENCE_MIN_ENTAILMENT` on a labeled pair set, and a missed cache hit is safe. **If a
negation case returns True, stop and report.** That would be a safety failure.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/claim_equivalence.py backend/tests/test_claim_equivalence.py
git commit -m "feat(cache): add claim equivalence check (number guard + bidirectional NLI) (C2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Top-k lookup that applies policy, then equivalence

**Files:**
- Create: `backend/app/db/migrations/04_cache_topk_and_ttl.sql`
- Modify: `backend/app/services/cache.py`
- Modify: `backend/app/agents/orchestrator.py` (the `lookup_cache` call)
- Test: `backend/tests/test_cache.py`, `backend/tests/test_orchestrator.py`, `eval/tests/test_cache_safety.py`

**Interfaces:**
- Consumes: `unservable_reason` (Task 1), `are_equivalent` (Task 2).
- Produces:
  - `lookup_cache(embedding: List[float], claim_text: str, similarity_threshold: float = 0.95, match_count: int = 3) -> Optional[Dict[str, Any]]`
  - RPC `match_claims(query_embedding, similarity_threshold, match_count)` returning `created_at` as well

- [ ] **Step 1: Write the failing tests**

1. In `backend/tests/test_cache.py`, add these imports at the top:

```python
from datetime import datetime, timedelta, timezone
```

Add this helper after the imports:

```python
def _servable_row(**overrides):
    row = {
        "claim_id": "test-uuid",
        "extracted_claim": "NASA water Mars",
        "similarity": 0.98,
        "c_total": 0.85,
        "confidence_level": "high",
        "evidence_status": "fact_checked",
        "agent_outputs": {
            "factcheck": {"degraded": False},
            "crossref": {"degraded": False},
            "linguistic": {"degraded": False},
        },
        "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }
    row.update(overrides)
    return row
```

2. Replace `test_lookup_cache_offline_miss` and `test_lookup_cache_offline_hit`, and the
`test_lookup_cache_legacy_row_without_evidence_status_is_miss` test added by Plan 1, with:

```python
def _supabase_returning(rows):
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return mock_supabase


@pytest.mark.anyio
async def test_lookup_cache_offline_miss():
    mock_supabase = _supabase_returning([])
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache([0.1] * 384, claim_text="NASA water Mars", similarity_threshold=0.95)
    assert result is None
    mock_supabase.rpc.assert_called_once_with(
        "match_claims",
        {"query_embedding": [0.1] * 384, "similarity_threshold": 0.95, "match_count": 3},
    )


@pytest.mark.anyio
async def test_lookup_cache_offline_hit():
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning([_servable_row()])), \
         patch("app.services.cache.are_equivalent", return_value=True):
        result = await lookup_cache([0.1] * 384, claim_text="NASA water Mars")
    assert result is not None
    assert result["claim_id"] == "test-uuid"
    assert result["warnings"] == []


@pytest.mark.anyio
async def test_lookup_cache_legacy_row_without_evidence_status_is_miss():
    """Rows scored before stance-aware consensus must never be served (C1)."""
    rows = [_servable_row(evidence_status=None)]
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning(rows)), \
         patch("app.services.cache.are_equivalent", return_value=True):
        assert await lookup_cache([0.1] * 384, claim_text="NASA water Mars") is None


@pytest.mark.anyio
async def test_non_equivalent_top_match_is_skipped_for_next_candidate():
    """C2: the closest embedding may be the negation; a lower-ranked equivalent row is served."""
    rows = [
        _servable_row(claim_id="negated", extracted_claim="NASA did not find water on Mars", similarity=0.99),
        _servable_row(claim_id="same", extracted_claim="NASA found water on Mars", similarity=0.96),
    ]
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning(rows)), \
         patch("app.services.cache.are_equivalent", side_effect=[False, True]) as mock_eq:
        result = await lookup_cache([0.1] * 384, claim_text="NASA has found water on Mars")
    assert result["claim_id"] == "same"
    assert mock_eq.call_args_list[0][0] == ("NASA has found water on Mars", "NASA did not find water on Mars")


@pytest.mark.anyio
async def test_policy_runs_before_equivalence():
    """Unservable rows (e.g. partial runs) never pay for an NLI call."""
    partial = _servable_row(agent_outputs={"factcheck": {"degraded": False},
                                           "crossref": {"degraded": True},
                                           "linguistic": {"degraded": False}})
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning([partial])), \
         patch("app.services.cache.are_equivalent") as mock_eq:
        assert await lookup_cache([0.1] * 384, claim_text="NASA water Mars") is None
    mock_eq.assert_not_called()


@pytest.mark.anyio
async def test_equivalence_error_fails_closed():
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning([_servable_row()])), \
         patch("app.services.cache.are_equivalent", side_effect=RuntimeError("NLI down")):
        assert await lookup_cache([0.1] * 384, claim_text="NASA water Mars") is None
```

3. In `backend/tests/test_orchestrator.py`, add this assertion at the end of
`test_verify_claim_cache_hit`:

```python
        assert mock_lookup.call_args.kwargs["claim_text"] == "The Earth orbits the Sun"
```

4. In `eval/tests/test_cache_safety.py`, replace `test_near_miss_similarity_misses_cache` with:

```python
@pytest.mark.anyio
async def test_near_miss_similarity_misses_cache():
    """Verify that a near-miss claim (e.g. 0.92 similarity) misses the 0.95 cache threshold."""
    mock_supabase = MagicMock()
    # match_claims RPC returns empty list when below similarity_threshold
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=[])

    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache(
            embedding=[0.2] * 384,
            claim_text="Sample novel claim",
            similarity_threshold=0.95
        )
        assert result is None
        mock_supabase.rpc.assert_called_once_with(
            "match_claims",
            {"query_embedding": [0.2] * 384, "similarity_threshold": 0.95, "match_count": 3}
        )


@pytest.mark.anyio
async def test_numeric_variant_is_never_served():
    """C2 guardrail: a cached '3%' claim must not answer an '8%' query even at 0.99 cosine."""
    from datetime import datetime, timezone
    row = {
        "claim_id": "c1", "extracted_claim": "Inflation hit 3% in May", "similarity": 0.99,
        "confidence_level": "high", "evidence_status": "fact_checked",
        "agent_outputs": {a: {"degraded": False} for a in ("factcheck", "crossref", "linguistic")},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=[row])
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        assert await lookup_cache([0.2] * 384, claim_text="Inflation hit 8% in May") is None


@pytest.mark.anyio
@pytest.mark.live
async def test_negated_claim_is_never_served():
    """C2 guardrail with the real NLI model: a negation must miss even at 0.99 cosine."""
    from datetime import datetime, timezone
    row = {
        "claim_id": "c1", "extracted_claim": "Vaccines cause autism", "similarity": 0.99,
        "confidence_level": "high", "evidence_status": "fact_checked",
        "agent_outputs": {a: {"degraded": False} for a in ("factcheck", "crossref", "linguistic")},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=[row])
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        assert await lookup_cache([0.2] * 384, claim_text="Vaccines do not cause autism") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `backend/`): `uv run pytest tests/test_cache.py tests/test_orchestrator.py -v`
Expected: FAIL. You should see `TypeError: lookup_cache() got an unexpected keyword argument 'claim_text'`, and `AttributeError` when patching `app.services.cache.are_equivalent`.

- [ ] **Step 3: Write the migration**

Create `backend/app/db/migrations/04_cache_topk_and_ttl.sql`:

```sql
-- VeriFact AI - Migration 04: top-k cache candidates + created_at for TTL (C2)
-- Spec: docs/superpowers/specs/2026-09-23-prd-tdd-review.md
-- Requires migration 03 (v_refute, evidence_status).

-- The signature gains match_count and the result gains created_at: drop both old shapes.
DROP FUNCTION IF EXISTS match_claims(vector, double precision);
DROP FUNCTION IF EXISTS match_claims(vector, double precision, integer);

CREATE FUNCTION match_claims(
    query_embedding vector(384),
    similarity_threshold float,
    match_count int DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    claim_id UUID,
    c_total FLOAT,
    s_fact FLOAT,
    b_ling FLOAT,
    v_consensus FLOAT,
    v_refute FLOAT,
    evidence_status VARCHAR(20),
    weights JSONB,
    confidence_level VARCHAR(15),
    agent_outputs JSONB,
    source_citations JSONB,
    linguistic_profile JSONB,
    warnings JSONB,
    processing_time_ms INT,
    created_at TIMESTAMPTZ,
    extracted_claim TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.claim_id,
        r.c_total,
        r.s_fact,
        r.b_ling,
        r.v_consensus,
        r.v_refute,
        r.evidence_status,
        r.weights,
        r.confidence_level,
        r.agent_outputs,
        r.source_citations,
        r.linguistic_profile,
        r.warnings,
        r.processing_time_ms,
        r.created_at,
        c.extracted_claim,
        (1 - (c.embedding <=> query_embedding))::float AS similarity
    FROM claims c
    JOIN results r ON r.claim_id = c.id
    WHERE (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;
```

- [ ] **Step 4: Rewrite `lookup_cache`**

In `backend/app/services/cache.py`:

1. Replace the import block with:

```python
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.db.connection import get_supabase
from app.services.cache_policy import unservable_reason
from app.services.claim_equivalence import are_equivalent
```

2. Replace the whole `lookup_cache` function with:

```python
async def lookup_cache(
    embedding: List[float],
    claim_text: str,
    similarity_threshold: float = 0.95,
    match_count: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Serve a cached result only when it is safe to (C2):
    1. pgvector returns up to `match_count` candidates with cosine >= similarity_threshold.
    2. Each candidate must pass the row policy: confidence, evidence_status, complete run, TTL.
    3. The first remaining candidate whose claim is equivalent to `claim_text` (same numbers
       and bidirectional NLI entailment) is served. Embeddings alone cannot tell a claim from
       its negation.
    Fails closed: any error is a cache miss.
    """
    if not embedding or all(v == 0.0 for v in embedding):
        logger.debug("Cannot lookup cache with empty or all-zero embedding.")
        return None

    try:
        supabase = get_supabase()
        # supabase-py is synchronous; run the blocking round-trip off the event loop
        response = await asyncio.to_thread(
            supabase.rpc(
                "match_claims",
                {
                    "query_embedding": embedding,
                    "similarity_threshold": similarity_threshold,
                    "match_count": match_count,
                }
            ).execute
        )

        now = datetime.now(timezone.utc)
        for match_data in response.data or []:
            reason = unservable_reason(match_data, now)
            if reason:
                logger.info(f"Semantic match skipped: {reason}.")
                continue

            cached_claim = match_data.get("extracted_claim") or ""
            if not await asyncio.to_thread(are_equivalent, claim_text, cached_claim):
                logger.info("Semantic match skipped: cached claim is not equivalent (negation, number or scope).")
                continue

            # Ensure warnings is always a list
            if match_data.get("warnings") is None:
                match_data["warnings"] = []
            logger.info(f"Semantic cache HIT. Match similarity: {match_data.get('similarity')}")
            return match_data

        logger.debug("Semantic cache MISS.")
        return None
    except Exception as exc:
        logger.error(f"Error during semantic cache lookup: {exc}")
        return None
```

- [ ] **Step 5: Update the orchestrator call**

In `backend/app/agents/orchestrator.py`, replace
`cached = await lookup_cache(embedding, similarity_threshold=0.95)` with:

```python
        cached = await lookup_cache(embedding, claim_text=cleaned_text, similarity_threshold=0.95)
```

Also update docstring step 4 to read: "Generate embedding and query semantic cache (cosine
≥ 0.95, then row policy and claim-equivalence checks; skipped when bypass_cache=True)."

- [ ] **Step 6: Run the tests to verify they pass**

Run (from `backend/`): `uv run pytest -q`
Run (repo root): `uv run pytest eval/tests/ -q`
Run (repo root): `uv run pytest eval/tests/test_cache_safety.py -m live -v`
Expected: all pass. The live negation guardrail **must** pass. If it fails, stop and report.

- [ ] **Step 7: Apply the migration (user action)**

Ask the user to run `04_cache_topk_and_ttl.sql` in the Supabase SQL editor, after migration
03. Then verify with:

```sql
SELECT pg_get_function_arguments('match_claims'::regproc);
```

Expected output includes `match_count integer DEFAULT 3`.

- [ ] **Step 8: Commit**

```bash
git add backend/app/db/migrations/04_cache_topk_and_ttl.sql backend/app/services/cache.py backend/app/agents/orchestrator.py backend/tests/test_cache.py backend/tests/test_orchestrator.py eval/tests/test_cache_safety.py
git commit -m "fix(cache): serve only equivalent, fresh, complete results from top-k candidates (C2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Spec updates

**Files:** `documents/VeriFact_AI_TDD.md` §1 diagram and §6, `documents/VeriFact_AI_PRD_v2.md` §10, `CLAUDE.md` §2

- [ ] **Step 1: TDD §1 diagram**

In the SEMANTIC CACHE LAYER box, replace the line
`cosine_similarity(input_embedding, cached) ≥ 0.95?` with
`top-3 cosine ≥ 0.95 → row policy → NLI equivalence?`.

- [ ] **Step 2: TDD §6**

Replace the "Cache Lookup Query" SQL block with the `match_claims` body from migration 04.
Then replace the **Cache invalidation** paragraph with:

```markdown
**Serving rules (C2).** Cosine similarity is only a candidate filter: bi-encoder embeddings
score a claim and its negation, or a changed number, above 0.95. `lookup_cache` fetches the
top 3 candidates and serves the first that passes, in order:

1. **Row policy** (`app/services/cache_policy.py`, pure):
   - confidence is not `low` or `insufficient`
   - `evidence_status` is present (legacy rows never serve)
   - every agent completed (no `degraded` output)
   - the row is within its TTL
2. **Equivalence** (`app/services/claim_equivalence.py`):
   - identical numbers, and
   - bidirectional NLI entailment ≥ 0.8 with the local cross-encoder.
   One-way entailment (a narrower or broader claim) and contradiction (a negation) both miss.

Any error is a miss (fail closed).

**TTL:** `fact_checked` results live 30 days (`CACHE_TTL_FACT_CHECKED_HOURS=720`). All other
statuses live 24 hours (`CACHE_TTL_DEFAULT_HOURS=24`), because news-driven evidence changes
quickly. Expired rows stay in the database (so `GET /api/results/{id}` keeps working) but are
never served as cache hits.
```

- [ ] **Step 3: PRD §10**

In the "Cache hit rate" row, change the Measurement cell to "pgvector query logs; hits
counted only after the row-policy and equivalence checks pass". Add a line under the table:
"Cache hits are served only for equivalent claims (TDD §6). A lower hit rate is an accepted
cost of never serving a negated or numerically different claim."

- [ ] **Step 4: CLAUDE.md §2**

Change the cache line to:

```
   ├─ Semantic cache (pgvector cosine ≥ 0.95 → row policy + NLI equivalence + TTL) → short-circuit on hit
```

- [ ] **Step 5: Commit**

```bash
git add documents/VeriFact_AI_TDD.md documents/VeriFact_AI_PRD_v2.md CLAUDE.md
git commit -m "docs: specify cache serving rules, equivalence check and TTL (C2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Review gate (CLAUDE.md §5 — mandatory)

- [ ] **Step 1:** Run all suites, including `-m live` for `tests/test_claim_equivalence.py` and `eval/tests/test_cache_safety.py`. Record the counts.
- [ ] **Step 2:** Code review of `git diff main...fix/cache-safety`: the `code-reviewer` subagent in Claude Code, or `/review` in Antigravity (README mapping). Check:
  - fail-closed on every path (RPC error, NLI error, bad timestamps, missing fields)
  - that the policy runs before equivalence
  - pair order and bidirectionality
  - that the migration drops both old signatures
  - cache-hit latency (at most 3 candidates × 2 pairs)
- [ ] **Step 3:** Run the eval gate (the `eval-guardian` subagent or skill). Confirm the cache-safety guardrails (negation and numeric) and that no eval metric regressed.
- [ ] **Step 4:** Fix blocking findings, writing a test first, then re-run Step 1. Record any non-blocking finding you defer, with a reason.
- [ ] **Step 5:** Report to the user:
  - test counts
  - review outcome
  - that migration 04 is needed after 03
  - that `EQUIVALENCE_MIN_ENTAILMENT` is provisional until Plan 3 calibrates it

## Known limitations (carry into Plan 3)

- Number normalization is lexical: "1,000"/"1000" and "3%"/"3 percent" match, but "three"
  and "3" do not. That errs toward a miss, which is safe.
- Named-entity swaps ("Biden visited Paris" vs "Macron visited Paris") rely on NLI alone.
  Plan 3's paraphrase/negation pair set should include entity swaps.
