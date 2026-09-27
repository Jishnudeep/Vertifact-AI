# Eval Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Antigravity) or superpowers:subagent-driven-development (Claude Code) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `docs/superpowers/plans/README.md` first.

**Goal:** Make eval numbers honest and affordable:
- Report metrics per dataset and per evidence status.
- Stop LIAR from grading PolitiFact with PolitiFact's own answer.
- Replay recorded API responses so runs are reproducible and don't burn the free-tier quota.
- Calibrate the NLI thresholds that Plans 1 and 2 introduced.

**Architecture:** Everything lives under `eval/`. Backend code does not change.
1. `eval/replay.py` monkeypatches the evidence and LLM call sites. It records responses to
   disk, replays them, and applies a per-claim **leave-source-out** domain filter on top of
   the cached results.
2. `eval/metrics/segments.py` computes pure per-segment metrics, including selective
   accuracy and coverage for the "unverified" abstention state (D2).
3. `run_eval.py` gains CLI flags, per-claim exclusion, replay-miss invalidation, and
   segmented output.
4. Dataset builders turn local LIAR and FEVER files into the existing `EvalClaim` schema.
5. `eval/calibrate_nli.py` picks thresholds from labeled pair sets, with a hard
   zero-false-positive rule for cache equivalence.

**Tech Stack:** Python 3.12, pytest + anyio, the existing `eval/metrics/*`, `app.services.nli` (Plan 1), `app.services.claim_equivalence` (Plan 2).

**Spec:** `docs/superpowers/specs/2026-09-23-prd-tdd-review.md` (C3, C4, D2, and the "Known limitations" sections of Plans 1 and 2). Also read `documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md` Part 2.

## Global Constraints

- **Depends on Plans 1, 2 and 4 being merged** (run order 1 → 2 → 4 → 3; see `README.md`). You need `confidence.evidence_status` in responses, `app.services.nli`, `app.services.claim_equivalence`, and Plan 4's `search_depth` (it changes replay cache keys).
- Python **3.12+**, `uv` only. Run eval commands from the **repo root** (`uv run pytest eval/tests/`, `uv run python eval/run_eval.py`).
- This plan changes **no backend code**. Leave-source-out and replay are eval-only monkeypatches that are active only inside `run_eval.py`.
- A zero-tolerance guardrail may never be reported green on an invalid run (existing rule R8.3, keep it).
- **Downloads need user permission.** Never fetch LIAR or FEVER yourself. Ask the user to place the files locally (Task 5 gives the exact files).
- Every live run costs Tavily credits (1,000 a month on the free tier; advanced search costs 2 credits per call; roughly 6–10 credits per claim) and Groq quota (1K requests a day). Record once, then replay.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (Claude Code) or the executor's own attribution line (Antigravity).

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `eval/datasets/claim_dataset_schema.py` | Modify | Add `source_dataset` and `origin_domain` |
| `eval/replay.py` | Create | Record/replay evidence and LLM calls; per-claim domain exclusion |
| `eval/metrics/segments.py` | Create | Per-dataset and per-evidence-status metrics; selective accuracy and coverage |
| `eval/metrics/__init__.py` | Modify | Export `compute_segment_metrics` |
| `eval/run_eval.py` | Modify | CLI, replay wiring, exclusion, `evidence_status` capture, segments, replay-miss invalidation |
| `eval/datasets/build_liar.py`, `eval/datasets/build_fever.py` | Create | Local file → `EvalClaim` JSON (stratified, seeded) |
| `eval/datasets/stance_pairs.json`, `eval/datasets/equivalence_pairs.json` | Create | Labeled NLI calibration pairs |
| `eval/calibrate_nli.py` | Create | Threshold sweep and recommendation |
| `eval/tests/test_replay.py`, `eval/tests/test_segments.py`, `eval/tests/test_dataset_builders.py`, `eval/tests/test_calibrate_nli.py` | Create | Offline tests (+1 live) |
| `eval/tests/test_metrics.py` | Modify | Test for the replay-miss invalid rule |
| `.gitignore` | Modify | Ignore `eval/results/evidence_cache/` and `eval/datasets/raw/` |
| `documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md`, `documents/VeriFact_AI_PRD_v2.md` §10 | Modify | Segmented targets and leave-source-out rule |

---

### Task 0: Pre-flight

- [ ] **Step 1:** Run `git status --short` and expect empty output. Confirm Plans 1, 2 and 4 are merged: `backend/app/services/claim_equivalence.py` and `backend/app/services/budget.py` must exist, and `grep evidence_status backend/app/agents/orchestrator.py` must match.
- [ ] **Step 2:** `git checkout -b fix/eval-integrity`
- [ ] **Step 3:** Run `uv run pytest eval/tests/ -q`. Expected: all pass. Record the count.

---

### Task 1: Dataset schema fields

**Files:** Modify `eval/datasets/claim_dataset_schema.py`. Test: `eval/tests/test_dataset_builders.py` (created here, extended in Task 5).

**Interfaces:** Produces `EvalClaim.source_dataset: Literal["curated", "liar", "fever"] = "curated"` and `EvalClaim.origin_domain: Optional[str] = None`.

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_dataset_builders.py`:

```python
import json
from pathlib import Path

from eval.datasets.claim_dataset_schema import EvalClaim

SEED = Path(__file__).resolve().parent.parent / "datasets" / "curated_claims_seed.json"


def test_existing_seed_defaults_to_curated_with_no_origin():
    for raw in json.loads(SEED.read_text(encoding="utf-8")):
        claim = EvalClaim(**raw)
        assert claim.source_dataset == "curated"
        assert claim.origin_domain is None


def test_origin_domain_round_trips():
    claim = EvalClaim(id="liar-1", claim_text="x", ground_truth_score=0.0, category="politics",
                      source_dataset="liar", origin_domain="politifact.com")
    assert claim.origin_domain == "politifact.com"
```

- [ ] **Step 2:** Run `uv run pytest eval/tests/test_dataset_builders.py -v`. Expected: FAIL with `AttributeError: 'EvalClaim' object has no attribute 'source_dataset'`.

- [ ] **Step 3: Implement.** Replace `eval/datasets/claim_dataset_schema.py` with:

```python
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class EvalClaim(BaseModel):
    id: str
    claim_text: str
    ground_truth_score: float = Field(ge=0.0, le=1.0)
    category: str
    claim_type: Literal["verifiable", "opinion", "question", "adversarial"] = "verifiable"
    expected_confidence: Literal["high", "medium", "low", "insufficient"] = "high"
    expected_verdict: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    # C3: which benchmark the claim came from; metrics are reported per source_dataset.
    source_dataset: Literal["curated", "liar", "fever"] = "curated"
    # C3 leave-source-out: the site that produced the label (LIAR -> politifact.com). Evidence
    # from this domain is filtered out while the claim is evaluated, so the system cannot
    # grade a claim with the labeller's own article.
    origin_domain: Optional[str] = None
```

- [ ] **Step 4:** Run `uv run pytest eval/tests/test_dataset_builders.py -v`. Expected: 2 passed.
- [ ] **Step 5: Commit**

```bash
git add eval/datasets/claim_dataset_schema.py eval/tests/test_dataset_builders.py
git commit -m "feat(eval): add source_dataset and origin_domain to EvalClaim (C3)"
```

---

### Task 2: Record/replay with leave-source-out exclusion

**Files:** Create `eval/replay.py`, `eval/tests/test_replay.py`. Modify `.gitignore`.

**Interfaces:**
- Consumes: the call sites `app.agents.factcheck_agent.search_factchecks`, `app.agents.factcheck_agent.search_web`, `app.agents.crossref_agent.search_web`, `app.agents.factcheck_agent.run_adk_agent`, `app.agents.linguistic_agent.run_adk_agent`; `EvidenceAPIError`; `extract_domain`.
- Produces:
  - `ReplayMode = Literal["record", "replay", "live"]`
  - `ReplayMissError(EvidenceAPIError)`
  - `class EvidenceReplay(cache_dir: Path, mode: ReplayMode)` with:
    - `.install() -> None`
    - `.uninstall() -> None`
    - `.misses: int`
    - `.recorded: int`
  - `excluding_domains(domains: Iterable[str])` (a context manager)
  - `filter_web_results(results, excluded) -> list`
  - `filter_factcheck_claims(claims, excluded) -> list`

- [ ] **Step 1: Write the failing tests**

Create `eval/tests/test_replay.py`:

```python
import pytest

import app.agents.crossref_agent as crossref_agent
import app.agents.factcheck_agent as factcheck_agent
from eval.replay import (
    EvidenceReplay, ReplayMissError, excluding_domains,
    filter_factcheck_claims, filter_web_results,
)
from app.services.errors import EvidenceAPIError


def test_filter_web_results_drops_excluded_domains():
    results = [{"url": "https://www.politifact.com/a"}, {"url": "https://www.reuters.com/b"}]
    assert filter_web_results(results, frozenset({"politifact.com"})) == [{"url": "https://www.reuters.com/b"}]


def test_filter_factcheck_claims_drops_reviews_and_empty_claims():
    claims = [
        {"text": "c1", "claimReview": [{"url": "https://www.politifact.com/x"}]},
        {"text": "c2", "claimReview": [{"url": "https://www.politifact.com/y"},
                                       {"url": "https://www.snopes.com/z"}]},
    ]
    kept = filter_factcheck_claims(claims, frozenset({"politifact.com"}))
    assert kept == [{"text": "c2", "claimReview": [{"url": "https://www.snopes.com/z"}]}]


def test_replay_miss_is_an_evidence_outage():
    assert issubclass(ReplayMissError, EvidenceAPIError)


@pytest.mark.anyio
async def test_record_then_replay_round_trip(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def fake_search_web(query, api_key, max_results=5, include_domains=None):
        calls["n"] += 1
        return [{"url": "https://www.reuters.com/x", "title": query}]

    monkeypatch.setattr(crossref_agent, "search_web", fake_search_web)
    monkeypatch.setattr(factcheck_agent, "search_web", fake_search_web)

    recorder = EvidenceReplay(tmp_path, "record")
    recorder.install()
    try:
        first = await crossref_agent.search_web(query="q1", api_key="k", max_results=5)
    finally:
        recorder.uninstall()
    assert recorder.recorded == 1 and calls["n"] == 1

    player = EvidenceReplay(tmp_path, "replay")
    player.install()
    try:
        again = await crossref_agent.search_web(query="q1", api_key="DIFFERENT-KEY", max_results=5)
        with pytest.raises(ReplayMissError):
            await crossref_agent.search_web(query="never-recorded", api_key="k", max_results=5)
    finally:
        player.uninstall()
    assert again == first
    assert calls["n"] == 1          # replay never touched the live function
    assert player.misses == 1
    assert crossref_agent.search_web is fake_search_web  # uninstall restored the original


@pytest.mark.anyio
async def test_exclusion_applies_on_replayed_results(tmp_path, monkeypatch):
    async def fake_search_web(query, api_key, max_results=5, include_domains=None):
        return [{"url": "https://www.politifact.com/x"}, {"url": "https://www.reuters.com/y"}]

    monkeypatch.setattr(crossref_agent, "search_web", fake_search_web)
    monkeypatch.setattr(factcheck_agent, "search_web", fake_search_web)
    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        with excluding_domains(["politifact.com"]):
            inside = await crossref_agent.search_web(query="q", api_key="k")
        outside = await crossref_agent.search_web(query="q", api_key="k")
    finally:
        replay.uninstall()
    assert [r["url"] for r in inside] == ["https://www.reuters.com/y"]
    assert len(outside) == 2        # the cache stores raw results; exclusion is only a view


@pytest.mark.anyio
async def test_live_errors_are_not_recorded(tmp_path, monkeypatch):
    async def failing(query, api_key, max_results=5, include_domains=None):
        raise EvidenceAPIError("HTTP 432", provider="tavily")

    monkeypatch.setattr(crossref_agent, "search_web", failing)
    monkeypatch.setattr(factcheck_agent, "search_web", failing)
    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        with pytest.raises(EvidenceAPIError):
            await crossref_agent.search_web(query="q", api_key="k")
    finally:
        replay.uninstall()
    assert replay.recorded == 0
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2:** Run `uv run pytest eval/tests/test_replay.py -v`. Expected: collection ERROR, `ModuleNotFoundError: No module named 'eval.replay'`.

- [ ] **Step 3: Implement.** Create `eval/replay.py`:

```python
"""
Eval-only record/replay for external calls, plus leave-source-out filtering (C3).

Never imported by the backend. run_eval.py installs it around a run:
  record: call live, store successful responses under cache_dir
  replay: serve from cache_dir only; a missing entry raises ReplayMissError (an outage)
  live:   call live, store nothing
Cached payloads are RAW results. Domain exclusion is applied on the way out, so one cache
serves runs with and without leave-source-out.
"""
import contextlib
import contextvars
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple

import app.agents.crossref_agent as crossref_agent
import app.agents.factcheck_agent as factcheck_agent
import app.agents.linguistic_agent as linguistic_agent
from app.config import settings
from app.services.errors import EvidenceAPIError
from app.utils.text_processing import extract_domain

ReplayMode = Literal["record", "replay", "live"]

_excluded: contextvars.ContextVar[frozenset] = contextvars.ContextVar("eval_excluded_domains", default=frozenset())


class ReplayMissError(EvidenceAPIError):
    """No recorded response for this call in replay mode. Agents treat it as an outage."""


@contextlib.contextmanager
def excluding_domains(domains: Iterable[str]):
    token = _excluded.set(frozenset(d.lower() for d in domains if d))
    try:
        yield
    finally:
        _excluded.reset(token)


def filter_web_results(results: List[Dict[str, Any]], excluded: frozenset) -> List[Dict[str, Any]]:
    return [r for r in results if extract_domain(r.get("url", "")) not in excluded]


def filter_factcheck_claims(claims: List[Dict[str, Any]], excluded: frozenset) -> List[Dict[str, Any]]:
    kept = []
    for claim in claims:
        reviews = [rv for rv in claim.get("claimReview", []) if extract_domain(rv.get("url", "")) not in excluded]
        if reviews:
            kept.append({**claim, "claimReview": reviews})
    return kept


def _key(provider: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps([provider, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class EvidenceReplay:
    def __init__(self, cache_dir: Path, mode: ReplayMode):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.misses = 0
        self.recorded = 0
        self._originals: List[Tuple[Any, str, Callable]] = []

    # --- cache plumbing -------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    async def _through_cache(self, provider: str, payload: Dict[str, Any], live_call):
        key = _key(provider, payload)
        path = self._path(key)
        if self.mode == "replay":
            if not path.exists():
                self.misses += 1
                raise ReplayMissError(f"No recorded {provider} response (key {key[:12]})", provider=provider)
            return json.loads(path.read_text(encoding="utf-8"))
        result = await live_call()  # errors propagate and are never recorded
        if self.mode == "record":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            self.recorded += 1
        return result

    # --- wrappers ---------------------------------------------------------
    def _wrap_search_web(self, original):
        # **kwargs passes through any newer search_web options (e.g. Plan 4's search_depth)
        # and makes them part of the cache key.
        async def search_web(query, api_key, max_results=5, include_domains=None, **kwargs):
            payload = {"query": query, "max_results": max_results,
                       "include_domains": sorted(include_domains or []), **kwargs}
            results = await self._through_cache(
                "tavily", payload,
                lambda: original(query=query, api_key=api_key, max_results=max_results,
                                 include_domains=include_domains, **kwargs),
            )
            return filter_web_results(results, _excluded.get())
        return search_web

    def _wrap_search_factchecks(self, original):
        async def search_factchecks(query, api_key):
            claims = await self._through_cache(
                "google_factcheck", {"query": query}, lambda: original(query, api_key)
            )
            return filter_factcheck_claims(claims, _excluded.get())
        return search_factchecks

    def _wrap_run_adk_agent(self, original):
        async def run_adk_agent(name, instruction, prompt, app_name):
            payload = {"model": settings.model_id, "name": name, "instruction": instruction,
                       "prompt": prompt, "app_name": app_name}
            return await self._through_cache(
                "llm", payload, lambda: original(name, instruction, prompt, app_name)
            )
        return run_adk_agent

    # --- install / uninstall ----------------------------------------------
    def _patch(self, module, attr: str, wrapper_factory) -> None:
        original = getattr(module, attr)
        self._originals.append((module, attr, original))
        setattr(module, attr, wrapper_factory(original))

    def install(self) -> None:
        if self.mode == "live":
            return
        self._patch(crossref_agent, "search_web", self._wrap_search_web)
        self._patch(factcheck_agent, "search_web", self._wrap_search_web)
        self._patch(factcheck_agent, "search_factchecks", self._wrap_search_factchecks)
        self._patch(factcheck_agent, "run_adk_agent", self._wrap_run_adk_agent)
        self._patch(linguistic_agent, "run_adk_agent", self._wrap_run_adk_agent)

    def uninstall(self) -> None:
        while self._originals:
            module, attr, original = self._originals.pop()
            setattr(module, attr, original)
```

**Note:** in `live` mode `install()` is a no-op, so leave-source-out would not apply.
`run_eval.py` (Task 4) therefore always installs in `record` or `replay` mode when any claim
has an `origin_domain`, and refuses `--replay live` for such datasets.

- [ ] **Step 4: Ignore the cache and raw downloads.** In `.gitignore`, directly after the `eval/metrics/*.png` line, add:

```
eval/results/evidence_cache/
eval/datasets/raw/
```

- [ ] **Step 5:** Run `uv run pytest eval/tests/test_replay.py -v`. Expected: 6 passed.
- [ ] **Step 6: Commit**

```bash
git add eval/replay.py eval/tests/test_replay.py .gitignore
git commit -m "feat(eval): record/replay evidence and LLM calls with leave-source-out filtering (C3, H2)"
```

---

### Task 3: Segmented metrics (per dataset, per evidence status, selective accuracy)

**Files:** Create `eval/metrics/segments.py`, `eval/tests/test_segments.py`. Modify `eval/metrics/__init__.py`.

**Interfaces:**
- Consumes: `compute_mae`, `compute_bucketed_accuracy`.
- Produces: `compute_segment_metrics(records: list[dict]) -> dict`. Each record has keys `source_dataset`, `evidence_status`, `predicted_c_total` and `ground_truth`. The result has this shape:

```
{
  "by_dataset": {name: {"n", "mae", "bucketed_accuracy"}},
  "by_evidence_status": {status: {"n", "mae", "bucketed_accuracy"}},
  "selective": {"coverage", "n_answered", "mae", "bucketed_accuracy"}
}
```

- [ ] **Step 1: Write the failing tests.** Create `eval/tests/test_segments.py`:

```python
from eval.metrics.segments import compute_segment_metrics


def _r(ds, status, pred, gt):
    return {"source_dataset": ds, "evidence_status": status, "predicted_c_total": pred, "ground_truth": gt}


def test_groups_by_dataset_and_status():
    records = [
        _r("curated", "fact_checked", 0.1, 0.0),
        _r("curated", "unverified", 0.5, 1.0),
        _r("liar", "fact_checked", 0.9, 1.0),
    ]
    out = compute_segment_metrics(records)
    assert out["by_dataset"]["curated"]["n"] == 2
    assert out["by_dataset"]["liar"]["mae"] == 0.1
    assert out["by_evidence_status"]["unverified"]["n"] == 1


def test_selective_metrics_exclude_unverified_and_report_coverage():
    records = [
        _r("curated", "fact_checked", 0.1, 0.0),   # answered, correct bucket
        _r("curated", "disputed", 0.3, 0.0),       # answered, correct bucket
        _r("curated", "unverified", 0.5, 1.0),     # abstained
        _r("curated", "unverified", 0.5, 0.0),     # abstained
    ]
    sel = compute_segment_metrics(records)["selective"]
    assert sel["coverage"] == 0.5
    assert sel["n_answered"] == 2
    assert sel["bucketed_accuracy"] == 1.0


def test_empty_input_reports_none_not_perfect():
    out = compute_segment_metrics([])
    assert out["selective"] == {"coverage": None, "n_answered": 0, "mae": None, "bucketed_accuracy": None}
    assert out["by_dataset"] == {} and out["by_evidence_status"] == {}


def test_missing_evidence_status_is_its_own_segment():
    out = compute_segment_metrics([_r("curated", None, 0.5, 0.5)])
    assert out["by_evidence_status"]["missing"]["n"] == 1
    assert out["selective"]["n_answered"] == 0   # missing status never counts as an answer
```

- [ ] **Step 2:** Run `uv run pytest eval/tests/test_segments.py -v`. Expected: collection ERROR, `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Create `eval/metrics/segments.py`:

```python
"""
Segmented metrics (C3/C4). Blended metrics hide two different systems: claims with a
matched fact-check (S_fact carries the verdict) and claims without one (score bounded to
roughly 0.25-0.75 by construction). Report them apart, and treat evidence_status
"unverified" as an abstention: selective accuracy over answered claims, plus coverage.
"""
from collections import defaultdict
from typing import Any, Dict, List

from eval.metrics.bucketed_accuracy import compute_bucketed_accuracy
from eval.metrics.mae import compute_mae

ANSWERED_STATUSES = frozenset({"fact_checked", "disputed", "corroborated"})


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [float(r["predicted_c_total"]) for r in records]
    gts = [float(r["ground_truth"]) for r in records]
    return {
        "n": len(records),
        "mae": compute_mae(preds, gts),
        "bucketed_accuracy": compute_bucketed_accuracy(preds, gts)["accuracy"],
    }


def compute_segment_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_status: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dataset[r.get("source_dataset") or "curated"].append(r)
        by_status[r.get("evidence_status") or "missing"].append(r)

    answered = [r for r in records if r.get("evidence_status") in ANSWERED_STATUSES]
    selective_summary = _summarize(answered)
    return {
        "by_dataset": {k: _summarize(v) for k, v in sorted(by_dataset.items())},
        "by_evidence_status": {k: _summarize(v) for k, v in sorted(by_status.items())},
        "selective": {
            "coverage": round(len(answered) / len(records), 4) if records else None,
            "n_answered": len(answered),
            "mae": selective_summary["mae"],
            "bucketed_accuracy": selective_summary["bucketed_accuracy"],
        },
    }
```

- [ ] **Step 4: Export.** In `eval/metrics/__init__.py`, add `from eval.metrics.segments import compute_segment_metrics` and add `"compute_segment_metrics"` to `__all__`.
- [ ] **Step 5:** Run `uv run pytest eval/tests/test_segments.py -v`. Expected: 4 passed.
- [ ] **Step 6: Commit**

```bash
git add eval/metrics/segments.py eval/metrics/__init__.py eval/tests/test_segments.py
git commit -m "feat(eval): per-dataset and per-evidence-status metrics with selective accuracy (C3, C4)"
```

---

### Task 4: Wire the harness (CLI, replay, exclusion, segments, replay-miss invalidation)

**Files:** Modify `eval/run_eval.py`, `eval/tests/test_metrics.py`.

**Interfaces:**
- Consumes: `EvidenceReplay`, `excluding_domains` (Task 2); `compute_segment_metrics` (Task 3); `EvalClaim.source_dataset` and `origin_domain` (Task 1); `res["confidence"]["evidence_status"]` (Plan 1).
- Produces:
  - `assess_run_validity(..., replay_misses: int = 0)`, where any miss means `INVALID_RUN`
  - `run_evaluation(dataset_paths: list[Path], output_path: Path, replay_mode: ReplayMode = "replay", cache_dir: Path = ..., claim_delay_s: float = 0.0)`
  - `summary["segments"]`, `summary["replay"]`

- [ ] **Step 1: Write the failing test.** Append to `eval/tests/test_metrics.py`:

```python
def test_replay_miss_invalidates_run():
    from eval.run_eval import assess_run_validity, STATUS_INVALID_RUN
    status, reasons = assess_run_validity(
        verifiable_preds=[0.2, 0.8], verifiable_targets=[0.0, 1.0],
        zero_source_rate=0.0, error_rate=0.0,
        bias_metrics={"passed": True}, bridging_violations=[], adversarial_results=[],
        replay_misses=3,
    )
    assert status == STATUS_INVALID_RUN
    assert any("replay" in r for r in reasons)
```

- [ ] **Step 2:** Run `uv run pytest eval/tests/test_metrics.py -v`. Expected: FAIL with `TypeError: ... unexpected keyword argument 'replay_misses'`.

- [ ] **Step 3: `assess_run_validity`.** Add the parameter `replay_misses: int = 0,` after `adversarial_results`. Directly before the line `if guardrail_reasons:`, add:

```python
    if replay_misses > 0:
        invalid_reasons.append(
            f"{replay_misses} external call(s) had no recorded response in replay mode - "
            f"re-run once with --replay record to populate eval/results/evidence_cache/"
        )
```

- [ ] **Step 4: Imports.** At the top of `eval/run_eval.py`, after the existing `from eval.metrics...` imports, add:

```python
import argparse
from eval.datasets.claim_dataset_schema import EvalClaim
from eval.metrics.segments import compute_segment_metrics
from eval.replay import EvidenceReplay, ReplayMode, excluding_domains

DEFAULT_CACHE_DIR = BASE_DIR / "eval" / "results" / "evidence_cache"
```

- [ ] **Step 5: `run_evaluation` signature and dataset loading.** Replace the function
header, and everything up to and including `load_bias_ratings()`, with:

```python
async def run_evaluation(
    dataset_paths: Optional[List[Path]] = None,
    output_path: Path = BASE_DIR / "eval" / "results" / "baseline_run.json",
    replay_mode: ReplayMode = "replay",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    claim_delay_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Run the evaluation harness over one or more EvalClaim JSON datasets.
    (See the module docstring for the metric list.) C3 additions:
    - replay_mode: record | replay | live. Replay serves recorded evidence/LLM responses and
      marks the run INVALID on any miss; record captures them; live calls through untouched.
    - leave-source-out: a claim's origin_domain is filtered out of every evidence call.
    - summary["segments"]: metrics per source_dataset and per evidence_status, plus
      selective accuracy/coverage treating evidence_status "unverified" as an abstention.
    """
    dataset_paths = dataset_paths or [BASE_DIR / "eval" / "datasets" / "curated_claims_seed.json"]
    claims_data: List[Dict[str, Any]] = []
    for path in dataset_paths:
        logger.info(f"Loading evaluation dataset from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            claims_data.extend(EvalClaim(**raw).model_dump() for raw in json.load(f))

    if replay_mode == "live" and any(c.get("origin_domain") for c in claims_data):
        raise ValueError("Leave-source-out needs --replay record or replay; refusing a live run "
                         "on a dataset with origin_domain set (C3).")

    replay = EvidenceReplay(cache_dir, replay_mode)
    replay.install()

    # Initialize bias ratings
    load_bias_ratings()
    segment_records: List[Dict[str, Any]] = []
```

Then wrap the rest of the function body, from `verifiable_preds: List[float] = []` down to
and including `return full_report`, in `try:` / `finally: replay.uninstall()`. Indent the
body one level. No other lines change.

- [ ] **Step 6: Per-claim exclusion, `evidence_status` capture and throttle.** Make four
changes inside the `for item in claims_data:` loop:

(a) Directly after `expected_conf = item.get("expected_confidence", "high")`, add:

```python
        excluded = [item["origin_domain"]] if item.get("origin_domain") else []
        if claim_delay_s:
            await asyncio.sleep(claim_delay_s)  # stay under Groq's 8K TPM free-tier limit
```

(b) Replace both `res = await verify_claim(text, bypass_cache=True)` lines (in the
adversarial branch and the verifiable branch) with:

```python
                with excluding_domains(excluded):
                    res = await verify_claim(text, bypass_cache=True)
```

Keep the indentation of the surrounding `try:` block. The adversarial branch is indented one
level deeper than the verifiable branch.

(c) In the verifiable branch, directly after `verifiable_targets.append(gt_score)`, add:

```python
            evidence_status = res["confidence"].get("evidence_status")
            segment_records.append({
                "source_dataset": item.get("source_dataset", "curated"),
                "evidence_status": evidence_status,
                "predicted_c_total": c_total,
                "ground_truth": gt_score,
            })
```

(d) In the same branch's `eval_results.append({...})`, add
`"evidence_status": evidence_status,` and `"source_dataset": item.get("source_dataset", "curated"),`.

- [ ] **Step 7: Summary.** Pass `replay_misses=replay.misses,` as the last argument to
`assess_run_validity(...)`. In the `summary` dict, directly after
`"calibration_bins": calib["bins"],`, add:

```python
        "segments": compute_segment_metrics(segment_records),
        "replay": {"mode": replay_mode, "misses": replay.misses, "recorded": replay.recorded},
```

In the printed summary, after the "Bucketed Accuracy" line, add:

```python
    seg = summary["segments"]["selective"]
    print(f"  • Selective Bucketed Acc.:     {fmt(seg['bucketed_accuracy'])} on answered claims (coverage {fmt(seg['coverage'], '{:.0%}')})")
    for ds, m in summary["segments"]["by_dataset"].items():
        print(f"  • [{ds}] n={m['n']} MAE={fmt(m['mae'])} bucketed={fmt(m['bucketed_accuracy'])}")
    print(f"  • Replay:                      mode={replay_mode} misses={replay.misses} recorded={replay.recorded}")
```

- [ ] **Step 8: CLI.** Replace the `if __name__ == "__main__":` block with:

```python
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VeriFact eval harness")
    parser.add_argument("--dataset", action="append", type=Path,
                        help="EvalClaim JSON file; repeatable. Default: curated seed.")
    parser.add_argument("--replay", choices=["record", "replay", "live"], default="replay",
                        help="record once (costs quota), then replay (free, reproducible).")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "eval" / "results" / "baseline_run.json")
    parser.add_argument("--claim-delay", type=float, default=0.0,
                        help="seconds between claims (use ~2.0 when recording on Groq free tier)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    report = asyncio.run(run_evaluation(
        dataset_paths=args.dataset, output_path=args.output,
        replay_mode=args.replay, claim_delay_s=args.claim_delay,
    ))
    sys.exit(0 if report["summary"]["run_status"] == STATUS_OK else 1)
```

- [ ] **Step 9:** Run `uv run pytest eval/tests/ -q`. Expected: all pass. Anything that calls `run_evaluation(dataset_path=...)` must be updated to `dataset_paths=[...]`. Run `grep -rn "run_evaluation(" eval/ backend/` to find those callers.

- [ ] **Step 10: Smoke-record the curated seed (user approval needed: it spends quota).** Ask the user first. This costs about 25 claims × roughly 8 Tavily credits plus about 100 Groq calls. Then run:

```bash
uv run python eval/run_eval.py --replay record --claim-delay 2
uv run python eval/run_eval.py --replay replay
```

Expected: the second run shows `misses=0`, and its metrics are identical to the recording
run. Save both summaries in the task report.

- [ ] **Step 11: Commit**

```bash
git add eval/run_eval.py eval/tests/test_metrics.py
git commit -m "feat(eval): replay-by-default harness with leave-source-out and segmented metrics (C3, C4, H2)"
```

---

### Task 5: LIAR and FEVER builders (local files only)

**Files:** Create `eval/datasets/build_liar.py`, `eval/datasets/build_fever.py`. Extend `eval/tests/test_dataset_builders.py`.

**Interfaces:**
- Produces:
  - `build_liar(tsv_path: Path, n: int, seed: int = 42) -> list[dict]`
  - `build_fever(jsonl_path: Path, n: int, seed: int = 42) -> list[dict]`

  Both return `EvalClaim`-valid dicts, stratified by label.
- Label maps (Eval Framework doc):
  - LIAR: pants-fire 0.0, false 0.15, barely-true 0.3, half-true 0.5, mostly-true 0.75, true 1.0
  - FEVER: SUPPORTS 1.0, REFUTES 0.0, NOT ENOUGH INFO 0.5

- [ ] **Step 1: Write the failing tests.** Append to `eval/tests/test_dataset_builders.py`:

```python
from eval.datasets.build_fever import build_fever
from eval.datasets.build_liar import build_liar

LIAR_ROWS = "\n".join([
    "1.json\tfalse\tSays the moon is cheese.\tscience\tjoe\t\t\t\t0\t0\t0\t0\t0\tspeech",
    "2.json\ttrue\tUnemployment fell to 3.5 percent in 2019.\teconomy\tann\t\t\t\t0\t0\t0\t0\t0\ttweet",
    "3.json\tpants-fire\tVaccines contain microchips.\thealth\tbob\t\t\t\t0\t0\t0\t0\t0\tad",
    "4.json\thalf-true\tThe bill cut taxes for most families.\ttaxes\tsue\t\t\t\t0\t0\t0\t0\t0\tdebate",
])

FEVER_ROWS = "\n".join([
    '{"id": 1, "label": "SUPPORTS", "claim": "Paris is the capital of France."}',
    '{"id": 2, "label": "REFUTES", "claim": "The Pacific is the smallest ocean."}',
    '{"id": 3, "label": "NOT ENOUGH INFO", "claim": "Some band released an album."}',
])


def test_build_liar_maps_labels_and_sets_origin(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(LIAR_ROWS, encoding="utf-8")
    rows = build_liar(path, n=4, seed=1)
    by_text = {r["claim_text"]: r for r in rows}
    assert by_text["Says the moon is cheese."]["ground_truth_score"] == 0.15
    assert by_text["Vaccines contain microchips."]["ground_truth_score"] == 0.0
    for r in rows:
        claim = EvalClaim(**r)
        assert claim.source_dataset == "liar"
        assert claim.origin_domain == "politifact.com"   # leave-source-out target


def test_build_liar_is_seeded_and_stratified(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(LIAR_ROWS, encoding="utf-8")
    assert build_liar(path, n=2, seed=7) == build_liar(path, n=2, seed=7)
    labels = {r["tags"][0] for r in build_liar(path, n=4, seed=7)}
    assert labels == {"false", "true", "pants-fire", "half-true"}


def test_build_fever_maps_labels_without_origin(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(FEVER_ROWS, encoding="utf-8")
    rows = build_fever(path, n=3, seed=1)
    scores = sorted(r["ground_truth_score"] for r in rows)
    assert scores == [0.0, 0.5, 1.0]
    for r in rows:
        claim = EvalClaim(**r)
        assert claim.source_dataset == "fever" and claim.origin_domain is None
```

- [ ] **Step 2:** Run `uv run pytest eval/tests/test_dataset_builders.py -v`. Expected: collection ERROR, `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Create `eval/datasets/build_liar.py`:

```python
"""
Build an EvalClaim dataset from a local LIAR split (Wang 2017; TSV, 14 columns:
id, label, statement, subject, speaker, ...). The user downloads the dataset; this script
never fetches anything. Every claim gets origin_domain="politifact.com": LIAR labels ARE
PolitiFact verdicts, so evidence from politifact.com is excluded during eval (C3).

Usage: uv run python -m eval.datasets.build_liar eval/datasets/raw/liar/test.tsv --n 40 \
         --out eval/datasets/liar_mapped.json
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

LIAR_SCORES = {
    "pants-fire": 0.0, "false": 0.15, "barely-true": 0.3,
    "half-true": 0.5, "mostly-true": 0.75, "true": 1.0,
}


def _stratified_sample(groups: Dict[str, List[dict]], n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    labels = sorted(groups)
    for label in labels:
        rng.shuffle(groups[label])
    picked: List[dict] = []
    i = 0
    while len(picked) < n and any(groups[l] for l in labels):
        label = labels[i % len(labels)]
        if groups[label]:
            picked.append(groups[label].pop())
        i += 1
    return picked


def build_liar(tsv_path: Path, n: int, seed: int = 42) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4 or row[1] not in LIAR_SCORES:
                continue
            groups[row[1]].append({
                "id": f"liar-{row[0].removesuffix('.json')}",
                "claim_text": row[2].strip(),
                "ground_truth_score": LIAR_SCORES[row[1]],
                "category": row[3].split(",")[0] or "politics",
                "claim_type": "verifiable",
                "expected_confidence": "medium",
                "tags": [row[1]],
                "source_dataset": "liar",
                "origin_domain": "politifact.com",
            })
    return _stratified_sample(groups, n, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/datasets/liar_mapped.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(build_liar(args.tsv, args.n, args.seed), indent=2), encoding="utf-8")
```

Create `eval/datasets/build_fever.py`:

```python
"""
Build an EvalClaim dataset from a local FEVER jsonl split (e.g. shared_task_dev.jsonl;
fields: id, label, claim). FEVER claims are Wikipedia-derived and rarely have ClaimReview
coverage: report them as their own segment, never blended into the headline (C3).

Usage: uv run python -m eval.datasets.build_fever eval/datasets/raw/fever/shared_task_dev.jsonl \
         --n 30 --out eval/datasets/fever_subset.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from eval.datasets.build_liar import _stratified_sample

FEVER_SCORES = {"SUPPORTS": 1.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 0.5}


def build_fever(jsonl_path: Path, n: int, seed: int = 42) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in FEVER_SCORES:
                continue
            groups[row["label"]].append({
                "id": f"fever-{row['id']}",
                "claim_text": row["claim"].strip(),
                "ground_truth_score": FEVER_SCORES[row["label"]],
                "category": "encyclopedic",
                "claim_type": "verifiable",
                "expected_confidence": "low",
                "tags": [row["label"].lower().replace(" ", "_")],
                "source_dataset": "fever",
                "origin_domain": None,
            })
    return _stratified_sample(groups, n, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/datasets/fever_subset.json"))
    args = parser.parse_args()
    args.out.write_text(json.dumps(build_fever(args.jsonl, args.n, args.seed), indent=2), encoding="utf-8")
```

- [ ] **Step 4:** Run `uv run pytest eval/tests/test_dataset_builders.py -v`. Expected: 5 passed.
- [ ] **Step 5: Obtain the raw files (user action).** Ask the user to download these into `eval/datasets/raw/`, which is gitignored:
  - LIAR `test.tsv` (UCSB "LIAR" dataset zip) → `eval/datasets/raw/liar/test.tsv`
  - FEVER `shared_task_dev.jsonl` (fever.ai) → `eval/datasets/raw/fever/shared_task_dev.jsonl`

  Then run both builders with their defaults (LIAR n=40, FEVER n=30). These sizes keep one
  recording pass inside a month's Tavily free tier alongside the curated set. Commit the
  generated `liar_mapped.json` and `fever_subset.json`: they are derived samples, not the
  raw dumps.
- [ ] **Step 6: Commit**

```bash
git add eval/datasets/build_liar.py eval/datasets/build_fever.py eval/tests/test_dataset_builders.py
git commit -m "feat(eval): seeded, stratified LIAR/FEVER builders with PolitiFact origin tagging (C3)"
```

---

### Task 6: Calibrate the NLI thresholds

**Files:** Create `eval/datasets/stance_pairs.json`, `eval/datasets/equivalence_pairs.json`, `eval/calibrate_nli.py`, `eval/tests/test_calibrate_nli.py`.

**Interfaces:**
- Consumes: `classify_pairs`, `NLIScores` (Plan 1); `label_stance`/`STANCE_MIN_PROB` (Plan 1); `numbers_match`/`EQUIVALENCE_MIN_ENTAILMENT` (Plan 2).
- Produces:
  - `sweep_stance(rows, scores, thresholds) -> list[dict]`
  - `sweep_equivalence(rows, scores, thresholds) -> list[dict]`
  - `recommend(stance_sweep, equivalence_sweep) -> dict`

  Recommendation rule: for equivalence, the lowest threshold with **zero false positives**
  (non-equivalent pairs marked equivalent). For stance, the threshold with the highest macro
  accuracy, where ties go to the higher threshold.

- [ ] **Step 1: Author the pair sets.** Create `eval/datasets/stance_pairs.json`. Each row is
`{premise, claim, label}` with `label ∈ supports|refutes|neutral`. This is the minimum set;
grow it over time with the same shape:

```json
[
  {"premise": "NASA scientists confirmed on Monday that liquid water was detected beneath the Martian surface.", "claim": "NASA confirms water found on Mars", "label": "supports"},
  {"premise": "Health officials warned that drinking bleach is dangerous and does not cure any infection.", "claim": "Drinking bleach cures the flu", "label": "refutes"},
  {"premise": "The city council met on Tuesday to discuss the new budget for road repairs.", "claim": "The mayor resigned last week", "label": "neutral"},
  {"premise": "A viral post claims the vaccine alters DNA. Experts say mRNA vaccines cannot change a person's DNA.", "claim": "COVID vaccines alter human DNA", "label": "refutes"},
  {"premise": "The Federal Reserve raised its benchmark rate by a quarter point on Wednesday.", "claim": "The Fed raised interest rates", "label": "supports"},
  {"premise": "Officials denied reports that the bridge had collapsed, saying it remains open to traffic.", "claim": "The Golden Gate Bridge collapsed", "label": "refutes"},
  {"premise": "The company reported quarterly revenue of $12 billion, up 8% from a year earlier.", "claim": "The company's revenue grew last quarter", "label": "supports"},
  {"premise": "Researchers are still studying whether the drug is effective; results are expected next year.", "claim": "The drug cures Alzheimer's disease", "label": "neutral"},
  {"premise": "Fact-checkers found no evidence that the election machines switched votes.", "claim": "Voting machines switched millions of votes", "label": "refutes"},
  {"premise": "The World Health Organization declared the outbreak a public health emergency on Thursday.", "claim": "WHO declared a public health emergency", "label": "supports"},
  {"premise": "Claims that 5G towers spread the virus have been repeatedly debunked by scientists.", "claim": "5G towers spread coronavirus", "label": "refutes"},
  {"premise": "The senator spoke about agriculture policy at a town hall in Iowa.", "claim": "The senator voted against the farm bill", "label": "neutral"},
  {"premise": "Wildfires have burned more than 2 million acres in California this year, officials said.", "claim": "California wildfires burned over a million acres this year", "label": "supports"},
  {"premise": "The Eiffel Tower remains in Paris, where it was completed in 1889.", "claim": "The Eiffel Tower was moved to London", "label": "refutes"},
  {"premise": "Analysts debated the merger's likely impact on consumer prices.", "claim": "The merger was blocked by regulators", "label": "neutral"},
  {"premise": "The Supreme Court ruled 6-3 to uphold the law on Friday.", "claim": "The Supreme Court upheld the law", "label": "supports"}
]
```

Create `eval/datasets/equivalence_pairs.json`. Each row is `{a, b, equivalent, kind}`:

```json
[
  {"a": "NASA confirms water found on Mars", "b": "NASA has confirmed that water was found on Mars", "equivalent": true, "kind": "paraphrase"},
  {"a": "The Fed raised interest rates", "b": "Interest rates were raised by the Federal Reserve", "equivalent": true, "kind": "paraphrase"},
  {"a": "Drinking bleach cures COVID-19", "b": "COVID-19 can be cured by drinking bleach", "equivalent": true, "kind": "paraphrase"},
  {"a": "The Eiffel Tower is in Paris", "b": "The Eiffel Tower is located in Paris", "equivalent": true, "kind": "paraphrase"},
  {"a": "Vaccines cause autism", "b": "Vaccines do not cause autism", "equivalent": false, "kind": "negation"},
  {"a": "The bridge is open", "b": "The bridge is not open", "equivalent": false, "kind": "negation"},
  {"a": "The senator supported the bill", "b": "The senator opposed the bill", "equivalent": false, "kind": "antonym"},
  {"a": "Crime rose last year", "b": "Crime fell last year", "equivalent": false, "kind": "antonym"},
  {"a": "Biden visited Paris", "b": "Macron visited Paris", "equivalent": false, "kind": "entity_swap"},
  {"a": "Apple acquired the startup", "b": "Google acquired the startup", "equivalent": false, "kind": "entity_swap"},
  {"a": "The president visited France", "b": "The president visited Paris", "equivalent": false, "kind": "scope"},
  {"a": "All schools closed", "b": "Some schools closed", "equivalent": false, "kind": "scope"},
  {"a": "Inflation hit 3% in May", "b": "Inflation hit 8% in May", "equivalent": false, "kind": "number"},
  {"a": "The law passed in 2019", "b": "The law passed in 2021", "equivalent": false, "kind": "number"},
  {"a": "Unemployment fell to 3.5%", "b": "Unemployment fell to 3.5 percent", "equivalent": true, "kind": "paraphrase"},
  {"a": "The company laid off workers", "b": "The company hired workers", "equivalent": false, "kind": "antonym"}
]
```

- [ ] **Step 2: Write the failing tests.** Create `eval/tests/test_calibrate_nli.py`:

```python
import pytest

from app.services.nli import NLIScores
from eval.calibrate_nli import recommend, sweep_equivalence, sweep_stance


def S(e, c, n):
    return NLIScores(entailment=e, contradiction=c, neutral=n)


def test_stance_sweep_accuracy_per_threshold():
    rows = [{"label": "supports"}, {"label": "refutes"}, {"label": "neutral"}]
    scores = [S(0.7, 0.2, 0.1), S(0.1, 0.65, 0.25), S(0.3, 0.3, 0.4)]
    sweep = {r["threshold"]: r for r in sweep_stance(rows, scores, [0.6, 0.7])}
    assert sweep[0.6]["accuracy"] == 1.0
    assert sweep[0.7]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)   # refutes at 0.65 falls to neutral


def test_equivalence_sweep_counts_false_positives():
    rows = [{"a": "x", "b": "y", "equivalent": True}, {"a": "p", "b": "q", "equivalent": False}]
    # each row has (forward, backward) scores
    scores = [(S(0.9, 0.05, 0.05), S(0.85, 0.05, 0.1)), (S(0.82, 0.1, 0.08), S(0.81, 0.1, 0.09))]
    sweep = {r["threshold"]: r for r in sweep_equivalence(rows, scores, [0.8, 0.83])}
    assert sweep[0.8]["false_positives"] == 1
    assert sweep[0.83]["false_positives"] == 0
    assert sweep[0.83]["true_positives"] == 1


def test_recommend_requires_zero_false_positives():
    stance = [{"threshold": 0.6, "accuracy": 0.9}, {"threshold": 0.7, "accuracy": 0.9}]
    equiv = [{"threshold": 0.8, "false_positives": 1, "true_positives": 5},
             {"threshold": 0.85, "false_positives": 0, "true_positives": 4},
             {"threshold": 0.9, "false_positives": 0, "true_positives": 2}]
    rec = recommend(stance, equiv)
    assert rec["stance_min_prob"] == 0.7          # tie -> higher threshold
    assert rec["equivalence_min_entailment"] == 0.85


def test_recommend_returns_none_when_no_safe_equivalence_threshold():
    rec = recommend([{"threshold": 0.6, "accuracy": 0.8}],
                    [{"threshold": 0.9, "false_positives": 2, "true_positives": 3}])
    assert rec["equivalence_min_entailment"] is None


@pytest.mark.live
def test_committed_thresholds_have_zero_equivalence_false_positives():
    """Guardrail with the real model: the shipped EQUIVALENCE_MIN_ENTAILMENT must never
    call a negation/antonym/entity-swap/scope/number pair equivalent."""
    from eval.calibrate_nli import run_calibration
    report = run_calibration()
    assert report["current"]["equivalence_false_positives"] == 0
```

- [ ] **Step 3:** Run `uv run pytest eval/tests/test_calibrate_nli.py -v`. Expected: collection ERROR, `ModuleNotFoundError`.

- [ ] **Step 4: Implement.** Create `eval/calibrate_nli.py`:

```python
"""
Calibrate the provisional NLI thresholds from Plans 1-2 on labeled pair sets.
  stance:       STANCE_MIN_PROB (crossref_agent) — maximize accuracy, ties -> stricter
  equivalence:  EQUIVALENCE_MIN_ENTAILMENT (claim_equivalence) — lowest threshold with ZERO
                false positives; a missed cache hit is cheap, a wrong one is a safety failure.
Usage: uv run python eval/calibrate_nli.py   (loads the real model; prints a JSON report)
Changing a threshold is a code change in the backend, made by hand and reviewed.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (BASE_DIR, BASE_DIR / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.services.nli import NLIScores, classify_pairs  # noqa: E402

DATASETS = BASE_DIR / "eval" / "datasets"
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def _stance_label(s: NLIScores, t: float) -> str:
    if s.entailment >= t and s.entailment > s.contradiction:
        return "supports"
    if s.contradiction >= t and s.contradiction > s.entailment:
        return "refutes"
    return "neutral"


def sweep_stance(rows: List[Dict[str, Any]], scores: Sequence[NLIScores], thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    out = []
    for t in thresholds:
        correct = sum(_stance_label(s, t) == r["label"] for r, s in zip(rows, scores))
        out.append({"threshold": t, "accuracy": round(correct / len(rows), 4)})
    return out


def sweep_equivalence(rows: List[Dict[str, Any]], scores: Sequence[Tuple[NLIScores, NLIScores]], thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    out = []
    for t in thresholds:
        tp = fp = 0
        for r, (fwd, bwd) in zip(rows, scores):
            predicted = fwd.entailment >= t and bwd.entailment >= t
            if predicted and r["equivalent"]:
                tp += 1
            elif predicted and not r["equivalent"]:
                fp += 1
        out.append({"threshold": t, "true_positives": tp, "false_positives": fp})
    return out


def recommend(stance_sweep: List[Dict[str, Any]], equivalence_sweep: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    best_stance = max(stance_sweep, key=lambda r: (r["accuracy"], r["threshold"]))["threshold"]
    safe = [r for r in equivalence_sweep if r["false_positives"] == 0]
    best_equiv = min(safe, key=lambda r: r["threshold"])["threshold"] if safe else None
    return {"stance_min_prob": best_stance, "equivalence_min_entailment": best_equiv}


def run_calibration() -> Dict[str, Any]:
    from app.agents.crossref_agent import STANCE_MIN_PROB
    from app.services.claim_equivalence import EQUIVALENCE_MIN_ENTAILMENT, numbers_match

    stance_rows = json.loads((DATASETS / "stance_pairs.json").read_text(encoding="utf-8"))
    equiv_rows = json.loads((DATASETS / "equivalence_pairs.json").read_text(encoding="utf-8"))

    stance_scores = classify_pairs([(r["premise"], r["claim"]) for r in stance_rows])
    # The number guard runs before NLI in production; mirror that here.
    nli_rows = [r for r in equiv_rows if numbers_match(r["a"], r["b"])]
    flat = classify_pairs([p for r in nli_rows for p in ((r["b"], r["a"]), (r["a"], r["b"]))])
    equiv_scores = list(zip(flat[0::2], flat[1::2]))

    stance_sweep = sweep_stance(stance_rows, stance_scores, THRESHOLDS)
    equiv_sweep = sweep_equivalence(nli_rows, equiv_scores, THRESHOLDS)
    current_equiv = sweep_equivalence(nli_rows, equiv_scores, [EQUIVALENCE_MIN_ENTAILMENT])[0]
    return {
        "stance_sweep": stance_sweep,
        "equivalence_sweep": equiv_sweep,
        "recommendation": recommend(stance_sweep, equiv_sweep),
        "current": {
            "stance_min_prob": STANCE_MIN_PROB,
            "equivalence_min_entailment": EQUIVALENCE_MIN_ENTAILMENT,
            "equivalence_false_positives": current_equiv["false_positives"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_calibration(), indent=2))
```

- [ ] **Step 5:** Run `uv run pytest eval/tests/test_calibrate_nli.py -v`. Expected: 4 passed, 1 deselected.
- [ ] **Step 6: Calibrate for real.** Run `uv run python eval/calibrate_nli.py` and `uv run pytest eval/tests/test_calibrate_nli.py -m live -v`. Paste the JSON report into the task report.
  - If `current.equivalence_false_positives > 0`: **stop**. Raise `EQUIVALENCE_MIN_ENTAILMENT` in `backend/app/services/claim_equivalence.py` to the recommended value, re-run the live test, and flag it for review. This is a safety fix.
  - If the recommended `stance_min_prob` differs from 0.6 by ≥ 0.1, propose the change to the user. Do not apply it silently: it moves every V_consensus score.
- [ ] **Step 7: Commit**

```bash
git add eval/datasets/stance_pairs.json eval/datasets/equivalence_pairs.json eval/calibrate_nli.py eval/tests/test_calibrate_nli.py
git commit -m "feat(eval): calibrate NLI stance/equivalence thresholds with zero-FP equivalence rule"
```

---

### Task 7: Spec updates (metric definitions)

**Files:** `documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md` (Part 2), `documents/VeriFact_AI_PRD_v2.md` §10.

- [ ] **Step 1: PRD §10.** Replace the two rows "Bucketed accuracy" and "MAE of C_total" with:

```markdown
| Selective bucketed accuracy (answered claims: evidence_status ≠ unverified) | ≥ 70% on the curated set | Eval harness `segments.selective` |
| Coverage (share of verifiable claims answered, not "unverified") | Reported; no target until a live baseline exists | Eval harness `segments.selective.coverage` |
| MAE of C_total, per dataset | ≤ 0.20 on curated `fact_checked`; others reported | Eval harness `segments.by_dataset` / `by_evidence_status` |
```

Add this line under the table: "Metrics are never blended across datasets. LIAR is evaluated
leave-source-out (politifact.com evidence excluded), because its labels *are* PolitiFact
verdicts. FEVER (Wikipedia-derived) is reported as its own segment."

- [ ] **Step 2: Eval Framework Part 2.** Add a subsection "Segmentation, leave-source-out and replay" that states:
  - the segment definitions and the answered set (`fact_checked`, `disputed`, `corroborated`)
  - the `origin_domain` rule
  - `--replay record|replay|live`, and that a run with any replay miss is `INVALID_RUN`
  - the LIAR/FEVER label maps (moved here verbatim from the dataset table)
  - that the "Ad Fontes" alternative in the bias-leakage line is removed: bias labels come from AllSides only (CLAUDE.md §4.6). Delete "/Ad Fontes" from that line.
- [ ] **Step 3: Commit**

```bash
git add documents/VeriFact_AI_OpenSource_Stack_and_Eval_Framework.md documents/VeriFact_AI_PRD_v2.md
git commit -m "docs: segmented eval targets, leave-source-out and replay (C3, C4)"
```

---

### Task 8: Review gate (mandatory)

- [ ] **Step 1:** Run `uv run pytest eval/tests/ -q` and (from `backend/`) `uv run pytest -q`. Record the counts.
- [ ] **Step 2:** Run the review gate as `README.md` §"Review gate" describes (`/review` + `eval-guardian`). Ask specifically:
  - Can replay ever silently serve live data?
  - Does exclusion apply on every evidence path, including Layer-1 Fact Check?
  - Is a replay miss always an invalid run?
  - Does anything blend segments back into a headline number?
- [ ] **Step 3:** Fix blocking findings, writing a test first. Report to the user:
  - test counts
  - calibration JSON
  - curated record/replay summaries
  - deferrals (for example, a LIAR/FEVER recording needs the user's downloads and quota)
