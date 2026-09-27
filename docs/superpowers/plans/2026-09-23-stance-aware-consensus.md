# Stance-Aware Consensus & Evidence Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Antigravity) or superpowers:subagent-driven-development (Claude Code) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read `docs/superpowers/plans/README.md` first (run order, review-gate mapping).

**Goal:** Make `V_consensus` count only reputable sources that *support* the claim. Add a
refutation signal that pulls disputed claims into the red band. Every response carries an
explicit `evidence_status`.

**Architecture:** A new `app/services/nli.py` loads one local NLI cross-encoder at startup.
The Cross-Ref agent scores every source with it (premise = article title + snippet,
hypothesis = claim). A pure `compute_consensus()` then turns the stances into
`(v_consensus, v_refute)`. The scoring engine stays pure and gets:
- a new `v_refute` input,
- a new capping Rule 4 ("disputed"),
- a pure `determine_evidence_status()`.

The PRD formula `C_total = w1·S_fact + w2·(1−B_ling) + w3·V_consensus` and its weights do
not change.

**Tech Stack:** Python 3.12, FastAPI, sentence-transformers `CrossEncoder`
(`cross-encoder/nli-deberta-v3-small`, Apache-2.0), pytest + anyio, Supabase Postgres (SQL
migration applied by hand).

**Spec:** `docs/superpowers/specs/2026-09-23-prd-tdd-review.md` (findings C1, decisions D1 and
D2). Also read `CLAUDE.md` §4–§5, `documents/VeriFact_AI_TDD.md` §4.3 and §7, and
`documents/VeriFact_AI_PRD_v2.md` §6.

## Global Constraints

- Python **3.12+**. Dependencies are managed with **`uv`** only (`uv add`), never bare `pip`
  and never by hand-editing `pyproject.toml` deps.
- Heavy models load **once at startup** in `app/main.py` `lifespan`, never per request (CLAUDE.md §4.7).
- Scoring engine stays **pure and deterministic**: no I/O, no model calls, no randomness. Invariants: `Σw = 1.0`, `w1 ≥ 0.4`, output clamped `[0,1]` (CLAUDE.md §4.4).
- Worker agents keep their **8-second timeout** and degrade to the documented neutral value (`v_consensus = 0.0`, `degraded=True`) on failure — never crash, never fabricate (CLAUDE.md §4.3).
- **No LLM in the stance path.** Bias labels still come **only** from AllSides (CLAUDE.md §4.6).
- The existing capping rules 1–3 and their thresholds stay unchanged.
- Run backend commands from `backend/`. Unit tests: `uv run pytest`. Eval tests (from repo root): `uv run pytest eval/tests/`. Live tests are opt-in with `-m live`.
- Every "Run:" step must actually be executed and its output recorded. Antigravity runs them directly. In Claude Code, the main-thread shell on this machine cannot run `uv`, so use a dispatched subagent there.
- Another agent (Antigravity) edits this repo in parallel. Before each task, run `git status` and re-read any file you are about to edit.
- Every commit message ends with the executor's attribution line. The commit blocks below show Claude Code's (`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`); Antigravity replaces it with its own or omits it.

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app/services/nli.py` | Create | Load the NLI cross-encoder once. `classify_pairs(premise, hypothesis) → NLIScores`. Reused by Plan 2. |
| `backend/app/config.py` | Modify | Add `nli_model_name` and `nli_max_length` settings |
| `backend/app/main.py` | Modify | Preload the NLI model in `lifespan` |
| `backend/app/agents/crossref_agent.py` | Modify | Per-source stance, `label_stance()`, `compute_consensus()`, `v_refute_score`, and degradation when the NLI model fails |
| `backend/app/scoring/confidence.py` | Modify | Rule 4 (disputed cap), `is_disputed()`, `determine_evidence_status()`, `v_refute` feeding the confidence level |
| `backend/app/scoring/engine.py` | Modify | Accept `v_refute`; return `evidence_status` |
| `backend/app/scoring/__init__.py` | Modify | Export `determine_evidence_status` |
| `backend/app/agents/orchestrator.py` | Modify | Pass `v_refute`; add `evidence_status`, `v_refute` and `stance` to the response; persist them |
| `backend/app/services/cache.py` | Modify | Persist `v_refute`/`evidence_status`; legacy rows (NULL status) become cache misses |
| `backend/app/api/schemas.py` | Modify | Response fields `v_refute`, `evidence_status`, `stance` |
| `backend/app/db/migrations/03_stance_aware_consensus.sql` | Create | New columns plus the `match_claims` return shape |
| `backend/tests/test_nli.py` | Create | NLI service unit tests plus one live test |
| `backend/tests/test_crossref_agent.py` | Modify | NLI stub fixture and new stance tests |
| `backend/tests/test_scoring_engine.py` | Modify | Rule 4 and evidence-status tests |
| `backend/tests/test_orchestrator.py` | Modify | Disputed-claim pipeline test and the new fields |
| `backend/tests/test_cache.py` | Modify | Legacy-row miss test; persistence of the new fields |
| `eval/tests/test_guardrails.py` | Modify | Guardrail: disputed claims never exceed the cap without a verdict |
| `documents/VeriFact_AI_PRD_v2.md`, `documents/VeriFact_AI_TDD.md`, `CLAUDE.md` | Modify | Spec updates (the docs must match the code) |

---

### Task 0: Pre-flight

**Files:** none

- [ ] **Step 1: Confirm a clean base and branch**

Run (repo root): `git status --short`
Expected: empty output. If there are uncommitted changes, **stop** and ask the user whether
to commit them first. They may be Antigravity's in-flight work (see `review-fixes-handoff.md`).

- [ ] **Step 2: Create the branch**

```bash
git checkout -b fix/stance-aware-consensus
```

- [ ] **Step 3: Baseline test run**

Run (from `backend/`): `uv run pytest -q`
Expected: all pass. Record the count. Any pre-existing failure must be reported to the user
before continuing.

---

### Task 1: NLI service (model loaded once, typed scores)

**Files:**
- Create: `backend/app/services/nli.py`
- Modify: `backend/app/config.py` (add the settings after `embedding_dimension`)
- Modify: `backend/app/main.py` (add a preload step 4 in `lifespan`)
- Test: `backend/tests/test_nli.py`

**Interfaces:**
- Produces:
  - `NLIScores` (frozen dataclass: `entailment: float`, `contradiction: float`, `neutral: float`; the three sum to 1.0)
  - `classify_pairs(pairs: Sequence[tuple[str, str]]) -> list[NLIScores]` (each pair is `(premise, hypothesis)`; synchronous and CPU-bound, so callers wrap it in `asyncio.to_thread`)
  - `get_nli_model()`
  - `settings.nli_model_name: str`
  - `settings.nli_max_length: int`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_nli.py`:

```python
import pytest

from app.services import nli
from app.services.nli import NLIScores, classify_pairs, _resolve_label_index


class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeHFModel:
    def __init__(self, id2label):
        self.config = _FakeConfig(id2label)


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: returns fixed logits."""

    def __init__(self, rows, id2label):
        self.rows = rows
        self.model = _FakeHFModel(id2label)
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.rows


@pytest.fixture
def fake_model(monkeypatch):
    def _install(rows, id2label=None):
        if id2label is None:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        model = _FakeCrossEncoder(rows, id2label)
        monkeypatch.setattr(nli, "_model", model)
        monkeypatch.setattr(nli, "_label_index", _resolve_label_index(model))
        return model
    return _install


def test_empty_pairs_do_not_load_model(monkeypatch):
    def _boom():
        raise AssertionError("model must not load for empty input")
    monkeypatch.setattr(nli, "get_nli_model", _boom)
    assert classify_pairs([]) == []


def test_logits_are_softmaxed_and_mapped_by_label(fake_model):
    fake_model([[0.0, 6.0, 0.0], [6.0, 0.0, 0.0]])
    scores = classify_pairs([("p1", "h1"), ("p2", "h2")])
    assert scores[0].entailment > 0.99
    assert scores[1].contradiction > 0.99
    for s in scores:
        assert isinstance(s, NLIScores)
        assert abs(s.entailment + s.contradiction + s.neutral - 1.0) < 1e-6


def test_label_order_follows_model_config(fake_model):
    # A model whose classification head is ordered differently must still map correctly.
    fake_model([[6.0, 0.0, 0.0]], id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
    [s] = classify_pairs([("p", "h")])
    assert s.entailment > 0.99


def test_generic_labels_fall_back_to_documented_order(fake_model):
    # Documented order for cross-encoder/nli-deberta-v3-small: contradiction, entailment, neutral.
    fake_model([[0.0, 6.0, 0.0]], id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    [s] = classify_pairs([("p", "h")])
    assert s.entailment > 0.99


def test_row_count_mismatch_raises(fake_model):
    fake_model([[0.0, 6.0, 0.0]])
    with pytest.raises(RuntimeError):
        classify_pairs([("p1", "h1"), ("p2", "h2")])


@pytest.mark.live
def test_real_model_entailment_and_contradiction():
    """Loads the real cross-encoder (downloads ~0.5GB on first run)."""
    supports, refutes = classify_pairs([
        ("A man is eating pizza in the park.", "A man eats something."),
        ("The bridge was closed to all traffic on Monday.", "The bridge was open on Monday."),
    ])
    assert supports.entailment > supports.contradiction
    assert refutes.contradiction > refutes.entailment
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nli.py -v`
Expected: ERROR at collection. `ModuleNotFoundError: No module named 'app.services.nli'`.

- [ ] **Step 3: Add the settings**

In `backend/app/config.py`, directly after the `embedding_dimension: int = 384` line, add:

```python

    # Local NLI cross-encoder for source stance (Cross-Ref agent) and claim equivalence
    # (semantic cache). Spec: docs/superpowers/specs/2026-09-23-prd-tdd-review.md D1.
    # Apache-2.0, ~0.1B params, CPU. Loaded once in main.py lifespan.
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"

    # Token budget per (premise, hypothesis) pair; longer snippets are truncated by the tokenizer.
    nli_max_length: int = 256
```

- [ ] **Step 4: Write the service**

Create `backend/app/services/nli.py`:

```python
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# Documented label order of cross-encoder/nli-deberta-v3-small. Used only when the model
# config carries generic labels (LABEL_0..2) instead of named ones.
_FALLBACK_LABELS = ("contradiction", "entailment", "neutral")
_EXPECTED_LABELS = frozenset(_FALLBACK_LABELS)

# Lazy loaded module-level CrossEncoder instance (preloaded once during startup lifespan)
_model = None
_label_index: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class NLIScores:
    """Softmax probabilities for one (premise, hypothesis) pair; they sum to 1.0."""
    entailment: float
    contradiction: float
    neutral: float


def _resolve_label_index(model) -> Dict[str, int]:
    """Map label name -> logit column, trusting the model config over the documented order."""
    id2label = None
    try:
        id2label = model.model.config.id2label
    except AttributeError:
        logger.warning("NLI model exposes no config.id2label; using documented label order.")

    labels = [str(id2label[i]).lower() for i in range(len(id2label))] if id2label else []
    if set(labels) != _EXPECTED_LABELS:
        logger.warning(f"NLI model labels {labels} are not named NLI labels; using documented order.")
        labels = list(_FALLBACK_LABELS)
    return {label: i for i, label in enumerate(labels)}


def get_nli_model():
    """
    Retrieve or initialize the NLI CrossEncoder.
    Loads the model once and caches it at the module level.
    """
    global _model, _label_index
    if _model is None:
        logger.info(f"Initializing NLI CrossEncoder: {settings.nli_model_name}")
        try:
            from sentence_transformers import CrossEncoder
            from torch import nn
            # Identity activation: predict() returns raw logits and we softmax them ourselves,
            # so the probabilities do not depend on the library's default activation.
            model = CrossEncoder(
                settings.nli_model_name,
                max_length=settings.nli_max_length,
                activation_fn=nn.Identity(),
            )
        except Exception as exc:
            logger.error(f"Failed to load NLI model {settings.nli_model_name}: {exc}")
            raise
        _label_index = _resolve_label_index(model)
        _model = model
    return _model


def _softmax(row: Sequence[float]) -> List[float]:
    peak = max(row)
    exps = [math.exp(x - peak) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def classify_pairs(pairs: Sequence[Tuple[str, str]]) -> List[NLIScores]:
    """
    Score each (premise, hypothesis) pair with the NLI model.
    CPU-bound and synchronous: call it via asyncio.to_thread from async code.

    Raises:
        RuntimeError: the model returned a different number of rows than pairs given.
        Exception: model load or inference failure (callers degrade, they never guess).
    """
    if not pairs:
        return []

    model = get_nli_model()
    raw = model.predict(list(pairs), convert_to_numpy=True, show_progress_bar=False)
    rows = [[float(x) for x in row] for row in raw]
    if len(rows) != len(pairs):
        raise RuntimeError(f"NLI model returned {len(rows)} rows for {len(pairs)} pairs")

    idx = _label_index
    results = []
    for row in rows:
        probs = _softmax(row)
        results.append(NLIScores(
            entailment=probs[idx["entailment"]],
            contradiction=probs[idx["contradiction"]],
            neutral=probs[idx["neutral"]],
        ))
    return results
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_nli.py -v`
Expected: 5 passed, 1 deselected (the live test).

- [ ] **Step 6: Run the live test against the real model**

Run: `uv run pytest tests/test_nli.py -m live -v`
Expected: 1 passed. If loading fails with an `ImportError` that mentions `sentencepiece` or
`protobuf` (the DeBERTa-v3 tokenizer needs them when no fast tokenizer is shipped), run
`uv add sentencepiece protobuf` and re-run. If the test fails on the assertions, **stop and
report**: it means the label mapping is wrong, and nothing downstream can be trusted.

- [ ] **Step 7: Preload at startup**

In `backend/app/main.py`, add the import next to the other service imports:

```python
from app.services.nli import get_nli_model
```

Then, in `lifespan`, directly after the "3. Preload SentenceTransformer embedding model"
block and before `logger.info("Startup complete. Pipeline is ready.")`, add:

```python
    # 4. Preload NLI cross-encoder (source stance + cache equivalence)
    logger.info("Preloading NLI model...")
    try:
        get_nli_model()
    except Exception as exc:
        logger.warning(f"NLI model preload warning: {exc}")
```

- [ ] **Step 8: Run the full unit suite**

Run: `uv run pytest -q`
Expected: the baseline count plus 5, all passing.

- [ ] **Step 9: Commit**

```bash
git add backend/app/services/nli.py backend/app/config.py backend/app/main.py backend/tests/test_nli.py backend/pyproject.toml backend/uv.lock
git commit -m "feat(nli): add local NLI cross-encoder service for stance and claim equivalence (C1, D1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`pyproject.toml`/`uv.lock` only change if Step 6 needed `uv add`. If `git add` reports them
unchanged, that's fine.)

---

### Task 2: Stance-aware consensus in the Cross-Ref agent

**Files:**
- Modify: `backend/app/agents/crossref_agent.py`
- Test: `backend/tests/test_crossref_agent.py`

**Interfaces:**
- Consumes: `NLIScores`, `classify_pairs` (Task 1); `CLAIMREVIEW_DOMAINS` from `app.services.search_api`.
- Produces:
  - `CrossRefResult.v_refute_score: float = 0.0`
  - each `sources_found` item gains `"stance": "supports" | "refutes" | "neutral"` and `"stance_confidence": float`
  - `label_stance(scores: NLIScores) -> tuple[str, float]`
  - `compute_consensus(reputable_sources: list[dict]) -> tuple[float, float]`, returning `(v_consensus, v_refute)`
  - constant `STANCE_MIN_PROB = 0.6`

- [ ] **Step 1: Add the NLI stub fixture and write the failing tests**

In `backend/tests/test_crossref_agent.py`:

1. Update the imports at the top to:

```python
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.crossref_agent import (
    run, CrossRefResult, make_neutral_crossref, label_stance, compute_consensus, STANCE_MIN_PROB
)
from app.services.errors import EvidenceAPIError
from app.services.bias_lookup import load_bias_ratings
from app.services.nli import NLIScores
from app.config import settings
```

2. Directly after the existing `setup_bias` fixture, add:

```python
SUPPORT = NLIScores(entailment=0.9, contradiction=0.05, neutral=0.05)
REFUTE = NLIScores(entailment=0.05, contradiction=0.9, neutral=0.05)
NEUTRAL = NLIScores(entailment=0.2, contradiction=0.2, neutral=0.6)


@pytest.fixture(autouse=True)
def stub_nli():
    """Default: every source supports the claim. Tests override side_effect as needed."""
    def _all_support(pairs):
        return [SUPPORT for _ in pairs]
    with patch("app.agents.crossref_agent.classify_pairs", side_effect=_all_support) as mock_nli:
        yield mock_nli
```

(Existing tests then keep their current numbers. For example,
`test_factual_claim_cross_reference` still expects `0.54` =
(0.92 + 0.96 + 0.82) / 5, because all three sources support the claim.)

3. At the end of the file, add:

```python
def test_label_stance_thresholds():
    assert label_stance(SUPPORT) == ("supports", 0.9)
    assert label_stance(REFUTE) == ("refutes", 0.9)
    assert label_stance(NEUTRAL)[0] == "neutral"
    # Below STANCE_MIN_PROB the model is not confident enough to take a side
    weak = NLIScores(entailment=STANCE_MIN_PROB - 0.01, contradiction=0.0, neutral=1 - (STANCE_MIN_PROB - 0.01))
    assert label_stance(weak)[0] == "neutral"


def test_compute_consensus_splits_support_and_refute():
    sources = [
        {"relevance_score": 0.9, "stance": "supports"},
        {"relevance_score": 0.8, "stance": "supports"},
        {"relevance_score": 0.7, "stance": "refutes"},
        {"relevance_score": 1.0, "stance": "neutral"},
    ]
    assert compute_consensus(sources) == (0.34, 0.14)


def test_compute_consensus_saturates_at_one():
    sources = [{"relevance_score": 1.0, "stance": "supports"} for _ in range(7)]
    assert compute_consensus(sources) == (1.0, 0.0)


@pytest.mark.anyio
async def test_refuting_coverage_does_not_raise_consensus(stub_nli):
    """C1: reputable outlets debunking a claim must NOT count as corroboration."""
    stub_nli.side_effect = lambda pairs: [REFUTE for _ in pairs]
    mock_results = [
        {"url": "https://www.reuters.com/fact-check/a", "title": "Claim is false", "content": "No evidence.", "score": 0.9},
        {"url": "https://www.foxnews.com/health/b", "title": "Experts reject claim", "content": "Debunked.", "score": 0.8},
        {"url": "https://www.nytimes.com/c", "title": "Viral post misleads", "content": "Untrue.", "score": 0.7},
    ]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("Drinking bleach cures the flu")
    assert result.degraded is False
    assert result.v_consensus_score == 0.0
    assert result.v_refute_score == 0.48  # (0.9 + 0.8 + 0.7) / 5
    assert all(s["stance"] == "refutes" for s in result.sources_found)


@pytest.mark.anyio
async def test_premise_is_source_text_and_hypothesis_is_claim(stub_nli):
    mock_results = [{"url": "https://www.reuters.com/x", "title": "T", "content": "Body text", "score": 0.9}]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        await run("NASA announces Mars landing")
    pairs = stub_nli.call_args[0][0]
    assert pairs == [("T. Body text", "NASA announces Mars landing")]


@pytest.mark.anyio
async def test_claimreview_domains_excluded_from_consensus(stub_nli):
    """Fact-check sites restate the claim they debunk; they belong to the Fact-Check agent."""
    mock_results = [
        {"url": "https://www.politifact.com/factchecks/x", "title": "Claim: X", "content": "Rating: False", "score": 0.95},
    ]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("Senator X voted against the bill")
    assert len(result.sources_found) == 1          # still shown to the user
    assert result.v_consensus_score == 0.0         # but never counted as corroboration
    assert result.v_refute_score == 0.0


@pytest.mark.anyio
async def test_nli_failure_degrades_instead_of_falling_back(stub_nli):
    """Without stance, coverage cannot be read as agreement: degrade, never use the old count."""
    stub_nli.side_effect = RuntimeError("model file missing")
    mock_results = [{"url": "https://www.reuters.com/x", "title": "T", "content": "C", "score": 0.9}]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("NASA announces Mars landing")
    assert result.degraded is True
    assert result.v_consensus_score == 0.0
    assert result.v_refute_score == 0.0
    assert "Stance model unavailable" in result.error


@pytest.mark.anyio
async def test_neutral_degraded_result_has_zero_refute():
    assert make_neutral_crossref(degraded=True).v_refute_score == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_crossref_agent.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'label_stance'`.

- [ ] **Step 3: Implement**

In `backend/app/agents/crossref_agent.py`:

1. Replace the import block (lines 1–12) with:

```python
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from app.config import settings
from app.agents.linguistic_agent import get_spacy_nlp
from app.services.search_api import search_web, CLAIMREVIEW_DOMAINS
from app.services.errors import EvidenceAPIError
from app.services.bias_lookup import get_bias_label, get_outlet_name
from app.services.nli import NLIScores, classify_pairs
from app.utils.errors import AgentErrorDetails, classify_provider_error
from app.utils.text_processing import extract_domain
```

2. Directly after `TIMEOUT = 8.0`, add:

```python

# C1: a source counts as supporting/refuting only when the NLI model is at least this sure;
# anything less is "neutral" and moves neither score.
# Track 3.1: calibrate STANCE_MIN_PROB against the eval stance fixtures (Plan 3).
STANCE_MIN_PROB = 0.6

# Premise = title + snippet, truncated before tokenization (the model also truncates at
# settings.nli_max_length tokens).
PREMISE_MAX_CHARS = 1000

# Fact-check sites belong to the Fact-Check agent. Their snippets restate the claim they
# debunk ("Claim: X. Rating: False"), which NLI reads as entailment, so counting them here
# would both double count and invert their stance.
_CLAIMREVIEW_DOMAIN_SET = frozenset(CLAIMREVIEW_DOMAINS)
```

3. In the `CrossRefResult` dataclass, directly after `v_consensus_score: float`, add:

```python
    v_refute_score: float = 0.0               # 0.0 to 1.0: relevance-weighted refuting reputable coverage
```

4. Replace `make_neutral_crossref` with a version that also sets the refute score:

```python
def make_neutral_crossref(
    degraded: bool = False,
    error: Optional[str] = None,
    error_details: Optional[AgentErrorDetails] = None
) -> CrossRefResult:
    """
    Factory creating a fresh neutral CrossRefResult.
    v_consensus_score and v_refute_score are strictly 0.0 on degradation or empty search.
    """
    return CrossRefResult(
        entities_extracted=[],
        sources_found=[],
        coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0,
        v_refute_score=0.0,
        degraded=degraded,
        error=error,
        error_details=error_details
    )
```

5. Directly after `_clean_publisher_name`, add the pure helpers:

```python
def label_stance(scores: NLIScores) -> Tuple[str, float]:
    """
    Map NLI probabilities (premise = source text, hypothesis = claim) to a stance label
    and the probability behind it.
    """
    if scores.entailment >= STANCE_MIN_PROB and scores.entailment > scores.contradiction:
        return "supports", round(scores.entailment, 3)
    if scores.contradiction >= STANCE_MIN_PROB and scores.contradiction > scores.entailment:
        return "refutes", round(scores.contradiction, 3)
    return "neutral", round(scores.neutral, 3)


def compute_consensus(reputable_sources: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Stance-aware consensus (C1). Returns (v_consensus, v_refute), each in [0, 1].
    Each supporting (refuting) reputable source adds its relevance score; five fully relevant
    sources saturate the scale, matching the PRD's "5+ reputable sources = 1.0".
    Neutral sources add to neither side.
    """
    support = sum(s["relevance_score"] for s in reputable_sources if s.get("stance") == "supports")
    refute = sum(s["relevance_score"] for s in reputable_sources if s.get("stance") == "refutes")
    return round(min(support / 5.0, 1.0), 3), round(min(refute / 5.0, 1.0), 3)


def _premise(article: Dict[str, Any]) -> str:
    title = (article.get("title") or "").strip()
    content = (article.get("content") or "").strip()
    return f"{title}. {content}".strip(". ")[:PREMISE_MAX_CHARS]
```

6. In `_analyze`, replace everything from the comment `# 5. Bias lookup and coverage
distribution calculation` through the `return CrossRefResult(...)` at the end of the
function with:

```python
    # 5. Stance per source (C1): premise = article text, hypothesis = claim.
    articles = list(domain_to_best_article.items())
    pairs = [(_premise(article), claim) for _, article in articles]
    try:
        nli_scores = await asyncio.to_thread(classify_pairs, pairs)
    except Exception as exc:
        # Without stance, coverage cannot be read as agreement: degrade, never fall back to
        # the coverage-only count (that is the bug C1 fixes).
        logger.error(f"Stance model failed during cross-reference: {exc}")
        return make_neutral_crossref(
            degraded=True,
            error=f"Stance model unavailable: {exc}",
            error_details=classify_provider_error(exc, provider="nli"),
        )

    # 6. Bias lookup and coverage distribution calculation
    coverage_distribution = {
        "Left": 0,
        "Lean Left": 0,
        "Center": 0,
        "Lean Right": 0,
        "Right": 0
    }

    sources_found = []
    reputable_sources = []

    for (domain, article), scores in zip(articles, nli_scores):
        bias_label = get_bias_label(domain)
        known_outlet = get_outlet_name(domain)
        publisher = known_outlet if known_outlet else _clean_publisher_name(domain)

        # Map search relevance score to 0.0 - 1.0 (Tavily score is typically 0.0 - 1.0)
        relevance_score = float(article.get("score", 0.5))
        stance, stance_confidence = label_stance(scores)

        source_dict = {
            "url": article.get("url"),
            "title": article.get("title", ""),
            "publisher": publisher,
            "bias_label": bias_label,
            "relevance_score": round(relevance_score, 2),
            "stance": stance,
            "stance_confidence": stance_confidence,
        }
        sources_found.append(source_dict)

        # Reputable check: AllSides rated outlets or institutional/primary domains per B4/D3.
        # Wire services (reuters/apnews/bbc) are intentionally NOT listed: they are AllSides-rated,
        # so the first clause already covers them.
        # Track 3.1: tighten institutional-domain trust (a .edu personal page is not a source)
        is_reputable = (
            bias_label in coverage_distribution
            or domain.endswith(".gov")
            or domain.endswith(".edu")
            # nasa.gov is intentionally NOT listed: `.gov` above already covers it.
            or domain in {"who.int"}
        )

        if bias_label in coverage_distribution:
            coverage_distribution[bias_label] += 1

        # Claim-review sites are shown but never counted (see _CLAIMREVIEW_DOMAIN_SET).
        if is_reputable and domain not in _CLAIMREVIEW_DOMAIN_SET:
            reputable_sources.append(source_dict)

    # Sort sources by relevance score descending
    sources_found.sort(key=lambda s: s["relevance_score"], reverse=True)

    # 7. Stance-aware consensus
    v_consensus_score, v_refute_score = compute_consensus(reputable_sources)

    return CrossRefResult(
        entities_extracted=entities,
        sources_found=sources_found,
        coverage_distribution=coverage_distribution,
        v_consensus_score=v_consensus_score,
        v_refute_score=v_refute_score,
        degraded=False,
        error=None
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_crossref_agent.py -v`
Expected: all pass (the existing tests plus 8 new ones; the live test is deselected). If
`test_factual_claim_cross_reference` fails on `0.54`, check that the `stub_nli` fixture is
autouse. Do **not** change the expected value.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/agents/crossref_agent.py backend/tests/test_crossref_agent.py
git commit -m "fix(crossref): count only supporting sources in V_consensus; add v_refute (C1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Scoring — Rule 4 (disputed cap) and evidence status

**Files:**
- Modify: `backend/app/scoring/confidence.py`
- Modify: `backend/app/scoring/engine.py`
- Modify: `backend/app/scoring/__init__.py`
- Test: `backend/tests/test_scoring_engine.py`
- Test: `eval/tests/test_guardrails.py`

**Interfaces:**
- Consumes: nothing new. This task is pure and independent of Tasks 1–2.
- Produces:
  - `ScoringEngine.compute(..., v_refute: float = 0.0)`
  - `ScoringResult.evidence_status: str` (one of `"fact_checked" | "disputed" | "corroborated" | "unverified"`)
  - `ScoreComponents.v_refute: float = 0.0`
  - `determine_evidence_status(fact_check_found: bool, verdict_found: bool, v_consensus: float, v_refute: float) -> str`
  - `is_disputed(v_consensus: float, v_refute: float, verdict_found: bool) -> bool`
  - constants `CORROBORATED_MIN = 0.3`, `DISPUTED_REFUTE_MIN = 0.3`, `DISPUTED_CAP = 0.33`

- [ ] **Step 1: Write the failing unit tests**

Append to `backend/tests/test_scoring_engine.py`:

```python
from app.scoring import determine_evidence_status
from app.scoring.confidence import DISPUTED_CAP


def test_disputed_claim_capped_into_red_band():
    """Rule 4: refuting reputable coverage and no definitive verdict -> capped at 0.33."""
    engine = ScoringEngine()
    # raw = 0.5*0.5 + 0.2*(1-0.0) + 0.3*0.0 = 0.45 -> capped to 0.33
    result = engine.compute(
        s_fact=0.5, b_ling=0.0, v_consensus=0.0, v_refute=0.6,
        agents_succeeded=3, fact_check_found=False, verdict_found=False,
    )
    assert result.c_total == DISPUTED_CAP
    assert result.evidence_status == "disputed"
    assert "Disputed: reputable sources contradict this claim" in result.warnings
    # Rule 2's "insufficient sources" warning is suppressed: Rule 4 is the accurate one
    assert not any(w.startswith("Low confidence") for w in result.warnings)
    assert result.confidence_level == "medium"
    assert result.components.v_refute == 0.6


def test_definitive_verdict_overrides_dispute():
    engine = ScoringEngine()
    # raw = 0.5*1.0 + 0.2*1.0 + 0.3*0.0 = 0.70; no Rule 4 because verdict_found
    result = engine.compute(
        s_fact=1.0, b_ling=0.0, v_consensus=0.0, v_refute=0.8,
        agents_succeeded=3, fact_check_found=True, verdict_found=True,
    )
    assert result.c_total == 0.7
    assert result.evidence_status == "fact_checked"
    assert not any(w.startswith("Disputed") for w in result.warnings)


def test_support_outweighing_refute_is_corroborated_not_disputed():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5, b_ling=0.2, v_consensus=0.5, v_refute=0.4,
        agents_succeeded=3, fact_check_found=False, verdict_found=False,
    )
    assert result.evidence_status == "corroborated"
    assert not any(w.startswith("Disputed") for w in result.warnings)
    # raw = 0.25 + 0.16 + 0.15 = 0.56, no cap applies
    assert result.c_total == 0.56


def test_refute_below_threshold_is_not_disputed():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5, b_ling=0.2, v_consensus=0.1, v_refute=0.2,
        agents_succeeded=3, fact_check_found=False, verdict_found=False,
    )
    assert result.evidence_status == "unverified"
    assert "Low confidence: no fact-checks found and insufficient corroborating sources" in result.warnings


def test_v_refute_is_clamped():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5, b_ling=0.0, v_consensus=0.0, v_refute=1.7,
        agents_succeeded=3,
    )
    assert result.components.v_refute == 1.0


def test_default_v_refute_keeps_existing_behaviour():
    """Callers that don't pass v_refute get exactly the pre-C1 result."""
    engine = ScoringEngine()
    result = engine.compute(s_fact=0.8, b_ling=0.1, v_consensus=0.7, agents_succeeded=3,
                            fact_check_found=True, verdict_found=True)
    assert result.c_total == 0.79
    assert result.evidence_status == "fact_checked"


@pytest.mark.parametrize("fact_check_found,verdict_found,v_consensus,v_refute,expected", [
    (True,  True,  0.0, 0.9, "fact_checked"),   # definitive verdict wins over everything
    (True,  False, 0.0, 0.5, "disputed"),       # "Half True" match + refuting coverage
    (True,  False, 0.0, 0.0, "fact_checked"),   # "Half True" match alone is still a fact-check
    (False, False, 0.4, 0.0, "corroborated"),
    (False, False, 0.3, 0.3, "corroborated"),   # tie is not disputed (refute must exceed support)
    (False, False, 0.29, 0.0, "unverified"),
    (False, False, 0.0, 0.0, "unverified"),
])
def test_determine_evidence_status(fact_check_found, verdict_found, v_consensus, v_refute, expected):
    assert determine_evidence_status(fact_check_found, verdict_found, v_consensus, v_refute) == expected
```

- [ ] **Step 2: Write the failing guardrail test**

Append to `eval/tests/test_guardrails.py`:

```python
def test_disputed_claims_never_exceed_cap_without_verdict():
    """C1 guardrail: when reputable coverage refutes a claim and no definitive verdict exists,
    no combination of the other inputs may lift C_total above DISPUTED_CAP."""
    from app.scoring import ScoringEngine
    from app.scoring.confidence import DISPUTED_CAP

    engine = ScoringEngine()
    grid = [i / 10 for i in range(11)]
    for s_fact in grid:
        for b_ling in grid:
            for v_consensus in [0.0, 0.1, 0.2, 0.29]:
                result = engine.compute(
                    s_fact=s_fact, b_ling=b_ling, v_consensus=v_consensus, v_refute=0.6,
                    agents_succeeded=3, fact_check_found=False, verdict_found=False,
                )
                assert result.c_total <= DISPUTED_CAP, (s_fact, b_ling, v_consensus, result.c_total)
                assert result.evidence_status == "disputed"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run (from `backend/`): `uv run pytest tests/test_scoring_engine.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'determine_evidence_status'`.

- [ ] **Step 4: Implement `confidence.py`**

Replace the whole of `backend/app/scoring/confidence.py` with:

```python
from typing import List, Tuple

# Stance-aware thresholds (C1 / D2). Track 3.1: calibrate against the eval set (Plan 3).
CORROBORATED_MIN = 0.3      # v_consensus at/above this counts as real corroboration
DISPUTED_REFUTE_MIN = 0.3   # v_refute at/above this, AND above v_consensus, marks a claim disputed
DISPUTED_CAP = 0.33         # top of the red gauge band (TDD §9.1: 0-33 = likely false)

EVIDENCE_STATUSES = ("fact_checked", "disputed", "corroborated", "unverified")


def is_disputed(v_consensus: float, v_refute: float, verdict_found: bool) -> bool:
    """Reputable coverage predominantly refutes the claim and no definitive verdict overrides it."""
    return (not verdict_found) and v_refute >= DISPUTED_REFUTE_MIN and v_refute > v_consensus


def determine_evidence_status(
    fact_check_found: bool,
    verdict_found: bool,
    v_consensus: float,
    v_refute: float,
) -> str:
    """
    Classify what kind of evidence backs the score (D2). The UI shows "Not enough evidence"
    instead of a gauge value when the status is "unverified".
    - fact_checked: a definitive verdict, or any matched review not contradicted by coverage
    - disputed:     reputable coverage refutes the claim and no definitive verdict exists
    - corroborated: no fact-check, but supporting reputable coverage >= CORROBORATED_MIN
    - unverified:   none of the above
    """
    if verdict_found:
        return "fact_checked"
    if is_disputed(v_consensus, v_refute, verdict_found):
        return "disputed"
    if fact_check_found:
        return "fact_checked"
    if v_consensus >= CORROBORATED_MIN:
        return "corroborated"
    return "unverified"


def determine_confidence_level(
    agents_succeeded: int,
    s_fact: float,
    v_consensus: float,
    total_agents: int = 3,
    fact_check_found: bool = False,
    verdict_found: bool = False,
    v_refute: float = 0.0,
) -> str:
    """
    Determine the confidence level of the scoring based on TDD Section 7.
    - high: all agents succeeded AND an explicit definitive fact-check verdict was found (verdict_found=True)
    - medium: at least 2 agents succeeded and there is real evidence either way
      (v_consensus >= 0.3, v_refute >= 0.3, or a matched fact-check)
    - low: at least 1 agent succeeded
    - insufficient: no agents succeeded
    """
    # High confidence requires all agents succeeded AND an explicit definitive fact-check verdict
    if agents_succeeded == total_agents and verdict_found:
        return "high"
    elif agents_succeeded >= 2 and (
        v_consensus >= CORROBORATED_MIN
        or v_refute >= DISPUTED_REFUTE_MIN
        or fact_check_found
        or verdict_found
    ):
        return "medium"
    elif agents_succeeded >= 1:
        return "low"
    else:
        return "insufficient"


def apply_confidence_capping(
    c_total: float,
    s_fact: float,
    v_consensus: float,
    agents_succeeded: int,
    total_agents: int = 3,
    fact_check_found: bool = False,
    verdict_found: bool = False,
    v_refute: float = 0.0,
) -> Tuple[float, List[str]]:
    """
    Applies the confidence capping rules from TDD Section 7.
    - Rule 1: If agents_succeeded < total_agents: cap c_total at 0.50 and warn
    - Rule 2: If not fact_check_found and v_consensus < 0.3 (and not disputed): cap c_total at 0.55 and warn
    - Rule 3: If verdict_found and s_fact <= 0.25: cap c_total at s_fact + 0.15 (max 0.25)
      so topical news coverage cannot inflate confirmed false claims.
    - Rule 4: If disputed (v_refute >= 0.3, v_refute > v_consensus, no definitive verdict):
      cap c_total at 0.33 and warn, so refuting coverage lands the claim in the red band.

    Returns:
        Tuple of (capped_c_total, list of warning strings)
    """
    warnings = []
    disputed = is_disputed(v_consensus, v_refute, verdict_found)

    # Rule 1: Partial analysis cap
    if agents_succeeded < total_agents:
        c_total = min(c_total, 0.50)
        warnings.append(f"Partial analysis: {agents_succeeded}/{total_agents} agents succeeded")

    # Rule 2: Low confidence cap (no fact check found and low consensus). Skipped when
    # disputed: Rule 4's cap is stricter and its warning is the accurate one.
    if not fact_check_found and v_consensus < CORROBORATED_MIN and not disputed:
        c_total = min(c_total, 0.55)
        warnings.append("Low confidence: no fact-checks found and insufficient corroborating sources")

    # Rule 3: Confirmed False Verdict Dominance Cap (R2.B1 / R4.A)
    # Uses verdict_found so dominance capping fires when a definitive false verdict exists
    if verdict_found and s_fact <= 0.25:
        c_total = min(c_total, s_fact + 0.15)
        c_total = min(c_total, 0.25)

    # Rule 4: Disputed cap (C1). Refuting reputable coverage must pull the claim into the red
    # band, never leave it amber; a definitive verdict (Rule 3 territory) overrides it.
    if disputed:
        c_total = min(c_total, DISPUTED_CAP)
        warnings.append("Disputed: reputable sources contradict this claim")

    return c_total, warnings
```

- [ ] **Step 5: Implement `engine.py`**

In `backend/app/scoring/engine.py`:

1. Change the confidence import to:

```python
from app.scoring.confidence import (
    determine_confidence_level,
    apply_confidence_capping,
    determine_evidence_status,
)
```

2. Replace the two dataclasses with:

```python
@dataclass
class ScoreComponents:
    s_fact: float
    b_ling: float
    v_consensus: float
    v_refute: float = 0.0

@dataclass
class ScoringResult:
    c_total: float
    components: ScoreComponents
    weights: Dict[str, float]
    confidence_level: str
    warnings: List[str]
    evidence_status: str = "unverified"
```

3. In `compute`, add `v_refute: float = 0.0,` as the last parameter (after `verdict_found`).
After the line `v_consensus_clamped = max(0.0, min(1.0, v_consensus))`, add:

```python
        v_refute_clamped = max(0.0, min(1.0, v_refute))
```

4. Pass `v_refute=v_refute_clamped,` as the last keyword argument to both
`apply_confidence_capping(...)` and `determine_confidence_level(...)`.

5. After the `determine_confidence_level` call, add:

```python
        evidence_status = determine_evidence_status(
            fact_check_found=fact_check_found,
            verdict_found=verdict_found,
            v_consensus=v_consensus_clamped,
            v_refute=v_refute_clamped,
        )
```

6. Replace the `return ScoringResult(...)` with:

```python
        return ScoringResult(
            c_total=c_total_final,
            components=ScoreComponents(
                s_fact=s_fact_clamped,
                b_ling=b_ling_clamped,
                v_consensus=v_consensus_clamped,
                v_refute=v_refute_clamped,
            ),
            weights={"w1": self.w1, "w2": self.w2, "w3": self.w3},
            confidence_level=confidence,
            warnings=warnings,
            evidence_status=evidence_status,
        )
```

- [ ] **Step 6: Export**

Replace `backend/app/scoring/__init__.py` with:

```python
from app.scoring.engine import ScoringEngine, ScoringResult, ScoreComponents
from app.scoring.weights import DEFAULT_W1, DEFAULT_W2, DEFAULT_W3, validate_weights
from app.scoring.confidence import (
    determine_confidence_level,
    apply_confidence_capping,
    determine_evidence_status,
)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run (from `backend/`): `uv run pytest tests/test_scoring_engine.py -v`
Expected: all pass (existing plus 13 new, counting the parametrized cases).

Run (repo root): `uv run pytest eval/tests/test_guardrails.py -v`
Expected: all pass, including `test_disputed_claims_never_exceed_cap_without_verdict`.

- [ ] **Step 8: Commit**

```bash
git add backend/app/scoring/ backend/tests/test_scoring_engine.py eval/tests/test_guardrails.py
git commit -m "feat(scoring): add disputed cap (Rule 4) and evidence_status (C1, D2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Wire through orchestrator, API schema, cache and DB

**Files:**
- Create: `backend/app/db/migrations/03_stance_aware_consensus.sql`
- Modify: `backend/app/agents/orchestrator.py`
- Modify: `backend/app/services/cache.py`
- Modify: `backend/app/api/schemas.py`
- Modify: `backend/app/api/routes.py` (`GET /api/results/{id}`)
- Test: `backend/tests/test_orchestrator.py`, `backend/tests/test_cache.py`, `backend/tests/test_api_routes.py`

**Interfaces:**
- Consumes:
  - `CrossRefResult.v_refute_score` and `sources_found[*]["stance"]` (Task 2)
  - `ScoringEngine.compute(..., v_refute=)`, `ScoringResult.evidence_status`, `ScoreComponents.v_refute` (Task 3)
- Produces:
  - Response `trust_score.v_refute: float`
  - `confidence.evidence_status: "fact_checked" | "disputed" | "corroborated" | "unverified"`
  - `sources[*].stance: "supports" | "refutes" | "neutral" | null`
  - `insert_cache(..., v_refute: float = 0.0, evidence_status: Optional[str] = None)`
  - `lookup_cache` returns `None` for rows whose `evidence_status` is NULL

- [ ] **Step 1: Write the failing tests**

1. In `backend/tests/test_cache.py`, find `test_lookup_cache_offline_hit` and change its
`mock_data` to include `"evidence_status": "fact_checked"`:

```python
    mock_data = [{
        "claim_id": "test-uuid",
        "extracted_claim": "NASA water Mars",
        "similarity": 0.98,
        "c_total": 0.85,
        "evidence_status": "fact_checked"
    }]
```

Then append:

```python
@pytest.mark.anyio
async def test_lookup_cache_legacy_row_without_evidence_status_is_miss():
    """Rows scored before stance-aware consensus (evidence_status NULL) must not be served (C1)."""
    mock_response = MagicMock()
    mock_response.data = [{
        "claim_id": "legacy-uuid",
        "similarity": 0.99,
        "confidence_level": "high",
        "evidence_status": None,
    }]
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = mock_response

    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        assert await lookup_cache([0.1] * 384, similarity_threshold=0.95) is None


@pytest.mark.anyio
async def test_insert_cache_persists_refute_and_evidence_status():
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute.side_effect = [
        MagicMock(data=[{"id": "claim-1"}]),
        MagicMock(data=[{"id": "result-1"}]),
    ]
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        claim_id = await insert_cache(
            input_type="text", raw_input="c", extracted_claim="c", embedding=[0.1] * 384,
            c_total=0.33, s_fact=0.5, b_ling=0.0, v_consensus=0.0,
            weights={"w1": 0.5, "w2": 0.2, "w3": 0.3}, confidence_level="medium",
            agent_outputs={}, source_citations=[], linguistic_profile={}, processing_time_ms=10,
            warnings=[], v_refute=0.6, evidence_status="disputed",
        )
    assert claim_id == "claim-1"
    result_row = mock_supabase.table.return_value.insert.call_args_list[1][0][0]
    assert result_row["v_refute"] == 0.6
    assert result_row["evidence_status"] == "disputed"
```

2. In `backend/tests/test_orchestrator.py`:

   (a) In `test_verify_claim_cache_hit`, add `"evidence_status": "fact_checked", "v_refute": 0.0,`
   to `mock_cached`, and add these assertions at the end of the test:

```python
        assert result["confidence"]["evidence_status"] == "fact_checked"
        assert result["trust_score"]["v_refute"] == 0.0
```

   (b) At the end of `test_verify_claim_cache_miss_pipeline_success`, add:

```python
        assert result["confidence"]["evidence_status"] == "fact_checked"
        assert result["trust_score"]["v_refute"] == 0.0
        assert mock_insert.call_args.kwargs["evidence_status"] == "fact_checked"
```

   (c) Append a new test:

```python
@pytest.mark.anyio
async def test_disputed_claim_pipeline():
    """C1 end-to-end: refuting reputable coverage and no fact-check -> red band, 'disputed'."""
    mock_fc = make_neutral_factcheck(degraded=False)
    mock_cr = CrossRefResult(
        entities_extracted=["flu"],
        sources_found=[{
            "url": "https://www.reuters.com/fact/x", "title": "Claim is false", "publisher": "Reuters",
            "bias_label": "Center", "relevance_score": 0.9, "stance": "refutes", "stance_confidence": 0.93,
        }],
        coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 1, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0,
        v_refute_score=0.6,
        degraded=False,
    )
    mock_ling = LinguisticResult(
        clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
        informational_density=1.0, b_ling_score=0.0, degraded=False,
    )

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", return_value=mock_fc), \
         patch("app.agents.crossref_agent.run", return_value=mock_cr), \
         patch("app.agents.linguistic_agent.run", return_value=mock_ling), \
         patch("app.agents.orchestrator.insert_cache", return_value="disputed-uuid") as mock_insert:

        result = await verify_claim("Drinking bleach cures the flu")

    assert result["trust_score"]["c_total"] <= 0.33
    assert result["trust_score"]["v_refute"] == 0.6
    assert result["confidence"]["evidence_status"] == "disputed"
    assert "Disputed: reputable sources contradict this claim" in result["confidence"]["warnings"]
    crossref_sources = [s for s in result["sources"] if s["type"] == "corroborating"]
    assert crossref_sources[0]["stance"] == "refutes"
    assert mock_insert.call_args.kwargs["v_refute"] == 0.6
    assert mock_insert.call_args.kwargs["evidence_status"] == "disputed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cache.py tests/test_orchestrator.py -v`
Expected: FAIL. The failures are `KeyError: 'evidence_status'` in the orchestrator tests and
`TypeError: insert_cache() got an unexpected keyword argument 'v_refute'` in the cache test.

- [ ] **Step 3: Write the migration**

Create `backend/app/db/migrations/03_stance_aware_consensus.sql`:

```sql
-- VeriFact AI - Migration 03: stance-aware consensus (C1) and evidence status (D2)
-- Spec: docs/superpowers/specs/2026-09-23-prd-tdd-review.md

-- 1. New result columns
ALTER TABLE results ADD COLUMN IF NOT EXISTS v_refute FLOAT NOT NULL DEFAULT 0;
ALTER TABLE results ADD COLUMN IF NOT EXISTS evidence_status VARCHAR(20)
    CHECK (evidence_status IN ('fact_checked', 'disputed', 'corroborated', 'unverified'));
-- Legacy rows keep evidence_status = NULL: they were scored with coverage-only consensus,
-- and lookup_cache treats them as cache misses so they are never served again.

-- 2. match_claims must return the new columns. The return type changes, so DROP first
--    (CREATE OR REPLACE cannot change a function's result columns).
DROP FUNCTION IF EXISTS match_claims(vector, double precision);

CREATE FUNCTION match_claims(
    query_embedding vector(384),
    similarity_threshold float
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
        c.extracted_claim,
        (1 - (c.embedding <=> query_embedding))::float AS similarity
    FROM claims c
    JOIN results r ON r.claim_id = c.id
    WHERE (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT 1;
END;
$$;
```

- [ ] **Step 4: Update `cache.py`**

1. In `lookup_cache`, directly after the `if conf in {"low", "insufficient"}:` block, add:

```python
            # C1: rows scored before stance-aware consensus have no evidence_status. Their
            # V_consensus counted refuting coverage as support, so they must never be served.
            if not match_data.get("evidence_status"):
                logger.info("Semantic match predates stance-aware scoring (no evidence_status). Treating as cache miss.")
                return None
```

2. Change the `insert_cache` signature's final parameter line from
`warnings: Optional[List[str]] = None` to:

```python
    warnings: Optional[List[str]] = None,
    v_refute: float = 0.0,
    evidence_status: Optional[str] = None,
```

3. In `insert_cache`, add these two keys to `result_data`, directly after `"v_consensus": v_consensus,`:

```python
            "v_refute": v_refute,
            "evidence_status": evidence_status,
```

- [ ] **Step 5: Update `schemas.py`**

1. In `TrustScoreModel`, after `v_consensus`, add:

```python
    v_refute: float = Field(default=0.0, ge=0.0, le=1.0)
```

2. In `ConfidenceModel`, after `verdict_found`, add:

```python
    evidence_status: Literal["fact_checked", "disputed", "corroborated", "unverified"] = "unverified"
```

3. In `SourceCitationModel`, after `type`, add:

```python
    stance: Optional[Literal["supports", "refutes", "neutral"]] = None
```

- [ ] **Step 6: Update `orchestrator.py`**

1. Cache-hit response. In the `trust_score` dict, after the `"v_consensus"` line, add:

```python
                    "v_refute": cached.get("v_refute") or 0.0,
```

In the cache-hit `confidence` dict, after `"verdict_found": ...`, add:

```python
                    "evidence_status": cached.get("evidence_status"),
```

2. Scoring call. Add `v_refute=cr_res.v_refute_score,` as the last argument to `engine.compute(...)`.

3. Fact-check sources. In the `sources.append({...})` inside `for m in fc_res.all_matches:`, add `"stance": None,` after `"type": src_type`.

4. Corroborating sources. In the `sources.append({...})` inside `for s in cr_res.sources_found:`, add `"stance": s.get("stance"),` after `"type": "corroborating"`.

5. Persistence. In the `agent_outputs["crossref"]` dict, after `"v_consensus_score": cr_res.v_consensus_score,`, add:

```python
                        "v_refute_score": cr_res.v_refute_score,
```

In the `insert_cache(...)` call, after `warnings=enriched_warnings`, add (put a comma after
`warnings=enriched_warnings` first):

```python
                v_refute=scoring_res.components.v_refute,
                evidence_status=scoring_res.evidence_status,
```

6. Final response. In the returned `trust_score` dict, after `"v_consensus"`, add
`"v_refute": scoring_res.components.v_refute,`. In the returned `confidence` dict, after
`"verdict_found": verdict_found`, add (with a comma after `verdict_found`):

```python
            "evidence_status": scoring_res.evidence_status,
```

- [ ] **Step 6b: Update `GET /api/results/{id}` in `backend/app/api/routes.py`**

First add a failing test to `backend/tests/test_api_routes.py`:

```python
def test_get_result_legacy_row_is_labelled():
    """Rows saved before stance-aware scoring have no evidence_status; the API must say so."""
    row = {
        "id": "11111111-1111-1111-1111-111111111111", "c_total": 0.7, "s_fact": 0.5, "b_ling": 0.2,
        "v_consensus": 0.6, "confidence_level": "medium", "warnings": [],
        "claims": {"extracted_claim": "Old claim"}, "evidence_status": None,
    }
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[row])
    with patch("app.api.routes.get_supabase", return_value=mock_supabase):
        response = client.get("/api/results/11111111-1111-1111-1111-111111111111")
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"]["evidence_status"] == "unverified"
    assert body["trust_score"]["v_refute"] == 0.0
    assert any("Legacy result" in w for w in body["confidence"]["warnings"])
```

Then, in `get_result_endpoint`, add `"v_refute": row.get("v_refute") or 0.0,` after the
`"v_consensus"` line of the returned `trust_score`. Replace the returned `confidence` dict with:

```python
            "confidence": {
                "level": row.get("confidence_level"),
                "warnings": (row.get("warnings") or []) + (
                    [] if row.get("evidence_status")
                    else ["Legacy result: scored before stance-aware consensus; treat corroboration as unverified"]
                ),
                "verdict_found": row.get("confidence_level") == "high",
                "evidence_status": row.get("evidence_status") or "unverified",
            },
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cache.py tests/test_orchestrator.py tests/test_api_routes.py -v`
Expected: all pass.

- [ ] **Step 8: Full unit and eval suites**

Run (from `backend/`): `uv run pytest -q`. Then (repo root): `uv run pytest eval/tests/ -q`
Expected: all pass. If `eval/tests/test_cache_safety.py` or `eval/tests/test_guardrails.py`
now fail because a mocked cache hit lacks `evidence_status`, add
`"evidence_status": "fact_checked"` to that mock. **Do not** relax the legacy-row check.

- [ ] **Step 9: Apply the migration (user action)**

Ask the user to run `backend/app/db/migrations/03_stance_aware_consensus.sql` in the Supabase
SQL editor. Then verify with:

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'results' AND column_name IN ('v_refute', 'evidence_status');
```

Expected: 2 rows.

- [ ] **Step 10: Commit**

```bash
git add backend/app/db/migrations/03_stance_aware_consensus.sql backend/app/agents/orchestrator.py backend/app/services/cache.py backend/app/api/schemas.py backend/app/api/routes.py backend/tests/test_cache.py backend/tests/test_orchestrator.py backend/tests/test_api_routes.py eval/tests/
git commit -m "feat(api): expose evidence_status, v_refute and per-source stance; retire legacy cache rows (C1, D2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Spec updates (docs must match code)

**Files:**
- Modify: `documents/VeriFact_AI_PRD_v2.md` §6, §7
- Modify: `documents/VeriFact_AI_TDD.md` §3.1, §4.3, §5, §7, §9.1
- Modify: `CLAUDE.md` §2

- [ ] **Step 1: PRD §6 — V_consensus definition**

Replace the **V_consensus** bullet block with:

```markdown
**V_consensus (Cross-Source Validation Density) — Range: 0.0 to 1.0**
- Counts only reputable sources whose text *supports* the claim, as judged by a local NLI
  model (premise = article title + snippet, hypothesis = claim; probability ≥ 0.6)
- Each supporting reputable source adds its search relevance; 5 fully relevant supporting sources = 1.0
- 0.0 = no supporting reputable source (coverage that refutes or merely mentions the claim adds nothing)
- Fact-check sites (the claim-review domains) are excluded: they are the Fact-Check agent's evidence

**V_refute (Refutation Density) — Range: 0.0 to 1.0** — not part of the weighted sum
- Same computation over reputable sources whose text *contradicts* the claim
- Used only by capping Rule 4 and the evidence status
```

- [ ] **Step 2: PRD §6 — capping rules and evidence status**

Append to the **Confidence Capping Rules** list:

```markdown
- If reputable coverage refutes the claim (V_refute ≥ 0.3 and V_refute > V_consensus) and no definitive fact-check verdict exists: Cap C_total at 0.33 and display "Disputed: reputable sources contradict this claim"
```

Then add a new subsection after it:

```markdown
### Evidence Status

Every result carries one of:

| Status | Meaning | UI |
|--------|---------|----|
| `fact_checked` | A fact-check review matched (definitive or not) | Gauge + verdict |
| `disputed` | Reputable coverage refutes the claim; no definitive verdict | Red gauge + "Disputed" |
| `corroborated` | No fact-check; supporting reputable coverage (V_consensus ≥ 0.3) | Gauge |
| `unverified` | None of the above | "Not enough evidence" in place of the gauge value |
```

- [ ] **Step 3: PRD §7 — results table**

Add these fields to the `results {}` block, after `v_consensus: FLOAT`:

```
  v_refute: FLOAT
  evidence_status: ENUM('fact_checked', 'disputed', 'corroborated', 'unverified')
```

- [ ] **Step 4: TDD §3.1 — response example**

In the response JSON:
- add `"v_refute": 0.0,` after `"v_consensus": 0.68,`
- add `"evidence_status": "fact_checked"` after `"verdict_found": true` (with a comma after `true`)
- add `"stance": "supports"` to the Reuters source object

After the `verdict_found` paragraph, add:
"`confidence.evidence_status` is defined in PRD §6 'Evidence Status'. `sources[].stance` is
`supports`/`refutes`/`neutral` for corroborating sources and `null` for fact-check sources."

- [ ] **Step 5: TDD §4.3 — Cross-Ref agent**

1. In the `CrossRefResult` dataclass block, add
`v_refute_score: float = 0.0  # 0.0 to 1.0, refuting reputable coverage` after
`v_consensus_score`.
2. Change the `sources_found` comment to
`# {url, title, publisher, bias_label, relevance_score, stance, stance_confidence}`.
3. Replace step 6 of **Logic** with:

```markdown
6. Classify stance for every source with the local NLI cross-encoder
   (`app/services/nli.py`, `cross-encoder/nli-deberta-v3-small`, loaded once at startup):
   premise = title + snippet (≤1000 chars), hypothesis = claim. `supports` / `refutes` when
   that probability ≥ 0.6 and exceeds the other; otherwise `neutral`.
7. Compute V_consensus / V_refute over reputable sources, excluding claim-review domains:
   - Reputable = AllSides-rated, or the institutional/primary allowlist (`.gov`, `.edu`, `who.int`).
   - `v_consensus = min(Σ relevance of supporting sources / 5, 1.0)`
   - `v_refute = min(Σ relevance of refuting sources / 5, 1.0)`
```

4. Add to the **Degradation** paragraph: "If the NLI model fails, the agent returns the neutral
result with `degraded=True`. It never falls back to counting coverage."

- [ ] **Step 6: TDD §5 — results table**

Add these lines after `v_consensus FLOAT NOT NULL,`:

```sql
    v_refute FLOAT NOT NULL DEFAULT 0,
    evidence_status VARCHAR(20)
        CHECK (evidence_status IN ('fact_checked', 'disputed', 'corroborated', 'unverified')),
```

- [ ] **Step 7: TDD §7 — scoring code**

Update the §7 code block to match `backend/app/scoring/confidence.py` and `engine.py` exactly
as committed in Task 3: the new `v_refute` parameter, Rule 4, the Rule 2 `not disputed` guard,
`determine_evidence_status`, and `evidence_status` on `ScoringResult`. Copy the code from
those files. Don't retype it.

- [ ] **Step 8: TDD §9.1 — Trust Gauge**

Append: "When `confidence.evidence_status == "unverified"`, render 'Not enough evidence' in
place of the numeric score and needle, and keep the confidence badge and warnings. When it is
`disputed`, show a 'Disputed' chip under the gauge. Source cards show a stance chip
(Supports / Refutes / Neutral)."

- [ ] **Step 9: CLAUDE.md §2**

In the architecture block, change the Cross-Ref line to:

```
   │     ├─ Cross-Ref agent     → spaCy NER + Tavily + AllSides + NLI stance → V_consensus, V_refute
```

and change the scoring line to:

```
   ├─ Scoring engine → C_total = w1·S_fact + w2·(1−B_ling) + w3·V_consensus  (+ Rule 4 disputed cap, evidence_status)
```

- [ ] **Step 10: Commit**

```bash
git add documents/VeriFact_AI_PRD_v2.md documents/VeriFact_AI_TDD.md CLAUDE.md
git commit -m "docs: specify stance-aware consensus, Rule 4 and evidence_status (C1, D2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Review gate (CLAUDE.md §5 — mandatory)

**Files:** none (fixes found here go back into the relevant task's files)

- [ ] **Step 1: Full verification**

Run (from `backend/`): `uv run pytest -q` and `uv run pytest tests/test_nli.py -m live -v`.
Run (repo root): `uv run pytest eval/tests/ -q`.
Expected: all green. Record the counts for the report.

- [ ] **Step 2: Code review.** In Claude Code, use the `code-reviewer` subagent. In Antigravity, use `/review` (the `code-review` skill). See the README mapping.

Scope: `git diff main...fix/stance-aware-consensus`. Check specifically:
- NLI direction (premise = source, hypothesis = claim)
- label-order safety
- that degradation never falls back to coverage counting
- that the scoring engine stays pure
- that Rule 4 never overrides a definitive verdict
- that legacy cache rows are unreachable
- that the migration's DROP/CREATE is safe to re-run

- [ ] **Step 3: Eval gate.** Use the `eval-guardian` subagent (Claude Code) or the `eval-guardian` skill (Antigravity).

Scope: the scoring, agent and guardrail changes. Confirm:
- zero bias leakage (the stance path uses no LLM, and bias labels still come only from AllSides)
- that the new guardrail test holds
- whether any existing eval metric moved

- [ ] **Step 4: Address findings**

Fix every blocking finding with a test first, then re-run Step 1. Record any non-blocking
finding you defer, with a reason.

- [ ] **Step 5: Report**

Tell the user:
- the test counts
- the review findings, and which were fixed or deferred
- that migration 03 must be applied in Supabase before deploy
- that the thresholds `STANCE_MIN_PROB`, `DISPUTED_REFUTE_MIN` and `DISPUTED_CAP` are provisional until Plan 3 calibrates them

## Known limitations (carry into Plan 3)

- Tavily snippets are short. A news article that quotes a false claim before debunking it can
  read as `supports`. Plan 3 adds a labeled stance fixture set to measure this, and to
  calibrate `STANCE_MIN_PROB`.
- NLI adds roughly 0.3–1 s of CPU time per request (about 10–15 pairs). This fits inside the
  8 s agent timeout; Plan 4's request deadline will account for it.
