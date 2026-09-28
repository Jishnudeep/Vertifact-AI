# VeriFact AI — Technical Design Document

**Version:** 1.0
**Author:** Jishnu Bhattacharjee
**Date:** May 31, 2026
**Status:** Implementation-Ready Draft
**Audience:** Developers building this system

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Input    │  │ Agent Status │  │ Trust     │  │ Linguistic│  │
│  │ Box      │  │ Pipeline     │  │ Gauge     │  │ Radar     │  │
│  └────┬─────┘  └──────▲───────┘  └─────▲─────┘  └─────▲─────┘  │
│       │               │               │               │         │
│       │          WebSocket             │  REST (JSON)  │         │
└───────┼───────────────┼───────────────┼───────────────┼─────────┘
        │               │               │               │
        ▼               │               │               │
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + ADK)                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  API Layer (FastAPI)                       │   │
│  │  POST /api/verify   POST /api/verify/image                │   │
│  │  GET  /api/results/{id}   WS /ws/status/{id}             │   │
│  └──────────────────────┬────────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼────────────────────────────────────┐   │
│  │              SEMANTIC CACHE LAYER (pgvector)               │   │
│  │  top-3 cosine ≥ 0.95 → row policy → NLI equivalence?      │   │
│  │  YES → return cached result    NO → proceed to agents      │   │
│  └──────────────────────┬────────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼────────────────────────────────────┐   │
│  │           ORCHESTRATOR AGENT (ADK Hub Node)                │   │
│  │  - Parses input (text/URL/image)                          │   │
│  │  - Dispatches parallel worker agents via asyncio.gather    │   │
│  │  - Collects results, computes C_total                     │   │
│  │  - Handles agent failures with graceful degradation        │   │
│  └────┬──────────────┬───────────────┬───────────────────────┘   │
│       │              │               │                           │
│       ▼              ▼               ▼                           │
│  ┌─────────┐  ┌────────────┐  ┌──────────────┐                  │
│  │ Fact-   │  │ Cross-Ref  │  │ Linguistic   │                  │
│  │ Check   │  │ & Source   │  │ Analysis     │                  │
│  │ Agent   │  │ Agent      │  │ Agent        │                  │
│  └────┬────┘  └─────┬──────┘  └──────┬───────┘                  │
│       │             │                │                           │
│       ▼             ▼                ▼                           │
│  ┌─────────┐  ┌────────────┐  ┌──────────────┐                  │
│  │ Google  │  │ Tavily     │  │ VADER +      │                  │
│  │ FC API  │  │ Search +   │  │ HuggingFace  │                  │
│  │ + Tavily│  │ spaCy NER  │  │ Classifiers  │                  │
│  └─────────┘  └────────────┘  └──────────────┘                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              SCORING ENGINE                                │   │
│  │  C_total = w1·S_fact + w2·(1-B_ling) + w3·V_consensus    │   │
│  │  + Confidence capping rules                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DATA LAYER (Supabase + pgvector)             │   │
│  │  claims | results | bias_ratings | eval_logs              │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    EVAL PIPELINE (Offline)                        │
│  DeepEval + custom harness → metrics report + calibration curve  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```
verifact-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── config.py                # Environment vars, API keys, model config
│   │   ├── api/
│   │   │   ├── routes.py            # /verify, /verify/image, /results
│   │   │   ├── schemas.py           # Pydantic request/response models
│   │   │   └── websocket.py         # WebSocket handler for agent status
│   │   ├── agents/
│   │   │   ├── orchestrator.py      # ADK orchestrator agent
│   │   │   ├── factcheck_agent.py   # Fact-Check Lookup Agent
│   │   │   ├── crossref_agent.py    # Cross-Reference & Source Agent
│   │   │   ├── linguistic_agent.py  # Linguistic Analysis Agent
│   │   │   ├── ocr_agent.py         # Surya OCR Agent
│   │   │   └── scraper_agent.py     # URL content extraction Agent
│   │   ├── scoring/
│   │   │   ├── engine.py            # C_total computation
│   │   │   ├── confidence.py        # Capping rules and warnings
│   │   │   └── weights.py           # Weight configuration and tuning
│   │   ├── services/
│   │   │   ├── factcheck_api.py     # Google Fact Check Tools API client
│   │   │   ├── search_api.py        # Tavily search client
│   │   │   ├── embedding.py         # BGE-M3 embedding generation
│   │   │   ├── cache.py             # Semantic cache (pgvector queries)
│   │   │   └── bias_lookup.py       # AllSides static data lookup
│   │   ├── db/
│   │   │   ├── connection.py        # Supabase client setup
│   │   │   ├── models.py            # SQLAlchemy / raw SQL models
│   │   │   └── migrations/          # Schema migrations
│   │   └── utils/
│   │       ├── text_processing.py   # URL stripping, input sanitization
│   │       └── validators.py        # Input type detection, claim vs opinion
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── InputBox.jsx         # Unified text/URL/image input
│   │   │   ├── TrustGauge.jsx       # SVG semicircular gauge
│   │   │   ├── LinguisticRadar.jsx  # Recharts radar chart
│   │   │   ├── SourceCoverage.jsx   # Source list with bias labels
│   │   │   ├── AgentPipeline.jsx    # Real-time agent status display
│   │   │   └── ConfidenceBadge.jsx  # High/Medium/Low/Insufficient indicator
│   │   ├── hooks/
│   │   │   ├── useVerify.js         # API call + WebSocket management
│   │   │   └── useAgentStatus.js    # WebSocket agent progress
│   │   └── styles/
│   │       └── globals.css          # Tailwind base + custom gauge animations
│   ├── package.json
│   └── vite.config.js
├── eval/
│   ├── datasets/
│   │   ├── liar_mapped.json
│   │   ├── fever_subset.json
│   │   └── manual_curated.json
│   ├── tests/
│   │   ├── test_sfact_retrieval.py
│   │   ├── test_bling_calibration.py
│   │   ├── test_ctotal_e2e.py
│   │   ├── test_guardrails.py
│   │   └── test_ocr_accuracy.py
│   ├── metrics/
│   │   ├── report_generator.py
│   │   └── calibration_plot.py
│   ├── conftest.py
│   └── run_eval.py
├── data/
│   ├── allsides_bias_ratings.csv
│   └── seed_claims.json             # Pre-seeded cache entries
├── docker-compose.yml
├── README.md
└── .github/
    └── workflows/
        └── eval.yml                  # CI: run eval on PR
```

---

## 3. API Specification

### 3.1 POST /api/verify

**Request:**
```json
{
  "input": "NASA confirms water found on Mars",
  "input_type": "text"
}
```

For URL input:
```json
{
  "input": "https://example.com/article/mars-water",
  "input_type": "url"
}
```

**Response (200 OK):**
```json
{
  "id": "a1b2c3d4-...",
  "claim": "NASA confirms water found on Mars",
  "trust_score": {
    "c_total": 0.73,
    "s_fact": 0.75,
    "b_ling": 0.22,
    "v_consensus": 0.68,
    "v_refute": 0.0,
    "weights": { "w1": 0.50, "w2": 0.20, "w3": 0.30 }
  },
  "confidence": {
    "level": "high",
    "warnings": [],
    "verdict_found": true,
    "evidence_status": "fact_checked"
  },
  "linguistic_profile": {
    "clickbait_syntax": 0.15,
    "emotional_framing": 0.30,
    "sensationalism": 0.25,
    "informational_density": 0.82
  },
  "sources": [
    {
      "url": "https://www.politifact.com/...",
      "title": "Did NASA confirm water on Mars?",
      "publisher": "PolitiFact",
      "rating": "Mostly True",
      "bias_label": "Lean Left",
      "type": "fact_check"
    },
    {
      "url": "https://www.reuters.com/...",
      "title": "Mars water discovery confirmed by NASA scientists",
      "publisher": "Reuters",
      "bias_label": "Center",
      "type": "corroborating",
      "stance": "supports"
    }
  ],
  "processing_time_ms": 7420,
  "cached": false
}
```

`confidence.verdict_found` is `true` only when a DEFINITIVE verdict was matched from a recognized
claim-review domain (see §7). It is the flag that gates `level: "high"`.

`confidence.evidence_status` is defined in PRD §6 'Evidence Status'. `sources[].stance` is
`supports`/`refutes`/`neutral` for corroborating sources and `null` for fact-check sources
(including claim-review sites returned by the Cross-Ref search).

**Request size limits (two distinct layers):**

- The Pydantic request schema caps `input` at `max_length=10000` characters. This is a transport-level
  DoS ceiling enforced by FastAPI validation, so anything larger is rejected with `422` before the
  pipeline runs.
- The business rule (CLAUDE.md §9) caps a claim at **2000 characters**, enforced by
  `validate_input_length` inside the orchestrator, and surfaced as `400`.
- Net effect: 2001–10000 chars → `400` ("too long"); >10000 chars → `422` (schema rejection).

**Error Responses:**

- `400` — Invalid input (empty, too long >2000 chars, unsupported format)
- `422` — Input classified as non-verifiable (opinion, question), or request body exceeds the 10000-char schema ceiling
- `429` — Rate limited
- `500` — All agents failed
- `503` — LLM provider unavailable

Every response carries `X-Process-Time-Ms`. The pipeline runs under a 9.5 s request deadline (`REQUEST_DEADLINE_S`). Agents get `min(8 s, remaining − 1 s)`, floored at 1 s (`MIN_AGENT_TIMEOUT`), so when the cache lookup or embedding has eaten most of the budget a request can overrun the deadline by up to about 1 s rather than give agents no time at all. A slow DB write returns the result with `id: null` while the write completes in the background; on shutdown the app waits up to 5 s for such writes. The header is listed in CORS `expose_headers` so the cross-origin frontend can read it.

### 3.2 POST /api/verify/image

**Request:** `multipart/form-data` with `image` field (PNG/JPEG, max 5MB)

**Response:** Same schema as `/api/verify`, with additional field:
```json
{
  "ocr_extracted_text": "The raw text Surya pulled from the image",
  "ocr_confidence": 0.92,
  ...
}
```

### 3.3 GET /api/results/{id}

Returns a previously computed result by ID. Used for sharing/bookmarking.

### 3.4 WS /ws/status/{request_id}

WebSocket endpoint for real-time agent pipeline status during verification.

**Server messages (sent as JSON):**
```json
{ "agent": "factcheck", "status": "running", "timestamp": "..." }
{ "agent": "factcheck", "status": "completed", "duration_ms": 2130 }
{ "agent": "crossref", "status": "running", "timestamp": "..." }
{ "agent": "linguistic", "status": "completed", "duration_ms": 840 }
{ "agent": "crossref", "status": "completed", "duration_ms": 3200 }
{ "agent": "scoring", "status": "completed", "result_id": "a1b2c3d4-..." }
```

---

## 4. Agent Specifications

### 4.1 Orchestrator Agent (ADK Hub Node)

**Role:** Single entry point. Parses input, manages parallel dispatch, aggregates results.

**Orchestration model.** The orchestrator is a deterministic Python coroutine
(`app/agents/orchestrator.py::verify_claim`), **not** an LLM agent. It fans out to the three
workers with `asyncio.gather` and gives each a per-agent timeout that fits the request
deadline (TDD §3.1). It then aggregates in code. Nothing is delegated by an LLM.

Google ADK is used at the **leaves** only. `app/utils/adk_helpers.py::run_adk_agent` wraps one
`LlmAgent` per prompt type (`rating_classifier`, `claim_aligner`, `verdict_extractor`,
`sensationalism_analyzer`), using `LiteLlm(model=settings.model_id, temperature=0, max_tokens=1024)`.
Runners are cached per prompt, and there is a fresh session per call.

Why not `LlmAgent(sub_agents=[...])`: in ADK that configures LLM-driven *transfer* to one
sub-agent, not parallel execution. `ParallelAgent` (`google.adk.agents`) runs branches
concurrently but has no per-branch timeout or typed-result contract. Explicit `asyncio` gives
both, plus deterministic degradation.

Model: `settings.model_id` (dev: `groq/openai/gpt-oss-120b`; see `app/config.py` for the
verified 404s on Groq's Llama models). It is never hardcoded in agent logic.

**Input processing logic:**
1. Receive raw input + type
2. If type == "image": dispatch to OCR agent first, extract text, then continue
3. If type == "url": dispatch to scraper agent first, extract article body + headline
4. Sanitize extracted claim (strip tracking params, normalize whitespace)
5. Generate BGE-M3 embedding for semantic cache check
6. If cache hit (cosine ≥ 0.95): return cached result immediately
7. If cache miss: dispatch factcheck, crossref, linguistic agents in parallel
8. Collect results with timeout (8 seconds per agent)
9. Pass to scoring engine
10. Persist to database
11. Return response

**Timeout handling:** Each worker agent gets an 8-second timeout. If an agent times out, the orchestrator uses a default neutral value for that component and flags the result as "Partial Analysis."

### 4.2 Fact-Check Lookup Agent

**Input:** Extracted claim text (string)
**Output:**
```python
@dataclass
class FactCheckResult:
    matches_found: int
    best_match: Optional[dict]  # {url, publisher, claim_reviewed, rating, review_date}
    all_matches: list[dict]
    s_fact_score: float         # 0.0 to 1.0
    search_queries_used: list[str]
    matched_review: bool = False  # True when any valid fact-check review was aligned/found
    verdict_found: bool = False   # True only on definitive verdicts from CLAIMREVIEW_DOMAINS
    degraded: bool = False
    error: Optional[str] = None
```

**Logic (two layers, fact-checking is the motive at every step):**
1. **Layer 1 — Google Fact Check Tools API** (free; highest precision). Call `claims:search` with the claim text. If results returned: parse ClaimReview objects, map `textualRating` through RATING_MAP to a numeric score. This is the preferred signal whenever it hits.
2. **Layer 2 — Tavily, scoped to fact-checking** (used only on a Layer-1 miss). Tavily is the model-agnostic search layer; we use it *with a fact-checking motive*, not as a general web search:
   - Construct fact-check-oriented queries: `"[claim text] fact check"`, `"[claim text] debunked"`, `"is [claim text] true"`.
   - Bias retrieval toward fact-check / reputable domains via Tavily's `include_domains` (e.g. snopes.com, politifact.com, factcheck.org, fullfact.org, apnews.com, reuters.com). Do not fall back to arbitrary blogs/SEO content.
3. If Layer 2 returns fact-check articles: use the LLM to extract the verdict **only from the retrieved snippet/article text** — the LLM may not introduce facts outside the sources (faithfulness guardrail, see eval §3.5). Map the extracted verdict through RATING_MAP.
4. If nothing is found in either layer: `s_fact = 0.5` (neutral/unknown). **Do not fabricate a verdict.** This neutral value triggers confidence capping downstream (scoring §7).

> **Rationale:** The Fact Check API stays as Layer 1 because it is free and returns structured fact-checker verdicts (PolitiFact/Snopes/etc.) — far higher precision than interpreting search snippets. We deliberately do **not** use ADK's built-in `google_search` tool: it is Gemini-only (incompatible with our LiteLLM→Groq dev model), cannot run inside a worker sub-agent, and is limited to one built-in tool per agent — all three break the parallel multi-agent design. Tavily provides free, model-agnostic, fact-check-scoped search instead.

**Rating mapping table:**
```python
RATING_MAP = {
    "true": 1.0, "correct": 1.0, "accurate": 1.0,
    "mostly true": 0.75, "mostly correct": 0.75,
    "half true": 0.5, "mixture": 0.5, "mixed": 0.5,
    "mostly false": 0.25, "mostly incorrect": 0.25,
    "false": 0.0, "incorrect": 0.0, "pants on fire": 0.0,
    "unproven": 0.5, "unverified": 0.5, "outdated": 0.4,
}
```

### 4.3 Cross-Reference & Source Agent

**Input:** Extracted claim text + spaCy entities
**Output:**
```python
@dataclass
class CrossRefResult:
    entities_extracted: list[str]   # People, organizations, locations
    sources_found: list[dict]       # {url, title, publisher, bias_label, relevance_score, stance, stance_confidence}
    coverage_distribution: dict     # {"Left": 2, "Center": 5, "Right": 1}
    v_consensus_score: float        # 0.0 to 1.0 (0.0 when no sources or degraded)
    v_refute_score: float = 0.0     # 0.0 to 1.0, refuting reputable coverage
    degraded: bool = False          # True on timeout, or when EVERY search failed (evidence-API outage)
    error: Optional[str] = None
```

**Logic:**
1. Run spaCy NER on claim text → extract named entities
2. Construct search queries: [claim text], [entity1 + key phrase], [entity2 + key phrase]
3. Call Tavily with each query (parallel async calls)
4. Deduplicate results by domain
5. For each source URL, look up domain in `bias_ratings` table
6. Classify stance for every source with the local NLI cross-encoder
   (`app/services/nli.py`, `cross-encoder/nli-deberta-v3-small`, loaded once at startup):
   premise = title + snippet (≤1000 chars), hypothesis = claim. `supports` / `refutes` when
   that probability ≥ 0.6 and exceeds the other; otherwise `neutral`. Claim-review domains
   (`CLAIMREVIEW_DOMAINS`) get `stance: null`: their snippets restate the claim they debunk,
   which NLI misreads as support, so they are shown without a chip and never counted.
7. Compute V_consensus / V_refute over reputable sources, excluding claim-review domains:
   - Reputable = AllSides-rated, or the institutional/primary allowlist (`.gov`, `.edu`, `who.int`).
   - `v_consensus = min(Σ relevance of supporting sources / 5, 1.0)`
   - `v_refute = min(Σ relevance of refuting sources / 5, 1.0)`

**Degradation (CLAUDE.md §4.3):** a search returning zero results means "no corroboration found"
(`degraded=False`). A search that could **not run** — every Tavily query raising `EvidenceAPIError`
(non-200/HTTP failure) — is an outage: the agent returns the neutral result with `degraded=True`, so
Rule 1 caps the score and the response carries the "Partial analysis" warning. The same split applies
to the Fact-Check agent across both of its evidence layers. If the NLI model fails, the agent returns the neutral
result with `degraded=True`. It never falls back to counting coverage.

### 4.4 Linguistic Analysis Agent

**Input:** Claim text (or full article body if URL was provided)
**Output:**
```python
@dataclass
class LinguisticResult:
    clickbait_syntax: float       # 0.0 (factual) to 1.0 (heavy clickbait)
    emotional_framing: float      # 0.0 (neutral) to 1.0 (emotionally charged)
    sensationalism: float         # 0.0 (measured) to 1.0 (sensational)
    informational_density: float  # 0.0 (sparse) to 1.0 (information-rich)
    b_ling_score: float           # Composite: mean of axes (with density inverted)
    flagged_phrases: list[str]    # Specific phrases that triggered high scores
```

**Logic:**
1. **VADER Sentiment:** Compute compound score. Map extreme pos/neg to emotional_framing.
2. **Clickbait detection:** Use a HuggingFace classifier (e.g., `roberta-base` fine-tuned on clickbait dataset) OR rule-based heuristics as MVP:
   - ALL CAPS words count
   - Question marks in headline
   - Superlatives ("BREAKING", "SHOCKING", "YOU WON'T BELIEVE")
   - Vague attribution ("experts say", "sources claim")
3. **Sensationalism:** LLM-as-judge prompt asking: "Rate the sensationalism of this text on a 0-10 scale. Consider word choice, tone, and whether claims are presented with appropriate nuance."
4. **Informational density:** Word count of factual assertions (numbers, dates, proper nouns, citations) divided by total word count.

**MVP approach:** Start with VADER + heuristic rules for all axes. Replace individual axes with trained classifiers iteratively in Phase 3.

### 4.5 OCR Agent (Phase 2)

**Input:** Image file (PNG/JPEG bytes)
**Output:**
```python
@dataclass
class OCRResult:
    extracted_text: str
    confidence: float        # 0.0 to 1.0
    bounding_boxes: list     # For debugging/visualization
    processing_time_ms: int
```

**Logic:**
1. Load image, resize if > 2048px on longest edge (memory optimization for 8GB RAM)
2. Run Surya OCR pipeline (detection → recognition → reading order)
3. Concatenate text blocks in reading order
4. If extracted text < 10 characters: return error "insufficient text detected"
5. Return cleaned text for downstream processing

---

## 5. Database Schema (SQL)

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Claims table
CREATE TABLE claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_type VARCHAR(10) NOT NULL CHECK (input_type IN ('text', 'url', 'image')),
    raw_input TEXT NOT NULL,
    extracted_claim TEXT NOT NULL,
    embedding vector(1024),  -- BGE-M3 embedding
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX idx_claims_embedding ON claims
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Results table
CREATE TABLE results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id),
    c_total FLOAT NOT NULL,
    s_fact FLOAT NOT NULL,
    b_ling FLOAT NOT NULL,
    v_consensus FLOAT NOT NULL,
    v_refute FLOAT NOT NULL DEFAULT 0,
    evidence_status VARCHAR(20)
        CHECK (evidence_status IN ('fact_checked', 'disputed', 'corroborated', 'unverified')),
    weights JSONB NOT NULL DEFAULT '{"w1": 0.5, "w2": 0.2, "w3": 0.3}',
    confidence_level VARCHAR(15) NOT NULL
        CHECK (confidence_level IN ('high', 'medium', 'low', 'insufficient')),
    agent_outputs JSONB,  -- Raw agent responses for debugging
    source_citations JSONB DEFAULT '[]',
    linguistic_profile JSONB,
    processing_time_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_results_claim_id ON results(claim_id);

-- Static bias ratings (pre-loaded from AllSides CSV)
CREATE TABLE bias_ratings (
    domain VARCHAR(255) PRIMARY KEY,
    outlet_name VARCHAR(255) NOT NULL,
    allsides_rating VARCHAR(15) NOT NULL
        CHECK (allsides_rating IN ('Left', 'Lean Left', 'Center', 'Lean Right', 'Right')),
    last_updated DATE
);

-- Evaluation logs (for tracking eval runs)
CREATE TABLE eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name VARCHAR(255),
    dataset_name VARCHAR(100),
    total_claims INT,
    mae FLOAT,
    bucketed_accuracy FLOAT,
    spearman_rho FLOAT,
    weights_used JSONB,
    detailed_results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 6. Embedding & Semantic Cache

### Embedding Generation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def generate_embedding(text: str) -> list[float]:
    """Generate 1024-dim BGE-M3 embedding for a claim."""
    return model.encode(text, normalize_embeddings=True).tolist()
```

**Important:** Load the model once at FastAPI startup (`@app.on_event("startup")`), not per-request. BGE-M3 takes ~2GB RAM when loaded. On 8GB RAM, this is not feasible alongside the rest of the stack. **Development config:** Use `all-MiniLM-L6-v2` (384-dim, ~100MB) and set `vector(384)` in the schema. **Production config:** Use `BAAI/bge-m3` (1024-dim) and set `vector(1024)`. Make the dimension a config variable so swapping requires no code changes, only a schema migration and re-embedding of cached claims.

### Cache Lookup Query

```sql
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
```

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

---

## 7. Scoring Engine Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Stance-aware thresholds (C1 / D2)
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
    # `high` keys on verdict_found (a definitive verdict), while Rule 2 keys on
    # fact_check_found (any matched review): a "Half True" match is real evidence — enough
    # to suppress the no-fact-check cap — but never enough to claim high confidence.
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

    # Rule 3: Confirmed False Verdict Dominance Cap
    if verdict_found and s_fact <= 0.25:
        c_total = min(c_total, s_fact + 0.15)
        c_total = min(c_total, 0.25)

    # Rule 4: Disputed cap (C1). Refuting reputable coverage must pull the claim into the red band
    if disputed:
        c_total = min(c_total, DISPUTED_CAP)
        warnings.append("Disputed: reputable sources contradict this claim")

    return c_total, warnings


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

class ScoringEngine:
    def __init__(self, w1: float = 0.50, w2: float = 0.20, w3: float = 0.30):
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1.0"
        assert w1 >= 0.4, "Fact-check weight must be dominant (≥0.4)"
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def compute(
        self,
        s_fact: float,
        b_ling: float,
        v_consensus: float,
        agents_succeeded: int,
        total_agents: int = 3,
        # `fact_check_found` (== FactCheckResult.matched_review): a review was matched,
        # INCLUDING "Half True" / "Unproven". Passed explicitly rather than inferred from
        # `s_fact == 0.5`, because those verdicts also map to exactly 0.5 and must NOT
        # trigger the no-fact-check cap (Rule 2).
        fact_check_found: bool = False,
        # `verdict_found` (== FactCheckResult.verdict_found): a DEFINITIVE verdict
        # (true / mostly true / mostly false / false) from a recognized claim-review
        # domain. Gates `high` confidence, the Rule-3 dominance cap, and overrides Rule 4:
        # only a definitive verdict may license top confidence or override coverage.
        # Invariant: verdict_found=True implies fact_check_found=True.
        verdict_found: bool = False,
        v_refute: float = 0.0,
    ) -> ScoringResult:
        assert not (verdict_found and not fact_check_found), "verdict_found implies fact_check_found"

        s_fact_clamped = max(0.0, min(1.0, s_fact))
        b_ling_clamped = max(0.0, min(1.0, b_ling))
        v_consensus_clamped = max(0.0, min(1.0, v_consensus))
        v_refute_clamped = max(0.0, min(1.0, v_refute))

        # Raw composite
        c_total_raw = (
            self.w1 * s_fact_clamped
            + self.w2 * (1.0 - b_ling_clamped)
            + self.w3 * v_consensus_clamped
        )

        # Apply confidence capping rules
        c_total_capped, warnings = apply_confidence_capping(
            c_total=c_total_raw,
            s_fact=s_fact_clamped,
            v_consensus=v_consensus_clamped,
            agents_succeeded=agents_succeeded,
            total_agents=total_agents,
            fact_check_found=fact_check_found,
            verdict_found=verdict_found,
            v_refute=v_refute_clamped,
        )

        # Determine confidence level
        confidence = determine_confidence_level(
            agents_succeeded=agents_succeeded,
            s_fact=s_fact_clamped,
            v_consensus=v_consensus_clamped,
            total_agents=total_agents,
            fact_check_found=fact_check_found,
            verdict_found=verdict_found,
            v_refute=v_refute_clamped,
        )

        evidence_status = determine_evidence_status(
            fact_check_found=fact_check_found,
            verdict_found=verdict_found,
            v_consensus=v_consensus_clamped,
            v_refute=v_refute_clamped,
        )

        c_total_final = round(max(0.0, min(1.0, c_total_capped)), 3)

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

---

## 8. External Service Integration

### 8.1 Google Fact Check Tools API

```python
import httpx

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

async def search_factchecks(query: str, api_key: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            FACTCHECK_API_URL,
            params={"query": query, "key": api_key, "languageCode": "en"},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("claims", [])
```

**Response parsing:** Each claim object contains `claimReview` array with `textualRating` and `url`. Map `textualRating` through the RATING_MAP in section 4.2.

### 8.2 Tavily Search

```python
from tavily import TavilyClient

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Broad Tavily retrieval domains for fact checking & news evidence
FACTCHECK_DOMAINS = [
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "leadstories.com", "checkyourfact.com", "healthfeedback.org",
    "climatefeedback.org", "afp.com", "factcheck.afp.com",
    "reuters.com", "apnews.com", "usatoday.com"
]

# Strict ClaimReview verdict-authority domains (Layer-2 verdict gating)
CLAIMREVIEW_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "leadstories.com", "checkyourfact.com", "healthfeedback.org",
    "climatefeedback.org", "afp.com", "factcheck.afp.com"
}

async def search_web(
    query: str,
    max_results: int = 5,
    include_domains: list[str] | None = None,
) -> list[dict]:
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        # Fact-Check agent passes FACTCHECK_DOMAINS (fact-check motive);
        # Cross-Ref agent passes None for broad cross-spectrum source discovery.
        include_domains=include_domains or [],
    )
    return response.get("results", [])
```

### 8.3 AllSides Bias Lookup

```python
# Pre-loaded in memory at startup from CSV
BIAS_RATINGS: dict[str, str] = {}  # domain → rating

def load_bias_ratings(csv_path: str):
    """Load AllSides ratings from CSV into memory dict."""
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = extract_domain(row["source_url"])
            BIAS_RATINGS[domain] = row["bias_rating"]

def get_bias_label(url: str) -> Optional[str]:
    domain = extract_domain(url)
    return BIAS_RATINGS.get(domain)
```

---

## 9. Frontend Component Specifications

### 9.1 Trust Gauge (SVG)

Semi-circular gauge from 0 to 100. Color transitions:
- 0-33: `#DC2626` (Crimson — likely false)
- 34-66: `#F59E0B` (Amber — mixed/uncertain)
- 67-100: `#10B981` (Emerald — likely true)

Needle animated with CSS `transition: transform 1.2s ease-out`.

Score displayed as large centered number. Confidence badge shown below.

When `confidence.evidence_status == "unverified"`, render 'Not enough evidence' in place of the numeric score and needle, and keep the confidence badge and warnings. When it is `disputed`, show a 'Disputed' chip under the gauge. Source cards show a stance chip (Supports / Refutes / Neutral).

### 9.2 Linguistic Volatility Radar (Recharts)

4-axis radar chart with axes: Clickbait Syntax, Emotional Framing, Sensationalism, Informational Density. Each axis 0-100. Filled area with 30% opacity. Tooltip shows flagged phrases per axis.

### 9.3 Source Coverage List

Vertical list of source cards, each showing: publisher name, article title (linked), AllSides bias label (colored chip: blue=Left, light blue=Lean Left, gray=Center, light red=Lean Right, red=Right), and type badge (Fact-Check or Corroborating).

---

## 10. Deployment Architecture

### Development (Local)

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - GOOGLE_FACTCHECK_API_KEY=${GOOGLE_FACTCHECK_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [backend]
```

**Database:** Use Supabase cloud free tier (not local). Saves RAM and avoids running PostgreSQL locally on 8GB machine.

**LLM Inference:** Groq cloud API (free tier) during development. No local model serving.

### Production / Demo

Deploy backend on **Railway** or **Render** (free tier). Frontend on **Vercel** (free). Database stays on Supabase cloud. LLM stays on Groq or switches to Together.ai for higher limits.

---

## 11. Security Considerations

- **Input sanitization:** strip `<script>`/HTML, tracking parameters (utm_*, fbclid, …) and JSON-answer syntax; limit input to 2000 characters. Only the sanitized text is persisted.
- **Rate limiting:** 10 requests/minute per client IP (slowapi). The key is `app/api/client_ip.py`: behind `TRUSTED_PROXY_HOPS` proxies it takes the N-th `X-Forwarded-For` entry from the right; with 0 hops it uses the socket peer. Set `TRUSTED_PROXY_HOPS=1` on Render/Railway/Fly.
- **Spend budgets:** per-UTC-day ceilings on Tavily credits (`TAVILY_DAILY_CREDIT_BUDGET`, default 30) and LLM calls (`LLM_DAILY_CALL_BUDGET`, default 900). When a budget is exhausted, the call raises an outage error, the agent degrades and the response carries "Partial analysis". It never returns a confident score. The counters are in-process (single instance). Two deliberate exceptions:
  - The optional LLM sensationalism rating (`ENABLE_LLM_SENSATIONALISM`, off by default) falls back to the heuristic score instead of degrading the Linguistic agent. Known issue: that path still reports `sensationalism_source="llm"`.
  - When Google Fact Check answers (even with no match) and only Tavily is refused, the Fact-Check agent reports "no fact-checks found" rather than degrading (R7.B4: degrade only when both layers are down). The Cross-Ref agent is refused by the same Tavily budget and degrades, so the response still carries "Partial analysis" and is not served from cache.
- **Prompt injection:** every LLM prompt fences untrusted text in XML tags, states an instruction hierarchy, strips JSON-answer syntax (`neutralize_for_prompt`), length-caps inputs, and enum-validates outputs. Adversarial claims in the eval set must never yield `verdict_found=true` or `high` confidence.
- **SSRF (URL inputs, P2-02):** the scraper must call `resolve_public_url` (`app/utils/url_safety.py`, via `asyncio.to_thread` because DNS resolution blocks) on the initial URL and on every redirect hop (at most 3, `follow_redirects=False`). It must connect to one of the IPs it returns rather than re-resolving the host (DNS rebinding), cap the response at 2 MB and 5 s, and accept only `text/html`. Every resolved IP must be globally routable and not multicast; IPv4-mapped and NAT64 (`64:ff9b::/96`) addresses are checked as the IPv4 address they carry. IPv4-compatible (`::/96`), SIIT (`::ffff:0:0:0/96`), local-use NAT64, Teredo, 6to4 and site-local addresses are rejected outright.
- **API keys:** never exposed to the frontend. All external calls happen server-side.
- **Image uploads:** validate the MIME type, reject files over 5 MB, process in memory (no disk writes), and never store image bytes.
- **Retention:** claims and results older than `RETENTION_DAYS` (default 90) are deleted by `purge_old_claims` (migration 05; pg_cron recommended). Shared result links expire with them.
- **Database roles:** the backend connects with the service-role key, and the frontend never talks to Supabase. Migration 06 therefore enables RLS with no policies on `claims`, `results`, `bias_ratings` and `eval_runs`, and revokes `EXECUTE` on `match_claims` and `purge_old_claims` from `PUBLIC`/`anon`/`authenticated`, so the publishable key can neither read the data nor call the purge. It also changes the schema's default privileges so functions created later are not executable by those roles. Any new table still needs `ENABLE ROW LEVEL SECURITY` in its own migration, and re-running 06 after recreating `match_claims` is harmless.
- **CORS:** restricted to the frontend domain in production.
