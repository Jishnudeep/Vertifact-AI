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
│  │  cosine_similarity(input_embedding, cached) ≥ 0.95?       │   │
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
    "weights": { "w1": 0.50, "w2": 0.20, "w3": 0.30 }
  },
  "confidence": {
    "level": "high",
    "warnings": []
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
      "type": "corroborating"
    }
  ],
  "processing_time_ms": 7420,
  "cached": false
}
```

**Error Responses:**

- `400` — Invalid input (empty, too long >2000 chars, unsupported format)
- `422` — Input classified as non-verifiable (opinion, question)
- `429` — Rate limited
- `500` — All agents failed
- `503` — LLM provider unavailable

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

**ADK Configuration:**
```python
from google.adk import Agent, ParallelAgent
import litellm

orchestrator = Agent(
    name="verifact_orchestrator",
    model="groq/llama-3.1-8b-instant",  # via LiteLLM
    description="Coordinates fact-checking pipeline",
    sub_agents=[factcheck_agent, crossref_agent, linguistic_agent],
    # ADK handles parallel dispatch natively
)
```

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
    sources_found: list[dict]       # {url, title, publisher, bias_label, relevance_score}
    coverage_distribution: dict     # {"Left": 2, "Center": 5, "Right": 1}
    v_consensus_score: float        # 0.0 to 1.0
```

**Logic:**
1. Run spaCy NER on claim text → extract named entities
2. Construct search queries: [claim text], [entity1 + key phrase], [entity2 + key phrase]
3. Call Tavily with each query (parallel async calls)
4. Deduplicate results by domain
5. For each source URL, look up domain in `bias_ratings` table
6. Compute V_consensus:
   - Count unique reputable domains (domains present in AllSides DB)
   - Weight by relevance score (how closely the article matches the claim)
   - Normalize to 0-1 range: `min(reputable_source_count / 5, 1.0)` adjusted by avg relevance

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
SELECT r.*, c.extracted_claim,
       1 - (c.embedding <=> $1::vector) AS similarity
FROM claims c
JOIN results r ON r.claim_id = c.id
WHERE 1 - (c.embedding <=> $1::vector) >= 0.95
ORDER BY similarity DESC
LIMIT 1;
```

**Cache invalidation:** No TTL for v2. Claims don't change truthfulness frequently enough to justify complexity. Revisit if the dataset grows past 10K entries.

---

## 7. Scoring Engine Implementation

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScoreComponents:
    s_fact: float
    b_ling: float
    v_consensus: float

@dataclass
class ScoringResult:
    c_total: float
    components: ScoreComponents
    weights: dict
    confidence_level: str
    warnings: list[str]

class ScoringEngine:
    def __init__(self, w1=0.50, w2=0.20, w3=0.30):
        assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1.0"
        assert w1 >= 0.4, "Fact-check weight must be dominant (≥0.4)"
        self.w1, self.w2, self.w3 = w1, w2, w3

    def compute(
        self,
        s_fact: float,
        b_ling: float,
        v_consensus: float,
        agents_succeeded: int,
        total_agents: int = 3,
    ) -> ScoringResult:
        warnings = []

        # Raw composite
        c_total = (
            self.w1 * s_fact
            + self.w2 * (1 - b_ling)
            + self.w3 * v_consensus
        )

        # Confidence capping
        if agents_succeeded < total_agents:
            c_total = min(c_total, 0.50)
            warnings.append(f"Partial analysis: {agents_succeeded}/{total_agents} agents succeeded")

        if s_fact == 0.5 and v_consensus < 0.3:
            c_total = min(c_total, 0.55)
            warnings.append("Low confidence: no fact-checks found and insufficient corroborating sources")

        # Determine confidence level
        if agents_succeeded == total_agents and (s_fact != 0.5 or v_consensus >= 0.5):
            confidence = "high"
        elif agents_succeeded >= 2 and v_consensus >= 0.3:
            confidence = "medium"
        elif agents_succeeded >= 1:
            confidence = "low"
        else:
            confidence = "insufficient"

        c_total = round(max(0.0, min(1.0, c_total)), 3)

        return ScoringResult(
            c_total=c_total,
            components=ScoreComponents(s_fact, b_ling, v_consensus),
            weights={"w1": self.w1, "w2": self.w2, "w3": self.w3},
            confidence_level=confidence,
            warnings=warnings,
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

# Reputable fact-check / wire-service domains. The Fact-Check agent (§4.2 Layer 2)
# passes these so Tavily retrieval is biased toward fact-checking, not general web.
FACTCHECK_DOMAINS = [
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "apnews.com", "reuters.com",
]

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

- **Input sanitization:** Strip `<script>` tags, tracking parameters (utm_*, fbclid, etc.), and limit input to 2000 characters
- **Rate limiting:** 10 requests/minute per IP (FastAPI middleware with `slowapi`)
- **API keys:** Never exposed to frontend. All external API calls happen server-side
- **Image uploads:** Validate MIME type, reject files > 5MB, process in memory (no disk writes)
- **CORS:** Restrict to frontend domain in production
