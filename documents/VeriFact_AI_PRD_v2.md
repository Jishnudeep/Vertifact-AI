# VeriFact AI — Product Requirements Document v2.0

**Product Name:** VeriFact AI
**Author:** Jishnu Bhattacharjee, Machine Learning Engineer
**Date:** May 31, 2026
**Status:** Draft for Development
**Target Platform:** Responsive Web Application (Mobile-First)

---

## 1. Problem Statement

Misinformation spreads 6x faster than factual corrections on social media. Existing fact-checking tools are either (a) academic-grade text reports that nobody reads on their phone, or (b) chatbot wrappers around a single LLM that hallucinate confidence without citing sources.

VeriFact AI solves a specific problem: **a user encounters a suspicious claim on social media, and within 8-10 seconds, gets a visual, source-backed credibility assessment they can act on.** Not a 2,000-word report. Not "according to my training data." A visual dashboard backed by verifiable external sources with transparent scoring.

### What We Are Not Building

- A general-purpose LLM chatbot
- A political bias detector that generates its own ratings
- A native mobile app (responsive web is sufficient for portfolio scope)
- A real-time social media monitoring tool

---

## 2. Target User

**Primary:** Social media users (18-45) who encounter questionable claims and want a fast credibility check before sharing or reacting.

**Secondary (portfolio context):** Hiring managers and technical interviewers evaluating ML engineering skills — the system's architecture, evaluation rigor, and production patterns matter as much as the end-user experience.

---

## 3. User Stories

### Epic 1: Multi-Modal Claim Ingestion

**US-1.1:** As a mobile user, I want to paste a headline or claim text into a search box and get results without creating an account.

**US-1.2:** As a mobile user, I want to upload a screenshot of a social media post and have the system extract the claim text automatically via OCR.

**US-1.3:** As a browser user, I want to paste a URL and have the system extract the article's primary claim for verification.

### Epic 2: Transparent Verification Pipeline

**US-2.1:** As a user waiting for results, I want to see which verification agents are currently running (fact-check lookup, source cross-referencing, linguistic analysis) so I understand the process isn't a black box.

**US-2.2:** As a user viewing results, I want every component of the trust score to link back to its source — which fact-check article matched, which outlets corroborated the claim, what linguistic patterns were detected.

### Epic 3: Visual Credibility Dashboard

**US-3.1:** As a user, I want a single Trust Gauge (0-100) with color coding (red/amber/green) so I can assess credibility at a glance.

**US-3.2:** As a user, I want a Linguistic Volatility Radar showing how sensationalized the claim's language is across 4 axes (clickbait syntax, emotional framing, sensationalism, informational density).

**US-3.3:** As a user, I want to see a Source Coverage Map showing which outlets reported on this claim, anchored to verified media bias ratings (not AI-generated ratings).

### Epic 4: Evaluation & Trust

**US-4.1:** As a user, I want to see a confidence indicator that tells me when the system has low confidence (e.g., no fact-checks found, insufficient sources) rather than fabricating certainty.

**US-4.2:** As a technically-minded user, I want to see how the trust score was computed — the weight of each component and why.

---

## 4. Phased Delivery

### Phase 1: Core Pipeline (Weeks 1-3)
**Goal:** Text input → multi-agent verification → scored JSON response via API.

| ID | Feature | Priority |
|----|---------|----------|
| P1-01 | FastAPI backend with health check and input validation | P0 |
| P1-02 | ADK orchestrator agent with LiteLLM integration (Llama 3.1 via Groq/Ollama) | P0 |
| P1-03 | Fact-Check Lookup Agent (Google Fact Check Tools API + Tavily web search) | P0 |
| P1-04 | Cross-Reference Agent (entity extraction via spaCy, source corroboration via search) | P0 |
| P1-05 | Linguistic Analysis Agent (VADER sentiment + HuggingFace sensationalism classifier) | P1 |
| P1-06 | Scoring engine (C_total composite with configurable weights) | P0 |
| P1-07 | Supabase PostgreSQL setup (claims table, results table, source citations) | P0 |
| P1-08 | pgvector semantic cache (short-circuit for duplicate/near-duplicate claims) | P1 |

**Phase 1 Exit Criteria:** API endpoint accepts text claims and returns structured JSON with trust score, source citations, and component breakdowns. Eval harness runs against 50+ labeled claims.

### Phase 2: Multi-Modal + Frontend (Weeks 4-6)
**Goal:** Add OCR, URL scraping, and the visual dashboard.

| ID | Feature | Priority |
|----|---------|----------|
| P2-01 | Surya OCR integration for screenshot uploads | P0 |
| P2-02 | URL scraping agent (BeautifulSoup/Playwright for article text extraction) | P1 |
| P2-03 | React frontend with Vite + Tailwind CSS | P0 |
| P2-04 | Trust Gauge component (SVG semicircular gauge, 0-100) | P0 |
| P2-05 | Linguistic Volatility Radar (Recharts radar chart, 4 axes) | P1 |
| P2-06 | Source Coverage display with AllSides bias anchoring | P1 |
| P2-07 | Agent pipeline progress animation (real-time WebSocket status) | P2 |

**Phase 2 Exit Criteria:** Users can submit text, URLs, or images. Dashboard renders trust score with all visual components. Works on mobile viewport (375px+).

### Phase 3: Evaluation, Guardrails & Polish (Weeks 7-8)
**Goal:** Prove the system works. Add safety rails. Polish for portfolio.

| ID | Feature | Priority |
|----|---------|----------|
| P3-01 | Full evaluation harness (LIAR + FEVER + manual test set) | P0 |
| P3-02 | DeepEval guardrails (faithfulness, hallucination detection on agent outputs) | P0 |
| P3-03 | Confidence calibration (cap scores when sources are insufficient) | P0 |
| P3-04 | Weight optimization via grid search on eval set | P1 |
| P3-05 | Input rejection for non-factual claims (opinions, questions) | P1 |
| P3-06 | Rate limiting and input sanitization | P1 |
| P3-07 | README, architecture diagram, eval results for GitHub | P0 |

**Phase 3 Exit Criteria:** Eval report with precision/recall, MAE, calibration curve. Guardrails catch >80% of hallucinated outputs. GitHub repo is portfolio-ready.

---

## 5. Technical Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend | Vite + React + Tailwind CSS | Fast builds, responsive-first, no SSR complexity |
| Visualization | Recharts | SVG-based, React-native, touch-responsive |
| Backend | FastAPI (Python 3.11+) | Native async, type hints, auto-docs |
| Agent Framework | Google ADK + LiteLLM | Model-agnostic orchestration, hub-and-spoke parallelism |
| LLM (Dev) | Llama 3.1 8B via Groq (free tier) | Fast inference, zero cost during development |
| LLM (Prod) | Qwen3-30B-A3B or Llama 3.1 70B via vLLM | Higher quality for demo/production |
| OCR | Surya | PyTorch-native, transformer-based, reading-order aware |
| Embeddings | BGE-M3 via sentence-transformers | Hybrid retrieval (dense + sparse), multilingual |
| Database | Supabase (PostgreSQL) | Free tier, auth, JSONB storage, real-time |
| Vector Search | pgvector (via Supabase) | Co-located with relational data, HNSW indexing |
| Fact-Check API | Google Fact Check Tools API | Free, ClaimReview standard, 100+ orgs |
| Web Search | Tavily | AI-agent-optimized results, 1K req/month free |
| Bias Data | AllSides CSV (static, pre-loaded) | No runtime API dependency |
| NLP | spaCy (NER) + VADER (sentiment) | Lightweight, no GPU needed |
| Eval Framework | DeepEval + custom harness | Hallucination detection, faithfulness, calibration |

---

## 6. Scoring Framework

### Composite Trust Score

```
C_total = w1 · S_fact + w2 · (1 - B_ling) + w3 · V_consensus
```

Where all weights satisfy Σwi = 1.0 and w1 ≥ 0.4 (fact-check match is always dominant).

### Component Definitions

**S_fact (Fact-Check Match Score) — Range: 0.0 to 1.0**
- 1.0 = Direct match from a verified fact-check org rating the claim as TRUE
- 0.0 = Direct match rating the claim as FALSE
- 0.5 = No fact-check found (neutral, not indicative)
- Intermediate values mapped from fact-checker labels (e.g., "Mostly True" = 0.75, "Half True" = 0.5, "Mostly False" = 0.25)
- When multiple fact-checks exist, take the weighted average (weighted by recency)

**B_ling (Linguistic Volatility Score) — Range: 0.0 to 1.0**
- Composite of 4 sub-scores (each 0.0-1.0): Clickbait Syntax, Emotional Framing, Sensationalism, Informational Density (inverted — higher density = lower volatility)
- B_ling = mean(clickbait, emotional, sensationalism, 1 - info_density)
- Note: (1 - B_ling) is used in C_total, so calm factual language increases the score

**V_consensus (Cross-Source Validation Density) — Range: 0.0 to 1.0**
- Based on how many reputable outlets corroborate the claim
- 0.0 = No corroborating sources found
- 0.5 = 1-2 sources with partial corroboration
- 1.0 = 5+ independent reputable sources with strong semantic alignment

### Confidence Capping Rules

- If S_fact = 0.5 (no fact-check found) AND V_consensus < 0.3: Cap C_total at 0.55 and display "Low Confidence — Insufficient Sources" warning
- If only 1 agent returns results (others timeout/fail): Cap C_total at 0.5 and display "Partial Analysis" warning
- If input is detected as opinion/question (not factual claim): Return no score, display "This doesn't appear to be a verifiable factual claim"

---

## 7. Data Model

### Claims Table
```
claims {
  id: UUID (PK)
  input_type: ENUM('text', 'url', 'image')
  raw_input: TEXT
  extracted_claim: TEXT
  embedding: VECTOR(1024)  -- BGE-M3 embedding for semantic cache
  created_at: TIMESTAMP
}
```

### Results Table
```
results {
  id: UUID (PK)
  claim_id: UUID (FK → claims)
  c_total: FLOAT
  s_fact: FLOAT
  b_ling: FLOAT
  v_consensus: FLOAT
  weights: JSONB  -- {w1: 0.5, w2: 0.2, w3: 0.3}
  confidence_level: ENUM('high', 'medium', 'low', 'insufficient')
  agent_outputs: JSONB  -- raw outputs from each agent
  source_citations: JSONB[]  -- array of {url, title, publisher, rating, bias_label}
  processing_time_ms: INT
  created_at: TIMESTAMP
}
```

### Bias Ratings Table (Static, pre-loaded)
```
bias_ratings {
  domain: TEXT (PK)
  outlet_name: TEXT
  allsides_rating: ENUM('Left', 'Lean Left', 'Center', 'Lean Right', 'Right')
  last_updated: DATE
}
```

---

## 8. Error Handling & Degradation

| Failure | Behavior |
|---------|----------|
| Fact-Check API timeout (>5s) | Skip S_fact, set to 0.5, flag as "Partial Analysis" |
| Tavily search returns 0 results | V_consensus = 0.0, confidence capped |
| OCR extracts no text from image | Return error: "Could not extract text. Try typing the claim instead." |
| URL scraping blocked (403/paywall) | Fall back to using the URL's title + meta description as claim text |
| LLM API rate limited | Queue request, show estimated wait time |
| All agents fail | Return error: "Verification unavailable. Please try again." — never return a fabricated score |

---

## 9. Cost Analysis (Development Phase)

| Service | Free Tier | Monthly Cost (Dev) |
|---------|-----------|-------------------|
| Groq (LLM inference) | 14,400 req/day | $0 |
| Google Fact Check API | Unlimited (API key) | $0 |
| Tavily (web search) | 1,000 req/month | $0 |
| Supabase (DB + auth) | 500MB DB, 50K rows | $0 |
| BGE-M3 embeddings | Local CPU inference | $0 |
| Surya OCR | Local CPU inference | $0 |
| **Total** | | **$0/month** |

Cloud GPU for eval runs (RunPod A10G): ~$0.50/hr × ~10 hrs/month = **$5/month when needed**.

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end latency (text input) | < 10 seconds (cold), < 500ms (cache hit) | P95 measured in FastAPI middleware |
| Bucketed accuracy (False/Mixed/True) | ≥ 70% on eval set | LIAR + FEVER + manual test set |
| MAE of C_total vs ground truth | ≤ 0.20 | Eval harness |
| Faithfulness (agent summaries) | ≥ 0.80 (DeepEval score) | DeepEval automated tests |
| Cache hit rate (semantic match) | ≥ 15% of repeat/similar queries | pgvector query logs |
| Mobile Lighthouse score | ≥ 85 (Performance) | Lighthouse CI |

---

## 11. Out of Scope (v2)

- User accounts and verification history (no auth required for MVP)
- Browser extension for inline fact-checking
- Real-time social media feed monitoring
- Multi-language claim support (English only for v2)
- Native mobile apps (Capacitor/iOS/Android)
- Custom model fine-tuning (deferred to Phase 4 stretch goal)

---

## 12. Open Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Google Fact Check API deprecation | Low | High | Abstract behind interface; Tavily + DataCommons as fallbacks |
| Groq free tier rate limits during demo | Medium | Medium | Cache aggressively; Together.ai as backup provider |
| Eval set too small for meaningful metrics | Medium | High | Commit to 200+ labeled claims before Phase 3 |
| Linguistic volatility model poorly calibrated | High | Medium | Start with VADER baseline; iterate with human-rated calibration set |
| Claims about very recent events (no fact-checks exist) | High | Medium | Confidence capping; transparent "no verified sources" messaging |
