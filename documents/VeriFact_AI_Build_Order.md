# VeriFact AI — Build Order & Implementation Guide

**Date:** May 31, 2026
**Estimated Total Duration:** 7-8 weeks (evenings/weekends pace)
**Prerequisites:** Python 3.11+, Node.js 18+, Git, VS Code, Supabase account, Groq API key

---

## Phase 0: Environment & Foundation (Day 1-2)

Everything else depends on this. Do not skip any step.

### Step 0.1 — Project Scaffolding
```
mkdir verifact-ai && cd verifact-ai
git init
mkdir -p backend/app/{api,agents,scoring,services,db,utils}
mkdir -p frontend/src/{components,hooks,styles}
mkdir -p eval/{datasets,tests,metrics}
mkdir -p data
```
Create `.gitignore`, `.env.example`, `README.md` stub.

### Step 0.2 — Backend Skeleton
- Initialize Python venv: `python -m venv venv`
- Install core deps: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `python-dotenv`
- Create `backend/app/main.py` with a health check endpoint (`GET /health`)
- Verify it runs: `uvicorn app.main:app --reload`

**Gate:** `curl localhost:8000/health` returns `{"status": "ok"}`

### Step 0.3 — Supabase Setup
- Create Supabase project (free tier)
- Run the SQL schema from TDD Section 5 (claims, results, bias_ratings, eval_runs tables)
- Enable pgvector extension
- Install `supabase` Python client
- Write `backend/app/db/connection.py` — verify you can insert and query a test row

**Gate:** Insert a dummy claim row and read it back via Python.

### Step 0.4 — API Keys
- Get Google Fact Check Tools API key (Google Cloud Console → enable Fact Check Tools API)
- Get Tavily API key (tavily.com, free signup)
- Get Groq API key (console.groq.com, free signup)
- Add all to `.env`, load via `backend/app/config.py` with Pydantic Settings

**Gate:** All 3 API keys work (test each with a simple curl/httpx call).

### Step 0.5 — Load Bias Ratings Data
- Download AllSides CSV from Kaggle
- Clean and normalize domain names
- Write `backend/app/services/bias_lookup.py` — load CSV, expose `get_bias_label(domain)` function
- Insert into `bias_ratings` table

**Gate:** `get_bias_label("nytimes.com")` returns `"Lean Left"`

---

## Phase 1: Core Verification Pipeline (Weeks 1-2)

Build each agent independently, test it independently, then wire them together. **Do not build the orchestrator first.** Build the leaves, then the hub.

### Step 1.1 — Fact-Check Agent (standalone)
**File:** `backend/app/agents/factcheck_agent.py`
**Depends on:** Step 0.4 (API keys)

1. Write `services/factcheck_api.py` — async wrapper for Google Fact Check Tools API
2. Write `services/search_api.py` — async wrapper for Tavily search
3. Implement fact-check agent logic (fact-checking is the motive at every step — see TDD §4.2):
   - Layer 1: Query the free Google FC API with claim text (highest-precision structured verdicts)
   - Layer 2 (on miss): query Tavily with fact-check-oriented queries ("[claim] fact check" / "[claim] debunked"), scoped to FACTCHECK_DOMAINS (snopes, politifact, factcheck.org, fullfact, AP, Reuters) via `include_domains` — not a general web search
   - LLM verdict extraction (Layer 2) must use ONLY retrieved snippet text (faithfulness guardrail, Step 3.5)
   - Map textual ratings to numeric scores using RATING_MAP
   - Note: do NOT use ADK's built-in `google_search` (Gemini-only, can't run in a sub-agent, one-built-in-tool limit — breaks the multi-agent design). Tavily is the model-agnostic search layer.
4. Write unit tests with known claims:
   - "The Earth is flat" → should find fact-check, s_fact near 0.0
   - "The Earth orbits the Sun" → should return s_fact near 1.0 or 0.5 (if no fact-check indexed)

**Gate:** Agent returns correct s_fact scores for 5 test claims. Tests pass.

### Step 1.2 — Linguistic Analysis Agent (standalone)
**File:** `backend/app/agents/linguistic_agent.py`
**Depends on:** Nothing (pure NLP, no external APIs)

1. Install `vaderSentiment`, `spacy` (download `en_core_web_sm`)
2. Implement the 4-axis analysis:
   - Clickbait: heuristic rules (ALL CAPS count, superlatives, question marks)
   - Emotional framing: VADER compound score mapped to 0-1
   - Sensationalism: count of trigger words from a curated list
   - Informational density: ratio of (numbers + proper nouns + dates) to total tokens
3. Compute b_ling as mean of axes (density inverted)
4. Test with contrasting inputs:
   - "GDP grew 2.3% in Q4 according to Bureau of Labor Statistics" → b_ling should be low (~0.1-0.2)
   - "SHOCKING: You Won't BELIEVE What Scientists Just Found!!!" → b_ling should be high (~0.7-0.9)

**Gate:** Agent correctly differentiates factual from sensational language on 10 test strings.

### Step 1.3 — Cross-Reference Agent (standalone)
**File:** `backend/app/agents/crossref_agent.py`
**Depends on:** Step 0.4 (Tavily API key), Step 0.5 (bias data)

1. Install spaCy (already done in 1.2)
2. Implement NER extraction → search query construction
3. Call Tavily search with constructed queries
4. Deduplicate by domain, look up bias labels
5. Compute v_consensus score
6. Test with a well-known news event — verify it finds multiple reputable sources

**Gate:** Agent returns source list with bias labels for 5 test claims.

### Step 1.4 — Scoring Engine
**File:** `backend/app/scoring/engine.py`
**Depends on:** Nothing (pure computation)

1. Implement `ScoringEngine` class from TDD Section 7
2. Implement confidence capping rules
3. Write thorough unit tests:
   - All agents succeed, fact-check found → high confidence
   - No fact-check, low consensus → capped score, low confidence
   - One agent fails → partial analysis
   - Edge cases: all zeros, all ones, boundary values

**Gate:** All unit tests pass. Scoring is deterministic and correct.

### Step 1.5 — Embedding Service + Semantic Cache
**File:** `backend/app/services/embedding.py`, `backend/app/services/cache.py`
**Depends on:** Step 0.3 (pgvector)

1. Install `sentence-transformers`
2. **Development optimization:** Use `all-MiniLM-L6-v2` (384-dim, ~100MB RAM) instead of BGE-M3 (2GB RAM) during development. Swap to BGE-M3 when deploying on better hardware. Update the vector dimension in your schema accordingly.
3. Write `generate_embedding(text)` function, load model once at module level
4. Write cache lookup query (cosine similarity ≥ 0.95)
5. Write cache insert (after successful verification)
6. Test: verify same claim returns cache hit, similar claim returns cache hit, different claim returns cache miss

**Gate:** Cache hit/miss works correctly. Embedding generation takes < 500ms.

### Step 1.6 — ADK Orchestrator (wire everything together)
**File:** `backend/app/agents/orchestrator.py`
**Depends on:** Steps 1.1-1.5 (all agents + scoring + cache)

1. Install `google-adk`, `litellm`
2. Configure ADK with LiteLLM pointing to Groq (Llama 3.1 8B)
3. Implement orchestrator flow:
   - Input → sanitize → embed → cache check
   - Cache miss → parallel dispatch agents via ADK
   - Collect results with per-agent timeout (8s)
   - Handle individual agent failures gracefully
   - Pass to scoring engine
   - Persist result to database
   - Return structured response
4. Test end-to-end with text claims

**Gate:** Full pipeline works. Text in → JSON out with all fields populated. Total latency < 12 seconds.

### Step 1.7 — API Routes
**File:** `backend/app/api/routes.py`, `backend/app/api/schemas.py`
**Depends on:** Step 1.6 (orchestrator)

1. Define Pydantic request/response schemas (see TDD Section 3)
2. Implement `POST /api/verify` — connects to orchestrator
3. Implement `GET /api/results/{id}` — DB lookup
4. Add input validation (max length, type detection)
5. Add rate limiting with `slowapi`
6. Add basic error handling middleware

**Gate:** API accepts text claims via POST, returns full JSON response. Rate limiting works.

---

### Phase 1 Checkpoint (end of Week 2)

Run a quick sanity check before moving on:
- [ ] 10 different text claims produce reasonable results
- [ ] Cache works (duplicate claim is instant)
- [ ] Failed agent doesn't crash the pipeline
- [ ] API returns proper error codes for bad input
- [ ] Total cold latency is under 12 seconds

---

## Phase 2: Multi-Modal Input + Frontend (Weeks 3-5)

### Step 2.1 — OCR Agent (Surya)
**File:** `backend/app/agents/ocr_agent.py`
**Depends on:** Phase 1 complete

1. Install `surya-ocr`
2. Implement image preprocessing (resize, format validation)
3. Implement OCR pipeline (detection → recognition → reading order)
4. Test with screenshots of tweets, news headlines, social media posts
5. **Memory management:** On 8GB RAM, load Surya model only when needed, unload after. Or run OCR in a subprocess.

**Gate:** OCR correctly extracts text from 10 test screenshots with >85% character accuracy.

### Step 2.2 — URL Scraper Agent
**File:** `backend/app/agents/scraper_agent.py`
**Depends on:** Phase 1 complete

1. Install `beautifulsoup4`, `httpx`
2. Implement article text extraction (try `<article>` tag, then `<p>` tags, then `<meta description>`)
3. Extract headline from `<title>` or `<h1>`
4. Handle common failures: 403 forbidden, paywalls, JavaScript-only pages
5. Fallback: use Tavily's `extract` feature to get page content

**Gate:** Successfully extracts article text from 10 test URLs (mix of news sites).

### Step 2.3 — Image Upload API Endpoint
**File:** Update `backend/app/api/routes.py`
**Depends on:** Step 2.1

1. Implement `POST /api/verify/image` (multipart form)
2. Validate file type (PNG/JPEG), size (< 5MB)
3. Route to OCR agent → extract text → feed to standard pipeline
4. Return response with `ocr_extracted_text` field

**Gate:** Upload a screenshot, get a verification result back.

### Step 2.4 — Frontend: Project Setup + Input Component
**Depends on:** API routes working

1. `npm create vite@latest frontend -- --template react`
2. Install Tailwind CSS, Recharts
3. Build `InputBox.jsx`:
   - Text input (default), URL input (auto-detected), Image upload (drag-drop or file picker)
   - Tab or auto-detect based on content (starts with http? → URL mode)
   - Submit button → calls backend API
4. Mobile-first responsive layout (375px viewport)

**Gate:** Input component renders, accepts text/URL/image, sends correct API request.

### Step 2.5 — Frontend: Trust Gauge Component
**Depends on:** Step 2.4

1. Build `TrustGauge.jsx` — SVG semicircular gauge
2. Implement color interpolation (red → amber → green based on score)
3. Animated needle with CSS transition
4. Large centered score number
5. Confidence badge below gauge (High/Medium/Low/Insufficient)

**Gate:** Gauge renders correctly for scores at 0, 25, 50, 75, 100. Animation smooth on mobile.

### Step 2.6 — Frontend: Linguistic Radar + Source Coverage
**Depends on:** Step 2.4

1. Build `LinguisticRadar.jsx` — Recharts `<RadarChart>` with 4 axes
2. Build `SourceCoverage.jsx` — source card list with bias label chips
3. Build `ConfidenceBadge.jsx`

**Gate:** All visualization components render correctly with sample data.

### Step 2.7 — Frontend: Results Page Assembly
**Depends on:** Steps 2.5-2.6

1. Build results page layout: Trust Gauge (hero), Linguistic Radar (below), Source Coverage (scrollable list)
2. Connect to API via `useVerify` hook (fetch on submit, loading state, error state)
3. Loading state: show agent pipeline status (basic version — just agent names with spinner/checkmark)
4. Error state: display user-friendly error messages
5. Mobile responsive pass (test at 375px, 428px, 768px, 1024px)

**Gate:** Full user flow works: enter claim → see loading → see results. Works on mobile viewport.

### Step 2.8 — WebSocket Agent Status (stretch)
**Depends on:** Step 2.7

1. Implement `WS /ws/status/{request_id}` in backend
2. Orchestrator sends status updates as agents start/complete
3. Frontend `useAgentStatus` hook connects to WebSocket
4. `AgentPipeline.jsx` shows real-time agent progress

**Gate:** User sees agents lighting up in real-time as pipeline runs. Non-blocking — skip if running behind schedule.

---

### Phase 2 Checkpoint (end of Week 5)

- [ ] All three input modes work (text, URL, image)
- [ ] Dashboard renders all visual components
- [ ] Works on mobile viewport (no horizontal scroll, touch targets ≥ 44px)
- [ ] Loading and error states are handled
- [ ] 20 different claims produce reasonable visual results

---

## Phase 3: Evaluation, Guardrails & Portfolio Polish (Weeks 6-8)

This phase is what separates a demo from a credible ML project. Do not skip it.

### Step 3.1 — Build Evaluation Dataset
**File:** `eval/datasets/`
**Depends on:** Phase 1 complete (need working pipeline to test against)

1. Download LIAR dataset → map 6-class labels to 0-1 scores → save as `liar_mapped.json`
2. Download FEVER dataset → sample 500 claims (balanced across SUPPORTED/REFUTED/NEI) → save as `fever_subset.json`
3. Manually curate 50-100 recent claims (2025-2026):
   - 20 clearly true (well-documented facts)
   - 20 clearly false (debunked misinformation)
   - 20 partially true (mixed claims, nuanced)
   - 10 ambiguous or opinion-like (should trigger "non-verifiable" detection)
   - 10-30 screenshot images of social media posts (for OCR testing)
4. Save as `manual_curated.json` with schema: `{claim, ground_truth_score, category, source_url, notes}`

**Gate:** 200+ labeled claims ready. Schema is consistent across all dataset files.

### Step 3.2 — Component-Level Evaluation
**Files:** `eval/tests/test_sfact_retrieval.py`, `test_bling_calibration.py`
**Depends on:** Step 3.1

1. **S_fact eval:** Run all claims through fact-check agent. Measure:
   - Recall@5 for claims with known fact-checks
   - Precision@5 (is the top result relevant?)
   - False positive rate on the semantic cache at 0.95 threshold
2. **B_ling eval:** Run 100 headlines through linguistic agent. Compare against human ratings (have 3 people rate 10 headlines on each axis, extrapolate). Measure Spearman correlation.
3. **OCR eval:** Run screenshot test images through OCR agent. Measure character error rate.

**Gate:** Recall@5 ≥ 0.6, B_ling Spearman ≥ 0.5 (iterate if not met), OCR CER < 15%.

### Step 3.3 — End-to-End Evaluation
**File:** `eval/tests/test_ctotal_e2e.py`
**Depends on:** Step 3.2

1. Run full pipeline on all 200+ claims
2. Compute:
   - MAE between C_total and ground truth
   - Spearman rank correlation
   - Bucketed accuracy (False/Mixed/True 3-class)
   - Calibration curve (plot expected vs actual accuracy per score bin)
3. Identify failure modes:
   - False claims scoring > 0.5 (system fooled)
   - True claims scoring < 0.5 (system too skeptical)
   - Log each failure for debugging

**Gate:** MAE ≤ 0.25 (stretch: ≤ 0.20), Bucketed accuracy ≥ 65% (stretch: ≥ 70%).

### Step 3.4 — Weight Optimization
**File:** `eval/metrics/`
**Depends on:** Step 3.3 (need baseline metrics first)

1. Run grid search over w1, w2, w3 combinations (w1 ≥ 0.4, Σ = 1.0)
2. Use 5-fold cross-validation on eval set
3. Select weights that minimize MAE
4. Re-run eval with optimized weights, report improvement

**Gate:** Optimized weights improve MAE by at least 0.02 over defaults. Report includes cross-validation variance.

### Step 3.5 — Guardrails (DeepEval)
**File:** `eval/tests/test_guardrails.py`
**Depends on:** Phase 1 complete

1. Install `deepeval`
2. Implement faithfulness test: for each claim, verify the orchestrator's summary only contains information from retrieved sources
3. Implement hallucination test: verify agent outputs don't introduce facts not in the source material
4. Implement bias leakage test: verify political bias ratings ONLY come from AllSides data, never generated by the LLM
5. Test input rejection: verify opinions/questions are flagged as non-verifiable

**Gate:** Faithfulness ≥ 0.80, hallucination rate ≤ 20%, bias leakage = 0 (zero tolerance).

### Step 3.6 — Confidence Calibration
**Depends on:** Steps 3.3-3.5

1. Plot calibration curve: among claims scored 0.7-0.8, what % are actually true?
2. If miscalibrated (e.g., 0.8 scores are only 60% accurate): adjust confidence capping rules
3. Adjust display thresholds for the Trust Gauge color bands based on calibration data

**Gate:** Calibration curve is monotonically increasing (higher scores = higher accuracy).

### Step 3.7 — GitHub Portfolio Polish
**Depends on:** Everything above

1. **README.md** — project overview, architecture diagram (Mermaid), tech stack, setup instructions, eval results summary
2. **Screenshots** — Trust Gauge, Linguistic Radar, full results page on mobile
3. **Eval results** — include MAE, accuracy, calibration plot as images in README
4. **CI workflow** — GitHub Action that runs eval suite on PR (`.github/workflows/eval.yml`)
5. **Demo** — Deploy to Railway/Render + Vercel. Include live demo link in README
6. **Blog post** (optional but high-impact) — write about the evaluation methodology and what you learned about scoring calibration

**Gate:** A stranger can clone the repo, follow the README, and have the project running in < 15 minutes. Eval results are visible in the README.

---

## Dependency Graph (Visual Summary)

```
Phase 0: Foundation
  0.1 Scaffolding ─┐
  0.2 FastAPI      ├──► 0.4 API Keys ──► Phase 1
  0.3 Supabase ────┤
  0.5 Bias Data ───┘

Phase 1: Core Pipeline
  1.1 Fact-Check Agent ──────┐
  1.2 Linguistic Agent ──────┼──► 1.6 Orchestrator ──► 1.7 API Routes
  1.3 Cross-Ref Agent ───────┤                             │
  1.4 Scoring Engine ────────┤                             │
  1.5 Embedding + Cache ─────┘                             │
                                                           │
Phase 2: Multi-Modal + Frontend                            │
  2.1 OCR Agent ──► 2.3 Image API ─┐                      │
  2.2 URL Scraper ─────────────────┤                      │
  2.4 Frontend Setup ──────────────┼──► 2.7 Results Page ──► 2.8 WebSocket
  2.5 Trust Gauge ─────────────────┤
  2.6 Radar + Sources ─────────────┘

Phase 3: Evaluation & Polish
  3.1 Build Eval Dataset ──► 3.2 Component Eval ──► 3.3 E2E Eval ──┐
                                                                     │
                              3.5 Guardrails ──────────────────────┤
                                                                     │
                              3.4 Weight Optimization ◄────────────┤
                                                                     │
                              3.6 Calibration ◄────────────────────┤
                                                                     │
                              3.7 GitHub Polish ◄──────────────────┘
```

---

## Common Pitfalls to Avoid

**1. Don't optimize prematurely.** Get the pipeline working end-to-end with ugly code before refactoring. A working ugly system beats a beautiful half-built one.

**2. Don't skip the eval dataset.** Building the 50-100 manually curated claims is tedious. You'll be tempted to skip it and just use LIAR/FEVER. Don't — those datasets are old, and your system needs to handle 2026 claims. This is the most important part of Phase 3.

**3. Don't load BGE-M3 on 8GB RAM during development.** Use all-MiniLM-L6-v2 (384-dim) for dev. Change the pgvector column dimension accordingly. Swap to BGE-M3 only when deploying on better hardware. Leave the swap as a config change, not a code change.

**4. Don't build the WebSocket agent status (Step 2.8) until everything else works.** It's the most demo-impressive feature but the least architecturally important. Build it last.

**5. Don't generate political bias ratings with the LLM.** This is a trust landmine. Use AllSides static data only. If a domain isn't in the AllSides database, label it "Unrated" — never let the LLM guess.

**6. Don't try to handle every language.** English only for v2. Multilingual OCR and claim verification is a massive scope increase with minimal portfolio value.

**7. Don't forget to handle the "no results" case gracefully.** Many real claims won't have fact-checks. Your system needs to say "I don't have enough information to verify this" instead of fabricating a score. The confidence capping rules exist for this reason — enforce them.
