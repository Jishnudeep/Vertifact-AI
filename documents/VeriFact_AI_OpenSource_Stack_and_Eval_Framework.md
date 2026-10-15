# VeriFact AI: Open-Source Stack & Evaluation Framework

**Author:** Jishnu | **Date:** May 25, 2026 | **Companion to:** VeriFact AI PRD v1.0

---

## Part 1: Open-Source Stack Options by Component

### 1.1 OCR Engine (replacing Google Cloud Vision API)

| Option | Why Consider | Tradeoff | Recommendation |
|--------|-------------|----------|----------------|
| **Surya** | Transformer-based, 90+ languages, built-in line detection + layout ordering + reading order analysis. Powers the Marker PDF-to-markdown tool. Best accuracy on complex layouts. | Newer project, smaller community than PaddleOCR. GPU recommended for speed. | **Top pick for VeriFact.** Screenshot inputs are typically single-claim social media posts, not multi-page documents. Surya's reading-order awareness handles messy screenshot layouts well. |
| **PaddleOCR** | 76K+ GitHub stars, production-grade, 80+ languages. PaddleOCR-VL 1.5 (2026) adds vision-language model for document parsing. Best for structured docs (invoices, tables). | Built on PaddlePaddle framework (not PyTorch). Adds a non-standard dependency. | Strong alternative. Use if you need table-heavy OCR or CJK language support. |
| **docTR** | Built on PyTorch AND TensorFlow. Layout-aware, spatial text output. Good middle ground. | Smaller community. Less battle-tested at scale. | Good if you want a PyTorch-native OCR pipeline without the PaddlePaddle dependency. |
| **EasyOCR** | Simple API, 80+ languages, PyTorch-based. Easy to get started. | Slower inference, lower accuracy on complex layouts compared to Surya/PaddleOCR. | Fine for prototyping, but you'll likely outgrow it. |

**Verdict:** Go with **Surya** for primary OCR. It's PyTorch-native (fills your resume gap), handles screenshot layouts well, and is technically impressive to talk about in interviews. Fall back to PaddleOCR if you hit edge cases with CJK text.

---

### 1.2 LLM (for orchestrator reasoning, claim decomposition, linguistic analysis)

ADK supports open-source models via **LiteLLM** integration. You can serve models locally with **Ollama** or **vLLM** and connect them to ADK through LiteLLM's OpenAI-compatible interface.

| Option | Parameters | Why Consider | Tradeoff |
|--------|-----------|-------------|----------|
| **Qwen3-30B-A3B** (MoE) | 30B total, ~3B active | Top-tier tool calling and agentic workflows. 1M context window in Qwen 3.6 Plus. Strong reasoning. | MoE architecture = larger disk footprint even though inference is efficient. |
| **Llama 3.1 8B / 70B** | 8B or 70B | Meta's workhorse. Native function calling. Strong on Berkeley Function Calling Leaderboard. Well-documented ADK integration via vLLM. | 8B may lack nuance for complex claim analysis. 70B needs serious GPU. |
| **Mistral 7B / Mixtral 8x7B** | 7B / 46.7B (MoE) | Excellent instruction following. Mixtral gives near-70B quality at MoE efficiency. | Smaller community ecosystem than Llama. |
| **Gemma 2 9B / 27B** | 9B or 27B | Google's open model, natural fit with ADK. Strong on reasoning benchmarks for its size. | Still Google ecosystem (if you're trying to diversify). |

**Verdict:** Use **`groq/openai/gpt-oss-120b`** (hosted Groq free tier) for development/testing. This supersedes the original "Llama 3.1 8B" pick: verified 2026-09-10, Groq returns `404 model_not_found` for both `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` on our key, and `GET /openai/v1/models` lists no Llama chat model at all — gpt-oss-120b is chosen over 20b for reliably emitting the strict JSON our agents parse (it is a reasoning model, hence `max_tokens=1024`). For production/demo quality, serve **Qwen3-30B-A3B** or **Llama 3.1 70B** via vLLM. The key is that your ADK agents are model-agnostic via LiteLLM, so you can swap without code changes.

**ADK + LiteLLM setup pattern:**
```python
# In your ADK agent config, point to local vLLM or Ollama endpoint
# via LiteLLM's OpenAI-compatible interface
model = "ollama/llama3.1:8b"  # or "vllm/qwen3-30b-a3b"
```

---

### 1.3 Embedding Model (for semantic cache, cross-reference vector search)

| Option | Dimensions | Why Consider | Tradeoff |
|--------|-----------|-------------|----------|
| **BGE-M3** (BAAI) | 1024 | Best open-source embedding for production. Supports dense + sparse + multi-vector retrieval in one model. Multilingual. | Heavier than MiniLM. |
| **all-MiniLM-L6-v2** | 384 | 22M params, fast inference, good baseline. Most downloaded on HuggingFace. | Lower accuracy on nuanced semantic matching vs. larger models. |
| **Nomic Embed v2** | 768 | Strong multilingual retrieval, competitive with OpenAI text-embedding-3-small. Open source with open training data. | Newer, less community validation. |
| **GTE-Large** (Alibaba) | 1024 | Top MTEB scores in its class. Good for English-focused tasks. | English-primary. |

**Verdict:** Use **BGE-M3** for your vector search pipeline. It natively supports hybrid retrieval (dense + sparse in one model), which directly fills the "hybrid search" gap from your resume. This is a single model that gives you both BM25-style keyword matching and dense semantic search. That's a strong interview talking point.

---

### 1.4 Vector Database / Search

Your PRD specifies pgvector via Supabase. That's fine for a side project, but here are the options:

| Option | Why Consider | Tradeoff |
|--------|-------------|----------|
| **pgvector (via Supabase)** | Already in your PRD. PostgreSQL extension, zero extra infra. Supabase gives you auth + storage + realtime for free tier. Supports HNSW and IVFFlat indexes. | Not purpose-built for vector search. Slower at scale than dedicated vector DBs. |
| **Qdrant** | Rust-based, purpose-built. Supports hybrid search natively (dense + sparse vectors). Filtering + payload storage. Docker single-container deployment. | Extra infrastructure to manage. |
| **Milvus** | The one Apple's JD specifically names. Distributed, high-performance, supports GPU-accelerated search. | Overkill for a side project. Complex deployment. |
| **ChromaDB** | Simplest setup. Embedded mode (no server needed). Good for prototyping. | Not production-grade. Limited filtering. |

**Verdict:** Stick with **pgvector via Supabase** for the project. It keeps your infra simple and the free tier is generous. But — and this matters for your resume — mention Qdrant or Milvus as alternatives you evaluated and explain *why* you chose pgvector (simplicity, sufficient for the scale, co-located with your relational data). That shows you know the landscape.

---

### 1.5 Fact-Check Data Sources (replacing fragile scraping)

This is where the PRD has its biggest practical gap. Here's what actually works:

| Source | Type | Access | What It Gives You |
|--------|------|--------|-------------------|
| **Google Fact Check Tools API** | REST API | Free, API key required | Searches ClaimReview markup from 100+ fact-check orgs worldwide. Returns claim text, claimant, rating, publisher, URL. This is your S_fact signal. |
| **DataCommons Fact Check Feed** | Downloadable dataset | Free, open | Historical ClaimReview data dump. Use for building your evaluation test set and populating your vector cache. |
| **LIAR Dataset** | Research dataset | Free (academic) | 12,836 claims from PolitiFact with 6-class labels (pants-fire to true). Use for evaluation, NOT for production lookups. |
| **FEVER Dataset** | Research dataset | Free (academic) | 185K claims verified against Wikipedia. Labels: SUPPORTED, REFUTED, NOT ENOUGH INFO. Best for training/evaluating claim verification models. |
| **FACT5 (2025)** | Research dataset | Free (academic) | 150 real-world claims with 5 ordinal truthfulness classes. Small but high-quality. Good for fine-grained eval. |
| **AllSides Media Bias Ratings** | Static dataset | Kaggle CSV + GitHub scraper | Domain-level bias ratings (Left, Lean Left, Center, Lean Right, Right). Use for REQ-004 anchored political rating. |
| **Ad Fontes Media Bias Chart** | Static dataset | Limited free access | Reliability + bias 2D ratings per outlet. More granular than AllSides. |

**Architecture recommendation:**
- **Runtime fact-checking:** Google Fact Check Tools API (reliable, maintained, free) + Google Search grounding via Gemini API for finding corroborating/contradicting sources.
- **Bias rating:** Static CSV lookup from AllSides dataset (Kaggle). No API calls needed. Pre-load into your PostgreSQL DB.
- **Evaluation ground truth:** LIAR + FEVER + FACT5 datasets. These never hit production — they're for measuring your system's accuracy.

---

### 1.6 Web Search API (for real-time source corroboration)

| Option | Free Tier | Why Consider |
|--------|-----------|-------------|
| **Google Search Grounding (Gemini)** | 5,000 req/month free | Native Google search grounding via ADK / Gemini API. Direct fresh index coverage and source citations. |
| **Tavily** | Previously used | Retired in Plan 6 in favor of Google Search grounding. |
| **Brave Search API** | 2,000 req/month free | Independent index (not Google/Bing reskin). Good result quality. |
| **SerpAPI** | 100 req/month free | Google results via API. Most accurate but expensive beyond free tier. |
| **Serper.dev** | 2,500 req/month free | Google results, cheaper than SerpAPI. |

**Verdict:** **Google Search Grounding (Gemini)** is the best fit (replaces Tavily in Plan 6). It provides 5,000 free searches/month, fresh Google search index grounding, and native ADK integration.

---

### 1.7 NLP / Linguistic Analysis Libraries

| Library | Use In VeriFact |
|---------|----------------|
| **spaCy** | Named entity recognition (NER) for extracting people, organizations, locations from claims. Use for entity-to-publication matching in REQ-004. |
| **TextBlob / VADER** | Baseline sentiment and subjectivity scoring. Feeds into B_ling (linguistic volatility). Lightweight, no GPU needed. |
| **Hugging Face Transformers** | Fine-tuned classifiers for sensationalism detection, clickbait detection, emotional framing. Use pre-trained models or fine-tune your own (PyTorch). |
| **sentence-transformers** | Sentence-level embeddings for semantic similarity. Powers the short-circuit cache and cross-reference matching. |

---

### 1.8 Full Recommended Stack Summary

```
Frontend:       Vite + React + Tailwind CSS + Recharts
Backend:        FastAPI (Python 3.11+)
Agent Framework: Google ADK + LiteLLM (model-agnostic)
LLM:            groq/openai/gpt-oss-120b (dev, hosted Groq) / Qwen3-30B-A3B (prod) via vLLM
OCR:            Surya
Embeddings:     BGE-M3 (via sentence-transformers)
Vector Search:  pgvector (via Supabase)
Database:       Supabase (PostgreSQL + auth + storage)
Fact-Check API: Google Fact Check Tools API
Web Search:     Google Search grounding (Gemini)
Bias Data:      AllSides CSV (static, pre-loaded)
NLP:            spaCy + VADER + HuggingFace Transformers
Eval Framework: DeepEval + custom harness (see Part 2)
```

---

## Part 2: Evaluation Framework Design

This is the section the PRD was completely missing. Without this, VeriFact AI is a demo. With it, it's a credible ML project.

### 2.1 The Core Problem

VeriFact produces a composite score (C_total) from 0.0 to 1.0. But what does 0.73 mean? Is that accurate? How do you know? You need to answer three questions:

1. **Does S_fact (fact-check lookup) correctly find matching claims?** (Information Retrieval eval)
2. **Does B_ling (linguistic volatility) correlate with actual sensationalism?** (Classification eval)  
3. **Does V_consensus (cross-source validation) reflect real coverage patterns?** (Retrieval + aggregation eval)
4. **Does C_total (the composite) actually predict claim truthfulness?** (End-to-end eval)

### 2.2 Ground Truth: Building Your Test Set

You need a labeled evaluation dataset. Here's how to construct it:

**Step 1: Source claims with known verdicts**

| Source | Claims | Labels | Use |
|--------|--------|--------|-----|
| LIAR Dataset | 12,836 | 6-class (pants-fire through true) | Map to your 0-1 scale. pants-fire=0.0, false=0.15, barely-true=0.3, half-true=0.5, mostly-true=0.75, true=1.0 |
| FEVER | 185,445 | 3-class (SUPPORTED, REFUTED, NOT_ENOUGH_INFO) | Use SUPPORTED=1.0, REFUTED=0.0, NEI=0.5 for a coarser signal |
| FACT5 | 150 | 5-class ordinal | High-quality claims for nuanced evaluation |
| Manual curation | 50-100 | Your own labels | Recent claims (2025-2026) that test temporal relevance. The academic datasets are older. |

**Step 2: Create input variants**

For each claim, create multiple input formats to test multi-modal ingestion:
- Raw text headline
- URL to a source article (if findable)
- Screenshot image of the claim (render text as image for OCR testing)

**Target: 200-300 evaluation claims** covering a mix of clearly true, clearly false, partially true, and ambiguous claims.

### 2.3 Component-Level Evaluation

#### S_fact: Fact-Check Retrieval Accuracy

**What you're measuring:** Given a claim, does the system find the correct fact-check article(s)?

**Metrics:**
- **Recall@K:** Of claims that have a known fact-check, how many does your system find in its top K results? Target: Recall@5 >= 0.7
- **Precision@K:** Of the fact-checks returned, how many are actually about the input claim (not a different claim)? Target: Precision@5 >= 0.6
- **MRR (Mean Reciprocal Rank):** Where does the correct fact-check appear in the ranked results? Higher = better.
- **Semantic match accuracy:** For the short-circuit cache (REQ-003), measure false positive rate at your 0.95 cosine threshold. A false match serves wrong cached results — this is catastrophic.

**Test protocol:**
```python
# Pseudocode for S_fact evaluation
for claim in eval_set:
    results = fact_check_agent.search(claim.text)
    top_k_urls = [r.url for r in results[:5]]
    
    # Does the known fact-check URL appear?
    recall = 1 if claim.known_factcheck_url in top_k_urls else 0
    
    # Is the top result relevant? (LLM-as-judge or manual)
    precision = judge_relevance(claim.text, results[0])
```

#### B_ling: Linguistic Volatility Calibration

**What you're measuring:** Does the sensationalism/clickbait score reflect human judgment?

**Metrics:**
- **Spearman rank correlation** between B_ling scores and human-annotated sensationalism ratings. Target: rho >= 0.6
- **Per-axis accuracy** for the 4-axis radar (Clickbait Syntax, Emotional Framing, Sensationalism, Informational Density). Each axis should independently correlate with human judgment.

**Calibration dataset:**
Take 100 headlines. Have 3 people rate each on a 1-5 scale for each axis. Use the average as ground truth. Compare against your model's output. This is straightforward and you can do it with friends/colleagues.

**Red flag detection:** Manually check claims where B_ling is very low (model says "calm, factual") but the claim is actually false. A false claim written in calm language should NOT get a high C_total just because it sounds professional.

#### V_consensus: Cross-Source Validation

**What you're measuring:** Does coverage density correlate with claim reliability?

**Metrics:**
- **Source diversity score:** Number of unique domains covering a claim. Verify against manual search.
- **Framing consistency:** Do the sources agree on the framing? Measure via embedding similarity of source summaries.
- **Absence detection:** For clearly false claims, verify that V_consensus is low (few reputable sources corroborate them).

### 2.4 End-to-End Evaluation

#### C_total vs. Ground Truth Verdict

**What you're measuring:** Does the final composite score predict claim truthfulness?

**Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| **MAE (Mean Absolute Error)** | Average distance between C_total and mapped ground truth score | <= 0.2 |
| **Spearman Correlation** | Rank correlation between C_total and ground truth | >= 0.65 |
| **Bucketed Accuracy** | Bin C_total into 3 buckets: False (0-0.33), Mixed (0.34-0.66), True (0.67-1.0). Measure classification accuracy. | >= 0.70 |
| **Calibration** | Among claims scored 0.8, are ~80% actually true? Plot calibration curve. | Monotonically increasing |

**Calibration curve (reliability diagram):** C_total is a credibility score, not a probability. The 'calibration curve' is a reliability diagram. Predicted C_total is binned (5 equal-width bins); each bin's mean continuous ground-truth score is plotted. It passes when populated bins (n ≥ 3) are monotonic. The Brier score is not used (the ground truth is continuous).

**Critical failure modes to test:**
- **Sophisticated misinformation:** False claims from reputable-looking sources with calm language. If C_total > 0.5, your system is fooled.
- **True but sensational:** Real facts reported with clickbait headlines. If C_total < 0.5, your B_ling weight is too high.
- **Novel claims:** True or false claims that no fact-checker has covered yet. S_fact returns nothing. Does the system handle gracefully or crash?
- **Stale cache:** A claim was true when cached but has since been debunked. Does the 0.95 cosine short-circuit serve outdated results?

### 2.5 Guardrails & Safety Evaluation

Use **DeepEval** for automated evaluation of LLM outputs within the pipeline:

```python
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Test: Does the evaluation agent's summary faithfully represent source data?
test_case = LLMTestCase(
    input="Is claim X true?",
    actual_output=agent_summary,
    retrieval_context=[source_1_text, source_2_text]  # what the agents retrieved
)

faithfulness = FaithfulnessMetric(threshold=0.8)
hallucination = HallucinationMetric(threshold=0.5)
```

**Guardrails to implement and test:**
- **Source attribution / Hallucination detection:** Every factual statement in the output must trace to a retrieved source. Guardrails catch >80% of hallucinated outputs (linked to PRD §4 Phase 3 exit criteria). Defined as: On a seeded set of ≥30 verdict explanations, each paired with its source snippets (half faithful, half with an injected unsupported fact), the DeepEval faithfulness check flags ≥80% of the injected ones and ≤20% of the faithful ones.
- **Confidence calibration:** If S_fact returns no results, C_total must be capped (e.g., max 0.5) with a "no verified sources found" warning.
- **Input rejection:** Detect and refuse non-factual inputs (opinions, questions, nonsense). Measure false rejection rate on valid claims.
- **Bias leakage:** The system should never generate its own political bias assessment. Verify that bias ratings ONLY come from the AllSides static data. Test with claims about politically charged topics.

### 2.6 Evaluation Pipeline Architecture

```
eval/
  datasets/
    liar_mapped.json          # LIAR dataset mapped to 0-1 scale
    fever_subset.json         # 500-claim FEVER subset
    manual_curated.json       # 50-100 recent hand-labeled claims
    screenshot_variants/      # Image versions of text claims
  
  tests/
    test_sfact_retrieval.py   # Fact-check lookup precision/recall
    test_bling_calibration.py # Linguistic volatility correlation
    test_vconsensus.py        # Cross-source validation checks
    test_ctotal_e2e.py        # End-to-end composite score accuracy
    test_guardrails.py        # DeepEval faithfulness + hallucination
    test_ocr_accuracy.py      # OCR text extraction accuracy on screenshots
    test_cache_safety.py      # Semantic cache false positive rate
  
  metrics/
    report_generator.py       # Generates eval report with all metrics
    calibration_plot.py       # Plots C_total calibration curve
  
  conftest.py                 # Shared fixtures, dataset loaders
  run_eval.py                 # Entry point: runs all evals, outputs report
```

### 2.7 Weight Tuning for C_total

The PRD says weights are "dynamically bounded" but doesn't explain how. Here's a concrete approach:

**Method: Grid search on eval set**

```python
import numpy as np
from itertools import product

best_mae = float('inf')
best_weights = None

# Grid search over weight combinations that sum to 1.0
for w1 in np.arange(0.2, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.5, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.1:
            continue
        
        predictions = w1 * s_fact + w2 * (1 - b_ling) + w3 * v_consensus
        mae = np.mean(np.abs(predictions - ground_truth))
        
        if mae < best_mae:
            best_mae = mae
            best_weights = (w1, w2, w3)
```

**Constraint:** w1 (fact-check match) should always be the dominant weight. A direct fact-check from PolitiFact should matter more than linguistic tone or coverage volume. Enforce w1 >= 0.4 in your grid search.

**Validation:** Use k-fold cross-validation on your eval set to prevent overfitting the weights to your test data. Report mean +/- std of MAE across folds.

### 2.8 Segmentation, Leave-Source-Out, and Replay (C3, C4)

#### Segment Definitions and Selective Accuracy
Blended metrics across heterogeneous claim types obscure pipeline behavior. Claims with an authoritative fact-check match (`S_fact`) carry high certainty, whereas claims without one rely on cross-referencing and tone analysis with confidence bounded to roughly 0.25–0.75 by construction.

VeriFact AI segments evaluation metrics by:
- **`source_dataset`:** `curated`, `liar`, or `fever`. Metrics are never blended across datasets.
- **`evidence_status`:** `fact_checked`, `disputed`, `corroborated`, `unverified`, or `insufficient_evidence`.

Claims with `evidence_status == "unverified"` are treated as abstentions:
- **Selective Bucketed Accuracy:** Bucketed accuracy computed strictly over the answered set (`fact_checked`, `disputed`, `corroborated`).
- **Coverage:** Proportion of verifiable claims that received an answered status (`n_answered / n_total`).

#### Leave-Source-Out Domain Exclusion
When evaluating claims derived from a fact-checking benchmark (such as LIAR, whose labels originate from PolitiFact), the pipeline must not grade a claim using the labeller's own article:
- Each claim may specify an `origin_domain` (e.g., `politifact.com` for LIAR).
- During evaluation, evidence retrieval calls filter out all results originating from `origin_domain` across both web search and fact-check endpoints.

#### Replay and Reproducibility
To prevent burning API quotas (Gemini, Groq) during repeated eval iterations and to guarantee deterministic test execution:
- `--replay record`: Calls live evidence and LLM APIs, saving raw responses to `eval/results/evidence_cache/`.
- `--replay replay`: Serves responses strictly from the cache. Any cache miss raises `ReplayMissError` and invalidates the run (`STATUS_INVALID_RUN`).
- `--replay live`: Calls external APIs live without caching. Refused when any claim in the dataset has `origin_domain` set (leave-source-out requires record or replay mode).

#### Benchmark Label Maps
- **LIAR (6-class):** `pants-fire` = 0.0, `false` = 0.15, `barely-true` = 0.3, `half-true` = 0.5, `mostly-true` = 0.75, `true` = 1.0.
- **FEVER (3-class):** `SUPPORTS` = 1.0, `REFUTES` = 0.0, `NOT ENOUGH INFO` = 0.5.

---

## Part 3: Resume Impact Checklist

When this project is done, your resume should be able to claim:

- [ ] Built multi-agent orchestration system using Google ADK with open-source LLMs (Llama/Qwen) via LiteLLM
- [ ] Implemented hybrid retrieval (dense + sparse) using BGE-M3 embeddings and pgvector
- [ ] Designed and executed evaluation framework measuring retrieval precision/recall, calibration, and faithfulness
- [ ] Built LLM guardrails using DeepEval for hallucination detection and source attribution
- [ ] Deployed PyTorch-based OCR pipeline (Surya) for multi-modal input processing
- [ ] Implemented semantic caching with cosine similarity thresholding, reducing API costs by X%
- [ ] Used FastAPI async architecture for concurrent agent execution with sub-N-second response times

Each of these directly addresses a gap identified in the Apple JD analysis.
