# PRD/TDD Review — Findings & Decisions (2026-09-23)

Source: full review of `documents/VeriFact_AI_PRD_v2.md` and `documents/VeriFact_AI_TDD.md`
against the current code, with external facts checked on 2026-09-23. This file is the spec
that the fix plans in `docs/superpowers/plans/2026-09-23-*.md` implement.

## Decisions (user, 2026-09-23)

- **D1 — Stance & claim equivalence use a local NLI cross-encoder**
  (`cross-encoder/nli-deberta-v3-small`, Apache-2.0, ~0.1B params, CPU). A single model,
  loaded once at startup, serves both source stance (C1) and cache equivalence (C2).
  No LLM in either path.
- **D2 — Add an explicit evidence status.** `C_total` and the PRD formula stay as they are.
  Every response also carries
  `confidence.evidence_status ∈ {fact_checked, disputed, corroborated, unverified}`.
  When the status is `unverified`, the UI shows "Not enough evidence" instead of an amber
  score. Eval metrics are reported separately per status.
- **D3 — Jev (TypeSafe) is tabled.** It will not be adopted now. Revisit behind a
  `ClaimClassifier` interface later.

## Findings

### Critical
- **C1 — `V_consensus` measures coverage, not agreement.** `crossref_agent.py` scores
  `min(count/5,1) × avg_relevance` over reputable domains, regardless of whether each
  article supports or refutes the claim. As a result, a widely debunked claim scores *high*.
- **C2 — The semantic cache can serve the opposite claim.** Bi-encoder embeddings
  (MiniLM especially) score negations and swapped numbers above 0.95 cosine, and raising
  the threshold does not fix this. There is also no TTL, and legacy rows scored under the
  old consensus logic are still served.
- **C3 — Eval leakage.** LIAR claims come from PolitiFact, so Layer 1/2 retrieves the exact
  article and S_fact becomes the label. FEVER is built from Wikipedia and rarely hits
  ClaimReview. Blended metrics hide both effects.
- **C4 — Score semantics.** With no fact-check, `C_total ∈ [0.25, 0.75]` (≤0.55 when
  V<0.3). Bucketed-accuracy/MAE targets are unreachable by construction and need
  segmentation (see D2).

### High
- **H1** — The latency budget doesn't close: 8 s per agent, plus serial pre- and post-steps,
  plus OCR/scrape. A single request deadline is needed.
- **H2** — The free tiers can't sustain the design. Tavily: 1,000 credits/mo, and advanced
  search costs 2 credits. Groq gpt-oss-120b free tier: 30 RPM / 1K RPD / 8K TPM. The PRD §9
  figure of "14,400 req/day" is stale.
- **H3** — Abuse/security gaps:
  - No global spend budget.
  - `get_remote_address` sees the proxy IP behind Render/Vercel.
  - No SSRF policy for the URL scraper.
  - Prompt-injection handling is not written into the spec.
- **H4** — No retention policy for raw inputs or OCR text.
- **H5** — Hosting and licensing:
  - The Render free tier is 512 MB and sleeps.
  - Railway has no permanent free tier.
  - Supabase free projects pause.
  - AllSides data is CC BY-NC 4.0 (attribution required).
  - Surya weights have a revenue cap.
  - The eval doc mentions "Ad Fontes", which conflicts with the AllSides-only rule.
- **H6** — The Fact Check API risk is underrated. It is `v1alpha1`, and Google removed
  ClaimReview snippets from Search in June 2025.

### Medium (doc/code drift)
- The docs and CLAUDE.md say `gpt-oss-20b`; `config.py` uses `gpt-oss-120b`.
- TDD §4.1's ADK snippet is wrong: `LlmAgent.sub_agents` is LLM-driven transfer, not
  parallel execution. The code uses `asyncio.gather` + `wait_for`.
- PRD and TDD disagree on the partial-analysis cap, the vector dimension, `JSONB[]`,
  the Python version, and LLM rate-limit behaviour. The TDD still lists `requirements.txt`.
- HTTP 422 is overloaded (opinion vs oversize body).
- `RATING_MAP` doesn't cover real `textualRating` values ("Misleading", "Missing context",
  "Satire", "Four Pinocchios").
- Not specified yet:
  - multi-claim inputs
  - primary-claim extraction for URLs
  - the hallucination-catch denominator
  - what a "calibration curve" means for a non-probability score
- `LinguisticResult` in the TDD lacks `degraded`/`error`.

## Plan map

Run order: Plan 1 → Plan 2 → Plan 4 → Plan 3 → Plan 5. See `plans/README.md`.

| Plan | File | Covers |
|------|------|--------|
| 1 | `plans/2026-09-23-stance-aware-consensus.md` | C1, D1, D2 |
| 2 | `plans/2026-09-23-cache-safety.md` | C2 |
| 3 | `plans/2026-09-23-eval-integrity.md` | C3, C4 metrics |
| 4 | `plans/2026-09-23-hardening.md` | H1–H4 |
| 5 | `plans/2026-09-23-doc-sync.md` | Medium drift, H5, H6 |
