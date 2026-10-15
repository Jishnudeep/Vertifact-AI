# VeriFact AI — Frontend MVP Design Brief (Plan 7 spec)

**Status:** draft for the project lead's approval
**Roadmap:** Phase B of `docs/superpowers/specs/2026-10-15-realigned-roadmap-design.md`
**Product truth:** `PRODUCT.md` (repo root). **API contract:** `backend/app/api/schemas.py`, TDD §3.
**Chosen composition:** `.impeccable/mocks/decision/r2-transit.html` (code-led sketch, picked 2026-10-15; `.impeccable/` is local-only and git-ignored, so it exists on the project lead's machine)
**Quality bar (finish level, not a layout to copy):** `.impeccable/quality-bar/board.webp`, `.impeccable/quality-bar/hero.webp`

This brief is the spec Plan 7 is written from. A builder (Antigravity or Claude Code) must be able to build from it without asking the project lead.

---

## 1. Job and audience

- **Primary:** someone on a phone who just saw a claim in their feed and wants to know, in one glance, whether it holds up, and why. Impatient, mid-scroll, often one-handed.
- **Secondary:** a hiring manager or interviewer judging engineering rigor. Their depth (which agent did what, timings, weights, the formula) sits one scroll or tap below the verdict and never blocks the glance.
- **Visitor mode:** Operate. Clarity, consistency and honest states outrank expression. The world lends type, palette, density and one signature move; layout, navigation and controls stay standard web controls.

## 2. Outcome and proof

- **Primary task:** paste a claim → get a verdict with its score, its confidence and its sources.
- **Success:** the user can say what VeriFact concluded, how sure it is, and name one source, within ~3 seconds of the result appearing.
- **Product-specific truth the screen must show:**
  - the composite score is built from three agents with fixed weights;
  - uncertainty is first-class: `unverified` shows **"Not enough evidence"** and no number;
  - bias labels come only from AllSides, `Unrated` included;
  - a degraded agent is shown, not hidden.
- **No fabricated proof:** no accuracy figures, user counts, or testimonials anywhere (PRODUCT.md "Evidence on Hand").

## 3. Selected direction: Midnight Transit Diagram

The project lead chose this direction on 2026-10-15 from a bolder re-roll. It is pinned: do not run a new concept roll.

**Sketch vs brief:** the chosen sketch fixes the *composition* (arrangement, density, diagram-then-verdict-then-sources order, type character). Its demo data and a few details are wrong and must not be copied: its Fact-Check line is amber, its "41 Disputed" plate is amber (disputed claims are capped at 0.33, so they are always in the crimson band), "capped at 45" is not a real cap, and it says "partial service". Where the sketch and this brief disagree, **this brief wins**. The finish review audits the build against the sketch's composition and this brief's rules.

### Direction contract

- **THESIS:** The score is an interchange: the three agent lines run in from the edges and meet at one porcelain circle that holds the verdict. Refuses the category default of a semicircle gauge on a stack of cards.
- **OWN-WORLD:**
  - **Ground:** midnight enamel `#0F1B3D` (shell and page). One raised surface tone `#16244D` for the details sections.
  - **Type and ticks:** porcelain `#F4F1EA`.
  - **Agent line inks:** Fact-Check lilac `#B9A2F2`, Cross-Ref teal `#2BB3A3`, Linguistic porcelain `#F4F1EA` with a navy casing.
  - **Line geometry:** 8px lines bending only at 0°, 45° and 90°; stations as porcelain-filled circles with an ink ring.
  - **Type:** one humanist transit grotesk (Hanken Grotesk, or the closest obtainable) at fixed sizes, tabular numerals.
  - **Verdict colour:** appears in exactly two places, the interchange ring and the verdict word.
- **STORY:** The visitor reads the call at the interchange. They can follow any line back to see what that agent found and how long it took, then open a station (a source) to read it. They leave knowing the verdict, how sure VeriFact is, and why.
- **FIRST VIEWPORT (375px):**
  - Header row: wordmark left, "New check" button right.
  - The claim in quotes, 2–3 lines.
  - The diagram, about 300px tall: three labelled lines (agent name · duration) bend into the interchange circle.
  - Inside the circle: score, verdict word, cap note when `cap_rule` is set (§8).
  - Directly below: verdict line plus the first warning, then the component legend (`score × weight` per line).
- **FORM:** Midnight Transit Diagram (catalog world `wayfinding-cartography-signage-midnight-transit-diagram`), chosen in the bolder re-roll; seed: none (unavailable).
- **FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

### Colour rules (binding)

| Role | Colours | Rule |
|---|---|---|
| Verdict bands (TDD §9.1 thresholds) | 0–33 `#DC2626` crimson "Likely false", 34–66 `#F59E0B` amber "Mixed", 67–100 `#10B981` emerald "Likely true"; `unverified` → slate `#8A94B0` "Not enough evidence" (no number) | Interchange ring and verdict word only. Always paired with the word. |
| Agent lines | lilac, teal, porcelain (above) | Never red, blue, amber or green: those belong to verdicts and AllSides chips. |
| AllSides chips (TDD §9.3) | Left blue, Lean Left light blue, Center gray, Lean Right light red, Right red, Unrated neutral outline | Rendered exactly as the API's `bias_label`; never computed client-side. |
| Stance chips | Supports / Refutes / Neutral | Words first; an outline style, not the verdict colours. |

### Copy rule

Transit is the visual language, never the vocabulary. Critical copy stays plain: "Partial analysis", not "partial service"; "Disputed", not "delayed". Do not use transit or game metaphors in verdicts, warnings or errors.

## 4. Scope and boundaries

**In scope (Plan 7):** the routes in §5, every state in §6, the backend additions in §8, responsive behavior in §7, tests in §10.

**Out of scope for Plan 7:**
- URL and screenshot input (Plan 8; the InputBox keeps a slot for them but ships text-only);
- the WebSocket status feed (stretch);
- the GDELT coverage panel (stretch);
- accounts, history, settings, dark/light toggle (the world is dark by design).

**Anti-goals (would make it wrong even if it looks polished):**
- a number shown for an `unverified` claim, or a score shown when all agents failed;
- a degraded agent drawn as healthy;
- a bias label inferred, guessed or recoloured;
- transit decoration that hides the verdict;
- fake per-agent progress that pretends to know which agent finished first.

## 5. Screens and routes

| Route | Screen | Content |
|---|---|---|
| `/` | Home | Claim input (multiline, 2000-char counter, Submit). Three tappable example claims (author them; plain factual claims). **Recently checked**: up to 10 rows from `GET /api/results/recent`, each with a small interchange dot in its verdict colour, the claim (2 lines max), the verdict word, relative time; a row opens `/r/{id}`. Empty state: examples only, no "nothing here" filler. |
| (in place) | Checking | The claim, then the diagram with all three lines drawn **dashed and moving** toward an empty interchange labelled "Checking…", plus an elapsed-seconds counter. No per-agent completion is shown, because the API does not stream it (honesty rule). When the response lands, lines resolve to their real state in one orchestrated transition. |
| `/r/:id` | Result | The first viewport from §3, then the sections in §5.1. Arrived from a submit (navigate here when the response has an `id`) or from a shared link (`GET /api/results/{id}`). When `id` is `null` (slow DB write), render the result in place without a share link. |
| `*` | Not found | Plain message plus "New check". |

### 5.1 Result page, below the first viewport

1. **How the score was built.** The formula with real numbers: `C = 0.50·S_fact + 0.20·(1 − B_ling) + 0.30·V_consensus`, using `trust_score.weights`, then the cap when `cap_rule` is set ("capped from raw_c_total"). Plus a table, one row per agent: status, duration, component value, weight, points.
2. **Sources**, grouped by the agent that found them, using the new `found_by` field (§8). Do not group by `type`: Fact-Check's web-search results come back as `type: "corroborating"`.

   Each source row has the same fixed furniture in fixed slots: publisher, AllSides chip, stance chip (`null` → none), linked title (opens in a new tab, `rel="noopener noreferrer"`), rating when present. Under the list: **"Media bias ratings: AllSides.com (CC BY-NC 4.0)"** as a link (license requirement).
3. **Language analysis.** A Recharts radar with 4 axes (Clickbait Syntax, Emotional Framing, Sensationalism, Informational Density): the API sends values 0–1, displayed ×100 (0–100). Porcelain strokes at 30% fill, with the exact value printed beside each axis label. The API returns no flagged phrases, so the tooltip shows the value only; do not invent phrases. When the Linguistic agent is degraded, the API's values are 0.5 placeholders: show "Language analysis unavailable for this check" instead of the radar.
4. **Google search suggestions.** When `search_suggestions_html` is present, render it unmodified in `<iframe sandbox="allow-popups allow-popups-to-escape-sandbox" srcdoc="…">` with a fitted height. Google's terms require it to be shown and unaltered. Label it "Related Google searches".
5. **Footer row:** processing time, the check date (`created_at`), "Cached result" only when a fresh submit was answered from the cache (`cached: true` on `POST /api/verify`), and a share button (Web Share API, falling back to copy-link). A shared page opened via `/r/:id` reads "Checked on <date>", not "Cached result".

## 6. States (every one must be built and testable with fixtures)

| State | Trigger | Rendering |
|---|---|---|
| Fact-checked | `evidence_status: "fact_checked"` | Ring and verdict word in the band colour; score shown. |
| Corroborated | `"corroborated"` | Same; the Cross-Ref line carries the emphasis. |
| Disputed | `"disputed"` | Band colour plus a **Disputed** chip under the interchange; the cap note when `cap_rule` is set. |
| Unverified | `"unverified"` | Slate ring; the circle reads **"Not enough evidence"**. **No composite score anywhere**: not in the circle, not in the Recently checked row, not as the formula's result or a points/total column. The per-agent inputs (component values and weights) and the warnings still render, labelled "Inputs measured — no score given". |
| Partial analysis | an agent degraded (from the §8 `agents` field) | That agent's line drawn dashed and dimmed, with its status word; the warning shows below the verdict. |
| Low / insufficient confidence | `confidence.level` | A confidence word next to the verdict ("High / Medium / Low / Insufficient confidence"), never hidden. |
| Cached | `cached: true` | "Cached result" note; everything else identical. |
| Loading | request in flight | §5 Checking. |
| Error `invalid_input` (400) | | Inline under the input: what was wrong (empty, too long). Keep the text. |
| Error `non_verifiable_claim` (422) | opinion or question | "VeriFact checks factual claims. This reads as an opinion or a question. Try a statement that can be true or false." Keep the text. |
| Error `request_validation` (422) | | Generic "Something about this input couldn't be read." |
| Error `rate_limited` (429) | | "Too many checks in a minute. Try again shortly." Disable submit for the `Retry-After` time when present. |
| Error `all_agents_failed` (500) | | "All three checks failed, so there is no score." No diagram numbers, no gauge. Offer retry. |
| Network / 503 | | "VeriFact can't reach its checking service right now." Offer retry. |
| Result not found | `GET /api/results/{id}` 404 | Not-found screen. |

## 7. Responsive behavior

Phone first, then up. Verify at **375, 428, 768, 1024, 1280 and 1440px**.

- **375–767:** single column. The diagram is an SVG with a fixed viewBox scaled to width. Station labels stay at a fixed size (≥11px) and the network never shrinks into an unreadable miniature: at narrow widths the label positions reflow (labels above lines), and the diagram never scales the text down.
- **768–1023 (tablet):** single column. The diagram grows to a maximum width of 640px, centered; sources sit in a two-column grid.
- **≥1024 (laptop, desktop):** two columns.
  - **Left (3/5):** claim, diagram, verdict, "How the score was built".
  - **Right (2/5):** sources rail, sticky under the header, then language analysis and search suggestions.
  - Home puts the input left and Recently checked right.
- **≥1440:** content max width 1280px, centered; the midnight ground runs full-bleed.
- No horizontal scroll at any width; touch targets ≥44px; the header stays reachable.

## 8. Backend additions in Plan 7

Both are additive and keep existing clients working. They follow CLAUDE.md §4 and §5.

1. **Per-agent status in `VerifyResponse`:** a new `agents` list, one entry per worker:
   - `name`: `factcheck` | `crossref` | `linguistic`
   - `status`: `ok` | `degraded`
   - `duration_ms`: int or null
   - `error_code`: string or null, from the existing `AgentErrorDetails`, never raw error text

   Timed in the orchestrator's `_call_agent`. Persisted in `results.agent_outputs` so `GET /api/results/{id}` and cached responses return it; older rows without timings return `duration_ms: null` and the UI then shows no duration.
2. **Cap transparency in `trust_score`:** add `raw_c_total` (the score before any cap) and `cap_rule` (`null`, or the id of the confidence rule that lowered it). These only expose values `ScoringEngine` already computes, and the scoring logic does not change. The UI shows "Capped from X to Y: <reason>" only when `cap_rule` is set. It never parses warning strings to detect caps.
3. **`found_by` on each source:** `factcheck` | `crossref`, set in `_consolidate_sources`.
4. **`GET /api/results/{id}` parity:**
   - It returns `agents`, `search_suggestions_html` (persisted by Plan 6), `found_by`, the cap fields and `created_at`.
   - It stops hard-coding `cached: true`; a stored result returns `cached: false`.
   - Its 404 for an unknown or malformed id stays as is; the frontend branches on the status code, not the body.
5. **No raw error text in warnings:** a degraded agent's warning uses a fixed message per `error_code`. For `UNKNOWN` that is "<Agent> check failed". The raw exception message stays in server logs only.
6. **`Retry-After: 60` on 429** responses from the custom rate-limit handler. The frontend falls back to 60 s when the header is missing.
7. **`GET /api/results/recent?limit=10`:**
   - Returns `[{id, claim, c_total, evidence_status, confidence_level, created_at}]` newest first.
   - Limit capped at 20.
   - Read-only and rate-limited like the other routes.
   - Never returns raw input, URLs, embeddings, or agent outputs.
   - Only rows with `featured = true` qualify (§11).

## 9. Constraints

- **Stack:** Vite + React + Tailwind + Recharts. **Nothing is scaffolded yet**: `frontend/` holds only empty `src/{components,hooks,styles}` folders, so Plan 7 starts by creating the Vite project, Tailwind, ESLint and the `npm run dev/build/lint/test` scripts. Router and test tooling are the builder's call; Vitest + Testing Library is the default.
- **API base URL** comes from `VITE_API_BASE_URL`. No secrets in the client. The client never calls third parties except the links the user taps.
- **Accessibility:**
  - Every verdict meaning is also in words.
  - The diagram has a text equivalent: the agent table in §5.1 or an `aria-describedby` summary ("Fact-Check: no verdict found, 2.1 seconds…").
  - Porcelain on midnight for body text (contrast checked ≥4.5:1).
  - A visible 2px porcelain focus ring.
  - `prefers-reduced-motion` turns the line draw and the checking animation into instant state changes.
- **Performance:** Lighthouse mobile Performance ≥85; fonts self-hosted or `font-display: swap`; Recharts loaded only on the result page.
- **Motion grammar:** lines draw along their path once (≈600 ms, ease-out) when a result appears; the interchange fills last. Nothing else animates except the checking state.
- **Design system output:** at the end, the builder runs impeccable's finish review and documenter so `DESIGN.md` and `.impeccable/design.json` exist (impeccable FINISH rule).

## 10. Tests and gate

- **Unit tests:**
  - score → band mapping at 0/33/34/66/67/100;
  - `unverified` renders no number;
  - each error code → its message;
  - bias chip renders the API string verbatim (including `Unrated`);
  - `stance: null` renders no stance chip;
  - a degraded agent renders dashed with its status word;
  - search suggestions render inside a sandboxed iframe only.
- **Fixtures:** one JSON fixture per state in §6 under `frontend/src/fixtures/`, built from the real `VerifyResponse` schema. A dev-only fixture switcher is allowed and must be excluded from production builds.
- **Backend:** pytest for:
  - the `agents` field: healthy, one degraded, cached, legacy row;
  - the cap fields: uncapped, each cap rule;
  - `found_by`: Layer-1 review, Layer-2 web result, Cross-Ref source;
  - `GET /api/results/{id}` parity: no hard-coded `cached`, suggestions present;
  - sanitized degradation warnings: `UNKNOWN` never leaks exception text;
  - the 429 `Retry-After` header;
  - `/api/results/recent`: limit cap, field allow-list, qualification rule.

  `eval-guardian` confirms the scoring outputs are byte-identical apart from the new fields.
- **Gate:**
  - `npm run lint && npm run build` green; frontend tests green; backend `uv run pytest` green;
  - verified at all §7 widths with screenshots;
  - the frontend-viz Done-when (`.agents/skills/frontend-viz/SKILL.md`);
  - impeccable detector + finish review;
  - `/review` (Antigravity) or the `code-reviewer` subagent (Claude Code);
  - `eval-guardian` for the backend change.

## 11. Open decisions (only the project lead can answer)

1. **Recently checked privacy — decided 2026-10-15: featured only.** A new `featured BOOLEAN NOT NULL DEFAULT false` column on `results` is set by hand in Supabase. `/api/results/recent` returns only featured rows. The migration is a new numbered file under `backend/app/db/migrations/` that the project lead runs by hand in the SQL editor (the same convention as migrations 03–06). Until any row is featured, the homepage shows only the example claims. Featured rows are **not** exempt from the 90-day retention purge (my call: the retention policy wins). When they expire, the list falls back to the examples.

No open decisions remain.

## 12. Doc updates in the same change

- `documents/VeriFact_AI_TDD.md` §9.1: the semicircle Trust Gauge is replaced by the interchange; band thresholds and colours unchanged; `unverified` behavior unchanged.
- TDD §3.1: add the `agents` field and `GET /api/results/recent`.
- `.agents/skills/frontend-viz/SKILL.md` and `.claude/skills/frontend-viz/`: component list updated (`TransitDiagram` replaces `TrustGauge`; `AgentPipeline` becomes the checking state of the diagram).
