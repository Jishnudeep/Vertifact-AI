# Plan 8 — URL and Screenshot Input Implementation Plan (runs last)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Antigravity: start with `/execute-plan` (`.agents/workflows/execute-plan.md`).

**Goal:** A user can submit a news link or a screenshot as well as text, and gets the same verdict screen (roadmap Phase C, which runs after Phases D and E).

**Architecture:**
- **URL path:** `POST /api/verify` with a URL runs the scraper agent before the normal pipeline. The scraper is built on the existing SSRF-guarded fetcher. The headline becomes the claim, and the article body goes only to the Linguistic agent (TDD §4.6).
- **Screenshot path:** a new `POST /api/verify/image` runs Surya OCR, which is loaded once at startup. A confident read is verified as the claim. A weak read comes back to the user to correct before anything is verified.
- **Frontend:** the Plan 7 InputBox gains link detection and an image picker, and the result page shows where the claim came from.

**Tech Stack:** FastAPI, httpx and BeautifulSoup (already present), `surya-ocr`, Pillow, `python-multipart`, pytest/anyio. Frontend: the Plan 7 stack.

**Spec:**
- `docs/superpowers/specs/2026-10-15-realigned-roadmap-design.md` §4 Phase C;
- `documents/VeriFact_AI_TDD.md` §3.2, §4.5 (OCR), §4.6 (scraper), §11 (SSRF);
- `documents/VeriFact_AI_Build_Order.md` Steps 2.1–2.3 (gates);
- `docs/superpowers/specs/2026-10-15-frontend-mvp-design.md` (visual rules, copy rules, states).

## Decisions this plan makes (were not pinned by the spec)

| # | Decision |
|---|---|
| P8-1 | **OCR models load once at startup** when `OCR_ENABLED=true` (the default). This follows CLAUDE.md §4.7, which overrides Build Order 2.1's "load and unload per request". On a low-RAM machine set `OCR_ENABLED=false`; the image endpoint then answers 503 `ocr_unavailable`. |
| P8-2 | **A weak OCR read is never verified.** When the text has fewer than 10 characters, the answer is 422 `insufficient_text`. When confidence < `OCR_MIN_CONFIDENCE` (0.6), the answer is 422 `ocr_low_confidence` carrying `extracted_text`, so the user corrects it and submits it as text. A confident read is verified directly (TDD §3.2 single call). The result offers "Edit and re-check". |
| P8-3 | **The scrape gets its own 5 s budget** (TDD §4.6) *before* the 9.5 s pipeline deadline starts. A URL request can therefore take up to about 15 s; the UI says "Reading the article…" first. |
| P8-4 | **URL detection:** an input is a URL when `input_type == "url"`, or when the trimmed input matches `^https?://\S+$`. Text that merely contains a link is verified as text. |
| P8-5 | **Nothing from an upload is stored except the extracted text.** Images are processed in memory and discarded. `raw_input` stores the sanitized URL or the OCR text. |

## Global Constraints

- **SSRF:** every URL fetch goes through `app/services/page_fetch.py`, which already enforces these rules:
  - `resolve_public_url` on every hop;
  - IP pinning, at most 3 redirects;
  - a 2 MB cap and `text/html` only.

  Never add a second HTTP fetch path.
- **Scraper limits** (TDD §4.6):
  - timeout `SCRAPE_TIMEOUT_S` = 5.0;
  - the body is the first 1,500 words;
  - the headline is `og:title`, else `<title>`, with the site-name suffix removed.
- **Image limits:**
  - PNG or JPEG only, checked by magic bytes **and** Pillow's detected format;
  - at most 5 MB, read in memory, never written to disk;
  - at most 25,000,000 pixels, checked from the header before decoding;
  - resized so the longest edge is at most 2048 px (TDD §4.5).
- **OCR thresholds:** `OCR_MIN_CHARS` = 10, `OCR_MIN_CONFIDENCE` = 0.6, `OCR_TIMEOUT_S` = 6.0. All live in `app/config.py`, never hardcoded.
- **New error codes** (`ErrorResponse.code`):
  - `url_unreadable` (422)
  - `insufficient_text` (422)
  - `ocr_low_confidence` (422, `extracted_text` set)
  - `ocr_unavailable` (503)
  - A bad image type or size is `invalid_input` (400), detail "Upload a PNG or JPEG under 5 MB."
- **Rate limit:** `10/minute` on `POST /api/verify/image`, the same as `/api/verify`.
- **Frontend copy** (brief §6 style), exact strings:
  - `url_unreadable`: "We couldn't read that page. Paste the headline or the claim instead."
  - `insufficient_text`: "We couldn't find enough text in that image. Paste the claim instead."
  - `ocr_low_confidence`: "We couldn't read the screenshot clearly. Check the text below, then verify."
  - `ocr_unavailable`: "Screenshot checking is turned off on this server. Paste the claim as text."
- **Dependencies:** `uv add surya-ocr pillow python-multipart`. Surya weights are Modified OpenRAIL-M (README license table already lists it).
- **Agent contract** (CLAUDE.md §4.3): `ScrapeResult` and `OCRResult` are dataclasses; failures return `degraded=True` with an error and never raise into the route.

## Review Focus

1. **A URL that redirects to a private address, or serves a PDF.** The answer is `url_unreadable` and nothing internal is fetched. → Task 1 `test_scrape_private_redirect_is_unreadable`, `test_scrape_non_html_is_unreadable`.
2. **A paywalled page with only `og:title` and `og:description`.** The headline is still used, and the body falls back to `og:title + og:description`. → Task 1 `test_scrape_paywall_uses_og_description`.
3. **A renamed file:** an SVG or PDF named `.png`, or a PNG declared as `image/jpeg`. Magic bytes decide; a mismatch is 400 `invalid_input`. → Task 4 `test_image_magic_bytes_decide`.
4. **A screenshot with almost no text** (a meme or emoji). The answer is `insufficient_text`, and no agent runs and no quota is spent. → Task 4 `test_insufficient_text_spends_nothing`.
5. **A decompression bomb:** a 20000×20000 PNG under 5 MB. It is rejected from the header before decoding. → Task 3 `test_pixel_limit_rejected_before_decode`.

---

### Task 0: Pre-flight

- [ ] **Step 1: Check that Plans 7, 9 and 10 are merged** (Plan 8 runs last). `docs/superpowers/plans/README.md` must show them `completed`, and `frontend/package.json` must exist. If not, stop and ask the project lead.
- [ ] **Step 2: Check the working tree and branch.** `git status --porcelain` must be empty. Then `git switch -c feat/url-and-screenshot-input`.
- [ ] **Step 3: Baseline.** Run the backend, eval (`-m "not live"`) and frontend test suites. All must be green; record the counts.
- [ ] **Step 4: Ask the project lead** whether this machine has the RAM for Surya (about 2 GB beyond the current backend). If not, Task 3 and Task 5 still run with `OCR_ENABLED=false`, and only their model-backed steps move to a machine that can.

---

### Task 1: Scraper agent on the shared fetcher

**Files:**
- Modify: `backend/app/services/page_fetch.py` (extract a public `fetch_html`; `fetch_page_text` keeps its behavior)
- Create: `backend/app/agents/scraper_agent.py`
- Modify: `backend/app/config.py` (`scrape_timeout_s: float = 5.0`)
- Test: `backend/tests/test_page_fetch.py` (unchanged tests must stay green), `backend/tests/test_scraper_agent.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class FetchedHTML: final_url: str; domain: str; html: str`
  - `async def fetch_html(url: str, expected_domain: Optional[str] = None, *, timeout_s: Optional[float] = None, transport=None, resolver=None) -> Optional[FetchedHTML]`
    - The same guards and the same never-raises contract as today.
    - `timeout_s` defaults to `settings.page_fetch_timeout_s`.
  - `fetch_page_text` becomes `fetch_html` plus the existing `_extract` and `MIN_TEXT_CHARS` check. Its behavior is unchanged.
  - `@dataclass class ScrapeResult: headline: str; body: str; final_url: str; degraded: bool = False; error: Optional[str] = None` (TDD §4.6)
  - `async def run(url: str) -> ScrapeResult` in `scraper_agent.py`. It calls `fetch_html(url, timeout_s=settings.scrape_timeout_s)`.
    - **Headline:** `og:title`, else `<title>`. The last segment after ` | `, ` – `, ` — ` or ` - ` is removed only when it equals `og:site_name` case-insensitively, or contains the domain's brand label (e.g. `reuters` for `reuters.com`).
    - **Body:** paragraph text from `<article>`, else all `<p>`, collapsed to whitespace and cut at 1,500 words. If that body is under 200 characters, the body becomes `og:title + " " + og:description`.
    - **Failure:** no fetch, or an empty headline → `degraded=True` with `error="unreadable"`.

- [ ] **Step 1: Write failing tests.** Use `httpx.MockTransport` and the `make_resolver` helpers from `test_page_fetch.py`.
  - `test_scrape_prefers_og_title_and_strips_site_suffix`: `og:title` "Fed holds rates steady - Reuters", `og:site_name` "Reuters" → headline "Fed holds rates steady".
  - `test_scrape_keeps_dash_inside_headline`: "Mid-term results - what changed" with `og:site_name` "Example News" → unchanged.
  - `test_scrape_falls_back_to_title_tag`.
  - `test_scrape_body_capped_at_1500_words`.
  - `test_scrape_paywall_uses_og_description`.
  - `test_scrape_private_redirect_is_unreadable`.
  - `test_scrape_non_html_is_unreadable`.
  - `test_scrape_timeout_uses_scrape_budget`: a slow handler under `scrape_timeout_s` succeeds where `page_fetch_timeout_s` (2.0) would fail.
- [ ] **Step 2: Run** `uv run pytest tests/test_scraper_agent.py tests/test_page_fetch.py -v`. Expected: the new tests FAIL, the existing ones PASS.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: all PASS.
- [ ] **Step 5: Commit** `feat(input): scraper agent on the SSRF-guarded fetcher`.

---

### Task 2: URL claims through `POST /api/verify`

**Files:**
- Modify: `backend/app/agents/orchestrator.py`:
  - `VerifyRequest.linguistic_text`;
  - `_linguistic` uses it;
  - the `verify_claim` keyword args;
  - the new `verify_url_claim`;
  - the new `UrlUnreadableError`.
- Modify: `backend/app/api/schemas.py`:
  - `VerifyResponse.input_type`, `source_url`, `ocr_extracted_text`, `ocr_confidence`;
  - the `ErrorResponse.code` additions;
  - `ErrorResponse.extracted_text`.
- Modify: `backend/app/api/routes.py` (dispatch on P8-4), `backend/app/main.py` (handler for `UrlUnreadableError` → 422 `url_unreadable`)
- Modify: `backend/app/api/result_format.py` (Plan 7): reads `agent_outputs["input"]`
- Test: `backend/tests/test_orchestrator_workflow.py`, `backend/tests/test_api_routes.py`

**Interfaces:**
- Consumes: Task 1 `scraper_agent.run`.
- Produces:
  - `is_url_input(text: str) -> bool` in `app/utils/validators.py` (P8-4 regex on the trimmed text).
  - `verify_claim(raw_input, input_type="text", bypass_cache=False, *, linguistic_text: Optional[str] = None, stored_raw_input: Optional[str] = None, input_extras: Optional[Dict[str, Any]] = None)`
    - `linguistic_text` is sent to the Linguistic agent only.
    - `stored_raw_input` is persisted as `raw_input` when given.
    - `input_extras` is merged into the response and persisted as `agent_outputs["input"]`.
  - `async def verify_url_claim(raw_url: str) -> Dict[str, Any]`:
    - it applies `sanitize_text` to the URL and runs the scraper;
    - when degraded it raises `UrlUnreadableError`;
    - otherwise it calls `verify_claim(headline, "url", linguistic_text=body, stored_raw_input=<sanitized url>, input_extras={"input_type": "url", "source_url": final_url})`.
  - `VerifyResponse.input_type: Literal["text","url","image"] = "text"`, `source_url: Optional[str] = None`, `ocr_extracted_text: Optional[str] = None`, `ocr_confidence: Optional[float] = None`.

- [ ] **Step 1: Write failing tests.**
  - `test_url_input_scrapes_and_verifies_headline`: the scraper mock returns headline H and body B → the agents' `claim == H`, and the linguistic mock received B.
  - `test_url_response_and_persistence`: `input_type == "url"`, `source_url` is the final URL; the `insert_cache` mock got `raw_input` = the URL and `agent_outputs["input"]["source_url"]`.
  - `test_text_containing_url_is_text`: "Check this https://x.com/a now" → the scraper is not called.
  - `test_url_unreadable_422`: the scraper is degraded → 422, `code == "url_unreadable"`, and no agent mock was called.
  - `test_stored_url_result_has_source_url`: `format_stored_result` returns `source_url` from `agent_outputs["input"]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_orchestrator_workflow.py tests/test_api_routes.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(api): verify a news link by its headline (TDD §4.6)`.

---

### Task 3: OCR service (Surya)

**Files:**
- Create: `backend/app/services/ocr.py`
- Create: `backend/app/agents/ocr_agent.py`
- Modify: `backend/app/config.py` (`ocr_enabled: bool = True`, `ocr_min_chars: int = 10`, `ocr_min_confidence: float = 0.6`, `ocr_timeout_s: float = 6.0`, `ocr_max_pixels: int = 25_000_000`)
- Modify: `backend/app/main.py` (lifespan: `load_ocr_models()` when `settings.ocr_enabled`)
- Modify: `backend/tests/conftest.py` (`OCR_ENABLED=false` for unit tests)
- Test: `backend/tests/test_ocr_agent.py`

**Interfaces:**
- Produces:
  - `services/ocr.py`:
    - `load_ocr_models() -> None` (idempotent; loads the Surya predictors once)
    - `ocr_available() -> bool`
    - `_predict_lines(image: PIL.Image.Image) -> List[Tuple[str, float, Tuple[float, float, float, float]]]`, returning `(text, confidence, bbox)` per line. This is the **only** function that touches Surya; tests patch it.
  - `class ImageRejected(ValueError)` in `ocr_agent.py`.
  - `@dataclass class OCRResult: extracted_text: str; confidence: float; bounding_boxes: list; processing_time_ms: int; degraded: bool = False; error: Optional[str] = None` (TDD §4.5 plus the agent contract).
  - `def preprocess(data: bytes) -> PIL.Image.Image` in `ocr_agent.py`. It raises `ImageRejected("type")` or `ImageRejected("pixels")`.
    - The magic bytes must match PNG `\x89PNG\r\n\x1a\n` or JPEG `\xff\xd8\xff`, and Pillow's `format` must be the same type.
    - `width * height > settings.ocr_max_pixels` is rejected from `Image.open(...).size` before `.load()`.
    - The result is converted to RGB and its longest edge resized to ≤2048.
  - `async def run(data: bytes) -> OCRResult`:
    - `preprocess`, then `_predict_lines` in `asyncio.to_thread` under `asyncio.wait_for(settings.ocr_timeout_s)`;
    - lines sorted top-to-bottom, then left-to-right within a row (rows: bbox vertical centres within half a median line height);
    - text joined with spaces, and `confidence` = the character-weighted mean of the line confidences;
    - a timeout or a model error → `degraded=True`.

- [ ] **Step 1: Install and confirm the API.**
  - `uv add surya-ocr pillow python-multipart`.
  - Read the installed Surya version's README for its predictor classes, and implement `load_ocr_models`/`_predict_lines` against that API.
  - Record the version in the task artifact.
- [ ] **Step 2: Write failing tests**, with `_predict_lines` patched:
  - `test_lines_joined_in_reading_order`: two columns and three rows given out of order → text in row-major order.
  - `test_confidence_is_char_weighted`.
  - `test_pixel_limit_rejected_before_decode`: a 20000×20000 PNG header → `ImageRejected("pixels")`, and `_predict_lines` is not called.
  - `test_resized_to_2048`.
  - `test_timeout_degrades`.
  - `test_ocr_unavailable_when_disabled`.
- [ ] **Step 3: Run** `uv run pytest tests/test_ocr_agent.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 6: Commit** `feat(input): Surya OCR service, loaded once at startup`.

---

### Task 4: `POST /api/verify/image`

**Files:**
- Modify: `backend/app/api/routes.py`
- Modify: `backend/app/agents/orchestrator.py` (`verify_image_claim`, `OcrTextError(code, extracted_text)`, `OcrUnavailableError`)
- Modify: `backend/app/main.py` (handlers: `OcrTextError` → 422 with `code` and `extracted_text`; `OcrUnavailableError` → 503 `ocr_unavailable`; `ImageRejected` → 400 `invalid_input`)
- Test: `backend/tests/test_api_routes.py`

**Interfaces:**
- Consumes: Task 3 `ocr_agent.run`, Task 2 `verify_claim` keywords.
- Produces:
  - `POST /api/verify/image` takes multipart field `image`. The route reads at most 5 MB + 1 bytes with `await image.read(5 * 1024 * 1024 + 1)`; more than 5 MB → 400.
  - `async def verify_image_claim(data: bytes) -> Dict[str, Any]`:
    - OCR unavailable → `OcrUnavailableError`;
    - OCR degraded → `OcrTextError("insufficient_text", "")`;
    - fewer than `ocr_min_chars` → `OcrTextError("insufficient_text", text)`;
    - confidence < `ocr_min_confidence` → `OcrTextError("ocr_low_confidence", text)`;
    - otherwise `verify_claim(text, "image", stored_raw_input=text, input_extras={"input_type": "image", "ocr_extracted_text": text, "ocr_confidence": conf})`.

- [ ] **Step 1: Write failing tests.**
  - `test_image_verifies_confident_text`: `ocr_agent.run` patched to return (text, 0.9) → 200, with `ocr_extracted_text` and `input_type == "image"`.
  - `test_low_confidence_returns_text_for_editing`: 0.4 → 422, `code == "ocr_low_confidence"`, `extracted_text` set.
  - `test_insufficient_text_spends_nothing`: "ok" → 422 `insufficient_text`, and the agent mocks were never called.
  - `test_image_too_large_400`.
  - `test_image_magic_bytes_decide`: an SVG body sent as `image/png` → 400 `invalid_input`.
  - `test_ocr_disabled_503`.
  - `test_image_rate_limited`.
- [ ] **Step 2: Run** `uv run pytest tests/test_api_routes.py -k image -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(api): screenshot verification endpoint (TDD §3.2)`.

---

### Task 5: OCR accuracy eval (Build Order 2.1 gate)

**Files:**
- Create: `eval/datasets/build_ocr_screenshots.py`. It renders 10 social-post-style PNGs (light and dark cards, 2–4 lines, sizes 600–1200 px wide) with Pillow's bundled font (`ImageFont.load_default(size=…)`). It writes `eval/datasets/ocr_screenshots/*.png` and `ground_truth.json`, mapping each `{file: text}`.
- Create: `eval/metrics/ocr_accuracy.py`: `char_accuracy(pred: str, truth: str) -> float` = `1 - levenshtein(norm(pred), norm(truth)) / max(1, len(norm(truth)))`, where `norm` collapses whitespace and lowercases. The Levenshtein DP is implemented here; no new dependency.
- Create: `eval/tests/test_ocr_accuracy.py`
- Modify: `eval/conftest.py` (register the `slow` marker: "loads a local model")

- [ ] **Step 1: Write the tests.**
  - `test_char_accuracy_metric`: `("hello", "hello")` → 1.0; `("hxllo", "hello")` → 0.8; empty prediction → 0.0.
  - `@pytest.mark.slow test_ocr_screenshots_meet_gate`: runs `ocr_agent.run` on every PNG and asserts the mean `char_accuracy` ≥ 0.85. It also checks `eval/datasets/ocr_screenshots/real/` with the same `ground_truth.json` shape when that folder exists; the project lead may add real screenshots there.
- [ ] **Step 2: Run** `uv run --project backend python eval/datasets/build_ocr_screenshots.py`. Expected: 10 PNGs and the JSON are written. Commit them.
- [ ] **Step 3: Run** `uv run --project backend pytest eval/tests/test_ocr_accuracy.py -v -m "slow or not slow"` on a machine with OCR enabled. Expected: PASS, with the mean accuracy printed. Paste it.
- [ ] **Step 4: Commit** `test(eval): OCR character-accuracy gate on synthetic screenshots`.

---

### Task 6: Frontend input and provenance

**Files:**
- Modify: `frontend/src/components/ClaimInput.jsx`, `frontend/src/hooks/useVerify.js`, `frontend/src/lib/api.js`, `frontend/src/lib/copy.js`
- Create: `frontend/src/components/{ImagePicker,InputProvenance}.jsx`, `frontend/src/lib/inputType.js`
- Modify: `frontend/src/components/ResultView` (render `InputProvenance` under the claim)
- Add fixtures: `url_result.json`, `image_result.json`, `ocr_low_confidence_error.json`
- Test: the matching `*.test.jsx`

**Interfaces:**
- Consumes: Plan 7 `useVerify`, `ERROR_COPY`, `TransitDiagram` checking mode.
- Produces:
  - `detectInputType(text: string) -> "url"|"text"`, the same rule as P8-4.
  - `verifyImage(file: File) -> Promise<VerifyResponse>`: multipart field `image`; throws `ApiError`, now carrying `extractedText` when present.
  - `useVerify().submit(text)` sends `input_type` from `detectInputType`; `useVerify().submitImage(file)` is new.
  - Checking-state lead line: "Reading the article…" (url), "Reading your screenshot…" (image), nothing for text. There is still no per-agent progress.
  - `ImagePicker`: a button "Add screenshot" (`accept="image/png,image/jpeg"`) plus drag-and-drop onto the input area. It rejects other types or files over 5 MB client-side with the `invalid_input` copy, before any request.
  - `InputProvenance`:
    - url → "Headline from <domain>", linked to `source_url` (`target="_blank" rel="noopener noreferrer"`);
    - image → a collapsible "Text read from your screenshot · <round(conf*100)>% confidence", with "Edit and re-check", which navigates to `/` with the text prefilled.
  - On `ocr_low_confidence`, the input is prefilled with `error.extractedText` and the message shown. The user verifies it as text.

- [ ] **Step 1: Write failing tests.**
  - `detectInputType`: a bare URL → url; text containing a URL → text; `"http://"` alone → text.
  - `ImagePicker rejects gif and 6 MB before request`.
  - `submitImage posts multipart image field`.
  - `ocr_low_confidence prefills input`.
  - `url provenance links source_url with rel`.
  - `image provenance shows confidence and edit action`.
  - `checking lead line per input type`.
- [ ] **Step 2: Run** `npm run test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** Follow the brief's colour, copy and 44px touch-target rules; drag-and-drop is enhancement only, and the button always works.
- [ ] **Step 4: Run** `npm run test -- --run && npm run lint && npm run build`. Expected: pass. Recapture 375 and 1440px screenshots of the URL and image results into `.impeccable/review/`, and run the impeccable detector on the changed files.
- [ ] **Step 5: Commit** `feat(frontend): link and screenshot input with provenance`.

---

### Task 7: Docs

- `documents/VeriFact_AI_TDD.md`:
  - §3.1: URL behavior (P8-3, P8-4) and the new response fields;
  - §3.2: the 422/503 codes and `extracted_text`;
  - §4.5: the load-once decision (P8-1) and thresholds;
  - §4.6: `fetch_html` reuse;
  - §11: image limits.
- `documents/VeriFact_AI_Build_Order.md` Steps 2.1–2.3: point to Plan 8; replace the "load/unload per request" note with P8-1; remove the Tavily-extract fallback (Tavily was removed in Plan 6).
- `documents/VeriFact_AI_PRD_v2.md` §4: P2-01, P2-02 done.
- `README.md`: `OCR_ENABLED` and the RAM note in setup, plus the "Project status" update (multi-modal input done).
- `docs/superpowers/plans/README.md`: Plan 8 status.
- [ ] **Step 1: Make the edits.** Commit `docs: URL and screenshot input (Plan 8)`.

---

### Task 8: Gate (Build Order Phase 2 checkpoint)

- [ ] **Step 1: Tests.**
  - `uv run pytest -q` (backend)
  - `uv run --project backend pytest eval/tests -m "not live" -q` (root)
  - `npm run test -- --run && npm run lint && npm run build`

  All green; paste the counts.
- [ ] **Step 2: Live scrape check (project lead's OK first).** 10 news URLs the project lead approves, listed in `backend/tests/fixtures/scrape_urls.txt`, run with `@pytest.mark.live test_scrape_live_urls`. Expected: a headline for ≥9 of 10 (Build Order 2.2 gate). Report the failures.
- [ ] **Step 3: Live end to end (project lead's OK first; it spends quota).** One text, one URL and one screenshot through the UI at 375 and 1440px. Screenshots go into `.impeccable/review/`.
- [ ] **Step 4: Review gate:**
  - `code-reviewer` or `/review` on the branch diff, with security focus on SSRF, upload handling and the `extracted_text` echo;
  - `eval-guardian`: `test_ocr_accuracy` result and guardrail tests green.

  Fix and re-review until approved.
- [ ] **Step 5: Log the outcome.** Update `docs/superpowers/plans/README.md` (Plan 8 `completed`) and write the outcome to `review-fixes-handoff.md`.
