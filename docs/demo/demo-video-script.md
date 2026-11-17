# VeriFact AI: Demo Video Script

**Target length:** 2:30–3:00 · **Format:** screen recording + voice-over, 1080p, linked from the README
**Runs on:** a local stack (backend `uv run uvicorn app.main:app`, frontend `npm run dev`)

## Before recording

1. **Wait for warm-up.** Start the backend and wait for "Startup complete" before the first take. The first screenshot loads the OCR models; either set `OCR_PRELOAD=true` or do one throwaway screenshot check first.
2. **Rehearse every claim below.** Run each one once and keep only those whose result tells the story you narrate. Live evidence changes, so a scripted outcome is not guaranteed.
   - Swap out any claim that doesn't behave as expected.
   - Never edit a result to fit the script.
3. **Speed up the waits.** Real latency is about 20 s for text, about 40 s for a URL, and more for a cold screenshot. Speed those segments up and **label them on screen** ("sped up 4×"). Don't cut them so it looks instant.
4. **Plan for the second run.** The second run of a claim is served from the semantic cache. Use that on purpose in scene 5, and record scenes 2–4 on claims you haven't run that day (or after the cache TTL has passed).
5. **Know the UI gap for URL and screenshot.** The UI has no URL or screenshot input yet (Plan 8 Task 6). Either finish that first or record scene 6 against the API docs at `localhost:8000/docs`. Don't fake it.
6. **Prepare the window.** Use a browser window at 1280×800 with no bookmarks bar, no other tabs and no personal data visible. Use dark mode if that's the look you want.

## Scenes

| # | Time | Screen | Voice-over |
|---|---|---|---|
| 1 | 0:00–0:15 | Home screen, cursor in the input box | "A screenshot shows up in your group chat. Is it true? VeriFact AI turns a claim into a source-backed credibility check. It isn't a chatbot's opinion: it's evidence you can inspect." |
| 2 | 0:15–0:50 | Paste a **widely fact-checked false claim** (candidate: *"5G towers spread COVID-19"*). Submit, wait (sped up), result appears | "Three agents run in parallel. One searches published fact-checks, one cross-references news coverage, and one analyses the language. Here a professional fact-check already exists, so the score is low and the verdict line says who rated it and links to it." |
| 3 | 0:50–1:20 | Scroll through the score breakdown, then the source list with stance and bias chips | "Every number is computed by code, not by the model. The score is a weighted mix of the fact-check, cross-source consensus and linguistic signals. Each source shows whether it supports or refutes the claim. Bias labels come only from AllSides ratings; if a site isn't rated, it says 'Unrated'. The AI never guesses political bias." |
| 4 | 1:20–1:45 | New claim: **something true but with no fact-check** (candidate: a recent, well-reported event). Show the confidence badge and warning | "Most claims have no fact-check at all. When the evidence is thin, VeriFact caps the confidence and says why. 'Not enough information' is an honest answer, and the system is built to give it rather than invent certainty." |
| 5 | 1:45–2:00 | Re-run the scene 2 claim with slightly different wording. Instant result, cached badge | "Rephrase a claim and the semantic cache recognises it. It still double-checks that the meaning matches, so a negated claim never gets someone else's answer." |
| 6 | 2:00–2:25 | Screenshot upload of a sensational post, then an article URL (UI, or `/docs` if Task 6 isn't done) | "It also reads screenshots, using on-device OCR so images are never stored, and article links, safely fetched with guards against internal-network tricks. The claim it extracted is shown, so you can check what it actually verified." |
| 7 | 2:25–2:40 | Language radar on a sensational claim | "The language radar shows emotional, sensational and absolutist wording. That's a reason to read carefully, not a verdict on its own." |
| 8 | 2:40–2:55 | Opinion input, e.g. *"Pineapple is the best pizza topping"*, refused | "And if you give it an opinion or a question, it refuses instead of scoring something that can't be fact-checked." |
| 9 | 2:55–3:00 | README / architecture diagram, repo link | "Built with Google ADK agents, FastAPI, React, and an evaluation suite that guards accuracy and bias leakage. The code is on GitHub." |

## Rules for the voice-over

- **Say only what the screen shows.** If a result differs from the script, describe the result you got.
- **No accuracy figures** unless they come from a valid published eval run in the README.
- **Keep the speed-up captions** wherever waits were shortened.

## After recording

- Export as MP4 at 1080p.
- Upload it as a GitHub release asset, or drag it into a README edit so GitHub hosts it.
- Embed it at the top of the README with a thumbnail and the actual length.
