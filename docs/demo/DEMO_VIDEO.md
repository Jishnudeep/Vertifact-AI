# VeriFact AI — demo video requirements

**Status:** requirements and shot list, not yet recorded. Values marked **(default)** are proposals; change them before recording.
**Why a video:** the backend runs on demand (Plan 10, decision D1), and the free Gemini key allows about 500 requests/day, so a live public demo would be off most of the time and drained within minutes. The recording is the public proof that the whole pipeline works. See [`../superpowers/specs/2026-10-20-plan10-design.md`](../superpowers/specs/2026-10-20-plan10-design.md).

## 1. Audience and goal

- **Primary viewer:** a hiring manager or technical interviewer with five minutes. They should leave knowing: it works end to end, the answer is explained and sourced, it admits when evidence is thin, and the author measures things.
- **Secondary viewer:** a phone user deciding whether the tool is worth trying. The product moment (claim in, verdict and evidence out) comes first; engineering depth comes after.
- **One-sentence pitch to land in the first 20 seconds:** "VeriFact turns one suspicious claim into a source-backed credibility score, using three parallel agents and a scoring engine that shows its work."

## 2. Format

| Item | Requirement |
|---|---|
| Length | 3:00 target, 3:30 hard maximum **(default)** |
| Resolution | 1920×1080, 30 fps, 16:9 |
| Layout | Desktop browser for most scenes; one short phone-width segment (375 px wide, the design baseline) |
| Audio | Voice-over plus burned-in captions, or captions only **(default: captions only, with a `.srt` file)**; no music over speech |
| Capture | OBS or the OS screen recorder at native resolution; browser zoom 110–125 % so text is readable at 720p |
| Delivery | `.mp4` (H.264), `.srt`, a 1280×720 thumbnail, and an unlisted hosting link for the README |

## 3. Honesty rules (these override any editing preference)

1. **No simulated progress.** The checking state today shows all three lines dashed with no per-agent progress; live per-agent status is deferred. Do not narrate "watch the agents run live". Show per-agent timings in the finished result, where they are real.
2. **Cuts of waiting time are labelled.** A cold claim takes as long as the pipeline takes. If the wait is cut, show a caption such as "wait shortened, real time 27 s" with the real elapsed time from the app's timer.
3. **No claims without a valid run.** Do not say "under 10 seconds", quote accuracy, MAE, calibration, latency percentiles or cache savings until a valid record run and latency bench have produced them (see [`../../eval/README.md`](../../eval/README.md)). Say "target" for the 10 s goal.
4. **No cherry-picked outcome.** Choose claims by a rehearsal that is not repeated until it looks good. Keep at least one result that is not a clean win: "Not enough evidence", or a degraded "Partial analysis".
5. **Real data only.** Every result on screen comes from the running backend, not from `?fixture=` previews. If a fixture is shown (to show a state that is hard to trigger), it is labelled "stored sample".
6. **Bias labels come from AllSides only.** Keep the attribution line "Media bias ratings: AllSides.com" visible in the source-list scene.
7. **Say what runs where.** One caption: "backend runs locally; the public site shows recorded results".

## 4. Pre-recording checklist

Environment

- [ ] `main` contains the code being shown; record from a clean checkout, `git status` clean.
- [ ] Supabase migrations 01–07 applied; at least three rows have `results.featured = true` so "Recently checked" is not empty.
- [ ] `.env` filled; **no terminal or editor tab that shows key values** is visible at any point. Use a fresh browser profile with no bookmarks bar, no extensions, notifications off, and a neutral desktop.
- [ ] Backend started and **warm** (models preloaded at boot): `cd backend && uv run uvicorn app.main:app`. Frontend: `cd frontend && npm run dev` with `VITE_API_BASE_URL=http://localhost:8000`.
- [ ] Quota plan written down. A cold claim costs about 8 Gemini requests; the free tier allows 15 per minute and about 500 per day. Budget **at most 8 cold claims in total** for rehearsals plus takes (about 64 requests), and pause between claims.
- [ ] Clear the semantic cache entries for the demo claims before the cold take, and keep one claim cached on purpose for the "instant repeat" scene.

Rehearsal (do before any take)

- [ ] Run each candidate claim once and note: evidence status, score, confidence, elapsed time, any degraded agent.
- [ ] Confirm each scene's claim produces the state the scene needs (table in §5). Swap claims, not the script's honesty.
- [ ] Confirm a second submission of the same claim returns the cached result (the app marks it).

## 5. Scenes

| # | Time | Scene | On screen | Caption / narration |
|---|---|---|---|---|
| 1 | 0:00–0:20 | Problem and pitch | Home page at phone width; empty claim box | The pitch sentence. "Misinformation spreads about 6× faster than corrections." |
| 2 | 0:20–1:10 | A cold claim, start to finish | Paste a claim with a likely fact-check; submit; the checking state; the result | Show the real elapsed timer; label any cut. Land on verdict, score ring and confidence. |
| 3 | 1:10–1:50 | Why that score | Scroll the result: score breakdown (S_fact, B_ling, V_consensus, weights 0.5 / 0.2 / 0.3), the cap rule if one applied, the language radar, sources with stance and AllSides chips, the attribution line, Google search suggestions | "Every component links to a source. Bias labels come only from AllSides data, never from the model." |
| 4 | 1:50–2:15 | Honest uncertainty | A claim with thin evidence → "Not enough evidence", no gauge number. If a degraded agent occurs naturally, show the dashed line and "Partial analysis" | "When evidence is thin it says so instead of inventing a score." |
| 5 | 2:15–2:30 | Instant repeat | Submit scene 2's claim again → cached result | Real response time shown. No savings percentage unless the cache bench has run. |
| 6 | 2:30–3:10 | Under the hood | Split screen: architecture diagram from the README; backend log lines for `cache_lookup`, `dispatch_agents timings_ms`, `score`; green test runs for backend and eval; the eval README's run statuses (`OK` / `INVALID_RUN` / `GUARDRAIL_FAIL`) | "Three agents in parallel, a failed one degrades instead of crashing. The scoring engine is pure code. The eval harness refuses to publish a run whose search was broken." |
| 7 | 3:10–3:30 | Close | Repo, README, link | What is built, what is next (agent ops, URL and screenshot input), and what is not claimed yet. |

If Plan 10 has shipped before recording, replace the log lines in scene 6 with the trace view and audit record, and say which request id links them.

### Candidate claims (verify in rehearsal; outcomes are not promised)

| Use | Candidates | Source |
|---|---|---|
| Scene 2 (cold, clear) | "Drinking industrial bleach cures and prevents COVID-19 viral infections." / "The Great Wall of China is visible from orbit with the unaided human eye." | Curated dataset, labelled false |
| Scene 2 alternative | "Consuming lemon water cures diabetes within 48 hours." | App example chips |
| Scene 4 (thin evidence) | "An obscure town council in rural Oregon held a local zoning meeting in March 2026." | Curated dataset, labelled 0.5 |
| Contested topic (optional) | "Nuclear power is a clean energy source." | Curated dataset, labelled 0.5; shows a mixed result |

Do not use a claim about a living private individual or an active election.

## 6. Acceptance criteria

- [ ] Within 3:30; every scene present; captions readable at 720p.
- [ ] No API key, `.env` content, Supabase URL or personal data visible in any frame (scrub the timeline frame by frame at scene changes).
- [ ] Every number on screen traces to a real run or a labelled cut.
- [ ] At least one non-win state shown.
- [ ] AllSides attribution visible.
- [ ] The README links the video and states that the backend runs on demand.

## 7. Open decisions for the project lead

1. Voice-over or captions only?
2. Hosting: unlisted YouTube, Loom or a file in the README?
3. Is the "under the hood" scene worth the time, or should the video stay on the product (the reviewer gets the depth from the README instead)?
4. Record before or after Plan 10? Before: no trace view, but the video exists now. After: richer scene 6, later.
