# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary:** social media users (18–45) who hit a questionable claim in their feed, usually on a phone, and want a fast credibility check before they share it or react to it. They arrive with one claim (text, a link, or a screenshot) and little patience.

**Secondary:** hiring managers and technical interviewers judging the author's ML engineering: architecture, evaluation rigor, production patterns. When the two audiences conflict on a screen, the phone user wins; the reviewer's depth (agents, weights, timings, evidence) is one deliberate step away, never in the way. (Confirmed 2026-10-15.)

## Product Purpose

Turn one suspicious claim into a visual, source-backed credibility assessment in under 10 seconds. Success is a person who can tell at a glance whether a claim holds up, why, and which sources say so, and who trusts the answer *because* it admits when evidence is thin.

## Positioning

Not a chatbot and not "according to my training data." A multi-agent pipeline (Fact-Check, Cross-Reference, Linguistic) whose composite trust score is transparent: every component links to its source, bias labels come only from AllSides, and the system says "not enough evidence" instead of inventing certainty.

## Operating Context

- One claim per visit; a shareable result URL (`GET /api/results/{id}`).
- The backend takes up to ~9.5 s; the wait is part of the experience.
- Inputs: text now; URL and screenshot (OCR) arrive in the MVP phase.
- Results carry Google Search suggestions HTML that Google's terms require be shown alongside grounded results.

## Capabilities and Constraints

- API contract: `backend/app/api/schemas.py` (`VerifyResponse`, `ErrorResponse`), TDD §3.
- Evidence states: `fact_checked`, `disputed`, `corroborated`, `unverified`; confidence levels `high`/`medium`/`low`/`insufficient`. `unverified` shows "Not enough evidence," never a number.
- Bias chips render only what the API returns; `Unrated` is a real state. The client never computes a bias label.
- AllSides attribution ("Media bias ratings: AllSides.com", CC BY-NC 4.0) must be visible under the source list.
- English only. No accounts. No secrets in the client.
- Stack (existing): Vite + React + Tailwind + Recharts.

## Brand Commitments

Name: VeriFact AI. No logo, palette, or voice exists; the visual world is created fresh. (Confirmed 2026-10-15.)

## Evidence on Hand

- Real API responses can be produced by the running backend; `backend/tests/fixtures/grounded_search_sample.json` is a synthetic search fixture.
- No users, testimonials, benchmarks, or published eval numbers exist yet. The only stored eval baseline is an invalid run. Do not show accuracy figures until a valid eval run produces them.

## Product Principles

1. Honesty over confidence: uncertainty is shown as prominently as a verdict.
2. Every number traces to a source.
3. Glance first, depth on demand.
4. The pipeline is visible, never a black box.
5. Never fabricate certainty, ratings, or proof.

## Accessibility & Inclusion

Mobile-first at 375px; touch targets ≥44px; verdict meaning never carried by color alone (red/amber/green bands always paired with words).
