# VeriFact fix plans (2026-09 PRD/TDD review)

These plans implement the findings in
[`../specs/2026-09-23-prd-tdd-review.md`](../specs/2026-09-23-prd-tdd-review.md). They are
written so that **Antigravity** or **Claude Code** can run them task-by-task without further
context. Start Antigravity with `/execute-plan` (`.agents/workflows/execute-plan.md`).

## Execution order and status

The plans refer to each other by the **Plan number** below. Run them in the **Run order**
column (1 → 2 → 4 → 3 → 5). Each plan's Task 0 checks its prerequisites.

| Run order | Plan | File | Covers | Needs merged first | Status |
|-----------|------|------|--------|--------------------|--------|
| 1st | Plan 1 | [stance-aware-consensus](2026-09-23-stance-aware-consensus.md) | C1, D1, D2: NLI stance, `v_refute`, Rule 4, `evidence_status` | — | completed |
| 2nd | Plan 2 | [cache-safety](2026-09-23-cache-safety.md) | C2: top-k, row policy, NLI equivalence, TTL | Plan 1 | completed |
| 3rd | Plan 4 | [hardening](2026-09-23-hardening.md) | H1–H4: deadline, budgets, proxy-aware limits, SSRF guard, retention | Plans 1, 2 | completed |
| 3rd+ | Review follow-ups | (no plan file; see below) | Fixes from the 2026-09-27 review of Plans 1, 2, 4 | Plans 1, 2, 4 | completed 2026-09-28 (Claude Code) |
| 4th | Plan 3 | [eval-integrity](2026-09-23-eval-integrity.md) | C3, C4: replay, leave-source-out, segments, LIAR/FEVER, NLI calibration | Plans 1, 2, 4 | completed 2026-09-28 (Antigravity) |
| 5th | Plan 5 | [doc-sync](2026-09-23-doc-sync.md) | Medium drift, H5, H6: error codes, rating map, docs, skills | Plans 1–4 | completed 2026-09-28 (Antigravity) |
| 6th | Plan 6 | [google-search-grounding](2026-10-14-google-search-grounding.md) | Google Search grounding replaces Tavily | Plans 1–5 | completed |
| 7th | Plan 7 | [frontend-mvp](2026-10-15-frontend-mvp.md) | Frontend MVP, text input end to end (Midnight Transit Diagram), plus API additions: per-agent status, cap fields, `found_by`, stored-result parity, `Retry-After`, featured recent results | Plan 6 | completed |
| 7th+ | Task A | [spec](../specs/2026-10-20-latency-thinking-budgets.md) (no plan file) | Per-agent thinking levels, model-turn caps, UTC date awareness, sub-stage timing | Plan 6 | merged 2026-10-22 (PR #9); **no valid latency measurement yet**, the A.4 bench ran with grounded search broken |
| 8th | Plan 9 | [evals](2026-10-16-evals.md) | Evals: valid baseline, DeepEval guardrail, ADK eval over traces, latency and cache metrics | Plan 7 | Tasks 0–6 built and tested; Task 7 (record run) blocked on quota; Plan 9 gate not claimed |
| 9th | Plan 10 | [agent-ops](2026-10-20-agent-ops.md) | Governance, tracing, audit record, shared Gemini budget, read API, Docker demo-on-demand deployment (spec: `../specs/2026-10-20-plan10-design.md`) | Plan 7 (Plan 9's live runs may stay on hold; Task A merged or rebased) | designed 2026-10-20 (spec and plan merged as PR #8); not started |
| 10th (last) | Plan 8 | [url-and-screenshot-input](2026-10-15-url-and-screenshot-input.md) | URL scraper, Surya OCR, `/api/verify/image`, frontend input | Plans 7, 9, 10 | not started; runs last by decision (2026-10-15) |

**Realignment (2026-10-15):** Plans 7–10 follow `../specs/2026-10-15-realigned-roadmap-design.md`. Plan 7's spec is `../specs/2026-10-15-frontend-mvp-design.md`. Plans 9 and 10 are written only after their own design sessions. Plan 8 keeps its number but runs last.

## Branching and pull requests

- **`main` is the integration branch.** Nothing is committed to it directly; every change lands through a pull request.
- **One branch per plan, cut from an up-to-date `main` after the previous plan's PR has merged.** Plans depend on each other's interfaces, so do not stack plan branches. The exception is the project lead explicitly asking to start a plan early: then branch from the previous plan's branch and rebase onto `main` once that PR merges.
- **Branch names:**

  | Kind | Branch |
  |---|---|
  | Plan 6 | `feat/google-search-grounding` |
  | Plan 7 | `feat/frontend-mvp` |
  | Plan 9 | `feat/evals` |
  | Plan 10 | `feat/agent-ops` |
  | Plan 8 | `feat/url-and-screenshot-input` |
  | Design or doc-only work | `docs/<topic>` |
  | Fixes from a review of an open plan PR | commits on that plan's branch |
  | Fixes after a plan has merged | `fix/<topic>` from `main` |

  Each plan's Task 0 creates its branch.
- **Commits:** one commit per plan task, using the message the plan gives (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`). Don't squash locally; the per-task history is the review trail.
- **Pull requests:** one per plan into `main`.
  - Open it as a **draft** once the branch has its first task commit, and push after every task.
  - Mark it ready only after the plan's Gate task passes: all suites green plus the `code-reviewer` / `/review` and `eval-guardian` gates.
  - The PR body links the plan and its spec, lists migrations the project lead must run, and lists any quota-spending steps that ran.
- **Merging:** use a merge commit, not squash, so `git bisect` and the review trail keep per-task commits. Delete the branch after merging. Apply any migration by hand right after the merge, then record it under "Before starting (user)".
- **Local-only files are never committed:** `.claude/`, `.agents/`, `CLAUDE.md`, `AGENTS.md`, `*-handoff.md`, and impeccable's working files in `.impeccable/` (sketches, quality-bar images, decision-page state). Only `.impeccable/design.json` is tracked, once Plan 7 creates it.
- **One-off exception:** the 2026-10-15 realignment docs and this section ride on the Plan 6 PR, because they were written while that branch was checked out.

Plan 4 runs before Plan 3 because it switches Cross-Ref to basic search depth, and that
changes the replay cache keys Plan 3 records.

### Review follow-ups (2026-09-28)

A code review and eval gate of Plans 1, 2 and 4 found issues that Claude Code fixed
test-first on `fix/review-followups`:
- **LLM budget:** an exhausted budget now degrades the Fact-Check agent instead of becoming an "unproven" `fact_checked` result (cached 30 days) or silently dropping real matches.
- **Stance chips:** claim-review sources in Cross-Ref carry `stance: null`.
- **Migration 06** (`06_warnings_and_lockdown.sql`): restores `results.warnings` (missing live, so every cache lookup and insert failed), enables RLS, and revokes public `EXECUTE` on `purge_old_claims` / `match_claims`.
- **Eval harness:** runs with spend budgets off, and the `live` marker is registered.
- **Smaller fixes:**
  - cache writes are kept alive through a cancelled request and drained on shutdown;
  - CORS exposes `X-Process-Time-Ms`;
  - the SSRF guard rejects multicast and NAT64-wrapped private IPs, and returns the validated IPs for pinning;
  - `.env.example` is updated;
  - TDD §3.1, §4.3, §7 and §11 are updated.

Plans 3 and 5 were amended to match. In particular, from the repo root use `uv run --project backend ...`.

## Before starting (user)

1. **Commit or stash the current working tree.** Every plan's Task 0 stops on a dirty tree.
2. **Supabase migrations run by hand**, in the SQL editor, in order: `03_stance_aware_consensus.sql` (Plan 1), `04_cache_topk_and_ttl.sql` (Plan 2), `05_retention.sql` (Plan 4), `06_warnings_and_lockdown.sql` (review follow-ups). 03–06 were applied. Never re-run 01–03 after 04: they recreate an old `match_claims` overload. `07_featured_results.sql` (Plan 7, Task 4) must be applied by hand after Plan 7 merges. Then feature rows by setting `results.featured = true` in Supabase.
3. **Quota spend needs your OK.** Any `--replay record` or `--live` eval run spends free-tier Gemini quota (about 8 requests per cold claim; 500/day cap). The LIAR/FEVER raw files are downloaded by you (Plan 3, Task 5).
4. ~~The dev model decision (Plan 5, Task 0)~~ is obsolete: workers run on Gemini (`MODEL_ID`, default `gemini-3.1-flash-lite`).

## Review gate mapping (CLAUDE.md §5 / AGENTS.md §5)

Every plan ends with a mandatory review task. Use whichever column matches your executor:

| Gate | Claude Code | Antigravity |
|------|-------------|-------------|
| Code review | `code-reviewer` subagent (`.claude/agents/code-reviewer.md`) or `/code-review` | `/review` workflow → `code-review` skill (`.agents/skills/code-review/`) |
| Metrics and guardrails | `eval-guardian` subagent (`.claude/agents/eval-guardian.md`) | `eval-guardian` skill (`.agents/skills/eval-guardian/`) |
| Security focus (Plan 4) | `/security-review` | `code-review` skill, with the plan's security questions |
| Plan execution | `superpowers:subagent-driven-development` | `superpowers:executing-plans` (via `.agents/skills.json`) + a task artifact |

Test commands in the plans (`uv run pytest ...`) must actually run. `uv` lives at
`C:\Users\user\.local\bin\uv.exe` (not on PATH). From the repo root, eval commands need
`--project backend`. Never mark a step done without pasting its real output.

## Skill parity: Claude Code ↔ Antigravity

| Skill / role | Claude Code | Antigravity | Notes |
|--------------|-------------|-------------|-------|
| verifact-agent, scoring-engine, eval-harness, api-endpoint, frontend-viz | `.claude/skills/` | `.agents/skills/` | Same facts, different house style. Plan 5, Task 7 syncs both. |
| code-review gate | `code-reviewer` agent + built-in `/code-review` | `.agents/skills/code-review` + `/review` | |
| eval-guardian gate | `.claude/agents/eval-guardian.md` | `.agents/skills/eval-guardian` (added 2026-09-27) | |
| backend/frontend-engineer roles | `.claude/agents/*.md` | `.agents/agents.md` roster | Antigravity spawns subagents dynamically |
| superpowers (executing-plans, TDD, verification, systematic-debugging, receiving-code-review, writing-plans, …) | plugin `superpowers@claude-plugins-official` | `.agents/skills.json` → `~/.claude/plugins/cache/claude-plugins-official/superpowers/6.3.0/skills` | Tool mapping for Antigravity: `using-superpowers/references/antigravity-tools.md`. **The path is versioned:** after a superpowers update, bump `6.3.0` in `.agents/skills.json`. The durable alternative is `agy plugin install https://github.com/obra/superpowers` (needs the `agy` CLI, which is not on PATH here). |
| impeccable (UI design), humanizer (prose) | `~/.claude/skills/` | `.agents/skills.json` (`include_only`) | Useful for the frontend phase and docs |
| Claude-desktop plugin skills (engineering:*, design:*, docx/pdf/pptx, productivity) | Claude app plugins | not shared | Not needed for these plans; Antigravity has its own built-ins. They live in a session-managed folder, so pointing at them would break. |
