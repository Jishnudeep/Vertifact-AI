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
2. **Supabase migrations run by hand**, in the SQL editor, in order: `03_stance_aware_consensus.sql` (Plan 1), `04_cache_topk_and_ttl.sql` (Plan 2), `05_retention.sql` (Plan 4), `06_warnings_and_lockdown.sql` (review follow-ups). 03–06 were applied. Never re-run 01–03 after 04: they recreate an old `match_claims` overload.
3. **Quota spend needs your OK.** The first eval `--replay record` run (Plan 3) spends Tavily and Groq quota. The LIAR/FEVER raw files are downloaded by you (Plan 3, Task 5).
4. **The dev model decision** (Plan 5, Task 0) is yours: `gpt-oss-120b` in config vs `gpt-oss-20b` in docs.

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
