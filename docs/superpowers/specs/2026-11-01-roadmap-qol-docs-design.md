# Roadmap QOL + Docs refresh — design

**Date:** 2026-11-01
**Status:** Draft — brainstorming complete, ready for builder check
**Scope:** Project (multi-track; quick-wins + plans for TODO backlog)

## Intent and why

ezpz is stable on Aurora/Polaris/Sunspot after #285 (status parity), but the
daily loop still has friction: polling `ezpz status` manually, profiling that
hangs, PBS wrappers that mask failures, stale preflight checks, no one-stop
CLI index, and 13 remaining TODO sections that have drifted since the June
2026 audit. This design captures the next tranche of QOL/docs/util work in a
single place so quick wins ship this week and larger refactors (jobs.py Slurm,
utils grab-bag, history) are scoped with concrete plans rather than vague
bullets.

Sources: `TODO.md` (18 sections, last audited 2026-06-24), open issues
(#275, #272, #270, #266, #252, #246, #243, #239, #237, #220), recent PRs #284–#285.

## Goals

- Document the full prior-brainstormed roadmap in durable form (TODO.md
  expansion + cross-links to issues/specs).
- Ship 4 quick wins on parallel branches: `watch/status --watch`,
  `profile.py` fixes, PBS wrapper exit-status fix, CLI docs index +
  preflight currency fix.
- Produce written plans (specs) for the 3 medium/larger tracks so they can
  be sequenced without re-brainstorming.

## Non-goals / anti-goals

- No new distributed training semantics (FSDP/TP/PP changes). This is
  tooling + docs + small bug fixes only.
- No API-breaking CLI changes — new flags only, never renames/removals.
- Not a release-blocking epic: each quick win merges independently; the
  larger plans do not gate the next patch release.

## Constraints

- Keep `TODO.md` as the canonical roadmap index (per AGENTS.md pattern);
  new specs live under `docs/superpowers/specs/`.
- Use git worktrees for parallel branches (per AGENTS.md / superpowers
  guidance) — never pile on `main`.
- `ezpz launch` must remain torchless-compatible (don't add hard torch
  imports to launch path).
- Subagent fanout: each quick-win gets its own agent/worktree; orchestrator
  only coordinates + reviews, doesn't write product code directly.

## Planned work — approved tracks

### A. TODO.md refresh (orchestrator, main worktree)

Add a dated "Queued — 2026-11-01 brainstorm" section to `TODO.md` that
lists the 12 ranked items from the prior session, grouped Quick wins /
Medium / Bigger bets, with links to open issues and pointers to the
existing TODO sections they map to. Mark #5 (history), #11 (tracker),
#12 (ANSI), #17 (yeet) as already partially scoped by prior shipped work
so they aren't re-described. Commit directly on `main` (doc-only).

### B. Quick wins — parallel subagent branches

| # | Branch | Scope | Enters as |
|---|--------|-------|-----------|
| 1 | `feat/status-watch` | `ezpz status --watch` / `ezpz watch` — poll with Rich, notify on state change | PR |
| 2 | `fix/profile-272-275` | Guard `--with-stack`, reorder export/key_averages, add range annotations; closes #272, #275 (and #270) | PR |
| 3 | `fix/pbs-exit-status-246` | Propagate `PIPESTATUS`/`ERR` trap in PBS wrapper; closes #246 | PR |
| 4 | `docs/cli-index-preflight` | `docs/cli/index.md` + nav cross-links; preflight currency honest fix | PR |

Each agent: read its issue/section, implement, add minimal test/doc,
open PR with `release:patch|skip`. No API renames.

### C. Written plans for larger tracks

Three `docs/superpowers/specs/*.md` files, one per track, drafted by a
subagent in a short-lived worktree and landed on `main`:

1. **`jobs.py is PBS-only` (TODO #4)** — unify PBS/SLURM behind
   `scheduler_for()` + `squeue` JSON parser mirroring the `qstat` path
   from #285. Spec should define the parser contract, error cases, and
   test fixture (fake `squeue`), and call out that `remote.py` already
   has the pattern.
2. **`doctor` polish** — `ssh -O check` staleness, `ControlPersist`
   hint, `~/.ssh/config` User mismatch detection (the `sam` vs `foremans`
   bug). Small spec: which checks, output format, non-interactive mode.
3. **`utils/__init__.py` grab-bag (TODO #8) + `history.py` slim (TODO #5/#11)**
   — re-export shim strategy, target module map, migration order
   (grab-bag split first, history extraction after), and a `docs/history.md`
   outline.

The `fetch`/`logs --follow` and `yeet`/CI/DeepSpeed ideas (#10–12 in the
original list) are intentionally deferred — they are real but depend on
the above and would make this design too large. They get a stub line in
TODO.md pointing to a future spec.

## How they decided (locked)

- Branch-per-win via worktrees, not stacked PRs on one branch.
- TODO.md is the live index; specs elaborate, not duplicate.
- Orchestrator doesn't code — subagents do.

## What's left to the builder

- Exact CLI flag naming for `watch` (e.g. `--interval`, `--notify`) — follow
  `status`'s existing flag style.
- Test granularity for each quick win (one happy-path test is sufficient).
- Docs wording — keep `AGENTS.md` voice rule.
- Order of the 3 plan-specs — any order is fine; author preference.

## Acceptance

- `TODO.md` has the new dated section with 12 items, grouped, with links.
- 4 quick-win branches exist with passing tests/docs, each with an open PR.
- 3 spec files exist on `main` and are coherent enough to hand to a builder.

Builder check: dispatch a fresh subagent with `builder-check-prompt.md`;
bring back only "theirs" questions in one round.
