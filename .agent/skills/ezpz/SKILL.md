---
name: ezpz
description: Use when running, testing, or benchmarking ezpz on an HPC system (Aurora, Sunspot, Polaris, Perlmutter) — submit, status, fetch, launch, launch flags, env/proxy gotchas, CCL hangs, and per-machine caveats. Router for the `ezpz` CLI surface.
---

# ezpz on HPC — router

This skill is the **HPC operator's dispatch table**. The canonical ambient doc is still `SKILL.md` at the repo root (one-liner, hard gotchas, `CCL_OP_SYNC`, machine table) — this file covers the CLI-facing workflows and when to reach for each subcommand.

## Quick dispatch

| intent | command | notes |
|---|---|---|
| smoke on login node | `source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env` → `ezpz launch python3 -m ezpz.examples.test` | 4-line contract, see below |
| remote smoke (one node) | `ezpz submit --remote <host> -N 1 -q debug --time 00:30:00 --workdir $HOME/$tdir -- python3 -m ezpz.examples.test` | then `ezpz status` / `ezpz fetch` |
| multi-node training | `ezpz submit -N 2 -q debug -A <acct> --filesystems home,flare -- python3 -m ezpz.examples.fsdp_tp --model large --tp 2` | add `--gpus-per-node`, `-C gpu` on Perlmutter |
| watch progress | `ezpz status --limit 10` (fanout), `ezpz status --remote <host> --limit 5`, `ezpz fetch $JID --remote <host> --lines 40` | fanout is zero-scrape; see `docs/cli/status.md`, `docs/cli/fetch.md` |
| cancel / extend | `ezpz cancel --dry-run`, `ezpz cancel $JID --remote <host>`, `ezpz extend $JID --time 02:00:00` | IRI only; see `docs/cli/cancel.md`, `docs/cli/extend.md` |
| local launch | `ezpz launch python3 -m ezpz.examples.fsdp --model small` | computes hostfile + `--cpu-bind`, `--timeout`; see `docs/cli/launch.md` |
| benchmark sweep | `python3 -c "from ezpz.cli import main; main()" benchmark --model small` | needs `ezpz` on PATH; see below |
| completions | `ezpz completion bash|zsh|fish` | see `docs/cli/completion.md` |

`ezpz submit` auto-wraps with `ezpz launch` (hostfile + CPU binding). Pass `--no-launch` only for custom launch flags. A redundant `ezpz launch` prefix is stripped with a warning. `ezpz submit --dry-run` prints the script without submitting.

## Smoke — the four-line contract

The smoke is intentionally tiny — no `PYTHONPATH`, no `bash -c` wrapper, no `--workdir` pinning to a shared cache. Each submit gets a fresh `ezpz-<timestamp>` workdir so its venv can't go stale:

```bash
# On a login node (interactive):
NOW=$(date "+%Y-%m-%d-%H%M%S"); tdir="ezpz-${NOW}"; mkdir "${tdir}" && cd "${tdir}" && pwd
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
ezpz launch python3 -m ezpz.examples.test

# Via ezpz submit (preferred — proxy + --login + workdir + launch handled):
NOW=$(date "+%Y-%m-%d-%H%M%S"); tdir="ezpz-${NOW}"; mkdir -p "${tdir}"
ezpz submit --remote aurora -N 1 -q debug --time 00:30:00 --workdir $HOME/$tdir \
  -- python3 -m ezpz.examples.test
# Then watch:
ezpz status --limit 10; ezpz fetch $JID --remote aurora --lines 40
```

Repeat the `tdir`+`submit` block per host; `ezpz status` fanout picks them all up in one table.

Anti-pattern: hand-building `qsub … -- /bin/bash $D/script.sh` with a separate `smoke-wrap.sh` + `smoke-job.sh` — that trio is how the harness grew stale-venv `127 ezpz: command not found` (`0.31.1`/`0.24.0`).

## Submit — flags worth knowing

```bash
ezpz submit --help
```

Key flags: `--remote` (or positional `ezpz submit aurora …`), `-N` nodes, `-q` queue, `-A` account, `-t`/`--time` walltime, `--workdir` (must exist; PBS defaults to `$PBS_O_WORKDIR/ezpz-$PBS_JOBID`, Slurm to `$SLURM_SUBMIT_DIR/ezpz-$SLURM_JOB_ID`), `--filesystems`, `--env` / `--env-setup-script` (explicit env overrides — no managed download), `--no-launch`, `--no-strict`, `--dry-run`, `--gpus-per-node`, `-C` constraint. Existing scheduler scripts passed as `ezpz submit job.sh -N 4` reject `--env` — put setup inside the script.

Full contract: `docs/guides/remote-submission.md`, `docs/cli/submit.md`.

## Status / fetch / logs

- `ezpz status` with no args fans out to every reachable host (IRI + SSH) — columns Host/Job ID/User/State/Queue/Output/Exit + running-nodes block like `qmert`. Filters: `--owner`, `--all-users`, `--historical`, `--states`, `--queue`.
- `ezpz status --remote <host>` — per-host. Add `--watch` to poll (`--interval 10`, `--once`, `--no-notify`); requires `--remote`.
- `ezpz fetch $JID --remote <host> --lines 40` — one PBS stdout snapshot, not a live tail. Dedupe is on by default (`--no-dedupe` for verbatim). Discovery may use IRI, reading still needs SSH. Flags `--out` / portable per-example `logs` are incomplete — don't promise them.
- `ezpz logs $JID --remote <host> --example test --lines 20` — per-example (site-specific workdir resolver).

See `docs/cli/status.md`, `docs/cli/fetch.md`, `docs/cli/logs.md`.

## Launch

`ezpz launch` computes hostfile, rank counts, and the `--cpu-bind=list:…` map for the machine. **Always use it** over bare `mpiexec`; a hand-written `mpiexec -n 24 -ppn 12` supplies no CPU binding and is a confound for timing-sensitive hangs (an entire #252 campaign ran that way).

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp --tp 2   # yes
mpiexec -n 24 -ppn 12 python3 -m ezpz.examples.fsdp_tp --tp 2   # no
```

Bare `mpiexec` is defensible only for non-training fan-out (unpacking a venv one-per-node) — say so in a comment when you do it.

See `docs/cli/launch.md`.

## Proxy, modules, and batch-shell preamble

All three still live in the ambient `SKILL.md` — proxy (`http_proxy=http://proxy.alcf.anl.gov:3128`), the `source /etc/profile` / no-`set -u` / `MODULEPATH` preamble, and the "login-node ≠ compute-node" warning. Read that file if the job dies with walltime `00:00:00` / empty `.o`/`.e` / `Exit_status=1`.

## When to use `alcf-preflight`

`/alcf-preflight` (`.agent/skills/alcf-preflight/SKILL.md`) gates hand-written PBS/Slurm scripts on `ZE_FLAT`, `--login`, `set -u`, shared-FS, stale worktree, and `ezpz.distributed` import. `ezpz submit` already covers most of this — use preflight for multi-arm harnesses or when you stage a worktree/venv the job will read at runtime.

## Related docs

- `SKILL.md` — ambient HPC gotchas, CCL hangs, `debugging a hang`, `writing an experiment script`
- `docs/guides/remote-submission.md` — the `submit`/`status`/`cancel` contract
- `docs/cli/{submit,status,fetch,launch,cancel,extend,resources,incidents,completion,benchmark}.md`
