---
name: alcf-preflight
description: Use BEFORE submitting any PBS/Slurm job on Aurora, Sunspot, Polaris, Perlmutter or Sophia. Gates on ZE_FLAT, MODULEPATH, shared-filesystem paths, stale worktree, and Python API signatures — every check catchable without a queue allocation.
---

# ALCF preflight

Six jobs were burned in one session to five omissions checkable before `qsub`. `SKILL.md` documents them as prose; this skill makes them a gate.

## Rule 0: prefer `ezpz submit`, do not hand-write a script

`ezpz submit` already emits `#!/bin/bash --login`, selects the workdir, and wraps with `ezpz launch` (hostfile + CPU binding). Start from it:

```bash
ezpz submit --dry-run -N 2 -q debug -A ACCOUNT -- python3 -m ezpz.examples.test --model debug
```

Hand-written scripts reproduce the same five omissions repeatedly. When you must write one, run `preflight.sh` first.

## Running the gate

```bash
.agents/skills/alcf-preflight/preflight.sh <worktree> [venv] [machine] [job-script]
```

Exits non-zero on any `BAD` line. Every gate below is enforced there.

## The gates a hand-written script forgets

| gate | `BAD` when | why it bites |
|------|-----------|--------------|
| `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | env var not `FLAT` on Aurora/Sunspot | PVC tile is hidden under `COMPOSITE`; `device index out of range [0, 6)` at 12 ranks/node |
| `#!/bin/bash --login` + `MODULEPATH` | script shebang lacks `--login`, or `MODULEPATH` empty | PBS has no `module` without login; `module load` becomes silent no-op, `CONDA_PREFIX still not set` downstream |
| shared FS | workdir or script path under `/tmp` | compute `/tmp` is node-local `tmpfs`; job starts then dies opening a second file |
| worktree current | `HEAD != @{u}` | stale checkout misses files a recent commit moved/added |
| `inspect.signature` | `bind()` on `setup_torch`/`get_torch_device` fails | renamed kwarg (`backend=`) passed locally, dies only on compute |

Python deps (`torch`, `mpi4py`, `ezpz`) are checked only when a venv is given; skip them for `ezpz submit` flows where the job bootstraps its own env.

## Before every submission

1. **Dry-run the APIs** — `inspect.signature(ezpz.distributed.setup_torch).bind(...)` with the exact kwargs the job will use. Presence checks are not enough.
2. **Preflight the worktree** — run `preflight.sh` in the checkout the job will read at runtime.
3. **Check the worktree is current** — a pinned worktree silently measures the wrong code.

## `ezpz submit` already covers most of this

`ezpz submit` generates the script locally so `--dry-run` shows exactly what runs. It does not need `preflight.sh` for its own output — but run preflight when you:

- hand-write a `.pbs`/`.sbatch` or multi-arm experiment harness
- stage a worktree/venv that the job will read at runtime
- change `setup_torch` kwargs or `ZE_FLAT` handling

## Machine facts

- **Aurora/Sunspot**: XPU, 6 devices `COMPOSITE` / 12 `FLAT`. `ZE_FLAT_DEVICE_HIERARCHY=FLAT` set by `utils.sh`.
- **Polaris**: CUDA, PBS, `cray-pals` + `libfabric`. `qsub` at `/opt/pbs/bin/qsub`.
- **Perlmutter**: CUDA, Slurm, `cudatoolkit/12.9`, `nccl/2.24.3`. Needs `--gpus-per-node 4 -C gpu`.
- **Sophia**: CUDA, PBS.
- `uv` and `qsub` not on non-interactive `PATH`: `$HOME/.local/bin/uv`, `/opt/pbs/bin/qsub`.
