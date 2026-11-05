---
name: ezpz
description: Use when running, testing, or benchmarking ezpz on an HPC system (Aurora, Sunspot, Polaris, Perlmutter) — job submission, environment setup, and the failure modes that silently produce wrong results. Covers submit/status/fetch/launch, CCL hangs, proxy/modules, and per-machine caveats.
---

# Running ezpz on HPC

Read this **before** writing a batch script or ssh'ing into a cluster.
Everything here was learned by losing allocations to it.

## The one-liner

On **every** ALCF system, in the repo root:

```bash
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
ezpz launch python3 -m ezpz.examples.test
```

`ezpz_setup_env` is the whole setup: it selects the python/venv, loads
the machine's module stack, and builds the hostfile. **Do not
hand-assemble it** from `ezpz_load_modules_<machine>` +
`ezpz_setup_conda_<machine>` + a venv activation. Four Sunspot
allocations were burned doing that; each failed differently and none
produced a result.

To exercise a working tree instead of the installed package:

```bash
uv pip install -e .   # editable install
```

## Prefer `ezpz submit` over hand-writing a batch script

`ezpz submit` generates the scheduler script for you (PBS/SLURM
auto-detected), emits `#!/bin/bash --login` + proxy + workdir, and
wraps with `ezpz launch` (hostfile + CPU binding). Always prefer it.

## Quick dispatch

| intent | command | notes |
|---|---|---|
| smoke on login node | `source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env` → `ezpz launch python3 -m ezpz.examples.test` | 4-line contract, see below |
| remote smoke (one node) | `ezpz submit --remote <host> -N 1 -q debug --time 00:30:00 -- python3 -m ezpz.examples.test` | then `ezpz status` / `ezpz fetch` |
| multi-node training | `ezpz submit -N 2 -q debug -A <acct> --filesystems home,flare -- python3 -m ezpz.examples.fsdp_tp --model large --tp 2` | add `--gpus-per-node`, `-C gpu` on Perlmutter |
| watch progress | `ezpz status --limit 10` (fanout), `ezpz status --remote <host> --limit 5`, `ezpz fetch $JID --remote <host> --lines 40` | fanout is zero-scrape; see `docs/cli/status.md`, `docs/cli/fetch.md` |
| cancel / extend | `ezpz cancel --dry-run`, `ezpz cancel $JID --remote <host>`, `ezpz extend $JID --time 02:00:00` | IRI only; see `docs/cli/cancel.md`, `docs/cli/extend.md` |
| local launch | `ezpz launch python3 -m ezpz.examples.fsdp --model small` | computes hostfile + `--cpu-bind`, `--timeout`; see `docs/cli/launch.md` |
| benchmark sweep | `python3 -c "from ezpz.cli import main; main()" benchmark --model small` | needs `ezpz` on PATH; see below |
| completions | `ezpz completion bash|zsh|fish` | see `docs/cli/completion.md` |

`ezpz submit` auto-wraps with `ezpz launch` (hostfile + CPU binding). Pass `--no-launch` only for custom launch flags. A redundant `ezpz launch` prefix is stripped with a warning. `ezpz submit --dry-run` prints the script without submitting.

### Smoke — the four-line contract

The smoke is intentionally tiny — no `PYTHONPATH`, no `bash -c` wrapper, no `--workdir` pinning to a shared cache. Each submit gets a fresh `ezpz-<timestamp>` workdir so its venv can't go stale:

```bash
# On a login node (interactive):
NOW=$(date "+%Y-%m-%d-%H%M%S"); tdir="ezpz-${NOW}"; mkdir "${tdir}" && cd "${tdir}" && pwd
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
ezpz launch python3 -m ezpz.examples.test

# Via ezpz submit (preferred — proxy + --login + workdir + launch handled):
# Default (no --workdir): job gets its own WORKDIR on shared FS — nothing to create.
ezpz submit --remote aurora -N 1 -q debug --time 00:30:00 -- python3 -m ezpz.examples.test
# Then watch:
ezpz status --limit 10; ezpz fetch $JID --remote aurora --lines 40
```

Repeat the submit per host; `ezpz status` fanout picks them all up in one table.
Only pass `--workdir` when you control the directory and have already created
it on the target host (otherwise `cd $WORKDIR` fails before setup).

Anti-pattern: hand-building `qsub … -- /bin/bash $D/script.sh` with a separate `smoke-wrap.sh` + `smoke-job.sh` — that trio is how the harness grew stale-venv `127 ezpz: command not found` (`0.31.1`/`0.24.0`).

### Submit — flags worth knowing

```bash
ezpz submit --help
```

Key flags: `--remote` (or positional `ezpz submit aurora …`), `-N` nodes, `-q` queue, `-A` account, `-t`/`--time` walltime, `--workdir` (must exist; PBS defaults to `$PBS_O_WORKDIR/ezpz-$PBS_JOBID`, Slurm to `$SLURM_SUBMIT_DIR/ezpz-$SLURM_JOB_ID`), `--filesystems`, `--env` / `--env-setup-script` (explicit env overrides — no managed download), `--no-launch`, `--no-strict`, `--dry-run`, `--gpus-per-node`, `-C` constraint. Existing scheduler scripts passed as `ezpz submit job.sh -N 4` reject `--env` — put setup inside the script.

Full contract: `docs/guides/remote-submission.md`, `docs/cli/submit.md`.

### Status / fetch / logs

- `ezpz status` with no args fans out to every reachable host (IRI + SSH) — columns Host/Job ID/User/State/Queue/Output/Exit + running-nodes block like `qmert`. Filters: `--owner`, `--all-users`, `--historical`, `--states`, `--queue`.
- `ezpz status --remote <host>` — per-host. Add `--watch` to poll (`--interval 10`, `--once`, `--no-notify`); requires `--remote`.
- `ezpz fetch $JID --remote <host> --lines 40` — one PBS stdout snapshot, not a live tail. Dedupe is on by default (`--no-dedupe` for verbatim). Discovery may use IRI, reading still needs SSH. Flags `--out` / portable per-example `logs` are incomplete — don't promise them.
- `ezpz logs $JID --remote <host> --example test --lines 20` — per-example (site-specific workdir resolver).

See `docs/cli/status.md`, `docs/cli/fetch.md`, `docs/cli/logs.md`.

### Launch

`ezpz launch` computes hostfile, rank counts, and the `--cpu-bind=list:…` map for the machine. **Always use it** over bare `mpiexec`; a hand-written `mpiexec -n 24 -ppn 12` supplies no CPU binding and is a confound for timing-sensitive hangs (an entire #252 campaign ran that way).

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp --tp 2   # yes
mpiexec -n 24 -ppn 12 python3 -m ezpz.examples.fsdp_tp --tp 2   # no
```

Bare `mpiexec` is defensible only for non-training fan-out (unpacking a venv one-per-node) — say so in a comment when you do it.

See `docs/cli/launch.md`.

## Inside a batch script, two things matter

Use `ezpz submit` and both are handled. The rest of this section is the "why".

### Compute nodes need the proxy; PBS scripts must be login shells

- Proxy: `http_proxy=http://proxy.alcf.anl.gov:3128` (and
  `https_proxy`/`no_proxy`) — `ezpz submit` emits this; for interactive
  shells put it in `~/.bashrc`/`~/.zshrc`.
- Shebang: `#!/bin/bash --login` sets `MODULEPATH` so `module load`
  works; `ezpz submit` already emits it.

HF caches (`hf`/`hf_trainer`/`fsdp_tp`/`diffusion`) must be pre-staged:
`ls "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"`.

### Login-node checks do not predict compute-node behaviour

`ezpz_setup_env` **fails on the Sunspot login node** (`CONDA_PREFIX
still not set`) because `module load frameworks` is a compute-node
thing. A login-node smoke test is not evidence the job will work —
and conversely, a login-node success is not evidence either: a
Perlmutter module combination that verified clean on the login node
still died with `ImportError: libcudart.so.13` on the compute nodes.

## Machine specifics

| | Sunspot | Aurora | Polaris | Perlmutter |
|---|---|---|---|---|
| accel | PVC (XPU) | PVC (XPU) | A100 | A100 |
| collectives | xccl | xccl | NCCL | NCCL |
| per node | 12 tiles | 12 tiles | 4 GPUs | 4 GPUs |
| scheduler | PBS | PBS | PBS | SLURM |

**Sunspot.** `qsub` is not on `$PATH` over plain ssh — use
`/opt/pbs/bin/qsub`. All four flags are required:

```
/opt/pbs/bin/qsub -l select=2 -l walltime=01:00:00 \
  -l filesystems=tegu:home -A datascience -q workq \
  -o $D/job.o -e $D/job.e -- /bin/bash $D/script.sh
```

- `-A datascience` — `Aurora_deployment` has no active Sunspot allocation.
- `-l filesystems=tegu:home` — required; `flare` is not valid here.
- Scripts **and** `-o`/`-e` must live on a shared filesystem. The login
  node's `/tmp` is not the compute node's `/tmp`.
- `$HOME/datascience` → `/lus/tegu/projects/datascience` (same dir).

**Polaris.** `module use /soft/modulefiles` first, then
`module load conda && conda activate base`. Available conda modules, as
of 2026-09-26: `2025-09-25` (the default, torch 2.8.0), `2025-09-28`,
`2026-09-17` (torch 2.14.0), and `2026-10-01` (torch 2.14.0).
`2026-10-01` is the newest and is **verified working end-to-end on
compute nodes** (job 7660718: module load, activate, FSDP+TP at tp=2 and
tp=4 across 2 nodes over NCCL). `2026-09-17` is likewise verified —
module load, activate, `ezpz_setup_job`, and a 2-node NCCL collective —
but neither is the site default
(`.modulerc.lua` still has its `module_version(..., "default")`
commented out), so `ezpz_setup_conda_polaris` still pins `2025-09-25`.
To use the newer one, preload it: `ezpz_setup_conda_polaris` early-returns
when `CONDA_PREFIX` is already set, which is the supported way to choose
a different conda.

**Perlmutter.** `module load nccl/2.24.3` or **NCCL silently falls back
to TCP** — 8.3× slower inter-node, no error, nothing in the log. Also
load `cudatoolkit/12.9` and do **not** swap it per torch build: the
NERSC NCCL plugin links `libcudart.so.12`, so under `cudatoolkit/13.0`
it fails with `Failed to initialize any NET plugin`.

!!! warning "That `module load nccl/…` may be inert"

    The NERSC PyTorch 2.13 install bundles its own `libnccl.so.2`
    (2.29.7) under `site-packages/nvidia/nccl/lib/`, which **wins via
    RPATH** — so the module load changes nothing and you are running a
    different NCCL than you asked for. Verify with `NCCL_DEBUG=INFO`
    rather than assuming; that mismatched pairing (aws-ofi-nccl 1.6.0
    against NCCL 2.29.7) is the leading suspect in ezpz #239.

## `CCL_OP_SYNC=1` on XPU: required for FSDP2 + TP>1

On oneCCL 2022.x (Aurora/Sunspot `frameworks/2026.1.0`), a 2D
`(dp_shard, tp)` mesh **hangs without `CCL_OP_SYNC=1`**. Measured on Sunspot,
24 ranks / 2 nodes, arms alternating within single allocations:

| setting | completions |
|---|---|
| `CCL_OP_SYNC=1` | **20/20** |
| unset (oneCCL default) | **2/20** |

~90% of runs hang with `SequenceParallel`, ~8% without it. The hang has **no
error, no traceback and no watchdog** — ranks stay alive inside a collective
that never returns, stopping at a different iteration each run (0, 2, 13, 54,
93, 117, 171 all observed).

**ezpz now defaults this to `1`** in every XPU setup entry point
(`ezpz_load_modules_*`, `ezpz_setup_xpu`, and `ezpz_setup_conda_*` — the last
being the one the recommended `ezpz_setup_env` flow actually reaches).
`tests/test_ccl_op_sync_default.sh` pins all five.

**An explicit caller value always wins, in both directions**, and is
preserved across `module load`:

```bash
export CCL_OP_SYNC=0   # async opt-in, for workloads that have measured it
export CCL_OP_SYNC=1   # explicit sync (same as the default)
```

Choose `0` deliberately only if you have measured your own workload: a
64-node / 768-rank TorchTitan run was ~27x slower synchronous, so at scale
the trade can invert. It is a default, not a policy.

`0` is oneCCL's own default, so `export CCL_OP_SYNC=0` is not a no-op — it is
the hanging configuration. oneCCL confirms the change in its log:

```
|CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
```

**Cost: ~11%** (240 steps: async 83-88 s, sync 92 s on every run —
synchronous completion also makes runtime near-deterministic). Note this is
the *opposite* trade from the TorchTitan measurement above: async is faster
when it works, and on oneCCL 2022.x with FSDP2+TP it frequently does not.

Five other oneCCL knobs were tested against a live control and **none** help:
`CCL_ATL_SYNC_COLL=1`, `CCL_ZE_DEPS_SYNC=1`, `CCL_ZE_SERIALIZE=1`,
`CCL_WORKER_WAIT=0`, `CCL_STRICT_ORDER=1`, `CCL_SYCL_OUTPUT_EVENT=0`.

!!! warning "The loudest message in the log is a red herring"

    Every hung run is full of

    ```
    |CCL_WARN| explicit dependencies are not supported for group calls: reduce_scatter
    ```

    It is not the mechanism. `CCL_ZE_DEPS_SYNC=1` — the knob that warning
    names — gives **0/8 completions**, no better than control. The warning
    also appears at the same rate (~24 per iteration) in runs that complete.

Background and full evidence: ezpz #252.

## Debugging a hang

**Lower `TORCH_DDP_TIMEOUT`.** It defaults to 3600 s, so a deadlock
outlives a debug allocation and looks like silence rather than a
watchdog dump. This is the single highest-value diagnostic:

```bash
export TORCH_DDP_TIMEOUT=300
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
```

**Actually set it.** A full session was spent on #252 with the default
3600 s timeout, so every hang ran its whole budget in silence and had to
be diagnosed by attaching `py-spy` to wedged ranks afterwards. The
watchdog would have named the stuck collective on its own.

On XPU the NCCL-specific variables above are inert; the timeout is not.
`py-spy dump --pid <rank>` remains the fallback, and it is what finally
showed ranks blocked in *different* collectives:

```bash
mapfile -t pids < <(pgrep -f "[e]zpz.examples.fsdp_tp")
for p in "${pids[@]}"; do py-spy dump --pid "$p" | head -30; done
```

Dump **every rank on every node**, not one node: an 8/4 split read from
a single node was really 8/16 across two, and the shape of the split is
the evidence.

## Writing an experiment script

Rules that come from real misreadings, not style preference:

- **Always launch with `ezpz launch`, never bare `mpiexec`.** This is
  the point of the library, and it is not cosmetic: `ezpz launch`
  computes the hostfile, rank counts and — critically — the
  `--cpu-bind=list:1-8:9-16:…` map for the machine. A hand-written
  `mpiexec -n 24 -ppn 12` supplies **no CPU binding at all**, so ranks
  land wherever the OS puts them. For a timing-sensitive,
  nondeterministic hang that is a confound, not a detail, and an entire
  #252 measurement campaign ran that way before anyone noticed.

  ```bash
  ezpz launch python3 -m ezpz.examples.fsdp_tp --tp 2   # yes
  mpiexec -n 24 -ppn 12 python3 -m ezpz.examples.fsdp_tp --tp 2   # no
  ```

  It also has `--timeout`, so there is no reason to wrap it in
  `timeout(1)` for the launcher's own sake. Bare `mpiexec` is defensible
  only for non-training fan-out (unpacking a venv one-per-node, reaping
  ranks) where there is no rank topology to compute — say so in a
  comment when you do it.
- **Reproduce a reference by copying its argv, not approximating it.**
  A run's own log prints the full command (`ezpz launch` logs
  `cmd_to_launch:`). Retyping the flags you think matter drops the ones
  you did not know about: omitting `--hf-assets-path` killed every rank
  with `FileNotFoundError: Tokenizer path './assets/hf/gemma-7b'`, and
  the "reference" being reproduced had also run only 5 steps against the
  240 it was being compared to.
- **Capture `$?` before any pipe.** `cmd | tail -60` then
  `echo "exit=$?"` reports *tail's* status, so a command that never
  started records `exit=0`. This turned a benchmark that died on its
  first line into a clean pass (Polaris job 7665390). Redirect to a
  file, save `rc=$?`, *then* tail:

  ```bash
  # `if`, not a bare call: under `set -e` a nonzero exit kills the shell
  # BEFORE `rc=$?` runs -- losing the status in exactly the failing case
  # you wanted to diagnose. Verified: `set -e; false; rc=$?` never
  # reaches the assignment.
  if timeout 900 python3 -c "from ezpz.cli import main; main()" \
          benchmark --model=small >"${W}/bench.log" 2>&1; then
      rc=0
  else
      rc=$?                  # captured BEFORE any pipe, and survives set -e
  fi
  tail -70 "${W}/bench.log"
  echo "benchmark exit=${rc}"
  ```

  `set -o pipefail` also fixes this (`false | tail -1` then reports 1,
  not 0) and the preamble above already sets it — the Polaris script
  that got bitten simply did not. Saving `rc=$?` off an unpiped command
  is the belt-and-braces version: it survives someone later dropping
  `pipefail`, and it keeps the log readable via `tail` regardless.

- **`python3 -m ezpz.cli` does not work.** `ezpz.cli` is a package with
  no `__main__`, so `-m` fails with *"'ezpz.cli' is a package and cannot
  be directly executed"*. The only entry point is the `ezpz` console
  script from `[project.scripts]`, which a bare `PYTHONPATH` clone
  (no `pip install`) does not provide. From an uninstalled checkout call
  the group directly:

  ```bash
  python3 -c "from ezpz.cli import main; main()" benchmark --model=small
  ```

  `python3 -m ezpz.examples.<name>` *is* fine — those modules do have a
  `__main__` guard. It is only the CLI group that needs this.

  **`ezpz benchmark` additionally needs an `ezpz` on `PATH`.** It spawns
  one child per example via `ezpz launch` (`run_all.py`), so entering the
  CLI is not sufficient — without the console script the first example
  dies with `FileNotFoundError: [Errno 2] No such file or directory:
  'ezpz'` (Perlmutter `pytorch/2.13.0`, job `59015143`). Polaris hides
  this: its conda env has ezpz installed. On an uninstalled checkout,
  either `pip install -e .` (not possible against a read-only module) or
  drop a shim ahead of it on `PATH`:

  ```bash
  if ! command -v ezpz >/dev/null 2>&1; then
      mkdir -p "${HOME}/.ezpz-shim"
      # QUOTED heredoc ('SHIM', not SHIM): an unquoted one -- or printf --
      # expands "$@" while writing the file, baking in the *caller's* args
      # and giving `Error: No such command '$@'`.
      cat >"${HOME}/.ezpz-shim/ezpz" <<'SHIM'
  #!/usr/bin/env bash
  exec python3 -c "from ezpz.cli import main; main()" "$@"
  SHIM
      chmod +x "${HOME}/.ezpz-shim/ezpz"
      export PATH="${HOME}/.ezpz-shim:${PATH}"
  fi
  ```

  ezpz ≥ the fix for #262 resolves the launcher itself and needs no shim.

- **Never classify a run by exit code.** A cell can exit `rc=1` having
  trained all 20 iterations (teardown failure), and hanging cells exit
  124 *or* 134. Classify on evidence: a watchdog line means HANG,
  reaching the plotting stage means TRAINED.
- **Do not gate on an `iter=` marker** — `fsdp_tp` runs do not emit one,
  and requiring it once labelled a clean 173 s pass INDETERMINATE.
- **Have a third verdict.** A crash, OOM, or bad environment is
  `INDETERMINATE` — it is *not* evidence for either arm. Print
  `(log is EMPTY)` when a cell produced no output, so the next reader
  knows the launch never started.
- **Capture `rc` before any pipe.** `| head` SIGPIPEs `tee` and clobbers
  `PIPESTATUS`, which once reported `rc=1` for a run that trained fine.
- **Assert the code under test by behaviour, not by name.** Checking an
  env var passes vacuously against a checkout that predates it — one
  job silently measured the opposite experimental arm that way. Import
  the function and assert what it returns.
- **Run the control in the same allocation** as the arm it controls, so
  a difference cannot be node or topology luck.
- **`timeout` kills the launcher, not the ranks.** MPI ranks survive it
  as orphans and keep holding accelerators. A later trial in the same job
  then starts on a contended node and hangs before iteration 1 — which
  looks exactly like the bug under test. Observed in job 8872503: 13
  orphaned torchtitan ranks plus 17 `fsdp_tp` ranks alive on a 12-tile
  node, 25 minutes after their `timeout 600` fired, and both trials of
  that job were void. Reap after **every** arm and assert clean before
  the next:

  ```bash
  RANK_PAT='[e]zpz\.examples\.fsdp_tp|[t]orchtitan\.experiments'
  reap()  { mpiexec -n "$NNODES" -ppn 1 bash -c "pkill -f '$RANK_PAT'; sleep 3; pkill -9 -f '$RANK_PAT'; exit 0" >/dev/null 2>&1; }
  count() { mpiexec -n "$NNODES" -ppn 1 bash -c "pgrep -cf '$RANK_PAT' || echo 0" 2>/dev/null | awk '{t+=$1} END {print t+0}'; }
  ```

  The **bracket form is load-bearing**: with a plain pattern, `pgrep -f`
  matches the counter's own `bash -c` argv and reports stragglers on a
  freshly-reaped node, so every trial is discarded as dirty. Abort the
  job if `count` is non-zero right after a reap — a gate that always
  trips is worse than no gate.
- **Zero iterations is two different outcomes.** A launch failure (dies
  in seconds) and a hang before the first iteration (runs the whole
  budget) both produce no `iter=` line. Scoring them alike turns a real
  hang into "NO-PROGRESS" and hides it. Discriminate on **elapsed time**
  against the budget, not on the iteration count alone.
- **A cache key includes the run config.** `--training.steps` feeds
  torchtitan's blendcorpus index hash, so an index warmed at 2 steps does
  not serve a 240-step run and 24 ranks then race to build the missing
  one — `FileNotFoundError` on a file that exists moments later. Warm the
  cache with the **exact** config the trials use, once, before measuring.
- **`date -d` is GNU-only.** It fails silently on macOS/BSD, so a
  `last_epoch=$(date -d "$ts" +%s 2>/dev/null || echo "$now")` fallback
  makes quiet-time zero and the STALLED branch unreachable — every hang
  reads as "still running". Try `date -j -f "%Y-%m-%d %H:%M:%S"` as well
  and emit `UNKNOWN-TIME` if both fail, rather than defaulting.
- **Test the classifier against recorded logs before trusting a run.**
  Feed it one known-hung and one known-completed log and check it says
  so. Six harness defects in a single measurement campaign each produced
  plausible output; three nearly became reported findings.

## Gotchas that produce silently wrong results

- **Never pipe `module load`.** A pipeline runs its stages in a
  **subshell**, so everything a modulefile exports via `execute{}` —
  including the `conda` shell function — is discarded when the subshell
  exits. Trimming a banner is enough to break it:

  ```bash
  module load conda/2026-09-17 | grep -v Lmod   # conda: NOTFOUND
  module load conda/2026-09-17 >/dev/null       # conda: conda   ✅
  ```

  A redirect is fine; a pipe is not. This cost a wrong bug report: the
  module looked broken (`conda: command not found`, `import torch`
  failing), the Lua `execute{cmd="source …/conda.sh"}` looked like it
  was not running, and the real cause was the `| grep` in the *test*.
  `module load conda && conda activate base` works exactly as
  documented.
- **Do not redirect the call you are debugging.** `cmd >/dev/null 2>&1`
  on a failing step throws away the only evidence of *why* it failed. A
  job that died with `exit 1` and no message cost two full rounds; the
  same job with output kept named its cause on the first try
  (`NGPUS: unbound variable`). Silence the noisy neighbours, never the
  suspect.
- **Non-editable installs.** The torchtitan venvs install `ezpz`
  non-editable, so `python3 -c "import ezpz"` imports the *installed*
  copy, not your working tree. Pytest is fine (`tests/conftest.py`
  prepends `src/`), which makes this confusing: unit tests pass while an
  inline probe raises `AttributeError` on a symbol you just added. Set
  `PYTHONPATH=$PWD/src`, and print `ezpz.__file__` to prove it.
- **`uv pip install torch` silently installs nothing.** `pyproject.toml`
  pins torch/torchvision/torchaudio/mpi4py to `sys_platform == 'never'`
  under `[tool.uv] override-dependencies`, and **uv applies that
  override to `uv pip install` too** — so it reports `Audited 1 package`
  and the venv still has no torch. Bootstrap pip and use bare pip, which
  ignores uv config (same trick as `.github/workflows/pytest.yml`):

  ```bash
  uv pip install --python "$V/bin/python" pip
  "$V/bin/python" -m pip install torch==2.13.0 \
      --index-url https://download.pytorch.org/whl/cu129
  ```

- **Under PALS, nothing derived from the interpreter's own location
  survives.** PBS/PALS copies the python binary into a per-job temp dir
  before launching ranks, which breaks every relative lookup. Three
  separate failures in one job, each looking unrelated:

  | what breaks | symptom | fix |
  |---|---|---|
  | `libpython` via `$ORIGIN/../lib` | `error while loading shared libraries: .../files/0/../lib/libpython3.12.so.1.0` | use a python with a real `RUNPATH` |
  | `site-packages` via `sys.prefix` → `pyvenv.cfg` | `ModuleNotFoundError: No module named 'torch'` *while the launcher clearly invoked the venv python* | put site-packages on `PYTHONPATH` |

  Note the first symptom quotes a **relative** path — that is an
  `$ORIGIN` lookup, and `LD_LIBRARY_PATH` cannot fix it. Reading it as a
  generic "library not found" cost an allocation.

  **Prefer Cray's python** (`/opt/cray/pe/python/3.12.12`) as the venv
  base on Polaris: it has `RUNPATH /opt/cray/pe/python/3.12.12/lib` and
  ships a working mpi4py, so it avoids both. A `uv`-managed CPython has
  **no** RUNPATH. Then belt-and-braces:

  ```bash
  _SP="$(echo "${V}"/lib/python3.*/site-packages)"
  export PYTHONPATH="${D}/src:${_SP}:${PYTHONPATH:-}"
  ```

- **On Polaris compute nodes, `import ezpz` hard-requires mpi4py.**
  `src/ezpz/__init__.py:17` does `if socket.getfqdn().startswith("x3")`
  → `from mpi4py import MPI`. Polaris compute nodes are `x3…`, login
  nodes are not — so ezpz imports fine on the login node and dies on the
  compute node with whatever is wrong with mpi4py. Do not conclude from
  a clean login-node import that the job will run. (Tracked in TODO.md
  §2 as a hardcoded hostname check.)

- **mpi4py on Polaris must be built against the MPI that exists.** A
  stock build can link `libmpi_gnu_123.so.12`, a soname absent from the
  current Cray PE — only `libmpi_gnu.so.12` is present. Symptom:
  `ImportError: libmpi_gnu_123.so.12: cannot open shared object file`.
  Forcing the wrong lib via `LD_LIBRARY_PATH` gives
  `*** stack smashing detected ***`, so rebuild rather than shim:

  ```bash
  module load craype cray-mpich PrgEnv-gnu
  export LDFLAGS="-L${MPICH_DIR}/lib -L/opt/cray/pe/lib64 -lmpi_gnu"
  export CFLAGS="-I${MPICH_DIR}/include"
  MPICC=cc python -m pip install --no-binary mpi4py --no-build-isolation mpi4py
  ```

  Check `ldd .../mpi4py/MPI*.so | grep mpi` shows no `not found`.

- **Downloads on ALCF need the proxy.** Nothing reaches the internet
  from a login node without:

  ```bash
  export http_proxy=http://proxy.alcf.anl.gov:3128
  export https_proxy=http://proxy.alcf.anl.gov:3128
  export no_proxy=localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov
  ```

- **torch 2.13 renamed FSDP2's collectives.**
  `all_gather_into_tensor` → `all_gather_single`,
  `reduce_scatter_tensor` → `reduce_scatter_single`. Any test that
  monkeypatches the old names records **zero** collectives on 2.13 — and
  `0 == 0` is symmetric, so a "collectives are balanced" assertion
  passes *vacuously*. Patch whichever pair exists, assert a hook landed,
  and pin absolute counts alongside any equality.
- **One SSH poller per cluster.** Parallel polling loops exhaust the
  login node's auth-attempt limit and lock you out
  (`Too many authentication failures`). An ssh-based watcher exiting
  255 means *connection*, not job failure — re-check `qstat`/`squeue`
  before reporting anything.


## Remote control — submit/status/cancel from anywhere

Read `docs/guides/remote-submission.md` + the CLI pages under `docs/cli/` for the full contract. Summary for agents:

- **Positional sugar:** `ezpz submit aurora ...` ≡ `ezpz submit --remote aurora ...`. Same for `status`/`cancel` (`ezpz status aurora --limit 5` lists your jobs on Aurora; job IDs are numeric, hosts alphabetic — no ambiguity).
- **Auto is IRI-first** on Aurora/Polaris/Crux, SSH elsewhere. Run `uvx alcf-tokens login iri` on the machine invoking ezpz, or supply `ALCF_IRI_TOKEN` there. Scheduler rejections must not trigger a retry. Verify ambiguous mutation failures before resubmitting.
- **Fan-out status:** `ezpz status` with no args fans out to every reachable machine (IRI + SSH) into one table (Host/Job ID/User/State/Queue/Output/Exit + running-nodes block like `qmert`). Add `--owner`/`--all-users`/`--historical`/`--states`.
- **No double launch:** `ezpz submit` already wraps with `ezpz launch`. A redundant `ezpz launch` prefix is auto-stripped with a warning; use `--no-launch` only for custom flags.
- **Choose the environment first:** `--env PATH` selects an existing venv/conda/Python prefix (or a Python executable under `PREFIX/bin/`). `--env-setup-script PATH` sources a target-side shell script **before** selecting `--env`; use both for module-dependent environments. Neither option uploads files. Scripts/environment paths must be visible to compute nodes; prefer absolute shared paths.
- **Explicit means explicit:** either environment option replaces the managed environment bootstrap, without injecting a curl download or package install. Env-only does not load site modules; script-only owns all setup. Install ezpz in the selected environment before using the launch wrapper. Existing scheduler scripts reject these flags; put setup inside the script. Migrate the old script/inline `--env` to `--env-setup-script`; the Python API's `env_setup` remains raw shell.
- **Remote workdir:** choose it before environment setup, even with overrides. PBS defaults to `$PBS_O_WORKDIR/ezpz-$PBS_JOBID`, Slurm to `$SLURM_SUBMIT_DIR/ezpz-$SLURM_JOB_ID`; both use a job-specific `$HOME` fallback. Bases must be shared FS, never assume node-local `/tmp` is shared. Explicit `--workdir` is used exactly (no suffix) and must already exist. `mkdir`/`cd` failures stop the job even with `--no-strict`. Local submissions retain their cwd.
- **Offline setup:** with neither environment option, the remote managed bootstrap still downloads `utils.sh` and may install ezpz. Compute-node internet is not guaranteed; prefer a staged setup script and prebuilt environment. Recognized ALCF targets get proxy settings before setup; non-ALCF targets do not.
- **Read stdout:** `ezpz fetch JOB_ID --remote sunspot --lines 80 --no-dedupe` takes one PBS stdout snapshot, not a live follow. IRI may discover the path but reading still needs SSH. The current fallback is PBS `qstat`, not Slurm. Do not promise `--out err/both` or portable per-example `logs`: stream selection is incomplete and the current `logs` workdir resolver is site-specific. See `docs/cli/fetch.md`.
- **Preflight (hand-written scripts):** `.agent/skills/alcf-preflight/preflight.sh <worktree> [venv] [machine] [job-script]` — see `/alcf-preflight`. `ezpz submit` already covers most gates; use preflight for experiment harnesses and worktree/venv staging.
- **Useful commands:** `ezpz status --json`, `ezpz cancel --dry-run`, `ezpz extend --time`, `ezpz resources`, `ezpz incidents`, `ezpz fetch`, `ezpz logs --example test`, `ezpz completion`.

## Known open issues

**#239** — LoRA + FSDP2 deadlocks in the first backward on Perlmutter
(A100/NCCL, torch 2.13). Boundary is exact: `--lora-rank 17` hangs,
`18` trains. **Workaround: `--lora-rank 18` or higher.** Six hypotheses
refuted so far; see `docs/guides/lora-fsdp-deadlock.md` before
proposing a seventh.

**#252** — FSDP + TP>1 hangs nondeterministically on XPU
(Aurora/Sunspot, oneAPI 2026.1.0). `--tp 1` is clean and emits zero CCL
warnings; tp=2 and tp=4 hang at a different point every run (observed:
0, 3, 17, 77, 120, 135, and once a clean 240/240 on the same machine and
commit). No error, no watchdog, full speed then a dead stop. `py-spy`
shows ranks blocked in **different** collectives — 8 in `all_gather`,
16 in `reduce_scatter` — i.e. a collective mismatch, not a crash.

Six hypotheses are refuted on the issue; read them before proposing a
seventh. Three died to defects in the *measurement*, not the system.
Two framing traps worth inheriting:

- The premise "it works in torchtitan, not ezpz" was never established.
  The torchtitan reference ran **5 steps**; ezpz has completed **240** on
  the same machine. At a ~50% failure rate over 240 steps, a 5-step run
  essentially never catches it.
- Because the failure is probabilistic, **single-run comparisons cannot
  separate two programs**. Measure a rate over repeated alternating
  trials and state the interpretation before looking at the data.

## Sub-skills

- `/ezpz:preflight` (`.agent/skills/ezpz/preflight`) — gates hand-written PBS/Slurm scripts on `ZE_FLAT`, `--login`, `set -u`, shared-FS, stale worktree, and `ezpz.distributed` import. See its `SKILL.md` for the checklist.
