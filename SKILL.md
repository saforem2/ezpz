---
name: ezpz
description: Use when running, testing, or benchmarking ezpz on an HPC system (Aurora, Sunspot, Polaris, Perlmutter) — job submission, environment setup, and the failure modes that silently produce wrong results.
---

# Running `ezpz` on HPC

Read this **before** writing a batch script or ssh'ing into a cluster.
Everything here was learned by losing allocations to it.

## The one-liner

On **every** ALCF system, in the repo root:

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
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
uvi -e .            # editable install
# or, without installing:
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Inside a batch script, three things change

### 1. Compute nodes have no outbound internet

`source <(curl -fsSL https://bit.ly/ezpz-utils)` **times out after
~270 s** and then silently leaves the environment unconfigured. Source
the repo's own copy — the same file:

```bash
source "${D}/src/ezpz/bin/utils.sh"
ezpz_setup_env || { echo "FATAL: ezpz_setup_env failed"; exit 1; }
```

### 2. PBS runs your script under a NON-login shell

`qsub ... -- /bin/bash script.sh` gives you a shell where `module` does
not exist. Worse, sourcing Lmod's `init/bash` alone defines `module` but
leaves **`MODULEPATH` empty**, so every `module load` becomes a silent
no-op ("No modules loaded") and `ezpz_setup_env` fails downstream with
the misleading `CONDA_PREFIX still not set`.

Source the login profile, and assert it worked.

**Do not use `set -u` in these scripts at all.** The module system is
not `set -u`-clean, in two separate places:

- `/etc/profile` references unset variables, so sourcing it under
  `set -u` aborts the script on that line — walltime `00:00:00`,
  `Exit_status=1`, **empty `.o` and `.e`**, not one line printed.
- Deferring `set -u` past the profile is *still* not enough: Lmod's
  `init/bash` reads `$ZSH_EVAL_CONTEXT`, and `ezpz_setup_env` re-enters
  the module machinery long after the preamble, so it dies mid-setup
  with `ZSH_EVAL_CONTEXT: unbound variable`.

Reproduce both with `bash -c 'set -u; source /etc/profile'`.

```bash
set -o pipefail                       # and NOT set -u, anywhere
source /etc/profile 2>/dev/null || true
command -v module >/dev/null 2>&1 || { echo "FATAL: no module cmd"; exit 1; }
[ -n "${MODULEPATH:-}" ] || { echo "FATAL: MODULEPATH empty"; exit 1; }
```

Use `${VAR:-default}` everywhere and explicit `[ -n ... ]` checks
instead — on these systems the shell strictness has to give way to the
module system, not the other way round.

An empty `.o`/`.e` with zero walltime always means the script died in
its own preamble. Check `qstat -xf <jobid>` for `Exit_status` and
`resources_used.walltime` before assuming anything about the run.

**Dry-run the preamble before submitting.** Everything above the first
`probe` call can be pasted into `ssh <host> bash -c '...'` and checked
in seconds. Six Sunspot submissions were spent discovering preamble bugs
one allocation at a time; each would have been caught by a five-second
dry run.

### 3. Login-node checks do not predict compute-node behaviour

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

**Perlmutter.** `module load nccl/2.24.3` or **NCCL silently falls back
to TCP** — 8.3× slower inter-node, no error, nothing in the log. Also
load `cudatoolkit/12.9` and do **not** swap it per torch build: the
NERSC NCCL plugin links `libcudart.so.12`, so under `cudatoolkit/13.0`
it fails with `Failed to initialize any NET plugin`.

## Debugging a hang

**Lower `TORCH_DDP_TIMEOUT`.** It defaults to 3600 s, so a deadlock
outlives a debug allocation and looks like silence rather than a
watchdog dump. This is the single highest-value diagnostic:

```bash
export TORCH_DDP_TIMEOUT=300
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
```

## Writing an experiment script

Rules that come from real misreadings, not style preference:

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

## Gotchas that produce silently wrong results

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

## Known open issue

**#239** — LoRA + FSDP2 deadlocks in the first backward on Perlmutter
(A100/NCCL, torch 2.13). Boundary is exact: `--lora-rank 17` hangs,
`18` trains. **Workaround: `--lora-rank 18` or higher.** Six hypotheses
refuted so far; see `docs/guides/lora-fsdp-deadlock.md` before
proposing a seventh.
