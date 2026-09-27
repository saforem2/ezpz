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
# or, without installing:
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Prefer `ezpz submit` over hand-writing a batch script

`ezpz submit` generates the scheduler script for you — PBS or SLURM,
detected automatically — and **wraps the command in `ezpz launch` by
default**, so the CPU-binding mistake below cannot happen:

```bash
ezpz submit -N 2 -q debug -A datascience --filesystems home,flare \
    -- python3 -m ezpz.examples.fsdp_tp --model large --tp 2

# Perlmutter (SLURM) needs the GPU directives:
ezpz submit -N 2 -q debug -A m4388_g -C gpu \
    --gpus-per-node 4 --ntasks-per-node 4 \
    -- python3 -m ezpz.examples.fsdp_tp --tp 2

ezpz submit --dry-run ...       # print the script, submit nothing
ezpz submit --no-strict ...     # omit `set -e` for multi-arm scripts
ezpz submit job.sh -N 4         # or submit an existing script
```

`--dry-run` first is worth the two seconds: it prints exactly what would
be submitted.

The rest of this section is for when you genuinely need a hand-written
script (multi-arm experiment harnesses, custom staging). Everything in
it is what `ezpz submit` already does correctly.

## Inside a batch script, three things change

### 1. Compute nodes have no outbound internet

`source <(curl -fsSL https://ezpz.cool/utils.sh)` **times out after
~270 s** and then silently leaves the environment unconfigured. Source
the repo's own copy — the same file:

```bash
source "${D}/src/ezpz/bin/utils.sh"
ezpz_setup_env || { echo "FATAL: ezpz_setup_env failed"; exit 1; }
```

The same applies to anything else you fetch. `curl`ing a script *inside*
the job returns an empty file, and the failure surfaces far downstream —
`rc=127`, every function undefined — which reads like the code is broken
rather than absent. Stage from a login node (which does have network)
onto a shared filesystem, then **assert it arrived**:

```bash
# on a login node, beforehand. ALCF login nodes need the proxy too --
# see "Downloads on ALCF need the proxy" below; without it this curl
# hangs rather than failing fast.
export https_proxy=http://proxy.alcf.anl.gov:3128   # ALCF only
mkdir -p "${HOME}/stage"

# Download to a temp file and move only on success: `curl` can be
# interrupted after writing some bytes, and a PARTIAL file passes a
# non-empty check while failing later with missing functions or a
# syntax error -- worse than an empty one, because it looks staged.
curl -fsSL --retry 3 \
    https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh \
    -o "${HOME}/stage/utils.sh.part" \
  && bash -n "${HOME}/stage/utils.sh.part" \
  && mv "${HOME}/stage/utils.sh.part" "${HOME}/stage/utils.sh" \
  || { echo "FATAL: staging failed"; rm -f "${HOME}/stage/utils.sh.part"; exit 1; }

# in the job:
U="${HOME}/stage/utils.sh"
[[ -s "$U" ]] || { echo "FATAL: ${U} missing/empty — stage it first"; exit 1; }
```

Also note the HF caches: `hf`/`hf_trainer`/`fsdp_tp`/`diffusion` examples
need their datasets and models downloaded ahead of time. **Check the
cache the libraries will actually read**, not a hard-coded path —
`HF_HOME`, `HF_HUB_CACHE` and `HF_DATASETS_CACHE` all override the
`~/.cache/huggingface` default, and the documented Perlmutter setup sets
`HF_HOME="$SCRATCH/.cache/hf"`:

```bash
ls "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
```

A cache miss on a compute node is an offline error, not a code bug.

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

**ezpz does not set this for you, by design.** `ezpz_load_modules_*` and
`ezpz_setup_xpu` preserve the caller's exact set/unset state
(`tests/test_ccl_op_sync_default.sh` enforces it), because collective
semantics belong to the application — matched TorchTitan controls found async
~6x faster there. So set it yourself for FSDP2 + TP>1:

```bash
export CCL_OP_SYNC=1
```

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
