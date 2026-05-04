# ezpz TODO

Tracked issues, improvements, and potential enhancements identified from a
full codebase review.

---

## 1. Architecture: `dist.py` vs `distributed.py` Duality [HIGH]

**Status**: DONE

Two modules doing the same thing:

- `dist.py` (~1800 lines): Original, with module-level side effects, eager
  imports, monolithic functions.
- `distributed.py` (~1400 lines): Clean rewrite with lazy imports, no
  side effects, clear `__all__`.

`__init__.py` lazy-loads `distributed` (not `dist`), but `dist.py` still
gets pulled in transitively by `train.py`, `history.py`, and others.

**Resolution**: `dist.py` is now a thin re-export shim (~420 lines, down
from ~2870). It re-exports all public symbols from `distributed.py` and
provides lightweight stubs for the ~26 legacy functions that only existed
in the old `dist.py` (none were imported by the main library). A
`DeprecationWarning` is emitted on `import ezpz.dist`. Next step: delete
it entirely once downstream callers have migrated.

---

## 2. Module-Level Side Effects [HIGH]

Several modules execute significant work at import time:

- `__init__.py:18-19`: Hardcoded `socket.getfqdn().startswith("x3")` check
  eagerly imports `mpi4py` and `torch`. Brittle hostname coupling.
- `jobs.py:18-34`: Calls `get_rank()` and `get_scheduler()` at module scope,
  freezing values at import time.
- `train.py:43-50`: Same pattern (`RANK = get_rank()`, `DEVICE = ...` at
  module scope).
- `dist.py:57-68`: Eagerly probes wandb API key at import time.
- `dist.py:84`: `os.environ["COLORTERM"] = "truecolor"` set unconditionally.

**Resolution**: Move to function-local or lazily-computed properties.

---

## 3. Security Concerns [MEDIUM]

- `utils/__init__.py:596-598`: `os.system(f"tar -cvf {output_filename} ...")`
  -- shell injection risk. Use `subprocess.run(["tar", ...])`.
- `pbs.py:475`: `os.system(jobenv["LAUNCH_CMD"])` -- executes arbitrary string
  through the shell.
- `configs.py:464-466`: `subprocess.check_output(..., shell=True)` --
  unnecessary for hardcoded commands.
- `launch.py:51-60`: `run_bash_command` uses `shell=True`.

---

## 4. `jobs.py` Is PBS-Only [MEDIUM]

Almost every function raises or asserts if scheduler is not PBS:

- `check_scheduler()` (line 58): raises `TypeError`
- `get_jobdir_from_env()` (line 64): calls `get_pbs_env()` unconditionally
- `get_jobenv()` (line 153): raises `ValueError`
- `get_jobid()` (line 83): hardcodes `PBS_JOBID`

SLURM has partial support in `slurm.py` and `launch.py`, but `jobs.py` is
a dead end for SLURM users.

**Resolution**: Add SLURM support or clearly document PBS-only scope.

---

## 5. `history.py` Complexity [MEDIUM]

~2600 lines. The `History` class mixes metric accumulation, plotting
(plotext, matplotlib, wandb), dataset serialization (xarray, h5py, netcdf),
timing, and logging.

- Unbounded memory growth (no windowing or flush-to-disk).
- `_tplot_metric_group()` is 335 lines.
- Duplicate section header bug at lines 420-422.
- `sns.kdeplot(..., shade=True)` uses deprecated seaborn API (line 1642).

**Resolution**: Consider splitting metric accumulation from
plotting/serialization.

---

## 6. Heavy Dependencies [MEDIUM]

For a library named "ezpz", the hard dependency list is heavy:

- `transformers==4.50.1`: Pinned exact version, pulls hundreds of transitive
  deps. Only used in example scripts.
- `evaluate`: Only used for HF training examples.
- `seaborn`: Only used for plotting utilities.
- `sh`: Only used in `pbs.py` and `slurm.py` for `qstat`/`sacct`.

**Resolution**: Move these to optional dependency groups.

---

## 7. Duplicated Code [MEDIUM]

- `write_deepspeed_zero12_auto_config` and `write_deepspeed_zero3_auto_config`
  in `utils/__init__.py` (lines 984-1129): ~140 lines of near-identical code.
- `cmd_exists` and `command_exists` both in `configs.py` (lines 89, 448).
- `DistributedPdb` and `ForkedPdb` in `utils/__init__.py`: identical classes.
- `_normalize_cpu_bind_value` in both `launch.py:127` and `pbs.py:231`.
- `_expand_slurm_nodelist` duplicated between `doctor.py` and `distributed.py`.

---

## 8. `utils/__init__.py` Is a Grab-Bag [LOW]

~1130 lines mixing Color/ANSI, debuggers, timestamps, model summaries,
GPU peak FLOPS, tensor conversion, tarball creation, xarray/h5py I/O,
and DeepSpeed config generation (500+ lines).

**Resolution**: Split DeepSpeed config helpers into `ezpz.deepspeed` module.

---

## 9. Concrete Bugs [HIGH]

- `utils/__init__.py`: Missing `import subprocess` -- `get_peak_flops()`
  will raise `NameError` at runtime.
- `dist.py:55`: `WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)`
  returns a string, so `not WANDB_DISABLED` is always `False` for any
  non-empty string. Wandb can never be disabled through this env var.
- `dist.py:103-104`: `ALREADY_PRINTED_DIST_SETUP` and
  `ALREADY_PRINTED_HOSTS` assigned but never referenced.
- `dist.py:1765-1786`: `barrier()` uses two `if` blocks (not `if/elif`),
  so both MPI and torch barriers can execute when `implementation is None`.
- `distributed.py:692`: DDP `device_ids` type inconsistency between CUDA
  (string) and XPU (int).

---

## 10. Test Organization [LOW]

38 test files with overlapping scope:

- `test_basic.py`, `test_simple_ezpz.py`, `test_ezpz.py`, `simple_test.py`,
  `comprehensive_test.py`
- `test_config.py` vs `test_configs.py`
- `test.py`, `test_test_dist_module.py`

Some appear to be one-off exploratory scripts rather than structured suites.

---

## 11. Tracker Improvements [MEDIUM]

- **CSV fallback on backend failure**: When a backend (e.g. MLflow) fails
  during `__init__`, automatically add a CSV backend so metrics are always
  captured locally. Currently a backend failure means those metrics are
  only in JSONL and wandb (if enabled).
- **TensorBoard backend**: Add `TensorBoardBackend` using
  `torch.utils.tensorboard.SummaryWriter`. Straightforward implementation
  (~50 lines), widely used, works offline.
- **AMSC MLflow server permissions**: The shared AMSC MLflow server
  intermittently returns 403 on `log-batch` / `log-parameter` endpoints.
  This is a server-side issue (not ezpz code), but worth documenting
  workarounds and investigating with the AMSC admin.

---

## 12. Log System: Auto-strip ANSI for Non-TTY Output [MEDIUM]

The global `ezpz.log` system should auto-detect when stdout is not a
TTY (e.g. redirected to a file, piped through `subprocess.PIPE`) and
strip ANSI escape codes. Currently `NO_COLOR=1` is the manual knob,
but the default should be smart about it. Care needed: users who pipe
through `tee` or `less -R` may want colors preserved. The benchmark
runner already strips ANSI for its log files as a targeted fix.

---

## 13. Potential Enhancements [LOW]

- Configurable timeout with better error messaging for `init_process_group`
  failures / NCCL hangs.
- Extend `ezpz doctor` to verify GPU-to-GPU connectivity.
- Retry logic for `scontrol`/`qstat` (transient failures on loaded login
  nodes).
- `cleanup()` that covers all backends (MPI finalize in all paths).

---

## 14. `bin/utils.sh` Cleanup [MEDIUM]

Audit findings from the yeet-refactor branch. Each item is independent
and should land as its own commit.

- **Dead-code logging wrappers** (`utils.sh:103-116`): `log_info`,
  `log_warn`, `log_error` are defined but never called anywhere in the
  codebase. Only `log_message` is used. Remove the three unused wrappers
  (~14 lines).
- **`EZPZ_LOG_LEVEL` is never honored as a threshold** (`utils.sh:118-138`):
  The variable is read at line 124 only as a fallback default for when
  the caller passes nothing — it's never used to filter output. Setting
  `EZPZ_LOG_LEVEL=ERROR` does not suppress `log_message INFO` calls. Add
  a numeric-rank comparison early in `log_message` so the env var
  actually filters by severity.
- **`ezpz_kill_mpi` overlaps with `ezpz kill`** (`utils.sh:162-167`):
  Once `ezpz kill` is in (see yeet-refactor spec), `ezpz_kill_mpi` is a
  less-precise alternative — it matches `pals|mpi|python` substring
  with no env-var marker. Replace its body with a one-liner that calls
  `ezpz kill --all-nodes` (or deprecate with a warning).
- **`ezpz_check_and_kill_if_running` hardcodes port 29500**
  (`utils.sh:730-738`): That's the default torch DDP rendezvous port,
  but only the default. Should accept an optional port arg:
  `ezpz_check_and_kill_if_running [port]`.
- **One file, 2,931 lines**: Biggest architectural smell. Natural split
  points exist in the function-name prefixes: `_setup_python_*`,
  `_setup_conda_*`, `_save_*_env`, `_get_pbs_*` / `_get_slurm_*`.
  Split into ~5–6 sourced files (`utils-logging.sh`, `utils-pbs.sh`,
  `utils-slurm.sh`, `utils-python.sh`, `utils-conda.sh`, with `utils.sh`
  as the entry point that sources them). Real refactor with regression
  risk — deserves its own spec.
- **Dead zsh branch** (`utils.sh:22-27`): The `# elif [[ ${EZPZ_SHELL_TYPE}
  == "zsh" ]]` block is fully commented out. Either implement zsh
  support or delete the dead branch.
- **Stale snapshot files**: `bin/utils-2025-12-12-093630.sh` and
  `bin/utils-2025-12-17-163108.sh` are timestamped backups of `utils.sh`.
  Git is the source of truth — delete them (or move out of the package
  if they need to be preserved for reference).

---

## 15. `wrap_model` Coverage of ZeRO Stages [LOW]

Today `wrap_model` (in `src/ezpz/distributed.py:747`) covers DDP,
ZeRO-2 (`reshard_after_forward=False`), ZeRO-3 (`True`), and
HYBRID_SHARD (`int`), but not ZeRO-1 (shard optimizer states only).

PyTorch FSDP doesn't have a native ZeRO-1 mode. The closest path is
`torch.distributed.optim.ZeroRedundancyOptimizer` wrapping a
DDP-wrapped model — params and grads stay replicated, optim state
gets sharded.

**Resolution sketch**: add a new value to `FSDP_SHARDING_STRATEGIES`
(e.g. `"shard-optim"`) that takes the DDP path in `wrap_model`, and
expose a thin helper `wrap_optimizer_for_zero1(opt)` that wraps it
with `ZeroRedundancyOptimizer`. Document the trade-off in
`docs/guides/distributed-training.md` (only saves optimizer memory;
params + grads still replicated; useful only when optimizer state is
the bottleneck — e.g. Adam at scale where it's 2× param size).

---

## 16. Explicit DeepSpeed Wrapper [LOW]

`ezpz` already has plumbing for DeepSpeed — `_init_deepspeed`
(`distributed.py:1570`), config builders in `utils/__init__.py:692`
and `841`, and example scripts under `examples/deepspeed/`. What's
missing is a `wrap_model_for_deepspeed(model, optimizer, zero_stage,
...)` sibling to `wrap_model_for_ddp` / `wrap_model_for_fsdp2`.

**Why a sibling, not a `wrap_model(use_deepspeed=True)` flag**:
`deepspeed.initialize(...)` returns `(engine, optimizer, _,
scheduler)` — a fundamentally different API shape than the FSDP path
which returns just the wrapped model. A unifying wrapper would either
lie about the return type or force callers to handle two APIs. Better
to be explicit.

**What to build**:

- `wrap_model_for_deepspeed(model, optimizer, *, zero_stage=2,
  config=None, ...) -> tuple[engine, optimizer, scheduler]`. Accepts
  either an explicit DS config dict or builds one from
  `get_deepspeed_config_json(stage=zero_stage, ...)`.
- Document in `docs/guides/distributed-training.md` next to the FSDP
  section, with a comparison table (FSDP2 vs DeepSpeed: when to pick
  which — DS still wins for ZeRO-Infinity / NVMe offload, MoE expert
  parallelism, and the HF Trainer ecosystem).
- Update the FSDP-vs-FSDP chart in the walkthrough to also show the
  DeepSpeed equivalent of each stage.

---

## 17. `ezpz yeet` Should Handle uv-Managed Python [MEDIUM]

`uv venv` (default mode, no `--copies`) builds a venv whose
`bin/python3` is a symlink into uv's interpreter cache at
`~/.local/share/uv/python/cpython-X.Y.Z-.../`. The stdlib lives in
that cache directory, NOT inside the venv.

After `ezpz yeet`:
- The venv contents land in `/tmp/.venv/`.
- `bin/python3`'s symlink target still points at the home-filesystem
  cache.
- Every `import` of a stdlib module (asyncio, threading, json,
  selectors, etc.) hits `~/.local/share/uv/...` over the network FS.
- At 4k+ ranks, this becomes the same import-storm tax that yeet was
  supposed to eliminate.

We currently document this as a user footgun in `docs/cli/yeet.md`
('Build your venv with `uv venv --copies`'), but the proper fix is
for yeet itself to handle it.

**Resolution sketch**:

1. Detect when the venv's `pyvenv.cfg` `home =` points under
   `~/.local/share/uv/python/`.
2. In that case, also yeet the uv Python directory:
   `~/.local/share/uv/python/cpython-X.Y.Z-.../` →
   `/tmp/cpython-X.Y.Z-.../`.
3. After local copy, rewrite `pyvenv.cfg`'s `home =` line to point
   at the new `/tmp/` location.
4. Re-link `.venv/bin/python3` to `/tmp/cpython-.../bin/python3`.

About 50-80 lines of changes to `_patch_venv_paths_local` plus a
second broadcast pass for the Python directory. Worth its own spec.
A `--include-python` flag on the CLI would let users opt in
explicitly while we shake out edge cases (uv venvs that share an
interpreter across multiple environments, etc.).
