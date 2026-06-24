# ezpz TODO

Tracked issues, improvements, and potential enhancements identified
from a full codebase review. **Last audited: 2026-06-24.**

When updating: include a `Status: DONE` line + commit/PR ref when an
item ships, then prune in the next audit pass.

**Shipped between 2026-06-10 and 2026-06-24** (PRs #166, #167):

- Examples: unified `s/m/l/xl/xxl/xxxl` size ladder across all 5
  example modules targeting 100M → 10B params (#166).
- Examples: `agpt-2b` / `agpt-20b` presets in `fsdp_tp` reproducing
  torchtitan's `agpt_configs` registry exactly (#166).
- Examples: HF model loading via `--model owner/repo` in `fsdp_tp`,
  with auto-tokenizer-default and `--tp 1` coercion (#166).
- Examples: `--activation-checkpoint {none,block,full,selective}`
  flag in `fsdp_tp` with HF-aware delegation to
  `gradient_checkpointing_enable` (#166).
- Examples: `--compile` + `--compile-mode` across all 5 example
  modules; wrapped AFTER FSDP/TP/AC (#166).
- Examples: `_arg_provided` consolidated into
  `ezpz.examples._presets` so `--flag=value` overrides correctly
  beat presets across every example (#166).
- Distributed: per-rank `setup_torch` log gated on `local_rank=0`
  — was 96 lines per launch, now 1 per host (#166).
- Distributed: `get_tensor_parallel_rank` / `_world_size` return
  identity defaults when TP isn't initialised — fixed `--tp=1`
  crash in `fsdp_tp` (#166).
- Library: `ezpz.utils` no longer hard-imports torch/xarray/torchinfo
  at module top → `ezpz launch` works in torchless venvs
  (partially addresses item 2; #166).
- Library: `ezpz.__getattr__` now surfaces underlying ImportErrors
  instead of silently swallowing them (#166).
- Logging: HF `datasets` library's two-channel spam (Python logging
  + internal verbosity router) silenced in `silence_noisy_loggers`
  (#166).
- Models: `hidden_dim` + `rope_theta` added to `ezpz.models.llama
  .ModelArgs`; threaded through `TransformerBlock` and
  `precompute_freqs_cis` call sites including the defensive
  fallback (#166).
- Shell: `utils.sh` `log_message` now writes to stderr instead of
  stdout — fixes `ezpz_setup_uv_venv` producing literal directories
  named after log lines when output is captured via `$(...)`
  (#167).

---

## 1. ~~Architecture: `dist.py` vs `distributed.py` Duality~~ [DONE]

`dist.py` is now a thin re-export shim (~383 lines, down from
~2870). It re-exports all public symbols from `distributed.py` and
provides lightweight stubs for the ~26 legacy functions that only
existed in the old `dist.py`. A `DeprecationWarning` is emitted on
`import ezpz.dist`.

**Next step**: delete `dist.py` entirely once downstream callers
have migrated. (Audit 2026-06-10: still in use as a shim, leave for
now.)

---

## 2. Module-Level Side Effects [HIGH, PARTIAL]

Several modules still execute work at import time:

- `src/ezpz/__init__.py:18-19`: hardcoded `socket.getfqdn().startswith("x3")`
  check eagerly imports `mpi4py` and `torch`. Brittle hostname
  coupling.
- `src/ezpz/jobs.py`: `RANK = get_rank()` at module scope.
- `src/ezpz/train.py`: `RANK = get_rank()`, `DEVICE = get_torch_device()`
  at module scope.

**Fixed** (no longer reproduces): `dist.py` eager wandb probe, the
unconditional `os.environ["COLORTERM"] = "truecolor"`. Also (2026-06,
PR #166) `ezpz.utils.__init__` no longer hard-imports
`torch`/`xarray`/`tqdm`/`torchinfo` at module top — each is wrapped
in `try/except ImportError` with a `None` fallback so the module
loads in torchless venvs. `ezpz.get_timestamp()` and other
torch-free helpers stay usable in environments that only run
`ezpz launch`. `ezpz.__getattr__` now also surfaces the underlying
ImportError when a submodule fails to load, so future regressions
of this shape are debuggable instead of silently raising "no
attribute".

**Resolution**: move remaining computations to function-local or
lazily-computed properties (still applies to `__init__.py:18-19`,
`jobs.py`, `train.py`).

---

## 3. Security: Shell Injection [MEDIUM, PARTIAL]

Still reproduces in current code:

- `src/ezpz/pbs.py:607`: `os.system(jobenv["LAUNCH_CMD"])` — executes
  arbitrary string through the shell. `LAUNCH_CMD` comes from
  per-machine PBS env discovery, but the function is reachable from
  user code via `pbs.launch_from_env()` so input isn't fully
  trusted.
- `src/ezpz/configs.py:478,480`: `subprocess.check_output(git_hash_cmd,
  shell=True)` — unnecessary for hardcoded `git rev-parse` / `git
  branch` calls. Switch to a list argv.
- `src/ezpz/launch.py:247`: `run_bash_command` uses `shell=True`.
  Documented as accepting "a shell command string" so the contract
  is by-design, but worth auditing all call sites to make sure
  none pass user-controlled input.

**Fixed** (no longer reproduces): `utils/__init__.py` no longer
shells out to `tar`.

---

## 4. `jobs.py` Is PBS-Only [MEDIUM, NOT FIXED]

Almost every function still raises or asserts when scheduler is
not PBS:

- `check_scheduler()` (line 37-58): `raise TypeError("{scheduler}
  not yet implemented!")`
- `get_jobenv()` (line 115-153): `raise ValueError("{scheduler} not
  yet implemented!")`
- `get_jobid()` (line 80): hardcodes `PBS_JOBID`.

SLURM has partial support in `slurm.py` and `launch.py`, but
`jobs.py` is a dead end for SLURM users.

**Resolution**: add SLURM support OR clearly document PBS-only
scope and rename to `pbs_jobs.py` to set expectations.

---

## 5. `history.py` Complexity [MEDIUM, MOSTLY NOT FIXED]

~2977 lines (audit 2026-06-10; grew from 2600 originally). The
`History` class still mixes metric accumulation, plotting (plotext,
matplotlib, wandb), dataset serialization (xarray, h5py, netcdf),
timing, and logging.

- Unbounded memory growth (no windowing or flush-to-disk).
- Still 2 `shade=True` calls (deprecated seaborn API).
- `_tplot_metric_group()` and similar mega-functions still present.

**Resolution**: consider splitting metric accumulation from
plotting/serialization. Big refactor; deserves its own spec.

---

## 6. Heavy Hard Dependencies [MEDIUM, NOT FIXED]

For a library named "ezpz", the hard dependency list (per current
`pyproject.toml`) is heavy:

- `transformers>=4.50`: pulls hundreds of transitive deps. Only used
  in example scripts.
- `evaluate`: only used for HF training examples.
- `seaborn`: only used for plotting utilities.
- `sh`: only used in `pbs.py` and `slurm.py` for `qstat`/`sacct`.

**Resolution**: move these to optional dependency groups (`[hf]`,
`[plot]`, `[scheduler]`). Existing `[project.optional-dependencies]`
already has `vit`, `mpi`, `hf`, `torch`, `docs` groups — drop the
above 4 from `dependencies` and add them to appropriate optional
groups.

Care needed: anything in `src/ezpz/` (not just `examples/`) that
imports these would break for users who don't install the extra.
Audit imports first.

---

## 7. Duplicated Code [MEDIUM, PARTIAL]

All of the originally-listed cases still present:

- `write_deepspeed_zero12_auto_config` and
  `write_deepspeed_zero3_auto_config` in `utils/__init__.py`
  (lines 1492, 1566): ~140 lines of near-identical code.
- `cmd_exists` (configs.py:94) and `command_exists` (configs.py:462)
  — two functions, same purpose.
- `DistributedPdb` (utils/__init__.py:119) and `ForkedPdb`
  (utils/__init__.py:155) — identical classes.
- `_normalize_cpu_bind_value` in both `launch.py:344` and
  `pbs.py:367`.
- `_expand_slurm_nodelist` duplicated between
  `distributed.py:1987` and `doctor.py:351`.

Easy individual cleanups; each could be its own ~30-min PR.

**Pattern reference** (since 2026-06, PR #166): `_arg_provided` had
been copy-pasted into all 5 `ezpz/examples/*.py` files and a fix in
one (`fsdp_tp`) silently regressed in the others. Consolidated into
`src/ezpz/examples/_presets.py` with a single `arg_provided`
function imported everywhere. Same playbook applies to the
duplicates above: extract to a leaf module, replace the copies with
imports, add a regression test that pins the cross-file contract.

---

## 8. `utils/__init__.py` Is a Grab-Bag [LOW, NOT FIXED]

1637 lines (audit 2026-06-10) mixing Color/ANSI, debuggers,
timestamps, model summaries, GPU peak FLOPS, tensor conversion,
tarball creation, xarray/h5py I/O, and DeepSpeed config generation
(500+ lines, items §7 above).

**Resolution**: split DeepSpeed config helpers into a new
`ezpz.deepspeed` module (which doesn't exist yet — confirmed via
`ls src/ezpz/deepspeed* → no matches`).

---

## 9. ~~Concrete Bugs~~ [DONE]

All 5 originally-listed bugs verified fixed as of 2026-06-10:

- `utils/__init__.py` `import subprocess` → present at line 11
- `dist.py:55` `WANDB_DISABLED` stringly-typed → no longer in
  `dist.py` (now a shim)
- `dist.py:103-104` dead `ALREADY_PRINTED_*` vars → gone
- `dist.py:1765-1786` `barrier()` double-fire → current
  `distributed.py:688` uses `try/return` to short-circuit on MPI
  success
- `distributed.py:692` DDP `device_ids` type inconsistency →
  current `distributed.py:887` passes `[local_rank]` (list-of-int)
  for both CUDA and XPU

---

## 10. Test Organization [LOW, PARTIAL]

Still present (audit 2026-06-10):

- `test_basic.py`, `test_simple_ezpz.py`, `test_ezpz.py`,
  `comprehensive_test.py` — overlapping scope
- `test_config.py` and `test_configs.py` coexist (different
  content, but the name collision is confusing)
- `test.py`, `test_test_dist_module.py` — odd names

**Fixed**: `simple_test.py` is gone.

**Resolution**: low priority. The PR #158 pytest CI workflow now
runs these every PR, so they're at least gated; consolidating is
cosmetic.

---

## 11. Tracker Improvements [MEDIUM, MIXED]

- **CSV fallback on backend failure**: when a backend (e.g.
  MLflow) fails during `__init__`, automatically add a CSV
  backend so metrics are always captured locally. Currently a
  backend failure means those metrics are only in JSONL and
  wandb (if enabled). NOT IMPLEMENTED.
- **TensorBoard backend**: add `TensorBoardBackend` using
  `torch.utils.tensorboard.SummaryWriter`. Straightforward
  (~50 lines), widely used, works offline. NOT IMPLEMENTED
  (no `TensorBoardBackend` class in `src/ezpz/tracker.py`).
- **AMSC MLflow server permissions**: shared MLflow server
  intermittently returns 403. Server-side, not ezpz code; worth
  documenting workarounds. Status: still happening per recent
  reports.

---

## 12. Log System: Auto-strip ANSI for Non-TTY Output [MEDIUM, PARTIAL]

`NO_COLOR` env honored (`log/__init__.py:220`, `log/config.py:50`,
`log/console.py:127`); `isatty()` checked in `log/console.py:142`.

But the global "log to a redirected file = no ANSI" behavior
isn't end-to-end automatic — some paths still emit colors when
piped. Audit + sweep needed.

---

## 13. Potential Enhancements [LOW]

- Configurable timeout with better error messaging for
  `init_process_group` failures / NCCL hangs.
  **Note**: PR #136 (v0.18.0) shipped `--timeout` for the launcher
  watchdog. The "better init_process_group error messages" sub-item
  is the remaining work.
- Extend `ezpz doctor` to verify GPU-to-GPU connectivity.
- Retry logic for `scontrol`/`qstat` (transient failures on loaded
  login nodes). **Partially done**: PR #142+ added retry for `qstat`
  via `_run_qstat_with_retry`. Same pattern could apply to
  `scontrol`.
- `cleanup()` that covers all backends (MPI finalize in all paths).

---

## 14. `bin/utils.sh` Cleanup [MEDIUM, PARTIAL]

`utils.sh` is now 3250 lines (audit 2026-06-10; grew from ~2931
originally).

**Shipped** (PR #167, 2026-06-12): `log_message` now writes to
**stderr** instead of stdout, so log lines don't get captured into
`$(some_helper)` substitutions. Pre-fix, `ezpz_setup_uv_venv` was
creating literal directories named after the log lines themselves
(e.g. `venvs/sunspot/'[2026-06-12-104735][I][:]   - Found python
root at: python3.12\n'/`). One-line patch + comment block to
prevent regression.

Specific remaining items:

- **Dead-code logging wrappers** (`utils.sh`): `log_info`, `log_warn`,
  `log_error` are defined but never called anywhere in the
  codebase. Only `log_message` is used. Remove the three unused
  wrappers (~14 lines).
- **`EZPZ_LOG_LEVEL` is never honored as a threshold**: the variable
  is read only as a fallback default for when the caller passes
  nothing — it's never used to filter output. Setting
  `EZPZ_LOG_LEVEL=ERROR` does not suppress `log_message INFO`
  calls. Add a numeric-rank comparison early in `log_message`.
- **`ezpz_kill_mpi` overlaps with `ezpz kill`**: once `ezpz kill`
  fully lands, replace `ezpz_kill_mpi`'s body with a one-liner that
  calls `ezpz kill --all-nodes` (or deprecate with a warning).
  Status: `ezpz_kill_mpi` still present at `utils.sh:162`.
- **`ezpz_check_and_kill_if_running` hardcodes port 29500**: that's
  the default torch DDP rendezvous port, but only the default.
  Should accept an optional port arg.
- **One file, 3250 lines**: biggest architectural smell. Natural
  split points exist in the function-name prefixes:
  `_setup_python_*`, `_setup_conda_*`, `_save_*_env`,
  `_get_pbs_*` / `_get_slurm_*`. Split into ~5–6 sourced files.
  Real refactor with regression risk — deserves its own spec.
- **Dead zsh branch**: the `# elif [[ ${EZPZ_SHELL_TYPE} == "zsh" ]]`
  block is commented out. Either implement zsh support or delete.
- **Stale snapshot files**: `bin/utils-2025-12-12-093630.sh` and
  `bin/utils-2025-12-17-163108.sh` are timestamped backups of
  `utils.sh`. Git is the source of truth — delete them.

---

## 15. `wrap_model` Coverage of ZeRO Stages [LOW, NOT FIXED]

Today `wrap_model` (in `src/ezpz/distributed.py:778`) covers DDP,
ZeRO-2 (`reshard_after_forward=False`), ZeRO-3 (`True`), and
HYBRID_SHARD (`int`), but not ZeRO-1 (shard optimizer states only).
No `shard-optim` entry in `FSDP_SHARDING_STRATEGIES`.

PyTorch FSDP doesn't have a native ZeRO-1 mode. Closest path:
`torch.distributed.optim.ZeroRedundancyOptimizer` wrapping a
DDP-wrapped model — params and grads stay replicated, optim state
gets sharded.

**Resolution sketch**: add a new value to `FSDP_SHARDING_STRATEGIES`
(e.g. `"shard-optim"`) that takes the DDP path in `wrap_model`,
expose a thin helper `wrap_optimizer_for_zero1(opt)` that wraps it
with `ZeroRedundancyOptimizer`. Document trade-offs in
`docs/guides/distributed-training.md`.

---

## 16. Explicit DeepSpeed Wrapper [LOW, NOT FIXED]

`ezpz` already has plumbing for DeepSpeed — `_init_deepspeed`
(`distributed.py:1570`-ish), config builders in
`utils/__init__.py:1492` and `1566`, example scripts under
`examples/deepspeed/`. What's missing is a
`wrap_model_for_deepspeed(model, optimizer, zero_stage, ...)`
sibling to `wrap_model_for_ddp` / `wrap_model_for_fsdp2`. Verified
not yet present.

**Why a sibling, not a `wrap_model(use_deepspeed=True)` flag**:
`deepspeed.initialize(...)` returns `(engine, optimizer, _,
scheduler)` — a different API shape than the FSDP path which
returns just the wrapped model. A unifying wrapper would either
lie about the return type or force callers to handle two APIs.

**What to build**:

- `wrap_model_for_deepspeed(model, optimizer, *, zero_stage=2,
  config=None, ...) -> tuple[engine, optimizer, scheduler]`.
  Accepts either an explicit DS config dict or builds one from
  `get_deepspeed_config_json(stage=zero_stage, ...)`.
- Document in `docs/guides/distributed-training.md` next to the
  FSDP section, with a comparison table.
- Update the FSDP-vs-FSDP chart in the walkthrough to show the
  DeepSpeed equivalent of each stage.

---

## 17. `ezpz yeet` Should Handle uv-Managed Python [MEDIUM, NOT FIXED]

`uv venv` (default mode, no `--copies`) builds a venv whose
`bin/python3` is a symlink into uv's interpreter cache at
`~/.local/share/uv/python/cpython-X.Y.Z-.../`. The stdlib lives
in that cache directory, NOT inside the venv.

After `ezpz yeet`:

- The venv contents land in `/tmp/.venv/`.
- `bin/python3`'s symlink target still points at the home-FS cache.
- Every `import` of a stdlib module (asyncio, threading, json,
  selectors, etc.) hits `~/.local/share/uv/...` over the network FS.
- At 4k+ ranks, this becomes the same import-storm tax yeet was
  supposed to eliminate.

Documented as a user footgun in `docs/cli/yeet.md` ("Build your
venv against a Python that exists on every node"), but the proper
fix is for yeet itself to handle uv-managed Python.

**Resolution sketch**:

1. Detect when the venv's `pyvenv.cfg` `home =` points under
   `~/.local/share/uv/python/`.
2. In that case, also yeet the uv Python directory:
   `~/.local/share/uv/python/cpython-X.Y.Z-.../` →
   `/tmp/cpython-X.Y.Z-.../`.
3. After local copy, rewrite `pyvenv.cfg`'s `home =` line to
   point at the new `/tmp/` location.
4. Re-link `.venv/bin/python3` to `/tmp/cpython-.../bin/python3`.

About 50-80 lines of changes to `_patch_venv_paths_local` plus a
second broadcast pass for the Python directory. A `--include-python`
flag on the CLI would let users opt in explicitly while we shake
out edge cases (uv venvs that share an interpreter across multiple
environments).

---

## 18. CI: Fix 5 deselected pytest tests [LOW, NOT FIXED]

PR #158 added a pytest CI workflow that runs on every PR. Five
tests are currently deselected because they fail in vanilla-CI
(ubuntu-latest, CPU torch + torchvision only) for environment
reasons unrelated to test correctness:

| Test | Failure | Proper fix |
|------|---------|------------|
| `tests/test_distributed.py::TestAllReduce::test_mpi_default` | `ModuleNotFoundError: No module named 'mpi4py'` | `pytest.importorskip("mpi4py")` at module top (or just on the test). Currently the test mocks via `fake_comm` fixture but the function-under-test does `from mpi4py import MPI` for the `op = MPI.SUM` default — the mock doesn't cover that path. |
| `tests/test_recipes.py::TestRecipeWandBLogging::test_setup_wandb_with_history` | `ValueError: cannot write NetCDF files because none of the suitable backend libraries (netCDF4, h5netcdf, scipy) are installed` | Either `pytest.importorskip("scipy")` or have `History.finalize()` fall back to JSON when no netCDF backend is available. |
| `tests/test_recipes.py::TestRecipeTrainingLoop::test_full_training_loop` | Same as above (calls `History.finalize()`) | Same — covered by the same fix |
| `tests/test_yeet_env.py::TestRun::test_generic_source_skips_venv_footer` | `assert 'Synced to' in ''` — captured stdout is empty | The test uses `capsys` but `yeet.run()` returns before reaching the `print()` footer (likely an exception swallowed somewhere). Needs investigation. |
| `tests/test_yeet_env.py::TestRun::test_venv_source_uses_venv_footer` | `assert 'To use this environment' in ''` — same root cause | Same |

Until fixed, the workflow `.github/workflows/pytest.yml` deselects
all 5. Remove the `--deselect` flags as each test is fixed.

Low priority because the 988 remaining tests provide real drift /
regression coverage.
