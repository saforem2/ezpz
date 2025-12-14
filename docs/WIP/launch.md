# CLI Reference (WIP)

Essential flags and env vars for `ezpz` CLIs.

## `ezpz-launch`
- Purpose: scheduler-aware launcher for Python commands/modules.
- Usage:
  ```bash
  ezpz-launch -m ezpz.test_dist --train-iters 100
  ezpz-launch -c "'import ezpz; ezpz.setup_torch()'"
  ezpz-launch python3 -m your.module --arg val
  ```
- Key flags:
  - `-m/--module`: run as `python -m <module>`.
  - `-c/--code`: run inline code string.
  - `--np`: override world size (processes).
  - `--ppn`: processes per node (pbs).
  - `--hostfile`: path to hostfile; overrides detection.
  - `--include-python`: force prepend python executable.
  - `--log-level`: override logging level for launcher.
- Env vars:
  - `WORLD_SIZE`, `RANK`, `LOCAL_RANK`: respected if set; otherwise derived.
  - `EZPZ_LOG_LEVEL`: `INFO`/`DEBUG`; controls filter verbosity.
  - `EZPZ_FILTERS`: additional comma-separated log filters.
  - `WANDB_MODE`: `offline` to avoid network.
  - Scheduler vars (`PBS_NODEFILE`, `PBS_JOBID`, `SLURM_*`) are auto-read.
- Behavior:
  - Detects scheduler; builds `mpiexec`/`srun`; falls back to `mpirun`.
  - Auto-injects python executable when not provided.
  - Filters known noisy log lines for Aurora/Sunspot; extendable via env.

## `ezpz-test`
- Purpose: quick smoke test for distributed setup (`ezpz.test.main`).
- Usage:
  ```bash
  ezpz-test --train-iters 20 --backend ddp --device cpu
  ```
- Flags: forwarded to `ezpz.test` (e.g., `--train-iters`, `--backend`, `--device`).
- Env vars: same logging/env conventions as `ezpz-launch`.

## `ezpz-yeet-env`
- Purpose: distribute a prebuilt env tarball to all worker nodes.
- Usage:
  ```bash
  ezpz-yeet-env --src /path/env.tar.gz --dst /tmp/env.tar.gz --d
  ```
- Key flags:
  - `--src`: source tarball path.
  - `--dst`: destination path on workers.
  - `--d/--decompress`: decompress after transfer.
  - `--chunk-size`: bytes per broadcast chunk.
- Behavior:
  - Uses `ezpz.launch` + scheduler detection to fan out to workers.
  - Uses torch distributed broadcast for transfer.

## `ezpz-tar-env`
- Purpose: package current env into a tarball.
- Usage:
  ```bash
  ezpz-tar-env --prefix $CONDA_PREFIX --output /tmp/env.tar.gz
  ```
- Key flags:
  - `--prefix`: env prefix to tar.
  - `--output`: destination tarball path.
  - `--exclude`: optional patterns to skip.

## Troubleshooting Flags
- `--hostfile` + `--np/--ppn`: override detection when scheduler vars are missing.
- `EZPZ_LOG_LEVEL=DEBUG`: show detection and command assembly.
- `WORLD_SIZE=1`: force single-process for debugging.
- `WANDB_MODE=offline`: disable network for logging.
