# 🍋 `ezpz` (Public-Facing Draft)

Write once, run anywhere: launch distributed PyTorch across NVIDIA/AMD/Intel/MPS from a single CLI.

## Why `ezpz`?
- Scheduler-aware launch (`pbs`/`slurm`/local fallback) with minimal flags.
- Automatic device/job discovery and sensible defaults for world size and hostfiles.
- Built-in logging, plotting, and wandb offline-first support.
- Utilities for packaging and distributing environments (`ezpz-tar-env`, `ezpz-yeet-env`).

## Install with `uv`
Requires Python 3.10–3.12 and `uv` installed.
```bash
git clone https://github.com/saforem2/ezpz
cd ezpz
uv venv .venv && source .venv/bin/activate
uv pip install ".[cpu]"      # or ".[cu128]" for CUDA 12.8, ".[xpu]" for Intel GPUs
```
For development:
```bash
uv pip install -e ".[dev]" ".[docs]"
```

## Quickstart
- Local smoke test:
  ```bash
  uv run python -m ezpz.test_dist --train-iters 10 --backend ddp --device cpu
  ```
- Launch via CLI (local fallback to `mpirun`):
  ```bash
  uv run ezpz-launch -m ezpz.test_dist --train-iters 10
  ```
- In a PBS/SLURM allocation:
  ```bash
  source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
  uv run ezpz-launch -m ezpz.test_dist --train-iters 100
  ```

## Feature Highlights
- Scheduler detection + launch command synthesis (mpiexec/srun/mpirun).
- Torch distributed setup with DDP/TP/PP helpers (`dist.py`, `tp/`).
- Logging + history: structured logs, wandb offline/online, plot generation to `outputs/`.
- Environment ops: tar/ship envs across nodes with `ezpz-tar-env` and `ezpz-yeet-env`.
- Integrations: HF trainer hooks and utilities for custom runners.

## Observe and Debug
- `EZPZ_LOG_LEVEL=DEBUG` to inspect scheduler detection and launch commands.
- `WANDB_MODE=offline` to avoid network usage; artifacts stay local.

## Next Steps (WIP)
- See `WIP/quickstart.md` for more detail.
- See `WIP/architecture.md` for component flow and extensibility notes.
- CLI/API tables coming soon for `ezpz-launch`, `ezpz-test`, `ezpz-yeet-env`, and module references.
