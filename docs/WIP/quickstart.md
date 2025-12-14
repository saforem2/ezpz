# Quickstart with `uv` (WIP)

Fast path to run `ezpz` locally or under a scheduler using `uv`.

## Prerequisites
- Python 3.10–3.12
- `uv` installed (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- GPU stack if applicable (CUDA/ROCm/oneAPI + matching torch build). For Intel XPU, ensure oneAPI modules are loaded; for CUDA, use a CUDA-enabled `torch` wheel.

## 1) Clone and enter
```bash
git clone https://github.com/saforem2/ezpz
cd ezpz
```

## 2) Create env + install with extras
Pick an extra matching your hardware:
- CPU only: `cpu`
- NVIDIA CUDA 12.8: `cu128`
- Intel XPU: `xpu`

```bash
uv venv .venv
source .venv/bin/activate
uv pip install ".[cpu]"        # or ".[cu128]" / ".[xpu]"
```

## 3) Optional: dev tools
```bash
uv pip install -e ".[dev]"        # editors, ruff, mypy, pytest, jupyter
uv pip install ".[docs]"          # mkdocs toolchain
```

## 4) Smoke test locally
```bash
uv run python -m ezpz.test_dist --train-iters 10 --backend ddp --device cpu
```
- Without scheduler metadata, `ezpz-launch` falls back to `mpirun`; set `WORLD_SIZE` if you want >1 process locally.

## 5) Launch via CLI (local fallback)
```bash
uv run ezpz-launch -c "'import ezpz; ezpz.setup_torch()'"
uv run ezpz-launch -m ezpz.test_dist --train-iters 10
```

## 6) Scheduler-backed run (PBS/SLURM)
Inside an allocation:
```bash
source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
uv run ezpz-launch -m ezpz.test_dist --train-iters 100
```
- `ezpz_setup_env` loads modules, activates the right conda+venv stack, and exports `launch` aliases.
- `ezpz-launch` will detect the scheduler (`pbs`/`slurm`), read hostfiles, and build `mpiexec`/`srun` accordingly.

## 7) Observability tips
- Set `EZPZ_LOG_LEVEL=DEBUG` to see scheduler detection and launch command assembly.
- Set `WANDB_MODE=offline` to avoid network use; runs log to `outputs/ezpz.*` and `wandb/`.

## 8) Next steps
- See `WIP/architecture.md` for how components connect.
- For CLI flags and env vars, consult the upcoming CLI tables (WIP).
