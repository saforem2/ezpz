# Systems Matrix (WIP)

Quick reference for supported environments and defaults.

## Matrix
| System      | Scheduler | Launcher | GPU type        | Hostfile source                    | Notes |
| ----------- | --------- | -------- | --------------- | ---------------------------------- | ----- |
| Aurora      | PBS Pro   | `mpiexec`| Intel XPU       | `/var/spool/pbs/aux/$PBS_JOBID.*`  | Uses `xccl`/`ccl` backend; module load via `ezpz_setup_env`. |
| Sunspot     | PBS Pro   | `mpiexec`| Intel XPU       | `/var/spool/pbs/aux/$PBS_JOBID.*`  | Similar to Aurora; filters for PBS noise. |
| Frontier    | SLURM     | `srun`   | AMD GPU         | SLURM env (`SLURM_NODELIST`)       | Load ROCm/RCCL modules before launch. |
| Perlmutter  | SLURM     | `srun`   | NVIDIA GPU      | SLURM env (`SLURM_NODELIST`)       | Ensure matching CUDA toolkit and `torch` wheel. |
| Local       | None      | `mpirun` | CPU/GPU (single)| None                               | Set `WORLD_SIZE`/`--np` for multi-proc; falls back to CPU if no GPU. |

## Overrides & Tips
- Missing hostfile? Pass `--hostfile /path/to/hostfile` or set `PBS_NODEFILE`.
- Override counts: `--np` (total ranks), `--ppn` (ranks per node) when autodetect is wrong.
- Backend choice: set `--backend` (`ddp`/`xccl`/`nccl`/`gloo`) in workload flags; `dist.py` will default by device type.
- Module loads:
  - Intel (Aurora/Sunspot): load oneAPI stack via `ezpz_setup_env`.
  - NVIDIA (Perlmutter): ensure CUDA driver/toolkit matches `torch` wheel.
  - AMD (Frontier): load ROCm/RCCL modules before launch.

## Known Failure Modes (preview)
- Scheduler not detected: `EZPZ_LOG_LEVEL=DEBUG`, check `PBS_NODEFILE`/`SLURM_NODELIST`; use `--hostfile`.
- `mpiexec`/`srun` not found: module load or use full path in `launch_cmd` override (future hook).
- Backend init failures (`xccl`/`nccl`): verify driver/modules, fall back to `gloo` for debugging.
- Wandb network issues: set `WANDB_MODE=offline`; sync later if needed.

## Example Launch Commands
- **Aurora (PBS)**:
  ```bash
  source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
  uv run ezpz-launch -m ezpz.examples.minimal --train-iters 50
  # or specify hostfile/ppn if needed:
  uv run ezpz-launch --hostfile $PBS_NODEFILE --ppn 12 -m ezpz.examples.minimal --train-iters 50
  ```
- **Sunspot (PBS)**:
  ```bash
  source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
  uv run ezpz-launch -m ezpz.examples.fsdp --epochs 1
  ```
- **Frontier (SLURM)**:
  ```bash
  # inside an allocation: salloc -N2 -t 00:30:00 ...
  uv run ezpz-launch -m ezpz.examples.vit --dataset fake --max_iters 50 --backend nccl
  ```
- **Perlmutter (SLURM)**:
  ```bash
  uv run ezpz-launch -m ezpz.examples.generate --model_name meta-llama/Llama-3.2-1B --dtype bfloat16
  ```
- **Local fallback (mpirun)**:
  ```bash
  WORLD_SIZE=2 uv run ezpz-launch -m ezpz.examples.minimal --train-iters 20
  ```
