# Systems Matrix (WIP)

Quick reference for supported environments and defaults.

## Matrix

| System     | Scheduler | Launcher  | GPU type         | Hostfile source                   | Notes                                                                |
| ---------- | --------- | --------- | ---------------- | --------------------------------- | -------------------------------------------------------------------- |
| Aurora     | PBS Pro   | `mpiexec` | Intel XPU        | `/var/spool/pbs/aux/$PBS_JOBID.*` | Uses `xccl`/`ccl` backend; module load via `ezpz_setup_env`.         |
| Sunspot    | PBS Pro   | `mpiexec` | Intel XPU        | `/var/spool/pbs/aux/$PBS_JOBID.*` | Similar to Aurora; filters for PBS noise.                            |
| Polaris    | PBS Pro   | `mpiexec` | NVIDIA GPU       | `/var/spool/pbs/aux/$PBS_JOBID.*` | Ensure matching CUDA toolkit and `torch` wheel.                      |
| Frontier   | SLURM     | `srun`    | AMD GPU          | SLURM env (`SLURM_NODELIST`)      | Load ROCm/RCCL modules before launch.                                |
| Perlmutter | SLURM     | `srun`    | NVIDIA GPU       | SLURM env (`SLURM_NODELIST`)      | Ensure matching CUDA toolkit and `torch` wheel.                      |
| Local      | None      | `mpirun`  | CPU/GPU (single) | None                              | Set `WORLD_SIZE`/`--np` for multi-proc; falls back to CPU if no GPU. |

## Overrides & Tips

- Want to use a custom hostfile?
  - Pass `--hostfile /path/to/hostfile` or set `HOSTFILE`.
- Override counts:
  - `-n` (total ranks)
  - `-ppn` (ranks per node)

## Known Failure Modes (preview)

- Scheduler not detected: `EZPZ_LOG_LEVEL=DEBUG`, check
  `PBS_NODEFILE`/`SLURM_NODELIST`; use `--hostfile`.
- `mpiexec`/`srun` not found: module load or use full path in `launch_cmd`
  override (future hook).
- Backend init failures (`xccl`/`nccl`): verify driver/modules, fall back to
  `gloo` for debugging.
- Wandb network issues: set `WANDB_MODE=offline`; sync later if needed.

## Example Launch Commands

- **Aurora (PBS)**:

  ```bash
  source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
  ezpz launch python3 -m ezpz.examples.diffusion
  # or specify hostfile/ppn if needed:
  ezpz launch python3 -n 4 -ppn 2 python3 ezpz.examples.diffusion
  ```

- **Sunspot (PBS)**:

  ```bash
  source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
  ezpz launch python3 -m ezpz.examples.diffusion
  ```

- **Frontier (SLURM)**:

  ```bash
  # inside an allocation: salloc -N2 -t 00:30:00 ...
  source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
  ezpz launch python3 -m ezpz.examples.diffusion
  ```

- **Perlmutter (SLURM)**:

  ```bash
  ezpz launch python3 -m ezpz.examples.diffusion
  ```

- **Local fallback (mpirun)**:

  ```bash
  WORLD_SIZE=2 uv run ezpz launch python3 -m ezpz.examples.diffusion
  ```
