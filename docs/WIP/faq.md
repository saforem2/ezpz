# FAQ & Troubleshooting (WIP)

## Scheduler detection fails
- Symptom: launcher falls back to `mpirun` when in a job.
- Checks:
  - `echo $PBS_NODEFILE` or `echo $SLURM_NODELIST` — should be set.
  - `ls /var/spool/pbs/aux/$PBS_JOBID.*` — hostfile exists.
- Fixes:
  - Pass `--hostfile /path/to/hostfile` and, if needed, `--np`/`--ppn`.
  - Set `EZPZ_LOG_LEVEL=DEBUG` to see detection steps.

## `mpiexec`/`srun` not found
- Ensure the scheduler module stack is loaded (use `ezpz_setup_env` on Aurora/Sunspot).
- Provide a full `mpiexec`/`srun` path via future `launch_cmd` override (planned); interim: load modules or prepend to `PATH`.

## Torch distributed backend errors (XCCL/NCCL)
- Symptoms: hangs on init or backend not available.
- Checks:
  - Right driver/toolkit modules loaded (oneAPI for XPU, CUDA for NCCL, ROCm for RCCL).
  - Backend matches device type (`--backend xccl` for Intel, `nccl` for CUDA, `gloo` as debug).
- Fixes:
  - For debugging: `--backend gloo` + `WORLD_SIZE=1`.
  - Ensure `MASTER_ADDR`/`MASTER_PORT` are set consistently if overriding.

## Wandb login/network issues
- Set `WANDB_MODE=offline` to avoid network; artifacts remain in `wandb/`.
- Re-enable later with `wandb online` or `WANDB_MODE=online`.

## Outputs and cleanup
- Outputs default to `outputs/ezpz.*` with plots and logs; clean manually when large.
- Wandb offline runs accumulate under `wandb/`; safe to delete if not syncing.

## Large logs and filtering
- Set `EZPZ_LOG_LEVEL=INFO` (default) to reduce noise.
- Add filters: `EZPZ_FILTERS="regex1,regex2"` to suppress known chatter.
- For Aurora/Sunspot, launcher auto-applies a baseline filter set.
