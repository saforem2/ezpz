# 🔧 Troubleshooting

Common issues and fixes for `ezpz`. Most problems fall into one of the
categories below. Each table lists the symptom, likely cause, and
recommended fix.

## Common Issues

### NCCL / CCL Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NCCL timeout` | Network config or firewall | Set `NCCL_DEBUG=INFO` for details; check `NCCL_SOCKET_IFNAME` |
| `CCL: ... error` | Version mismatch | Ensure `oneccl_bindings_pt` matches PyTorch version |
| `NCCL error: unhandled system error` | Driver/NCCL mismatch | Update GPU drivers; verify `nvidia-smi` works |

### Device Not Found

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No CUDA/XPU device` | Missing drivers or modules | Load GPU modules (`module load cuda`); check `nvidia-smi` / `xpu-smi` |
| Wrong device selected | Env override | Set `TORCH_DEVICE=cuda` (or `xpu`, `cpu`) explicitly |
| `RuntimeError: CUDA out of memory` | GPU memory exhausted | Reduce batch size; enable gradient accumulation; use FSDP |

### MPI Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: mpi4py` | Package not installed | `pip install mpi4py` (needs MPI headers) |
| `mpiexec: command not found` | MPI not on PATH | Load MPI module or install OpenMPI/MPICH |
| Hangs on init | Firewall / network | Check `MASTER_ADDR` and `MASTER_PORT` env vars |

### Scheduler Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `UNKNOWN` scheduler | No PBS/SLURM env vars | Run inside a job allocation; or set `PBS_JOBID`/`SLURM_JOB_ID` manually for testing |
| Hostfile not found | Scheduler didn't create it | ezpz auto-creates one; check `HOSTFILE` env var |
| Wrong number of ranks | Misconfigured job | Verify `--ntasks` (SLURM) or node count (PBS) matches expectations |

### Weights & Biases Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hangs on `wandb.init()` | No internet on compute nodes | Set `WANDB_MODE=offline` |
| Auth errors | Missing API key | Run `wandb login` or set `WANDB_API_KEY` |

### Import Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: ezpz` | Not installed | `pip install git+https://github.com/saforem2/ezpz` |
| `ImportError: ... .so` | Binary incompatibility | Rebuild from source; match Python/PyTorch versions |

## Diagnostic Tools

Run `ezpz doctor` to automatically check your environment for common
problems:

```bash
ezpz doctor
```

This will verify GPU availability, MPI configuration, environment
variables, and installed package versions.

!!! tip "ALCF-Specific Issues"
    For issues specific to ALCF systems (Polaris, Aurora, Sunspot), see the [FAQ](./notes/faq.md).
