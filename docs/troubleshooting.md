# 🔧 Troubleshooting

Common issues and fixes for `ezpz`. Most problems fall into one of the
categories below. Each table lists the symptom, likely cause, and
recommended fix.

## Common Issues

### NCCL / CCL Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NCCL timeout` | Network config or firewall | Set `NCCL_DEBUG=INFO` for details. `NCCL_SOCKET_IFNAME` controls which network interface NCCL uses — check available interfaces with `ip link show` and try e.g. `NCCL_SOCKET_IFNAME=eth0` |
| `CCL: ... error` | Version mismatch | Ensure `oneccl_bindings_pt` matches PyTorch version |
| `NCCL error: unhandled system error` | Driver/NCCL mismatch | Update GPU drivers; verify `nvidia-smi` works |

### Device Not Found

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No CUDA/XPU device` | Missing drivers or modules | Load GPU modules (`module load cuda`); check `nvidia-smi` / `xpu-smi` |
| Wrong device selected | Env override | Set `TORCH_DEVICE=cuda` (or `xpu`, `cpu`) explicitly |
| `RuntimeError: CUDA out of memory` | GPU memory exhausted | Try in order: (1) reduce batch size, (2) enable gradient accumulation, (3) switch to FSDP via `wrap_model(model, use_fsdp=True)` |

### MPI Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: mpi4py` | Package not installed | `pip install mpi4py` (needs MPI headers) |
| `mpiexec: command not found` | MPI not on PATH | Load MPI module or install OpenMPI/MPICH |
| Hangs on init | Firewall / network | ezpz auto-detects `MASTER_ADDR`/`MASTER_PORT` from the scheduler. Verify with `echo $MASTER_ADDR` and ensure the address is reachable from all nodes. For non-standard networks, set them manually |

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

### Distributed Training Hangs / Deadlocks

Training freezes with no error output are usually caused by a
communication problem between ranks.

| Symptom | Cause | Fix |
|---------|-------|-----|
| All ranks freeze during init | Master address unreachable | Verify hostfile, check `MASTER_ADDR` is reachable from all nodes |
| Hangs during `backward()` or `all_reduce()` | One rank crashed silently | Set `NCCL_DEBUG=INFO` to see communication logs; check all ranks are alive |
| Freezes after several steps | Network timeout | Increase timeout: `TORCH_DDP_TIMEOUT=7200` |
| Only hangs at scale (>1 node) | Mismatched world_size or hostfile | Verify `PBS_NODEFILE` / `SLURM_NODELIST` matches expected node count |
| Intermittent hangs | Firewall or NIC misconfiguration | Set `NCCL_SOCKET_IFNAME` to the correct interface (check with `ip link show`) |

**Debugging steps:**

```bash
# 1. Enable NCCL debug logging
NCCL_DEBUG=INFO ezpz launch -- python3 train.py 2>&1 | tee nccl_debug.log

# 2. Log from all ranks to find which one is stuck
LOG_FROM_ALL_RANKS=1 ezpz launch --line-buffer -- python3 train.py

# 3. Increase the timeout to rule out slow initialization
TORCH_DDP_TIMEOUT=7200 ezpz launch -- python3 train.py
```

### FSDP-Specific Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during forward pass | All-gather materializes full parameters | Use `reshard_after_forward=True` (ZeRO-3) or reduce batch size |
| `FSDP not supported on mps` | Apple MPS doesn't support FSDP | Expected — ezpz auto-falls back to DDP; or use `use_fsdp=False` |
| Checkpoint saved with FSDP won't load in DDP | State dict format mismatch | Save a portable checkpoint (see below) |
| `RuntimeError: ... DeviceMesh` | PyTorch version mismatch | FSDP2 requires PyTorch 2.4+; check `python -c "import torch; print(torch.__version__)"` |

??? example "Saving portable FSDP checkpoints"

    !!! note "FSDP1 API"

        The example below uses the **FSDP1** API (`torch.distributed.fsdp`).
        If you are using **FSDP2** (PyTorch 2.4+, `torch.distributed.fsdp2`),
        use `torch.distributed.checkpoint` instead — see the
        [PyTorch docs](https://pytorch.org/docs/stable/distributed.checkpoint.html).

    FSDP checkpoints are sharded by default and can't be loaded outside FSDP.
    To save a checkpoint that works with both FSDP and DDP:

    ```python
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state = model.state_dict()
        if ezpz.get_rank() == 0:
            torch.save(state, "checkpoint.pt")
    ```

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

## Debugging Workflow

When something goes wrong, work through these steps in order:

!!! tip "Step-by-step"

    1. **Run `ezpz doctor`** — checks GPU availability, MPI configuration,
       environment variables, and installed package versions. Fix anything it
       flags before continuing.

    2. **Set `NCCL_DEBUG=INFO`** and reproduce the issue. This shows connection
       setup, transport selection, and basic error context:

        ```bash
        NCCL_DEBUG=INFO ezpz launch python3 -m your_app.train
        ```

    3. **Escalate to `NCCL_DEBUG=TRACE`** if INFO didn't reveal the problem.
       TRACE logs every collective call with timing — produces a lot of output,
       so redirect to a file:

        ```bash
        NCCL_DEBUG=TRACE ezpz launch python3 -m your_app.train 2>&1 | tee nccl_debug.log
        ```

    4. **Set `EZPZ_LOG_LEVEL=DEBUG`** for ezpz-internal decisions (device
       selection, backend choice, hostfile resolution):

        ```bash
        EZPZ_LOG_LEVEL=DEBUG ezpz launch python3 -m your_app.train
        ```

    5. **Enable all-rank logging** to see output from every process, not just
       rank 0:

        ```bash
        LOG_FROM_ALL_RANKS=1 ezpz launch --line-buffer python3 -m your_app.train
        ```
