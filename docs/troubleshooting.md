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
| `ImportError: libsycl.so.9: undefined symbol: urDeviceWaitExp` (XPU) | System oneAPI libs shadow the venv's bundled ones | Prepend the venv libs — use `ezpz_activate_venv` (see below) |

#### `libsycl.so` undefined symbol on XPU

When a `torch+xpu` wheel is installed in a venv, it ships its own
`libsycl` / `libur_loader` under `${VIRTUAL_ENV}/lib`. If you load a
system oneAPI module (e.g. via `ezpz_load_modules`) **after** activating
the venv, the module's older copy lands first on `LD_LIBRARY_PATH` and
shadows the wheel's, so `import torch` dies with:

```
ImportError: .../torch/lib/../../../../libsycl.so.9: undefined symbol: urDeviceWaitExp, version LIBUR_LOADER_0.12
```

`LD_LIBRARY_PATH` wins over a library's own `RUNPATH`, so the fix is to
ensure `${VIRTUAL_ENV}/lib` is **first**. Use `ezpz_activate_venv`
(activates the venv and re-prepends its lib dir), and put it **last** in
the setup chain so it runs after module loading:

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils) \
  && ezpz_setup_job && ezpz_load_modules && ezpz_activate_venv
```

See [`ezpz_activate_venv`](notes/shell-environment.md#ezpz_activate_venv-activate-fix-ld_library_path)
for details.

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
| **FSDP2 hangs on FIRST `all_gather_into_tensor`** on Aurora/Sunspot (XPU) | Process group bound to wrong device — fixed in `ezpz>=0.18.4` | See [XPU FSDP2 First-Step Hang](#xpu-fsdp2-first-step-hang) below |

### XPU FSDP2 First-Step Hang

!!! info "Fixed in v0.18.4"

    The two-part fix landed in
    [v0.18.4](https://github.com/saforem2/ezpz/releases/tag/v0.18.4)
    (PR [#149](https://github.com/saforem2/ezpz/pull/149)): `setup_torch`
    now calls `set_device(local_rank)` before `_setup_ddp`, and
    `_setup_ddp` binds `device_id=` for XPU just like CUDA. Upgrade is
    the recommended action.

**Symptom:** A `fully_shard`-wrapped model deadlocks on the very first
`all_gather_into_tensor` call inside `pre_forward`. A `py-spy` dump shows
rank 0 stuck in `sched_yield → ur::level_zero::urEventWait` and every
other local rank in `pthread_cond_wait`.

**Root cause (pre-0.18.4):** `setup_torch` called `init_process_group`
*before* `torch.xpu.set_device(local_rank)`. On XPU the "current device"
at PG-construction time was `xpu:0` on every rank; later `set_device`
switched the current device but the PG stayed bound to `xpu:0`.
`xccl/foreach_all_gather` then routed some ranks' collectives to `xpu:0`
and others to `xpu:LOCAL_RANK` — they never met up.

**Fix:** Upgrade to `ezpz>=0.18.4`. `setup_torch` now calls
`set_device(local_rank)` *before* `_setup_ddp` and binds `device_id=` on
XPU process groups (mirroring CUDA). If you must stay on an older
version, work around it by calling `torch.xpu.set_device(local_rank)`
manually before `setup_torch` — but the proper fix is the upgrade.

**Verifying:** an `ezpz>=0.18.4` run will print
`init_process_group: ... device_id=xpu:N` from `setup_torch`. If you
see `device_id` *absent* from that log line on XPU, you're on the old
behavior.

### Custom DeviceMesh on XPU

!!! info "Helper added in v0.18.4"

    `ezpz.init_device_mesh_safe()` is the drop-in replacement, added in
    [v0.18.4](https://github.com/saforem2/ezpz/releases/tag/v0.18.4)
    (PR [#149](https://github.com/saforem2/ezpz/pull/149)). On versions
    older than this, build the mesh dimensions yourself via
    `torch.distributed.new_group(ranks=...)`.

**Symptom:** Calling `torch.distributed.init_device_mesh(...)` directly
on Aurora/Sunspot raises:

```
RuntimeError: No backend for the parent process group or its backend
              does not support splitting
```

**Root cause:** When `setup_torch` binds the default PG to a device (via
`init_process_group(device_id=...)`, required for FSDP2 — see above),
torch's `DeviceMesh._init_one_process_group` prefers the `split_group`
code path. The current xccl backend reports `supports_splitting=False`
and raises.

**Fix:** Use `ezpz.init_device_mesh_safe(...)` instead — same signature
as torch's, but it round-trips `bound_device_id` around the call so
torch falls back to the `new_group(ranks, ...)` path (which xccl
supports). No-op on CUDA. See the [distributed reference](python/Code-Reference/distributed.md#multi-dimensional-devicemesh-xpu-safe)
for details.

```python
# Before (raises on XPU):
from torch.distributed.device_mesh import init_device_mesh
mesh = init_device_mesh("xpu", (dp, tp), mesh_dim_names=("dp", "tp"))

# After:
import ezpz
mesh = ezpz.init_device_mesh_safe(
    "xpu", (dp, tp), mesh_dim_names=("dp", "tp")
)
```

`ezpz.wrap_model`'s auto-created 1D mesh and `ezpz.examples.fsdp_tp`
already route through `init_device_mesh_safe`, so users on those paths
get the workaround for free.

### Hugging Face Datasets — `ArrowInvalid` / `FileNotFoundError` in multi-rank loading

!!! info "Auto-handled in `get_hf_text_dataset` since v0.18.4"

    The rank-0-first barrier landed in
    [v0.18.4](https://github.com/saforem2/ezpz/releases/tag/v0.18.4)
    (PR [#149](https://github.com/saforem2/ezpz/pull/149)). If you're on
    `ezpz.data.hf.get_hf_text_dataset` you get the fix for free; if
    you're writing your own data loader, use the pattern below.

**Symptom:** Multi-rank job using `datasets.load_dataset` or
`Dataset.map` on a shared filesystem (`/home`, `/lus`) crashes with one
or both of:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/home/.../datasets/.../cache-<hash>.arrow'
   ── arrow_dataset.py: os.chmod(cache_file_name, ...)

pyarrow.lib.ArrowInvalid: Tried reading schema message, was null or length 0
   ── opened a partially-written Arrow stream
```

You may also see N parallel "Tokenizing HF dataset" progress bars, one
per rank.

**Root cause:** Every rank computes the same fingerprint and races to
write the same Arrow cache file. One rank's `os.chmod` lands after
another rank renames the file (`FileNotFoundError`); another rank opens
a stream the writer hasn't finished (`ArrowInvalid`).

**Fix:** Have rank 0 populate the cache first, then release the others
to read it. `ezpz.get_hf_text_dataset` does this automatically as of
`ezpz>=0.18.4` (`_main_process_first()` barrier around both
`load_dataset` and `Dataset.map`).

For a **custom** data loader, use the same pattern with
`torch.distributed.barrier()`:

```python
import torch.distributed as dist

if dist.is_initialized() and dist.get_rank() != 0:
    dist.barrier()  # non-rank-0 waits

dataset = datasets.load_dataset(name, split=split)
tokenized = dataset.map(tokenize_fn, batched=True)

if dist.is_initialized() and dist.get_rank() == 0:
    dist.barrier()  # rank 0 releases everyone
```

!!! note "Shared vs node-local cache"
    The pattern above gates on **global** rank 0, which is correct for
    a shared cache dir (the default on ALCF — `/home`, `/lus`).  If
    you set a **node-local** `HF_DATASETS_CACHE` (e.g. `$TMPDIR`),
    every node needs its own rank-0 to populate its local cache —
    switch the gate to `ezpz.get_local_rank() == 0`.

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
