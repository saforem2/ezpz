# `ezpz.distributed`

- See [`ezpz/distributed.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/distributed.py)

The core distributed training module. Provides initialization, device/backend
detection, collective operations, and model wrapping for distributed PyTorch
training across any supported hardware.

## Setup

The primary entry point is `setup_torch()`, which bootstraps the entire
distributed environment in a single call:

```python
import ezpz

rank = ezpz.setup_torch(seed=42)
```

### Advanced Parameters

`setup_torch()` accepts keyword-only parameters for multi-dimensional
parallelism:

```python
rank = ezpz.setup_torch(
    seed=42,
    tensor_parallel_size=2,    # Split model across 2 GPUs per TP group
    pipeline_parallel_size=1,  # No pipeline parallelism
    context_parallel_size=1,   # No context parallelism
)
```

| Parameter                    | Default   | Description                                      |
| ---------------------------- | --------- | ------------------------------------------------ |
| `port`                       | `None`    | Rendezvous port (auto-detected if not set)        |
| `seed`                       | `None`    | Random seed for reproducibility                   |
| `timeout`                    | `None`    | DDP init timeout in seconds (env `TORCH_DDP_TIMEOUT`, default 3600) |
| `verbose`                    | `False`   | Enable verbose logging during setup                |
| `tensor_parallel_size`*      | `1`       | Number of ranks per tensor-parallel group          |
| `pipeline_parallel_size`*    | `1`       | Number of pipeline stages                          |
| `context_parallel_size`*     | `1`       | Number of context-parallel ranks                   |
| `tensor_parallel_backend`*   | `None`    | Backend for TP groups (auto-detected)              |
| `pipeline_parallel_backend`* | `None`    | Backend for PP groups (auto-detected)              |
| `context_parallel_backend`*  | `None`    | Backend for CP groups (auto-detected)              |
| `data_parallel_backend`*     | `None`    | Backend for DP groups (auto-detected)              |
| `device_id`*                 | `None`    | Override the per-rank device index. Defaults to `LOCAL_RANK`. When set, this is the device the process group binds to (`init_process_group(device_id=...)`) AND the device `setup_torch` activates before init. On XPU this binding is load-bearing — see [Multi-dimensional DeviceMesh](#multi-dimensional-devicemesh-xpu-safe) for why. |

Parameters marked with `*` are keyword-only.

## Model Wrapping

### High-level API

`wrap_model()` selects the appropriate wrapping strategy based on the arguments:

```python
model = MyModel().to(ezpz.get_torch_device())

# FSDP wrapping (default)
model = ezpz.wrap_model(model, use_fsdp=True, dtype="bf16")

# DDP wrapping
model = ezpz.wrap_model(model, use_fsdp=False)
```

### Explicit DDP

```python
model = ezpz.wrap_model_for_ddp(model)
```

### FSDP (v1)

```python
model = ezpz.wrap_model_for_fsdp(model, dtype="bfloat16")
```

### FSDP2

FSDP2 uses `torch.distributed._composable.fsdp.fully_shard` for per-layer
sharding with optional device mesh support:

```python
model = ezpz.wrap_model_for_fsdp2(
    model,
    dtype="bf16",
    device_mesh=my_mesh,  # optional DeviceMesh for multi-dim parallelism
)
```

## Multi-dimensional DeviceMesh (XPU-safe)

!!! info "Since v0.18.4"

    `ezpz.init_device_mesh_safe()` was added in
    [v0.18.4](https://github.com/saforem2/ezpz/releases/tag/v0.18.4)
    (PR [#149](https://github.com/saforem2/ezpz/pull/149)) — see
    [troubleshooting](../../troubleshooting.md#custom-devicemesh-on-xpu)
    for the underlying xccl `split_group` limitation it works around.

Building a `DeviceMesh` directly on Aurora/Sunspot (xccl) requires a small
workaround — torch's `DeviceMesh._init_one_process_group` prefers the
`split_group` path when the default PG is device-bound, but the current
xccl backend reports `supports_splitting=False` and raises:

```
RuntimeError: No backend for the parent process group or its backend
              does not support splitting
```

`ezpz.init_device_mesh_safe()` is a drop-in for `torch.distributed.init_device_mesh`
that round-trips `bound_device_id` around the call so torch takes the
`new_group(ranks, ...)` fallback (which xccl supports), then restores the
binding so FSDP2's per-device PG resolution still works. No-op on CUDA/NCCL
(which supports `split_group` natively).

```python
import ezpz

# 1D mesh
mesh = ezpz.init_device_mesh_safe("xpu", (world_size,))

# 2D (dp, tp) mesh — see ezpz.examples.fsdp_tp
mesh = ezpz.init_device_mesh_safe(
    str(ezpz.get_torch_device()),
    (dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),
)
```

`ezpz.wrap_model`'s auto-created 1D mesh and `ezpz.examples.fsdp_tp` both
route through this helper, so callers using those paths get the workaround
for free. Reach for `init_device_mesh_safe` directly when you're building
your own mesh (TP, PP, CP, EP, 2D/3D combinations).

## Hostfile Helpers

Functions for managing hostfiles used by MPI launchers:

```python
# Read nodes from an existing hostfile
nodes = ezpz.get_nodes_from_hostfile("/path/to/hostfile")

# Find or create a hostfile with fallback logic
hostfile = ezpz.get_hostfile_with_fallback()

# Write the current hostname to a hostfile
ezpz.write_localhost_to_hostfile("/tmp/hostfile")

# Write a list of hosts to a hostfile
ezpz.write_hostfile_from_list_of_hosts(
    ["node1", "node2", "node3"],
    "/tmp/hostfile"
)
```

## Dtype Map

The `TORCH_DTYPES_MAP` dictionary maps string dtype names to `torch.dtype`
objects:

```python
>>> ezpz.TORCH_DTYPES_MAP["bf16"]
torch.bfloat16
>>> ezpz.TORCH_DTYPES_MAP["fp32"]
torch.float32
```

Supported keys: `"bf16"`, `"bfloat16"`, `"fp16"`, `"float16"`, `"half"`,
`"fp32"`, `"float32"`.

::: ezpz.distributed
