# 🧑‍🍳 Recipes

Short, copy-pasteable patterns for common `ezpz` tasks. For a full
walkthrough with progressive examples, see the
[Distributed Training Guide](./guides/distributed-training.md).

## FSDP Training

Set up FSDP with a single flag change from DDP.

=== "`recipe_fsdp.py`"

    ```python
    import torch
    import ezpz

    rank = ezpz.setup_torch()
    model = torch.nn.Linear(32, 16).to(ezpz.get_torch_device())
    model = ezpz.wrap_model(model, use_fsdp=True)  # use_fsdp=False for DDP
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"[rank {rank}] model wrapped, optimizer ready")
    ezpz.cleanup()
    ```

=== "Output (2 CPUs)"

    ```bash
    $ ezpz launch -np 2 -- python3 recipe_fsdp.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/launch] mpirun -np 2 python3 recipe_fsdp.py
    [I][ezpz/launch] Execution started...
    Using [2 / 2] available "cpu" devices !!
    FSDP is not supported on cpu devices; falling back to DDP.
    [rank 0] model wrapped, optimizer ready
    [rank 1] model wrapped, optimizer ready
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 2.97 seconds.
    ```

=== "Output: Polaris 8 [=2x4] GPUs"

    ```bash
    $ ezpz launch -np 8 -- python3 recipe_fsdp.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/pbs] ✅ Using [8/8] GPUs [2 hosts] x [4 GPU/host]
    [I][ezpz/launch] mpiexec --np=8 --ppn=4 python3 recipe_fsdp.py
    [I][ezpz/launch] Execution started...
    Using [8 / 8] available "cuda" devices !!
    [rank 0] model wrapped, optimizer ready
    [rank 1] model wrapped, optimizer ready
    [rank 2] model wrapped, optimizer ready
    [rank 3] model wrapped, optimizer ready
    [rank 4] model wrapped, optimizer ready
    [rank 5] model wrapped, optimizer ready
    [rank 6] model wrapped, optimizer ready
    [rank 7] model wrapped, optimizer ready
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 9.49 seconds.
    ```

## W&B Logging

`History` dispatches metrics to Weights & Biases by default (via the
`EZPZ_TRACKER_BACKENDS` env var, which defaults to `wandb`).

=== "`recipe_wandb.py`"

    ```python
    import ezpz

    rank = ezpz.setup_torch()

    history = ezpz.History(
        project_name="ezpz-wandb-recipe",
        # backends defaults to EZPZ_TRACKER_BACKENDS env var, then "wandb"
    )
    num_steps = 10
    for step in range(num_steps):
        loss_val = 1.0 / (step + 1)
        lr_val = 1e-3
        history.update({"loss": loss_val, "lr": lr_val}, step=step)

    if rank == 0:
        history.finalize(outdir="wandb-recipe-outputs", plot=False)
    ```

=== "Output (2x MPS)"

    ```bash
    $ ezpz launch -np 2 -- python3 recipe_wandb.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/launch] mpirun -np 2 python3 recipe_wandb.py
    [I][ezpz/launch] Execution started...
    Using [2 / 2] available "mps" devices !!
    [I][ezpz/history] Using History with distributed_history=True
    [I][utils] Saving dataset to: ./outputs/dataset_dataset.nc
    [I][ezpz/history] Saving history report to ./outputs/report.md
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 2.76 seconds.
    ```

=== "Output: Polaris 8 [=2x4] GPUs"

    ```bash
    $ ezpz launch -np 8 -- python3 recipe_wandb.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/pbs] ✅ Using [8/8] GPUs [2 hosts] x [4 GPU/host]
    [I][ezpz/launch] mpiexec --np=8 --ppn=4 python3 recipe_wandb.py
    [I][ezpz/launch] Execution started...
    Using [8 / 8] available "cuda" devices !!
    [I][ezpz/history] Using History with distributed_history=True
    [I][utils] Saving dataset to: wandb-recipe-outputs/dataset_dataset.h5
    [I][ezpz/history] Saving history report to wandb-recipe-outputs/report.md
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 9.80 seconds.
    ```

## MLflow Tracking

Same as W&B — just change the backend. Or use both at once.

```bash
# MLflow only
EZPZ_TRACKER_BACKENDS=mlflow ezpz launch -np 2 -- python3 recipe_wandb.py

# Both W&B and MLflow
EZPZ_TRACKER_BACKENDS=wandb,mlflow ezpz launch -np 2 -- python3 recipe_wandb.py
```

No code changes needed — when `backends=` is not passed explicitly,
`History` reads the `EZPZ_TRACKER_BACKENDS` env var automatically
(defaulting to `wandb`). Set your tracking server in
`~/.amsc.env` or the project `.env`:

```bash title="~/.amsc.env"
AMSC_API_KEY=your-api-key
MLFLOW_TRACKING_URI=https://mlflow.american-science-cloud.org
MLFLOW_TRACKING_INSECURE_TLS=true
```

See [Experiment Tracking > MLflow](./history.md#mlflow) for the full setup guide.

## Custom Hostfile

Use `--hostfile` with `ezpz launch` to target specific nodes.

```bash
# hostfile.txt — one hostname per line, with slots
# node01 slots=4
# node02 slots=4

ezpz launch --hostfile hostfile.txt -- python3 train.py
```

## Forcing a Specific Device/Backend

Override auto-detection with `TORCH_DEVICE` and `TORCH_BACKEND` environment variables.

```bash
# Force CPU + gloo (useful for debugging on a GPU node)
TORCH_DEVICE=cpu TORCH_BACKEND=gloo ezpz launch -np 2 -- python3 train.py

# Force XPU + xccl on Intel GPUs
TORCH_DEVICE=xpu TORCH_BACKEND=xccl ezpz launch -np 4 -- python3 train.py
```

## Timing with `ezpz.synchronize()`

Use `ezpz.synchronize()` for correct cross-backend timing that works on CUDA, XPU, MPS, and CPU.

=== "`recipe_timing.py`"

    ```python
    import time
    import torch
    import ezpz

    rank = ezpz.setup_torch()
    model = torch.nn.Linear(32, 16).to(ezpz.get_torch_device())
    batch = torch.randn(8, 32, device=ezpz.get_torch_device())

    ezpz.synchronize()
    t0 = time.perf_counter()

    output = model(batch)
    loss = output.sum()
    loss.backward()

    ezpz.synchronize()
    dt = time.perf_counter() - t0
    print(f"[rank {rank}] step time: {dt:.4f}s")
    ezpz.cleanup()
    ```

=== "Output (2x MPS)"

    ```bash
    $ ezpz launch -np 2 -- python3 recipe_timing.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/launch] mpirun -np 2 python3 recipe_timing.py
    [I][ezpz/launch] Execution started...
    Using [2 / 2] available "mps" devices !!
    [rank 1] step time: 0.2058s
    [rank 0] step time: 0.3189s
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 2.41 seconds.
    ```

=== "Output: Polaris 8 [=2x4] GPUs"

    ```bash
    $ ezpz launch -np 8 -- python3 recipe_timing.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/pbs] ✅ Using [8/8] GPUs [2 hosts] x [4 GPU/host]
    [I][ezpz/launch] mpiexec --np=8 --ppn=4 python3 recipe_timing.py
    [I][ezpz/launch] Execution started...
    Using [8 / 8] available "cuda" devices !!
    [rank 0] step time: 0.0988s
    [rank 1] step time: 0.0893s
    [rank 2] step time: 0.0907s
    [rank 3] step time: 0.0957s
    [rank 4] step time: 0.0866s
    [rank 5] step time: 0.0912s
    [rank 6] step time: 0.0838s
    [rank 7] step time: 0.0910s
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 7.11 seconds.
    ```

## Multi-Node Launch Patterns

Use `-np` (total processes), `-ppn` (processes per node), and `--nhosts` to control placement.

```bash
# 4 GPUs on a single node
ezpz launch -np 4 -- python3 train.py

# 2 nodes, 4 GPUs each (8 total)
ezpz launch -np 8 -ppn 4 --nhosts 2 -- python3 train.py

# Pass extra env vars to workers
ezpz launch -np 8 -ppn 4 -x NCCL_DEBUG=INFO -- python3 train.py
```

## Disabling Distributed History

Set `EZPZ_NO_DISTRIBUTED_HISTORY=1` to skip cross-rank metric aggregation on large runs where the all-gather overhead is noticeable.

```bash
# Disable distributed history aggregation (auto-disabled above 384 ranks)
EZPZ_NO_DISTRIBUTED_HISTORY=1 ezpz launch -np 512 -- python3 train.py
```

=== "`recipe_no_dist_history.py`"

    ```python
    import ezpz

    rank = ezpz.setup_torch()
    history = ezpz.History(distributed_history=False)

    for step in range(5):
        history.update({"loss": 1.0 / (step + 1)}, step=step)

    print(f"[rank {rank}] distributed_history={history.distributed_history}")
    ezpz.cleanup()
    ```

=== "Output (2x MPS)"

    ```bash
    $ ezpz launch -np 2 -- python3 recipe_no_dist_history.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/launch] mpirun -np 2 python3 recipe_no_dist_history.py
    [I][ezpz/launch] Execution started...
    Using [2 / 2] available "mps" devices !!
    [I][ezpz/history] Using History with distributed_history=False
    [rank 0] distributed_history=False
    [rank 1] distributed_history=False
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 2.95 seconds.
    ```

=== "Output: Polaris 8 [=2x4] GPUs"

    ```bash
    $ ezpz launch -np 8 -- python3 recipe_no_dist_history.py
    [I][ezpz/launch] ----[🍋 ezpz.launch][started]----
    [I][ezpz/pbs] ✅ Using [8/8] GPUs [2 hosts] x [4 GPU/host]
    [I][ezpz/launch] mpiexec --np=8 --ppn=4 python3 recipe_no_dist_history.py
    [I][ezpz/launch] Execution started...
    Using [8 / 8] available "cuda" devices !!
    [I][ezpz/history] Using History with distributed_history=False
    [rank 0] distributed_history=False
    [rank 1] distributed_history=False
    [rank 2] distributed_history=False
    [rank 3] distributed_history=False
    [rank 4] distributed_history=False
    [rank 5] distributed_history=False
    [rank 6] distributed_history=False
    [rank 7] distributed_history=False
    [I][ezpz/launch] ----[🍋 ezpz.launch][stop]----
    [I][ezpz/launch] Execution finished with 0.
    [I][ezpz/launch] Executing finished in 8.38 seconds.
    ```

## Distributed Data Loading

Use `DistributedSampler` to shard data across ranks. The key detail:
call `sampler.set_epoch(epoch)` before each epoch so every rank gets a
different shuffle.

```python title="recipe_dataloader.py"
import torch
from torch.utils.data import DataLoader, DistributedSampler
import ezpz

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()

# Any standard torch Dataset works
dataset = torch.utils.data.TensorDataset(
    torch.randn(1000, 32),  # inputs
    torch.randint(0, 10, (1000,)),  # labels
)

sampler = DistributedSampler(dataset) if ezpz.get_world_size() > 1 else None

dataloader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    shuffle=(sampler is None),  # don't shuffle when using sampler
    drop_last=True,  # consistent batch size across ranks
)

model = torch.nn.Linear(32, 10).to(device)
model = ezpz.wrap_model(model)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(3):
    if sampler is not None:
        sampler.set_epoch(epoch)  # re-shuffle each epoch
    for batch_inputs, batch_labels in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        loss = torch.nn.functional.cross_entropy(model(batch_inputs), batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"[rank {rank}] epoch {epoch} done")

ezpz.cleanup()
```

!!! tip "When to skip the sampler"

    For single-process runs (`world_size == 1`), a sampler isn't needed.
    The `if ezpz.get_world_size() > 1` guard keeps the same code working
    on a laptop and a cluster.

## Checkpointing

Save and load model checkpoints that work across DDP and FSDP.

### DDP Checkpointing

With DDP, every rank holds a full copy, so save from rank 0:

```python title="recipe_checkpoint_ddp.py"
import torch
import ezpz

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()

model = torch.nn.Linear(32, 10).to(device)
model = ezpz.wrap_model(model, use_fsdp=False)
optimizer = torch.optim.Adam(model.parameters())

# ... training loop ...

# Save (rank 0 only)
if rank == 0:
    torch.save({
        "model": model.module.state_dict(),  # unwrap DDP
        "optimizer": optimizer.state_dict(),
        "epoch": 10,
    }, "checkpoint.pt")

# Load (all ranks)
torch.distributed.barrier()
ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=True)
model.module.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])

ezpz.cleanup()
```

### FSDP Checkpointing

FSDP shards parameters, so you need to gather the full state dict first:

```python title="recipe_checkpoint_fsdp.py"
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
import ezpz

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()

model = torch.nn.Linear(32, 10).to(device)
model = ezpz.wrap_model(model)  # FSDP by default
optimizer = torch.optim.Adam(model.parameters())

# ... training loop ...

# Save — gather full state dict, then save from rank 0
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state = model.state_dict()
    if rank == 0:
        torch.save(state, "checkpoint_fsdp.pt")

# Load — load into full state dict, then scatter back
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state = torch.load("checkpoint_fsdp.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

ezpz.cleanup()
```

!!! note "FSDP2 (PyTorch 2.4+)"

    If you're using FSDP2, use `torch.distributed.checkpoint` instead
    of `FSDP.state_dict_type()`. See the
    [PyTorch docs](https://pytorch.org/docs/stable/distributed.checkpoint.html).

## Gradient Accumulation

Accumulate gradients over multiple micro-batches before stepping. This
effectively increases batch size without increasing per-GPU memory.

```python title="recipe_grad_accum.py"
import torch
import ezpz

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()

model = torch.nn.Linear(32, 10).to(device)
model = ezpz.wrap_model(model)
optimizer = torch.optim.Adam(model.parameters())

accum_steps = 4  # accumulate over 4 micro-batches
effective_batch_size = 16 * accum_steps  # = 64

for step in range(100):
    optimizer.zero_grad()
    for micro_step in range(accum_steps):
        x = torch.randn(16, 32, device=device)
        loss = model(x).sum() / accum_steps  # scale loss
        loss.backward()  # gradients accumulate
    optimizer.step()
    if step % 10 == 0:
        print(f"[rank {rank}] step {step}, loss={loss.item() * accum_steps:.4f}")

ezpz.cleanup()
```

!!! tip "With FSDP: use `no_sync()`"

    FSDP synchronizes gradients on every `backward()` by default.
    Wrap the accumulation micro-steps in `model.no_sync()` to defer
    the all-reduce until the final micro-step:

    ```python
    from contextlib import nullcontext

    for micro_step in range(accum_steps):
        ctx = model.no_sync() if micro_step < accum_steps - 1 else nullcontext()
        with ctx:
            loss = model(x).sum() / accum_steps
            loss.backward()
    optimizer.step()
    ```

## MFU Tracking

Track Model FLOPS Utilization — what fraction of the hardware's
peak compute your model actually uses.

```python title="recipe_mfu.py"
import time
import torch
import ezpz
from ezpz.flops import estimate_model_flops, compute_mfu

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()

model = torch.nn.Linear(4096, 4096).to(device)
model = ezpz.wrap_model(model)
optimizer = torch.optim.Adam(model.parameters())

# Count FLOPS once before training
model_flops = estimate_model_flops(model, input_shape=(32, 4096))
if rank == 0:
    print(f"Model FLOPS (fwd+bwd): {model_flops:.2e}")

for step in range(100):
    ezpz.synchronize()
    t0 = time.perf_counter()
    x = torch.randn(32, 4096, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    ezpz.synchronize()
    dt = time.perf_counter() - t0

    mfu = compute_mfu(model_flops, dt)
    if step % 10 == 0 and rank == 0:
        print(f"step={step} loss={loss.item():.4f} mfu={mfu:.2f}%")

ezpz.cleanup()
```

`compute_mfu` auto-detects the device and world size. Supported
accelerators: NVIDIA (A100, H100, H200, B200, L40S), AMD (MI250X,
MI300X, MI325X, MI355X), and Intel PVC.

!!! tip "When to use MFU"

    MFU measures compute efficiency, not communication efficiency.
    Low MFU can mean:

    - **Memory-bound model** — the model doesn't have enough compute
      per byte of data movement (e.g. small batch size)
    - **Communication overhead** — gradient all-reduce takes too long
      (try FSDP or reduce world size)
    - **Kernel launch overhead** — too many small ops (try
      `torch.compile`)
