# 🧑‍🍳 Recipes

Short, copy-pasteable patterns for common `ezpz` tasks.

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

Pair `setup_wandb` with `History` for automatic metric tracking and logging.

=== "`recipe_wandb.py`"

    ```python
    import ezpz

    rank = ezpz.setup_torch()
    if rank == 0:
        ezpz.setup_wandb(project_name="ezpz-wandb-recipe")

    history = ezpz.History()
    num_steps = 10
    for step in range(num_steps):
        loss_val = 1.0 / (step + 1)
        lr_val = 1e-3
        history.update({"loss": loss_val, "lr": lr_val})

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
        history.update({"loss": 1.0 / (step + 1)})

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
