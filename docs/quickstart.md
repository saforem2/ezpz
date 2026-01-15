# 🚀 Quickstart

🍋 `ezpz` provides a set of dynamic, light weight utilities that simplify
running experiments with distributed PyTorch.

These can be broken down, roughly into two distinct categories:

1. **Shell Environment and Setup**:

    - [ezpz/bin/`utils.sh`](https://github.com/saforem2/ezpz/blob/main/utils/utils.sh)

          Use via:

          ```bash
          source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
          ```

        ??? details "What's in `utils.sh`?"

              This script contains utilities for automatic:

              - Job scheduler detection with Slurm and PBS
              - Module loading and base Python environment setup
              - Virtual environment creation and activation
                ... _and more_!
              - Check out [🏖️ Shell Environment](./notes/shell-environment.md) for
              additional information.


1. [**Python Library**]:
    1. Launching and running distributed PyTorch code (_from python!_)
    1. Device Management, and running on different
       {`cuda`, `xpu`, `mps`, `cpu`} devices
    1. Experiment Tracking and tools for automatically
       recording, saving and plotting metrics.

---

## 🌐 Write Hardware Agnostic Distributed PyTorch Code

???+ tip inline end "Pick and Choose"

    Each of these components are designed so that you can pick and choose only
    those tools that are useful for you.

    For example, if you're only interested in the automatic device detection,
    all you need is:

    ```python
    import ezpz
    device = ezpz.get_torch_device()
    ```


- **Accelerator detection:** `ezpz.get_torch_device_type()` and
  `ezpz.setup_torch()` normalize CUDA/XPU/MPS/CPU selection.

- **Scheduler smarts:** detects PBS/Slurm automatically;  
  Otherwise falls back to `mpirun` with sensible env forwarding.
  For launcher-only flags/env (e.g., `-x FOO=bar`), place them before `--`;
  everything after `--` is the command to run:

    ```bash
    ezpz launch <launch flags> -- <command to run> <command args>
    ```

    ??? abstract "Examples"

        e.g.:

        ```bash
        ezpz launch -- python3 -m ezpz.examples.fsdp
        ```

        or, specify `-n 8` processes, forward a specific `PYTHONPATH`, and set
        `EZPZ_LOG_LEVEL=DEBUG`:

        ```bash
        ezpz launch -n 8 \
            -x PYTHONPATH=/tmp/.venv/bin:${PYTHONPATH} \
            -x EZPZ_LOG_LEVEL=DEBUG \
            -- \
            python3 -m ezpz.examples.fsdp
        ```
<br>

### 🤝 Using `ezpz` in Your Application

The real usefulness of `ezpz` comes from its usefulness in _other_ applications.


- `ezpz.setup_torch()` replaces manual `torch.distributed` initialization:

    ```diff
    - torch.distributed.init_process_group(backend="nccl", ...)
    + ezpz.setup_torch()
    ```

- `ezpz.get_local_rank()` replaces manual `os.environ["LOCAL_RANK"]`:

    ```diff
    - local_rank = int(os.environ["LOCAL_RANK"])
    + local_rank = ezpz.get_local_rank()
    ```

- `ezpz.get_rank()` replaces manual `os.environ["RANK"]`:

    ```diff
    - rank = int(os.environ["RANK"])
    + rank = ezpz.get_rank()
    ```

- `ezpz.get_world_size()` replaces manual `os.environ["WORLD_SIZE"]`:

    ```diff
    - world_size = int(os.environ["WORLD_SIZE"])
    + world_size = ezpz.get_world_size()
    ```

- `ezpz.get_torch_device()` replaces manual device assignment:

    ```diff
    - device = torch.device(f"cuda")
    + device = ezpz.get_torch_device()
    ```

    ```diff
      model = build_model(...)

    - model.to("cuda")
    + model.to(ezpz.get_torch_device())
    ```

- `ezpz.wrap_model()` replaces manual `DistributedDataParallel` wrapping:

    ```diff
    - model = torch.nn.parallel.DistributedDataParallel(
    -     model,
    -     device_ids=[local_rank],
    -     output_device=local_rank
    - )

    + model = ezpz.wrap_model(use_fsdp=False)
    ```

- `ezpz.synchronize()` replaces manual device synchronization:

    ```diff
      for iter, batch in enumerate(dataloader):
    -     batch = batch.to("cuda")
    +     batch = batch.to(ezpz.get_torch_device())
          t0 = time.perf_counter()
          loss = train_step(...)
    -     torch.cuda.synchronize()
    +     ezpz.synchronize()
          metrics = {
              "dt": time.perf_counter() - t0,
              "loss": loss.item(),
              # ...
          }
    ```

<!--
We can generalize this script to run on any distributed setup with `ezpz` as follows:


```python
import ezpz

_ = setup_torch()
local_rank = ezpz.get_local_rank()

model = build_model(...)
model.to(ezpz.get_device())
model = ezpz.wrap_model(use_fsdp=False)

for iter, batch in enumerate(dataloader):
    batch = batch.to(ezpz.get_device())
    t0 = time.perf_counter()
    loss = train_step(...)
    ezpz.synchronize()
    metrics = {
        "dt": time.perf_counter() - t0,
        "loss": loss.item(),
        # ...
    }
```
-->

<!--

- **Required modifications**:

    ```diff
    # your_app/train.py

    + import ezpz
    + ezpz.setup_torch()

        # optional but useful: get logger to log from only rank 0 by default
    + logger = ezpz.get_logger(__name__)

        model = build_model(...)

    - model.to("cuda")
    + model.to(ezpz.get_torch_device_type())
    ```

    then, we can launch `your_app/train.py` with:

    ```bash
    ezpz launch -n 4 -- python3 -m your_app.train --additional-args ...
    ```

For example, say you have PyTorch code with explicit:

```python
# manual backend + device setup
master_addr = os.environ.get("MASTER_ADDR", "localhost")
# ... manually initialize and broadcast as needed ...
torch.distributed.init_process_group(backend="nccl", ...)

# manual device assignment, etc
model = build_model(...)
model.to("cuda")

# manual device syncs, etc
for step in range(num_steps):
    t0 = time.perf_counter()
    loss = train_step(...)
    torch.cuda.synchronize()
    metrics = {
        "dt": time.perf_counter() - t0,
        "loss": loss.item(),
        # ...
    }
```

-->

## 📈 Track metrics with `ezpz.History`

Capture metrics across all ranks, persist JSONL, generate text/PNG plots, and
(when configured) log to Weights & Biases—no extra code on worker ranks.

```python
import ezpz
from ezpz import History

# single process logging, automatically!
logger = ezpz.get_logger(__name__)

ezpz.setup_torch()
history = History()

for step in range(num_steps):
    t0 = time.perf_counter()
    loss, acc = train_step(...)
    ezpz.synchronize()
    dt = time.perf_counter() - t0

    logger.info(
            history.update(
                {
                    "train/step": step,
                    "train/loss": loss,
                    "train/acc": acc
                }
            )
    )

# Aggregated statistics (mean/min/max/std) are recorded across all MPI ranks,
# and plots + JSONL logs land in outputs/ by default.
if ezpz.get_rank() == 0:
    history.finalize()
```

## Complete Example

```python
import ezpz

logger = ezpz.get_logger(__name__)

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()
model = build_model()
model.to(device)

history = ezpz.History()

for iter, batch in enumerate(dataloader):
    batch = batch.to(device)
    output = model(batch)
    loss = calc_loss(output, batch)
    metrics = calc_metrics(output, batch)
    logger.info(
        history.update(
            {
                "iter": iter,
                "loss": loss.item(),
                **metrics,
            }
        )
    )

if rank == 0:
    history.finalize()

ezpz.cleanup()
```

