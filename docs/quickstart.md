# 🏃‍♂️ Quick Start

Everything you need to get started: install, write a script, launch it, track
metrics, and a complete API cheat sheet with before/after diffs.

For a complete runnable example with terminal output, see the
[Reference](./reference.md).

## 📦 Install

```bash
uv pip install git+https://github.com/saforem2/ezpz
```

!!! tip "Editable install for development"

    ```bash
    git clone https://github.com/saforem2/ezpz.git
    cd ezpz
    uv pip install -e .
    ```

??? tip "Try _without installing_ via `uv run`"

    If you already have a Python environment with
    {`torch`, `mpi4py`} installed, you can try `ezpz` without installing
    it:

    ```bash
    # pip install uv first, if needed
    uv run --with "git+https://github.com/saforem2/ezpz" ezpz doctor

    TMPDIR=$(pwd) uv run --with "git+https://github.com/saforem2/ezpz" \
        --python=$(which python3) \
        ezpz test
    ```

??? example "Verify: `ezpz test`"

    After installing, run a quick smoke test to verify distributed
    functionality and device detection:

    ```bash
    ezpz test
    ```

    This trains a simple MLP on MNIST using DDP and reports timing
    metrics. See the
    [W&B Report](https://api.wandb.ai/links/aurora_gpt/q56ai28l)
    for example output.

??? question "[Optional] Shell Environment and Setup"

    1. [**Shell Environment and Setup**](./notes/shell-environment.md):

        ??? warning "Deprecation Notice"

            I plan to deprecate `utils.sh` in favor of a uv native approach.
            This shell script was originally developed for personal use, and I
            don't plan to officially support this script in the long term.

        - [ezpz/bin/`utils.sh`](https://github.com/saforem2/ezpz/blob/main/utils/utils.sh):
        Shell script containing a collection of functions that I've accumulated
        over time and found to be useful.
        To use these, we can source the file directly from the command line:

            ```bash
            source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
            ```

        - [`savejobenv`](https://github.com/saforem2/ezpz/blob/main/utils/savejobenv): Shell
        function that will save relevant job-specific environment
        variables to a file for later use.

        - Running from a PBS job: `savejobenv` will automatically
            save the relevant environment variables to a file
            `~/.pbsenv` which can be later sourced via `source
            ~/.pbsenv` from, e.g., another terminal:
            ```bash
            $ qsub -A <ALLOCATION> -q <QUEUE> \
                -l select=2 \
                -l walltime=01:00:00,filesystems=eagle:home \
                -I
            $ source <(curl -fsSL https://bit.ly/ezpz-utils) && savejobenv
            ```

## 🚂 Distributed Training Script

```python title="train.py"
import ezpz
import torch

rank = ezpz.setup_torch()  # auto-detects device + backend, returns global rank

device = ezpz.get_torch_device()

model = torch.nn.Linear(128, 64).to(device)
model = ezpz.wrap_model(
    model,
    use_fsdp=True,  # False will use DDP
    dtype=torch.bfloat16,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(100):
    x = torch.randn(32, 128, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

ezpz.cleanup()
```

That's it — `ezpz` detects the available backend, initializes the process
group, wraps your model in FSDP/DDP, and assigns each rank to the correct
device.

!!! tip "`ezpz.synchronize()`"

    Use `ezpz.synchronize()` instead of `torch.cuda.synchronize()` to get
    correct timing measurements on any backend (CUDA, XPU, MPS, CPU).

## 🚀 Launch It

```bash
ezpz launch python3 train.py
```

This **single command works everywhere** because the launcher detects the
active job scheduler automatically:

| Environment | What `ezpz launch` runs |
|-------------|------------------------|
| PBS job (Polaris, Aurora, Sunspot) | `mpiexec` with hostfile from `$PBS_NODEFILE` |
| SLURM job (Frontier, Perlmutter) | `srun` with SLURM topology |
| No scheduler (laptop / workstation) | `mpirun -np <ngpus>` fallback |


!!! note "Flag aliases"

    `-n`, `-np`, and `--nproc` are all aliases for the same flag (number of
    processes). Similarly, `-ppn` and `--nproc_per_node` are aliases.

??? abstract "Advanced Launcher Examples"

    To pass arguments through to the launcher[^launcher]:

    ```bash
    $ ezpz launch -- python3 -m ezpz.examples.fsdp

    # pass --line-buffer through to mpiexec:
    $ ezpz launch --line-buffer -- python3 \
          -m ezpz.examples.vit --compile --fsdp

    # Create and use a custom hostfile
    $ head -n 2 "${PBS_NODEFILE}" > hostfile0-2
    $ ezpz launch --hostfile hostfile0-2 -- python3 \
        -m ezpz.examples.fsdp_tp

    # use explicit np/ppn/nhosts
    $ ezpz launch \
          -np 4 \
          -ppn 2 \
          --nhosts 2 \
          --hostfile hostfile0-2 \
          -- \
          python3 -m ezpz.examples.diffusion

    # forward the PYTHONPATH environment variable
    $ ezpz launch -x PYTHONPATH=/tmp/.venv/bin:${PYTHONPATH} \
          -- \
          python3 -m ezpz.examples.fsdp
    ```

[^launcher]: This will be `srun` if a Slurm scheduler is detected, `mpirun` /
    `mpiexec` otherwise.

For pass-through launcher flags, custom hostfiles, and advanced usage, see
the [Reference launcher section](./reference.md#scheduler-aware-launcher-ezpz-launch)
and the [CLI reference](./cli/launch/index.md).

## 🛠️ API Cheat Sheet

Each `ezpz` component can be used independently — pick only what you need.

#### Setup & Distributed Init

```diff
- import os, torch.distributed as dist
- dist.init_process_group(backend="nccl", ...)
- rank = int(os.environ["RANK"])
- local_rank = int(os.environ["LOCAL_RANK"])
- world_size = int(os.environ["WORLD_SIZE"])

+ import ezpz
+ rank = ezpz.setup_torch()          # returns global rank
+ local_rank = ezpz.get_local_rank()
+ world_size = ezpz.get_world_size()
```

#### Device Management

```diff
- device = torch.device("cuda")
- model.to("cuda")
- batch = batch.to("cuda")

+ device = ezpz.get_torch_device()   # cuda, xpu, mps, or cpu
+ model.to(device)
+ batch = batch.to(device)
```

#### Model Wrapping

```diff
- from torch.nn.parallel import DistributedDataParallel as DDP
- model = DDP(model, device_ids=[local_rank], output_device=local_rank)

+ model = ezpz.wrap_model(model)                  # DDP (default)
+ model = ezpz.wrap_model(model, use_fsdp=True)   # FSDP
```

#### Training Loop

```diff
  for step, batch in enumerate(dataloader):
-     batch = batch.to("cuda")
+     batch = batch.to(ezpz.get_torch_device())
      t0 = time.perf_counter()
      loss = train_step(...)
-     torch.cuda.synchronize()
+     ezpz.synchronize()
      dt = time.perf_counter() - t0
```

#### Metric Tracking

```python
import ezpz

logger = ezpz.get_logger(__name__)
ezpz.setup_wandb(project_name="my-project")  # optional
history = ezpz.History()

for step in range(100):
    loss = train_step(...)
    logger.info(history.update({"step": step, "loss": loss.item()}))

history.finalize(outdir="./outputs")  # saves dataset + plots
```

For full details on `History`, see the
[Metric Tracking guide](./history.md).

## 📊 Track Metrics

Use the built-in `History` class to record, save, and plot training metrics:

```python
from ezpz.history import History

# Optional: enable W&B logging before creating History
ezpz.setup_wandb(project_name="my-project")

history = History()

for step in range(100):
    # ... training loop ...
    metrics = {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
    # update() returns a summary string suitable for logging
    summary = history.update(metrics)
    logger.info(summary)  # e.g. "loss=0.123456 lr=0.001000"

history.finalize(outdir="./outputs")  # saves dataset + generates plots
```

`History` automatically computes distributed statistics (min, max, mean, std)
across all ranks for every recorded metric — no extra code needed on worker
ranks.

!!! tip "What `finalize` produces"

    Calling `history.finalize()` writes a summary dataset and generates
    loss curves and other plots — ready for inspection or inclusion in
    reports. See the [Reference complete example](./reference.md#complete-example-with-history)
    for sample output with terminal plots.

For the full History API — distributed aggregation, environment variables,
`StopWatch`, and more — see the [Metric Tracking guide](./history.md).

## 🔗 Next Steps

- **[Reference](./reference.md)** — complete runnable example with terminal output
- **[Metric Tracking](./history.md)** — full `History` guide: distributed stats, W&B, plots
- **[Examples](./examples/index.md)** — end-to-end training scripts (FSDP, ViT, Diffusion, etc.)
- **[CLI Reference](./cli/index.md)** — full `ezpz launch` usage and flags
- **[Configuration](./configuration.md)** — environment variables and config dataclasses
- **[Architecture](./architecture.md)** — how `ezpz` works under the hood
