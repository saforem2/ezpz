# 🏃‍♂️‍➡️ Quick Start

Everything you need to get started: install, write a script, launch it, track
metrics, and a complete API cheat sheet with before/after diffs.

For a full walkthrough with real terminal output, see the
[End-to-End Walkthrough](./reference.md).

## 📦 Install

```bash
uv pip install git+https://github.com/saforem2/ezpz
```

??? tip "Don't have `uv`?"

    Install it first (one-liner, no Python needed):

    ```bash linenums='0'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Or use plain `pip`:

    ```bash linenums='0'
    pip install git+https://github.com/saforem2/ezpz
    ```

!!! tip "Editable install for development"

    ```bash linenums='0'
    git clone https://github.com/saforem2/ezpz.git
    cd ezpz
    uv pip install -e .
    ```

??? tip "Try _without installing_ via `uv run`"

    If you already have a Python environment with
    {`torch`, `mpi4py`} installed, you can try `ezpz` without installing
    it:

    ```bash linenums='0'
    # pip install uv first, if needed
    uv run --with "git+https://github.com/saforem2/ezpz" ezpz doctor

    TMPDIR=$(pwd) uv run --with "git+https://github.com/saforem2/ezpz" \
        --python=$(which python3) \
        ezpz test
    ```

??? example "Verify: `ezpz test`"

    After installing, run a quick smoke test to verify distributed
    functionality and device detection:

    ```bash linenums='0'
    ezpz test
    ```

    This trains a simple MLP on MNIST using DDP and reports timing
    metrics. See the
    [W&B Report](https://api.wandb.ai/links/aurora_gpt/q56ai28l)
    for example output.

??? question "[Optional] Shell Environment and Setup"

    1. [**Shell Environment and Setup**](./notes/shell-environment.md):

        - [ezpz/bin/`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):
        Shell script containing a collection of functions that I've accumulated
        over time and found to be useful.
        To use these, we can source the file directly from the command line:

            ```bash linenums='0'
            source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
            ```

        - [`savejobenv`](https://github.com/saforem2/ezpz/blob/main/utils/savejobenv): Shell
        function that will save relevant job-specific environment
        variables to a file for later use.

        - Running from a PBS job: `savejobenv` will automatically
            save the relevant environment variables to a file
            `~/.pbsenv` which can be later sourced via `source
            ~/.pbsenv` from, e.g., another terminal:
            ```bash linenums='0'
            $ qsub -A <ALLOCATION> -q <QUEUE> \
                -l select=2 \
                -l walltime=01:00:00,filesystems=eagle:home \
                -I
            $ source <(curl -fsSL https://bit.ly/ezpz-utils) && savejobenv
            ```

## 🚂 Distributed Training Script

```python title="train.py" linenums='0'
import ezpz
import torch

rank = ezpz.setup_torch()  # auto-detects device + backend, returns global rank

device = ezpz.get_torch_device()

model = torch.nn.Linear(128, 64).to(device)
model = ezpz.wrap_model(model, dtype="bfloat16")  # FSDP by default; use_fsdp=False for DDP
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
device. For help choosing between DDP, FSDP, and FSDP+TP, see the
[Distributed Training Guide](./guides/distributed-training.md#quick-reference).

!!! tip "`ezpz.synchronize()`"

    Use `ezpz.synchronize()` instead of `torch.cuda.synchronize()` to get
    correct timing measurements on any backend (CUDA, XPU, MPS, CPU).

## 🚀 Launch It

```bash linenums='0'
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

    ```bash linenums='0'
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
the [CLI reference](./cli/launch/index.md).

## 📤 Submit It

`ezpz launch` runs inside an existing allocation. To submit a **batch job**
to the scheduler queue, use `ezpz submit`:

```bash
ezpz submit -N 2 -q debug -t 01:00:00 -A <project> \
    -- python3 train.py
```

This auto-generates a PBS or SLURM job script, wraps your command with
`ezpz launch`, and submits it. Preview the generated script first with
`--dry-run`:

```bash
ezpz submit -N 2 -q debug -t 01:00:00 --dry-run -- python3 train.py
```

You can also submit an existing job script directly:

```bash
ezpz submit job.sh
```

See the [CLI reference](./cli/submit.md) for the full option list, or the
[Distributed Training Guide](./guides/distributed-training.md#going-to-production-with-ezpz-submit)
for a complete production workflow walkthrough.

## 🛠️ API Cheat Sheet

Each `ezpz` component can be used independently — pick only what you need.

#### Setup & Distributed Init

```diff linenums='0'
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

```diff linenums='0'
- device = torch.device("cuda")
- model.to("cuda")
- batch = batch.to("cuda")

+ device = ezpz.get_torch_device()   # cuda, xpu, mps, or cpu
+ model.to(device)
+ batch = batch.to(device)
```

#### Model Wrapping

```diff linenums='0'
- from torch.nn.parallel import DistributedDataParallel as DDP
- model = DDP(model, device_ids=[local_rank], output_device=local_rank)

+ model = ezpz.wrap_model(model)                   # FSDP (default)
+ model = ezpz.wrap_model(model, use_fsdp=False)   # DDP
```

#### Training Loop

```diff linenums='0'
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

```python linenums='0'
import ezpz

logger = ezpz.get_logger(__name__)
history = ezpz.History(
    project_name="my-project",   # optional
    backends="wandb",            # or "mlflow", "wandb,mlflow,csv", etc.
)

for step in range(100):
    loss = train_step(...)
    logger.info(history.update({"loss": loss.item()}, step=step))

history.finalize(outdir="./outputs")  # saves dataset + plots
```

`History` automatically computes distributed statistics (min, max, mean, std)
across all ranks — no extra code needed on worker ranks.

!!! tip "What `finalize` produces"

    Calling `history.finalize()` writes a summary dataset and generates
    loss curves and other plots — ready for inspection or inclusion in
    reports. See the [Walkthrough](./reference.md#full-example-with-history)
    for sample output with terminal plots.

For the full History API — distributed aggregation, environment variables,
`StopWatch`, and more — see the [Metric Tracking guide](./history.md).

## 🔗 Next Steps

**Read next →** [Distributed Training Guide](./guides/distributed-training.md):
the progressive tutorial that takes the hello-world above through
to production-grade FSDP + tensor parallelism.

??? note "Other references"

    - **[Recipes](./recipes.md)** — copy-pasteable patterns (data loading,
      checkpointing, gradient accumulation)
    - **[End-to-End Walkthrough](./reference.md)** — full runnable example
      with real terminal output
    - **[Experiment Tracking](./history.md)** — `History` guide:
      distributed stats, multi-backend logging, plots
    - **[Examples](./examples/index.md)** — end-to-end training scripts
      (FSDP, ViT, Diffusion, etc.)
    - **[CLI Reference](./cli/index.md)** — `ezpz launch`, `ezpz submit`,
      and more
    - **[Configuration](./configuration.md)** — environment variables and
      config dataclasses
    - **[Comparison vs. alternatives](./compare.md)** — vs. raw torchrun,
      Accelerate, DeepSpeed
    - **[Architecture](./architecture.md)** — how `ezpz` works under the hood
