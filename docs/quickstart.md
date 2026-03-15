# 🏃‍♂️ Quick Start

🍋 `ezpz` provides a set of dynamic, light weight utilities that simplify
running experiments with distributed PyTorch.

<!-- 1. [**Python Library**](./python/Code-Reference/index.md): -->

These can be broken down, roughly into a few distinct categories:

1. Launching and running distributed PyTorch code:

    ```bash
    ezpz launch python3 -m ezpz.examples.test  # bash
    ```

    ```python
    >>> import ezpz.examples.test
    >>> ezpz.examples.test.main()  # launch from python!
    ```



1. Device Management, and running on different
    {`cuda`, `xpu`, `mps`, `cpu`} devices
1. Experiment Tracking and tools for automatically
    recording, saving and plotting metrics.

??? warning "[Deprecated] Shell Environment and Setup"

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

            ??? details "What's in `utils.sh`?"

                This script contains utilities for automatic:

                - Job scheduler detection with Slurm and PBS
                - Module loading and base Python environment setup
                - Virtual environment creation and activation
                ... _and more_!
                - Check out [🏖️ Shell Environment](./notes/shell-environment.md) for
                additional information.

---

## 🪄 Using `ezpz`

The real usefulness of `ezpz` comes from its usefulness in _other_
applications.

With `ezpz`, we can write hardware-agnostic distributed PyTorch code that
can be launched easily on different cluster environments.

This allows us to focus on the important parts of our application, without
having to worry about the boilerplate code required to set up distributed
training, device management, and metric tracking.


- **Accelerator detection:** `ezpz.get_torch_device_type()` and
  `ezpz.setup_torch()` normalize CUDA/XPU/MPS/CPU selection.

### 🚀 Scheduler-Aware Launcher: `ezpz launch`

- **Scheduler smarts:** detects PBS / Slurm automatically!  
  `ezpz launch` will, by default, determine the appropriate launcher based on
    the detected job scheduler environment.
    - **Sensible Fallback**: Sensible fallback to `mpirun -np` when running /
      testing locally

- **Flexible resource specification:** `-np`, `-ppn`, `--nhosts`, `--hostfile`,
  etc.  
  Including the ability to pass 

- **Pass-through arguments:** Ability 
  For launcher-only flags/env (e.g., `-x FOO=bar`), place them before `--`;
  everything after `--` is the command to run:

    ```bash
    ezpz launch <launch flags> -- <command to run> <command args>
    ```

    ??? abstract "Launcher Examples"

        To pass arguments through to the launcher[^launcher]

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

### 🛠️ Using `ezpz` Components in Your Application

<!-- ???+ tip "🤝 Use `ezpz` in Your Application" -->

Each of these components are designed so that you can pick and choose only
those tools that are useful for you.

For example, if you're only interested in:

- Automatic device detection:

    ```python
    import ezpz
    device = ezpz.get_torch_device()
    ```

- Metric / Experiment Tracking:

    ```python
    import ezpz

    logger = ezpz.get_logger(__name__)
    # ezpz WandB setup and automatic integration with History
    run = ezpz.setup_wandb(project_name="ezpz-quickstart")
    history = ezpz.History()
    for i in range(10):
        loss = forward_step(...)
        logger.info(history.update({"step": i, "loss": loss}))
    ```

    ??? abstract "Output"

        ```bash
        (.venv)
        #[01/17/26 @ 12:56:36][~/v/s/ezpz][dev][$!?]
        ✦ ; cat << EOF > ~/python/quickstart.py
        import ezpz
        logger = ezpz.get_logger('quickstart')
        history = ezpz.History()
        for i in range(10):
            logger.info(history.update({"step": i}))
        EOF

        python3 ~/python/quickstart.py
        [2026-01-17 12:56:40,604302][I][ezpz/history:214:__init__] Not using distributed metrics! Will only be tracked from a single rank...
        [2026-01-17 12:56:40,606251][I][ezpz/history:220:__init__] Using History with distributed_history=False
        [2026-01-17 12:56:40,607138][I][python/quickstart:5:<module>] step=0
        [2026-01-17 12:56:40,607740][I][python/quickstart:5:<module>] step=1
        [2026-01-17 12:56:40,608065][I][python/quickstart:5:<module>] step=2
        [2026-01-17 12:56:40,608377][I][python/quickstart:5:<module>] step=3
        [2026-01-17 12:56:40,608652][I][python/quickstart:5:<module>] step=4
        [2026-01-17 12:56:40,608926][I][python/quickstart:5:<module>] step=5
        [2026-01-17 12:56:40,609311][I][python/quickstart:5:<module>] step=6
        [2026-01-17 12:56:40,609709][I][python/quickstart:5:<module>] step=7
        [2026-01-17 12:56:40,610036][I][python/quickstart:5:<module>] step=8
        [2026-01-17 12:56:40,610323][I][python/quickstart:5:<module>] step=9
        [2026-01-17-112101] Execution time: 3s sec
        ```


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

## ✅ Complete Example

Capture metrics across all ranks, persist JSONL, generate text/PNG plots, and
(when configured) log to Weights & Biases—no extra code on worker ranks.


```python title="example.py"
import ezpz
import torch

from ezpz.models.minimal import SequentialLinearNet

import time

logger = ezpz.get_logger(__name__)

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()
model = SequentialLinearNet(
    input_dim=16,
    output_dim=32,
    sizes=[4, 8, 12]
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters())

history = ezpz.History()

for i in range(10):
    t0 = time.perf_counter()
    batch = torch.randn(1, 16)
    batch = batch.to(device)
    output = model(batch)
    pred = torch.randn(output.shape)
    loss = ((output - pred.to(device)) ** 2).sum()
    loss.backward()
    optimizer.step()
    logger.info(
        history.update(
            {
                "iter": i,
                "loss": loss,
                "dt": time.perf_counter() - t0,
            }
        )
    )

if rank == 0:
    history.finalize()

ezpz.cleanup()
```

??? info "🪵 Logs"

    ??? success "Single Process"

        Launching in a single process via `python`:

        <div class="ansi-block">
        <pre class="terminal">
        <code>
        <span class='shell'>&gt; </span><span class='cmd'>python3</span> <span class='arg'>example.py</span>
        \[2026-01-15 16:29:59,463919\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1451</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch_distributed</span></i>\]<span style='color:var(--white)'> </span>Using <span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--magenta)'>mps</span> with <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--magenta)'>gloo</span>
        \[2026-01-15 16:29:59,475974\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1316</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch_DDP</span></i>\]<span style='color:var(--white)'> </span>Caught <span style='color:var(--bright-blue)'>MASTER_PORT</span>=<b><span style='color:var(--cyan)'>61496</span></b> from environment!
        \[2026-01-15 16:29:59,477538\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1332</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch_DDP</span></i>\]<span style='color:var(--white)'> </span>Using torch.distributed.init_process_group with
        - <span style='color:var(--bright-blue)'>master_addr</span>=<span style='color:var(--green)'>&#39;Sams-MacBook-Pro-2.local&#39;</span>
        - <span style='color:var(--bright-blue)'>master_port</span>=<span style='color:var(--green)'>&#39;61496&#39;</span>
        - <span style='color:var(--bright-blue)'>world_size</span>=<b><span style='color:var(--cyan)'>1</span></b>
        - <span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>0</span></b>
        - <span style='color:var(--bright-blue)'>local_rank</span>=<b><span style='color:var(--cyan)'>0</span></b>
        - <span style='color:var(--bright-blue)'>timeout</span>=<b><span style='color:var(--magenta)'>datetime.timedelta</span></b>(<span style='color:var(--bright-blue)'>seconds</span>=<b><span style='color:var(--cyan)'>3600</span></b>)
        - <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--green)'>&#39;gloo&#39;</span>
        \[2026-01-15 16:29:59,478263\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>964</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>init_process_group</span></i>\]<span style='color:var(--white)'> </span>Calling torch.distributed.init_process_group_with: <span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>0</span></b> <span style='color:var(--bright-blue)'>world_size</span>=<b><span style='color:var(--cyan)'>1</span></b> <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--magenta)'>gloo</span>
        \[2026-01-15 16:29:59,789459\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1699</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch</span></i>\]<span style='color:var(--white)'> </span>Using <span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--green)'>&#39;mps&#39;</span> with <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--green)'>&#39;gloo&#39;</span> + <span style='color:var(--green)'>&#39;gloo&#39;</span> for distributed training.
        \[2026-01-15 16:29:59,872685\]\[<span style='color:var(--bright-yellow)'>W</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>502</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>print_dist_setup</span></i>\]<span style='color:var(--white)'> </span>Using \[<b><span style='color:var(--cyan)'>1</span></b> <span style='color:var(--magenta)'>/</span> <b><span style='color:var(--cyan)'>1</span></b>\] available <span style='color:var(--green)'>&quot;mps&quot;</span> devices !!
        \[2026-01-15 16:29:59,873382\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1746</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch</span></i>\]<span style='color:var(--white)'> </span>\[<span style='color:var(--green)'>&#39;Sams-MacBook-Pro-2.local&#39;</span>\]\[<span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--green)'>&#39;mps&#39;</span>\]\[<span style='color:var(--bright-blue)'>node</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>0</span></b>\]\[<span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>0</span></b>\]\[<span style='color:var(--bright-blue)'>local_rank</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>0</span></b>\]
        \[2026-01-15 16:30:01,875023\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>214</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>__init__</span></i>\]<span style='color:var(--white)'> </span>Not using distributed metrics! Will only be tracked from a single rank<span style='color:var(--bright-yellow)'>...</span>
        \[2026-01-15 16:30:01,875595\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>220</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>__init__</span></i>\]<span style='color:var(--white)'> </span>Using History with <span style='color:var(--bright-blue)'>distributed_history</span>=<i><span style='color:var(--bright-red)'>False</span></i>
        \[2026-01-15 16:30:02,316946\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>0</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>31.003010</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.435792</span></b>
        \[2026-01-15 16:30:02,330593\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>1</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>57.543598</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.008874</span></b>
        \[2026-01-15 16:30:02,337684\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>2</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>28.547897</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003079</span></b>
        \[2026-01-15 16:30:02,346325\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>3</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>22.243866</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.002852</span></b>
        \[2026-01-15 16:30:02,353276\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>4</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>25.085716</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003102</span></b>
        \[2026-01-15 16:30:02,359662\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>5</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>27.327484</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.002849</span></b>
        \[2026-01-15 16:30:02,364890\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>6</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>19.950121</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003308</span></b>
        \[2026-01-15 16:30:02,371596\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>7</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>36.892731</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.005253</span></b>
        \[2026-01-15 16:30:02,378344\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>8</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>28.500504</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.002372</span></b>
        \[2026-01-15 16:30:02,384270\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>example</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>9</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>33.020760</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.002239</span></b>
        /Users/samforeman/vibes/saforem2/ezpz/src/ezpz/history.py:2223: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
        Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:837.)
        x = torch.Tensor(x).numpy(force=True)
        \[2026-01-15 16:30:02,458225\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>2385</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>finalize</span></i>\]<span style='color:var(--white)'> </span>Saving plots to <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/</span><span style='color:var(--bright-magenta)'>mplot</span> (matplotlib) and <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/</span><span style='color:var(--bright-magenta)'>tplot</span> (tplot)
        \[2026-01-15 16:30:03,822720\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>tplot</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>321</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>tplot</span></i>\]<span style='color:var(--white)'> </span>Using plot type: line
        \[2026-01-15 16:30:03,823148\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>tplot</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>323</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>tplot</span></i>\]<span style='color:var(--white)'> </span>Using plot marker: hd
                                 dt vs iter                       
             ┌─────────────────────────────────────────────────────┐
        0.436┤▌                                                    │
             │▚                                                    │
        0.364┤▝▖                                                   │
             │ ▌                                                   │
             │ ▐                                                   │
        0.291┤  ▌                                                  │
             │  ▚                                                  │
        0.219┤  ▝▖                                                 │
             │   ▚                                                 │
        0.147┤   ▐                                                 │
             │    ▌                                                │
             │    ▐                                                │
        0.074┤    ▝▖                                               │
             │     ▚                                               │
        0.002┤     ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
             └┬─────┬─────┬────┬─────┬─────┬─────┬────┬─────┬──────┘
        0     1     2     3    4     5     6     7    8     9       
        dt                            iter                          
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot/dt.txt</span>
        \[2026-01-15 16:30:03,827907\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>tplot</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>321</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>tplot</span></i>\]<span style='color:var(--white)'> </span>Using plot type: hist
        \[2026-01-15 16:30:03,828187\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>tplot</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>323</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>tplot</span></i>\]<span style='color:var(--white)'> </span>Using plot marker: hd
                                freq vs dt                        
           ┌───────────────────────────────────────────────────────┐
        9.0┤█████                                                  │
           │█████                                                  │
        7.5┤█████                                                  │
           │█████                                                  │
           │█████                                                  │
        6.0┤█████                                                  │
           │█████                                                  │
        4.5┤█████                                                  │
           │█████                                                  │
        3.0┤█████                                                  │
           │█████                                                  │
           │█████                                                  │
        1.5┤█████                                             █████│
           │█████                                             █████│
        0.0┤█████                                             █████│
           └┬─────────────┬────────────┬─────────────┬────────────┬┘
           -0.02         0.10         0.22          0.34        0.46 
        freq                          dt                            
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot/dt-hist.txt</span>
        \[2026-01-15 16:30:03,833010\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>tplot</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>321</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>tplot</span></i>\]<span style='color:var(--white)'> </span>Using plot type: line
        \[2026-01-15 16:30:03,833296\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>tplot</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>323</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>tplot</span></i>\]<span style='color:var(--white)'> </span>Using plot marker: hd
                                loss vs iter                      
            ┌──────────────────────────────────────────────────────┐
        57.5┤     ▗▌                                               │
            │     ▌▐                                               │
        51.3┤    ▐  ▌                                              │
            │   ▗▘  ▐                                              │
            │   ▞    ▌                                             │
        45.0┤  ▗▘    ▝▖                                            │
            │  ▌      ▚                                            │
        38.7┤ ▐       ▝▖                                           │
            │▗▘        ▚                              ▞▄           │
        32.5┤▞         ▝▖                            ▞  ▀▄        ▗│
            │▘          ▚                           ▞     ▀▄  ▗▄▞▀▘│
            │            ▚▖               ▗        ▞        ▀▀▘    │
        26.2┤             ▝▚▄        ▄▄▄▀▀▘▀▄     ▞                │
            │                ▀▄▄▄▄▀▀▀        ▀▄  ▞                 │
        20.0┤                                  ▀▟                  │
            └┬─────┬─────┬─────┬─────┬────┬─────┬─────┬─────┬──────┘
            1     2     3     4     5    6     7     8     9       
        loss                          iter                          
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot/loss.txt</span>
        \[2026-01-15 16:30:03,837141\]\[<span style='color:var(--bright-yellow)'>W</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>2420</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>finalize</span></i>\]<span style='color:var(--white)'> </span>h5py not found! Saving dataset as netCDF instead.
        \[2026-01-15 16:30:03,837503\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>utils</span>/<span style='color:var(--magenta)'>__init__</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>636</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>save_dataset</span></i>\]<span style='color:var(--white)'> </span>Saving dataset to: <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/</span><span style='color:var(--bright-magenta)'>dataset_dataset.nc</span>
        \[2026-01-15 16:30:03,885343\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>2433</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>finalize</span></i>\]<span style='color:var(--white)'> </span>Saving history report to <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/</span><span style='color:var(--bright-magenta)'>report.md</span>
        <span class='shell'>&gt; </span><span class='caret'> </span>
        </pre>
        </code>
        </div>


    ??? success "`ezpz launch`"

        Launching via `ezpz launch` (fallback with 2 processes on MacBookPro):

        <div class="language-bash highlight">
        <pre class="terminal">
        <code>
        <span class='shell'>&gt; </span><span class='cmd'>ezpz</span> <span class='arg'>launch</span> <span class='arg'>python3</span> <span class='arg'>/tmp/test.py</span>
        \[2026-01-15 16:25:45,611138\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>launch</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>515</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>run</span></i>\]<span style='color:var(--white)'> </span>No active scheduler detected; falling back to local mpirun: mpirun -np <b><span style='color:var(--cyan)'>2</span></b> python3 <span style='color:var(--magenta)'>/tmp/</span><span style='color:var(--bright-magenta)'>test.py</span>
        \[2026-01-15 16:25:47,138854\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1451</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch_distributed</span></i>\]<span style='color:var(--white)'> </span>Using <span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--magenta)'>mps</span> with <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--magenta)'>gloo</span>
        \[2026-01-15 16:25:47,149140\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1316</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch_DDP</span></i>\]<span style='color:var(--white)'> </span>Caught <span style='color:var(--bright-blue)'>MASTER_PORT</span>=<b><span style='color:var(--cyan)'>60839</span></b> from environment!
        \[2026-01-15 16:25:47,150476\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1332</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch_DDP</span></i>\]<span style='color:var(--white)'> </span>Using torch.distributed.init_process_group with
        - <span style='color:var(--bright-blue)'>master_addr</span>=<span style='color:var(--green)'>&#39;Sams-MacBook-Pro-2.local&#39;</span>
        - <span style='color:var(--bright-blue)'>master_port</span>=<span style='color:var(--green)'>&#39;60839&#39;</span>
        - <span style='color:var(--bright-blue)'>world_size</span>=<b><span style='color:var(--cyan)'>2</span></b>
        - <span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>0</span></b>
        - <span style='color:var(--bright-blue)'>local_rank</span>=<b><span style='color:var(--cyan)'>0</span></b>
        - <span style='color:var(--bright-blue)'>timeout</span>=<b><span style='color:var(--magenta)'>datetime.timedelta</span></b>(<span style='color:var(--bright-blue)'>seconds</span>=<b><span style='color:var(--cyan)'>3600</span></b>)
        - <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--green)'>&#39;gloo&#39;</span>
        \[2026-01-15 16:25:47,151050\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>964</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>init_process_group</span></i>\]<span style='color:var(--white)'> </span>Calling torch.distributed.init_process_group_with: <span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>0</span></b> <span style='color:var(--bright-blue)'>world_size</span>=<b><span style='color:var(--cyan)'>2</span></b> <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--magenta)'>gloo</span>
        \[2026-01-15 16:25:47,242104\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1699</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch</span></i>\]<span style='color:var(--white)'> </span>Using <span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--green)'>&#39;mps&#39;</span> with <span style='color:var(--bright-blue)'>backend</span>=<span style='color:var(--green)'>&#39;gloo&#39;</span> + <span style='color:var(--green)'>&#39;gloo&#39;</span> for distributed training.
        \[2026-01-15 16:25:47,261869\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1746</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch</span></i>\]<span style='color:var(--white)'> </span>\[<span style='color:var(--green)'>&#39;Sams-MacBook-Pro-2.local&#39;</span>\]\[<span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--green)'>&#39;mps&#39;</span>\]\[<span style='color:var(--bright-blue)'>node</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>0</span></b>\]\[<span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>1</span></b>/<b><span style='color:var(--cyan)'>1</span></b>\]\[<span style='color:var(--bright-blue)'>local_rank</span>=<b><span style='color:var(--cyan)'>1</span></b>/<b><span style='color:var(--cyan)'>1</span></b>\]
        \[2026-01-15 16:25:47,289930\]\[<span style='color:var(--bright-yellow)'>W</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>502</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>print_dist_setup</span></i>\]<span style='color:var(--white)'> </span>Using \[<b><span style='color:var(--cyan)'>2</span></b> <span style='color:var(--magenta)'>/</span> <b><span style='color:var(--cyan)'>2</span></b>\] available <span style='color:var(--green)'>&quot;mps&quot;</span> devices !!
        \[2026-01-15 16:25:47,290348\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>dist</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>1746</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>setup_torch</span></i>\]<span style='color:var(--white)'> </span>\[<span style='color:var(--green)'>&#39;Sams-MacBook-Pro-2.local&#39;</span>\]\[<span style='color:var(--bright-blue)'>device</span>=<span style='color:var(--green)'>&#39;mps&#39;</span>\]\[<span style='color:var(--bright-blue)'>node</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>0</span></b>\]\[<span style='color:var(--bright-blue)'>rank</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>1</span></b>\]\[<span style='color:var(--bright-blue)'>local_rank</span>=<b><span style='color:var(--cyan)'>0</span></b>/<b><span style='color:var(--cyan)'>1</span></b>\]
        \[2026-01-15 16:25:48,882995\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>220</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>__init__</span></i>\]<span style='color:var(--white)'> </span>Using History with <span style='color:var(--bright-blue)'>distributed_history</span>=<i><span style='color:var(--bright-green)'>True</span></i>
        \[2026-01-15 16:25:49,293872\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>0</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>14.438349</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.383613</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>18.930481</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>23.422613</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>14.438349</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>4.492133</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.383651</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.383690</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.383613</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000000</span></b>
        \[2026-01-15 16:25:49,310545\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>1</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>38.289841</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.006327</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>37.768768</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>38.289841</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>37.247700</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.521159</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.006445</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.006563</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.006327</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000118</span></b>
        \[2026-01-15 16:25:49,323389\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>2</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>15.649942</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003752</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>26.894470</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>38.138996</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>15.649942</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>11.244525</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.003934</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.004116</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.003752</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000182</span></b>
        \[2026-01-15 16:25:49,335400\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>3</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>21.518583</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.006340</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>38.892834</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>56.267082</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>21.518583</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>17.374252</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.006604</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.006869</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.006340</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000264</span></b>
        \[2026-01-15 16:25:49,343467\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>4</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>43.398060</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003205</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>41.371902</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>43.398060</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>39.345749</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>2.026196</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.002617</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.003205</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.002029</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000588</span></b>
        \[2026-01-15 16:25:49,351912\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>5</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>43.348061</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.002345</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>39.714069</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>43.348061</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>36.080078</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>3.633997</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.002180</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.002345</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.002014</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000166</span></b>
        \[2026-01-15 16:25:49,360378\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>6</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>40.937546</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003073</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>36.756641</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>40.937546</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>32.575737</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>4.180907</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.002433</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.003073</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.001794</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000640</span></b>
        \[2026-01-15 16:25:49,368605\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>7</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>30.643730</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.002785</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>32.207088</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>33.770447</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>30.643730</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>1.563398</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.002315</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.002785</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.001844</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000470</span></b>
        \[2026-01-15 16:25:49,377235\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>8</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>26.110786</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.003046</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>33.217815</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>40.324844</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>26.110786</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>7.107031</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.002361</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.003046</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.001676</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000685</span></b>
        \[2026-01-15 16:25:49,384409\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>tmp</span>/<span style='color:var(--magenta)'>test</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>30</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>&lt;module&gt;</span></i>\]<span style='color:var(--white)'> </span><span style='color:var(--bright-blue)'>iter</span>=<b><span style='color:var(--cyan)'>9</span></b> <span style='color:var(--bright-blue)'>loss</span>=<b><span style='color:var(--cyan)'>22.861826</span></b> <span style='color:var(--bright-blue)'>dt</span>=<b><span style='color:var(--cyan)'>0.001886</span></b> loss/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>25.471987</span></b> loss/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>28.082148</span></b> loss/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>22.861826</span></b> loss/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>2.610158</span></b> dt/<span style='color:var(--bright-blue)'>mean</span>=<b><span style='color:var(--cyan)'>0.002179</span></b> dt/<span style='color:var(--bright-blue)'>max</span>=<b><span style='color:var(--cyan)'>0.002472</span></b> dt/<span style='color:var(--bright-blue)'>min</span>=<b><span style='color:var(--cyan)'>0.001886</span></b> dt/<span style='color:var(--bright-blue)'>std</span>=<b><span style='color:var(--cyan)'>0.000293</span></b>
        /Users/samforeman/vibes/saforem2/ezpz/src/ezpz/history.py:2223: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
        Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:837.)
        x = torch.Tensor(x).numpy(force=True)
        \[2026-01-15 16:25:49,455888\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>2385</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>finalize</span></i>\]<span style='color:var(--white)'> </span>Saving plots to <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/</span><span style='color:var(--bright-magenta)'>mplot</span> (matplotlib) and <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/</span><span style='color:var(--bright-magenta)'>tplot</span> (tplot)
                            dt                                    dt/min
             ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
        0.384┤▌                                │0.384┤<span style='color:var(--cyan)'>-</span>                                │
        0.320┤▐                                │0.129┤ <span style='color:var(--cyan)'>--------------------------------</span>│
        0.256┤ ▚                               │     └┬───────┬───────┬───────┬───────┬┘
        0.129┤ ▝▖                              │     1.0     3.2     5.5     7.8   10.0 
        0.066┤  ▐                              │dt/min              iter
        0.002┤   ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│                    dt/std
             └┬───────┬───────┬───────┬───────┬┘       ┌───────────────────────────────┐
             1.0     3.2     5.5     7.8   10.0 0.00068┤             <span style='color:var(--magenta)'>\*</span>      <span style='color:var(--magenta)'>\*</span>      <span style='color:var(--magenta)'>\*</span>   │
        dt                  iter                0.00046┤       <span style='color:var(--magenta)'>\*\*\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*</span>   <span style='color:var(--magenta)'>\*</span> <span style='color:var(--magenta)'>\*\*\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*\*</span>│
                        dt/mean                 0.00011┤<span style='color:var(--magenta)'>\*\*\*\*\*\*\*</span>         <span style='color:var(--magenta)'>\*\*\*</span>            │
             ┌─────────────────────────────────┐       └┬───────┬──────┬───────┬──────┬┘
        0.384┤<span style='color:var(--green)'>·</span>                                │       1.0     3.2    5.5     7.8  10.0 
        0.320┤<span style='color:var(--green)'>·</span>                                │dt/std               iter
        0.256┤ <span style='color:var(--green)'>·</span>                               │                   dt/max
        0.193┤  <span style='color:var(--green)'>·</span>                              │     ┌─────────────────────────────────┐
        0.129┤  <span style='color:var(--green)'>·</span>                              │0.384┤<span style='color:var(--red)'>+</span>                                │
        0.066┤   <span style='color:var(--green)'>·</span>                             │0.257┤ <span style='color:var(--red)'>++</span>                              │
        0.002┤    <span style='color:var(--green)'>·····························</span>│0.066┤   <span style='color:var(--red)'>++++++++++++++++++++++++++++++</span>│
             └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
            1.0     3.2     5.5     7.8   10.0      1.0     3.2     5.5     7.8   10.0 
        dt/mean             iter                dt/max              iter                
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt.txt</span>
             ┌─────────────────────────────────────────────────────────────────────────┐
        0.384┤ <span style='color:var(--red)'>++</span> dt/max                                                               │
             │ <span style='color:var(--cyan)'>--</span> dt/min                                                               │
             │ <span style='color:var(--green)'>··</span> dt/mean                                                              │
        0.320┤ ▞▞ dt                                                                   │
             │ ▐                                                                       │
             │  ▌                                                                      │
        0.256┤  ▚                                                                      │
             │  ▝▖                                                                     │
             │   ▌                                                                     │
        0.193┤   ▐                                                                     │
             │    ▌                                                                    │
             │    ▐                                                                    │
             │    ▝▖                                                                   │
        0.129┤     ▚                                                                   │
             │     ▐                                                                   │
             │      ▌                                                                  │
        0.065┤      ▐                                                                  │
             │      ▝▖                                                                 │
             │       ▚                                                                 │
        0.002┤       ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
             └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
             1.0               3.2               5.5               7.8             10.0 
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt_summary.txt</span>
                       dt/mean hist                             dt/max hist             
           ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
        9.0┤████                               │9.0┤████                               │
        7.5┤████                               │7.5┤████                               │
        6.0┤████                               │6.0┤████                               │
        4.5┤████                               │4.5┤████                               │
        3.0┤████                               │3.0┤████                               │
        1.5┤████                           ████│1.5┤████                           ████│
        0.0┤███                            ████│0.0┤███                            ████│
           └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
           -0.01    0.09    0.19     0.30   0.40   -0.01    0.09    0.19     0.30   0.40 
                        dt/min hist                              dt/std hist
           ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
        9.0┤████                               │2.00┤       ███                    ████│
        7.5┤████                               │1.67┤       ███                    ████│
        6.0┤████                               │1.33┤       ███                    ████│
        4.5┤████                               │1.00┤█████████████████   ████   ███████│
           │████                               │    │█████████████████   ████   ███████│
        3.0┤████                               │0.67┤█████████████████   ████   ███████│
        1.5┤████                           ████│0.33┤█████████████████   ████   ███████│
        0.0┤███                            ████│0.00┤█████████████████   ████   ███████│
           └┬────────┬───────┬────────┬───────┬┘    └┬───────┬────────┬───────┬────────┘
           -0.02    0.09    0.19     0.30   0.40   -0.00003 0.00016  0.00034 0.00053     
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt_hist.txt</span>
                            loss                                  loss/min              
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        43.4┤              ▗▀▀▀▀▄▄▄▄           │39.3┤    <span style='color:var(--cyan)'>-</span>          <span style='color:var(--cyan)'>------------</span>       │
        38.6┤   ▟         ▗▘        ▚▖         │22.7┤<span style='color:var(--cyan)'>----</span> <span style='color:var(--cyan)'>----------</span>            <span style='color:var(--cyan)'>-------</span>│
        33.7┤  ▞ ▚       ▗▘          ▝▚▖       │    └┬───────┬────────┬───────┬───────┬┘
        24.1┤ ▐   ▚     ▗▘             ▝▀▚▄▖   │    1.0     3.2      5.5     7.8   10.0 
        19.3┤▗▘    ▚   ▄▘                  ▝▀▀▀│loss/min            iter
        14.4┤▌      ▚▄▀                        │                  loss/std
            └┬───────┬────────┬───────┬───────┬┘    ┌──────────────────────────────────┐
            1.0     3.2      5.5     7.8   10.0 17.4┤           <span style='color:var(--magenta)'>\*</span>                      │
        loss                iter                11.8┤       <span style='color:var(--magenta)'>\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*</span>               <span style='color:var(--magenta)'>\*</span>    │
                          loss/mean              3.3┤<span style='color:var(--magenta)'>\*\*\*\*\*\*\*</span>       <span style='color:var(--magenta)'>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*</span> <span style='color:var(--magenta)'>\*\*\*\*</span>│
            ┌──────────────────────────────────┐    └┬───────┬────────┬───────┬───────┬┘
        41.4┤               <span style='color:var(--green)'>····</span>               │    1.0     3.2      5.5     7.8   10.0 
        37.6┤    <span style='color:var(--green)'>·</span>      <span style='color:var(--green)'>····</span>    <span style='color:var(--green)'>····</span>           │loss/std            iter                
        33.9┤   <span style='color:var(--green)'>·</span> <span style='color:var(--green)'>·</span>    <span style='color:var(--green)'>·</span>            <span style='color:var(--green)'>·······</span>    │                  loss/max              
        30.2┤  <span style='color:var(--green)'>·</span>   <span style='color:var(--green)'>·</span>  <span style='color:var(--green)'>·</span>                    <span style='color:var(--green)'>··</span>  │    ┌──────────────────────────────────┐
        26.4┤ <span style='color:var(--green)'>·</span>     <span style='color:var(--green)'>··</span>                       <span style='color:var(--green)'>··</span>│56.3┤           <span style='color:var(--red)'>+</span>                      │
        22.7┤<span style='color:var(--green)'>·</span>                                 │45.3┤    <span style='color:var(--red)'>+++++++</span> <span style='color:var(--red)'>++++++++++++++++++</span>    │
        18.9┤<span style='color:var(--green)'>·</span>                                 │28.9┤<span style='color:var(--red)'>++++</span>                          <span style='color:var(--red)'>++++</span>│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
            1.0     3.2      5.5     7.8   10.0     1.0     3.2      5.5     7.8   10.0 
        loss/mean           iter                loss/max            iter
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss.txt</span>
            ┌──────────────────────────────────────────────────────────────────────────┐
        56.3┤ <span style='color:var(--red)'>++</span> loss/max            <span style='color:var(--red)'>+</span>                                                 │
            │ <span style='color:var(--cyan)'>--</span> loss/min           <span style='color:var(--red)'>+</span> <span style='color:var(--red)'>+</span>                                                │
            │ <span style='color:var(--green)'>··</span> loss/mean         <span style='color:var(--red)'>+</span>   <span style='color:var(--red)'>+</span>                                               │
        49.3┤ ▞▞ loss             <span style='color:var(--red)'>+</span>     <span style='color:var(--red)'>++</span>                                             │
            │                    <span style='color:var(--red)'>+</span>        <span style='color:var(--red)'>+</span>                                            │
            │                   <span style='color:var(--red)'>+</span>          <span style='color:var(--red)'>+</span>                                           │
        42.3┤                  <span style='color:var(--red)'>+</span>            <span style='color:var(--red)'>+</span>▞▀▀▀▀▀▀▀▀▚▄▄▄▖                            │
            │                 <span style='color:var(--red)'>+</span>             ▞<span style='color:var(--green)'>·</span>         <span style='color:var(--red)'>+++</span>▝▀▀▀▚               <span style='color:var(--red)'>+</span>        │
            │        ▖<span style='color:var(--red)'>++++++++</span>       <span style='color:var(--green)'>······</span>▐<span style='color:var(--green)'>·</span><span style='color:var(--cyan)'>-</span><span style='color:var(--green)'>·········</span>        ▀▖           <span style='color:var(--red)'>++</span> <span style='color:var(--red)'>+</span>       │
        35.4┤       ▞▚<span style='color:var(--green)'>·</span>             <span style='color:var(--green)'>·</span>     ▗▘<span style='color:var(--cyan)'>-</span> <span style='color:var(--cyan)'>---------</span><span style='color:var(--green)'>········</span> ▝▚▖<span style='color:var(--red)'>+</span>     <span style='color:var(--red)'>+++</span>    <span style='color:var(--red)'>+</span>      │
            │      ▐<span style='color:var(--cyan)'>--</span>▚<span style='color:var(--green)'>··</span>         <span style='color:var(--green)'>··</span>     ▗▘<span style='color:var(--cyan)'>-</span>           <span style='color:var(--cyan)'>----</span>    <span style='color:var(--green)'>···</span>▝▄<span style='color:var(--red)'>+++++</span>     <span style='color:var(--green)'>·</span>  <span style='color:var(--red)'>++</span>    │
            │     ▗▘  <span style='color:var(--cyan)'>-</span>▌ <span style='color:var(--green)'>·</span>       <span style='color:var(--green)'>·</span>       ▞<span style='color:var(--cyan)'>-</span>                <span style='color:var(--cyan)'>----</span>    <span style='color:var(--green)'>·</span>▚▖<span style='color:var(--green)'>········</span> <span style='color:var(--green)'>··</span>  <span style='color:var(--red)'>+</span>   │
            │    <span style='color:var(--green)'>·</span>▌    ▝▖ <span style='color:var(--green)'>··</span>   <span style='color:var(--green)'>··</span>       ▞<span style='color:var(--cyan)'>-</span>                     <span style='color:var(--cyan)'>------</span>▝▚▄▖        <span style='color:var(--green)'>··</span> <span style='color:var(--red)'>+</span>  │
        28.4┤   <span style='color:var(--green)'>·</span>▞      ▝▖  <span style='color:var(--green)'>···</span>        ▐<span style='color:var(--cyan)'>-</span>                              <span style='color:var(--cyan)'>-</span>▝▀▚▄▖      <span style='color:var(--green)'>··</span><span style='color:var(--red)'>++</span>│
            │  <span style='color:var(--green)'>·</span>▗▘       ▚            ▗▘                                   <span style='color:var(--cyan)'>-</span>▝▀▀▄▄▖   <span style='color:var(--green)'>··</span>│
            │<span style='color:var(--red)'>+</span><span style='color:var(--green)'>·</span>▗▘         ▚          ▗▘                                        <span style='color:var(--cyan)'>--</span>▝▀▀▄▄▄│
        21.4┤<span style='color:var(--green)'>·</span> ▞           ▌        ▗▞                                                 │
            │<span style='color:var(--green)'>·</span>▞            ▝▖    ▗▄▀▘                                                  │
            │▗▘             ▝▖<span style='color:var(--cyan)'>-</span>▄▞▘                                                     │
        14.4┤▌               ▝▀                                                        │
            └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
            1.0               3.2                5.5               7.8             10.0 
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss_summary.txt</span>
                    loss/mean hist                           loss/max hist           
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        2.00┤                           ███████│2.00┤             ███████████          │
        1.67┤                           ███████│1.67┤             ███████████          │
        1.33┤                           ███████│1.33┤             ███████████          │
        1.00┤████   ███████   █████████████████│1.00┤███████   ██████████████      ████│
        0.67┤████   ███████   █████████████████│0.67┤███████   ██████████████      ████│
        0.33┤████   ███████   █████████████████│0.33┤███████   ██████████████      ████│
        0.00┤███    ██████    █████████████████│0.00┤███████   ██████████████      ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
        17.9    24.0     30.2    36.3   42.4    22.0    30.9     39.8    48.8   57.7 
                        loss/min hist                           loss/std hist           
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        2.00┤████                          ████│3.00┤████                              │
        1.67┤████                          ████│2.50┤████                              │
        1.33┤████                          ████│2.00┤██████████                        │
        1.00┤████   ██████████   ██████████████│1.50┤██████████                        │
            │████   ██████████   ██████████████│    │██████████                        │
        0.67┤████   ██████████   ██████████████│1.00┤██████████████      ████      ████│
        0.33┤████   ██████████   ██████████████│0.50┤██████████████      ████      ████│
        0.00┤███    ██████████   ██████████████│0.00┤█████████████       ████      ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
        13.3    20.1     26.9    33.7   40.5    -0.2     4.4      8.9    13.5   18.1 
        <b><span style='color:var(--bright-green)'>text saved in</span></b> <span style='opacity:0.67'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss_hist.txt</span>
        \[2026-01-15 16:25:50,768264\]\[<span style='color:var(--bright-yellow)'>W</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>2420</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>finalize</span></i>\]<span style='color:var(--white)'> </span>h5py not found! Saving dataset as netCDF instead.
        \[2026-01-15 16:25:50,768640\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>utils</span>/<span style='color:var(--magenta)'>__init__</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>636</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>save_dataset</span></i>\]<span style='color:var(--white)'> </span>Saving dataset to: <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/</span><span style='color:var(--bright-magenta)'>dataset_dataset.nc</span>
        \[2026-01-15 16:25:50,817704\]\[<span style='color:var(--green)'>I</span>\]\[<span style='color:var(--bright-magenta)'>ezpz</span>/<span style='color:var(--magenta)'>history</span><span style='opacity:0.67'>:</span><span style='color:var(--red)'>2433</span><span style='opacity:0.67'>:</span><i><span style='color:var(--bright-green)'>finalize</span></i>\]<span style='color:var(--white)'> </span>Saving history report to <span style='color:var(--magenta)'>/Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/</span><span style='color:var(--bright-magenta)'>report.md</span>
        <span class='shell'>&gt; </span><span class='caret'> </span>
        </code>
        </pre>
        </div>
