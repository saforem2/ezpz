# 🍋 ezpz

> Write once, run anywhere.

`ezpz` makes distributed PyTorch launches portable across any supported
hardware {NVIDIA, AMD, Intel, MPS, CPU} with **zero code changes**.

This lets us write _a single distributed PyTorch script_ that can be run
anywhere, at any scale (with built-in support for HPC Job Schedulers, e.g. PBS,
Slurm)

## Overview

Explicitly, `ezpz` provides:

1. 🧰 [**CLI**](./cli/index.md): `ezpz <command>`  
   Utilities for launching distributed PyTorch applications:
    - [`ezpz doctor`](./cli/doctor.md): Health check your environment
    - [`ezpz test`](./cli/test.md): Run simple distributed smoke test
    - [`ezpz launch`](./cli/launch.md): Launch arbitrary distributed commands  
      with _automatic **job scheduler** detection_ (PBS, Slurm) !!

1. 🐍 [**Python library**](./python/Code-Reference/index.md): `import ezpz`  
   Python API for writing hardware-agnostic, distributed PyTorch code.  
    - See [**Features**](#features) for a list of core features and functionality
      provided by `ezpz`.
    - The complete API reference for `ezpz` is available at:
        - [`ezpz`](https://saforem2.github.io/ezpz/python/Code-Reference/):
          Entry point for the `ezpz` library.
        - [`ezpz.dist`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/):
          Contains the bulk of the important logic related to device detection
          and distributed initialization.

- 📝 [**Complete Examples**](./examples/index.md): `ezpz.examples.*`  
    A collection of ready-to-go distributed training examples that can
    be run at _**any scale**_, on **_any hardware_**:
    - [Train MLP with DDP on MNIST](https://saforem2.github.io/ezpz/examples/test-dist/): [`ezpz.examples.test_dist`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)
    - [Train CNN with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/fsdp/): [`ezpz.examples.fsdp`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)
    - [Train ViT with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/vit/): [`ezpz.examples.vit`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)
    - [Train Transformer with FSDP and TP on HF Datasets](https://saforem2.github.io/ezpz/examples/fsdp-tp/): [`ezpz.examples.fsdp_tp`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)
    - [Train Diffusion LLM with FSDP on HF Datasets](https://saforem2.github.io/ezpz/examples/diffusion/): [`ezpz.examples.diffusion`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py)
    - [Train LLM with FSDP and HF Trainer on HF Datasets](https://saforem2.github.io/ezpz/examples/hf-trainer/): [`ezpz.examples.hf_trainer`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)
    - Simple example demonstrating generic initialization logic:

        ```python
        # demo.py
        import ezpz

        # automatic device + backend setup for distributed PyTorch
        _ = ezpz.setup_torch()  # CUDA/NCCL, XPU/XCCL, {MPS, CPU}/GLOO, ...

        device = ezpz.get_torch_device() # {cuda, xpu, mps, cpu, ...}
        rank = ezpz.get_rank()
        world_size = ezpz.get_world_size()
        # ...etc

        if rank == 0:
            print(f"Hello from rank {rank} / {world_size} on {device}!")
        ```

        We can launch this script with:

        ```bash
        ezpz launch python3 demo.py
        ```

        <!-- - <details closed><summary>Output(s):</summary> -->
        <!-- <details closed><summary>MacBook Pro:</summary> -->
        <!-- <details closed><summary>Aurora (2 nodes):</summary> -->

        /// details | Output(s)
            type: success

        /// details | MacBook Pro
            type: example

        ```bash
        # from MacBook Pro
        $ ezpz launch python3 demo.py
        [2026-01-08 07:22:31,989741][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 /Users/samforeman/python/ezpz_demo.py
        Using [2 / 2] available "mps" devices !!
        Hello from rank 0 / 2 on mps!
        ```

        ///

        /// details | Aurora (2 nodes)
            type: example

        ```bash
        # from 2 nodes of Aurora:
        #[aurora_frameworks-2025.2.0](foremans-aurora_frameworks-2025.2.0)[C v7.5.0-gcc][43s]
        #[01/08/26,07:26:10][x4604c5s2b0n0][~]
        ; ezpz launch python3 demo.py

        [2026-01-08 07:26:19,723138][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
        [2026-01-08 07:26:19,725453][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
        [2026-01-08 07:26:19,725932][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
        [2026-01-08 07:26:20,290222][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-072620]----
        [2026-01-08 07:26:21,566797][I][ezpz/launch:416:launch] Job ID: 8246832
        [2026-01-08 07:26:21,567684][I][ezpz/launch:417:launch] nodelist: ['x4604c5s2b0n0', 'x4604c5s3b0n0']
        [2026-01-08 07:26:21,568082][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        [2026-01-08 07:26:21,568770][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
        [2026-01-08 07:26:21,569557][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
        [2026-01-08 07:26:21,569959][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
        [2026-01-08 07:26:21,570821][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 demo.py
        [2026-01-08 07:26:21,571548][I][ezpz/launch:433:launch] Took: 2.11 seconds to build command.
        [2026-01-08 07:26:21,571918][I][ezpz/launch:436:launch] Executing:
        mpiexec
        --envall
        --np=24
        --ppn=12
        --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        --no-vni
        --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
        python3
        demo.py
        [2026-01-08 07:26:21,573262][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
        [2026-01-08 07:26:21,573781][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-072621...
        [2026-01-08 07:26:21,574195][I][ezpz/launch:138:run_command] Caught 24 filters
        [2026-01-08 07:26:21,574532][I][ezpz/launch:139:run_command] Running command:
        mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8246832.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 demo.py
        cpubind:list x4604c5s3b0n0 pid 131587 rank 12 0: mask 0x1c
        cpubind:list x4604c5s3b0n0 pid 131588 rank 13 1: mask 0x1c00
        cpubind:list x4604c5s3b0n0 pid 131589 rank 14 2: mask 0x1c0000
        cpubind:list x4604c5s3b0n0 pid 131590 rank 15 3: mask 0x1c000000
        cpubind:list x4604c5s3b0n0 pid 131591 rank 16 4: mask 0x1c00000000
        cpubind:list x4604c5s3b0n0 pid 131592 rank 17 5: mask 0x1c0000000000
        cpubind:list x4604c5s3b0n0 pid 131593 rank 18 6: mask 0x1c0000000000000
        cpubind:list x4604c5s3b0n0 pid 131594 rank 19 7: mask 0x1c000000000000000
        cpubind:list x4604c5s3b0n0 pid 131595 rank 20 8: mask 0x1c00000000000000000
        cpubind:list x4604c5s3b0n0 pid 131596 rank 21 9: mask 0x1c0000000000000000000
        cpubind:list x4604c5s3b0n0 pid 131597 rank 22 10: mask 0x1c000000000000000000000
        cpubind:list x4604c5s3b0n0 pid 131598 rank 23 11: mask 0x1c00000000000000000000000
        cpubind:list x4604c5s2b0n0 pid 121225 rank 0 0: mask 0x1c
        cpubind:list x4604c5s2b0n0 pid 121226 rank 1 1: mask 0x1c00
        cpubind:list x4604c5s2b0n0 pid 121227 rank 2 2: mask 0x1c0000
        cpubind:list x4604c5s2b0n0 pid 121228 rank 3 3: mask 0x1c000000
        cpubind:list x4604c5s2b0n0 pid 121229 rank 4 4: mask 0x1c00000000
        cpubind:list x4604c5s2b0n0 pid 121230 rank 5 5: mask 0x1c0000000000
        cpubind:list x4604c5s2b0n0 pid 121231 rank 6 6: mask 0x1c0000000000000
        cpubind:list x4604c5s2b0n0 pid 121232 rank 7 7: mask 0x1c000000000000000
        cpubind:list x4604c5s2b0n0 pid 121233 rank 8 8: mask 0x1c00000000000000000
        cpubind:list x4604c5s2b0n0 pid 121234 rank 9 9: mask 0x1c0000000000000000000
        cpubind:list x4604c5s2b0n0 pid 121235 rank 10 10: mask 0x1c000000000000000000000
        cpubind:list x4604c5s2b0n0 pid 121236 rank 11 11: mask 0x1c00000000000000000000000
        Using [24 / 24] available "xpu" devices !!
        Hello from rank 0 / 24 on xpu!
        [2026-01-08 07:26:33,060432][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-072633]----
        [2026-01-08 07:26:33,061512][I][ezpz/launch:448:launch] Execution finished with 0.
        [2026-01-08 07:26:33,062045][I][ezpz/launch:449:launch] Executing finished in 11.49 seconds.
        [2026-01-08 07:26:33,062531][I][ezpz/launch:450:launch] Took 11.49 seconds to run. Exiting.
        took: 22s
        ```

        ///

        ///

[^distributed-history]: The `ezpz.History` class automatically computes
    distributed statistics (min, max, mean, std. dev) across ranks for all
    recorded metrics.  
    **NOTE**: This is automatically disabled when
    `ezpz.get_world_size() >= 384` (e.g. >= {32, 96} {Aurora, Polaris} nodes)
    due to the additional overhead introduced (but can be manually enabled, if
    desired).

## Getting Started

To use `ezpz`, we first need:

1. A suitable MPI implementation (MPICH, OpenMPI), and
2. A Python environment (preferably _virtual_) with {`torch`, `mpi4py`}

_If you already have both of these things: skip directly to (2.) **Install**_.

1. <details closed><summary><b>[Optional]</b>: Setup Python environment</summary>  

    - We can use the provided
      [src/ezpz/bin/utils.sh](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)[^bitly]
      to setup our environment:

        ```bash
        source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
        ```

        /// details | [**Optional**]
            type: abstract

        **Note**: This is _technically_ optional, but recommended.<br>
        Especially if you happen to be running behind a job scheduler (e.g.
        PBS/Slurm) at any of {ALCF, OLCF, NERSC}, this will automatically 
        load the appropriate modules and use these to bootstrap a virtual
        environment.  
        However, if you already have a Python environment with
        {`torch`, `mpi4py`} installed and would prefer to use that, skip
        directly to (2.) installing `ezpz` below

        ///

    <!-- </details> -->
    <!-- - <details closed><summary>... <i>or try without installing</i>!</summary> -->

1. **Install `ezpz`[^uvi]**:

    ```bash
    uv pip install "git+https://github.com/saforem2/ezpz"
    ```
    <!-- - <details closed><summary>Need <code>torch</code> or <code>mpi4py</code>?</summary> -->
    /// details | Need `torch` or `mpi4py`?
        type: question

    If you don't already have PyTorch or `mpi4py` installed,
    you can specify these as additional dependencies:

    ```bash
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz[torch,mpi]"
    ```

    ///

    /// details | _or try without installing_!
        type: tip

    If you already have a Python environment with
    {`torch`, `mpi4py`} installed, you can try `ezpz` without installing
    it:

    ```bash
    # pip install uv first, if needed
    uv run --with "git+https://github.com/saforem2/ezpz" ezpz doctor

    TMPDIR=$(pwd) uv run --with "git+https://github.com/saforem2/ezpz" \
        --python=$(which python3) \
        ezpz test

    TMPDIR=$(pwd) uv run --with "git+https://github.com/saforem2/ezpz" \
        --python=$(which python3) \
        ezpz launch \
            python3 -m ezpz.examples.fsdp_tp
    ```

    ///

    /// details | `ezpz test`
        type: example

    After installing, we can run a simple smoke test to verify distributed
    functionality and device detection:

    - [`ezpz test`](./cli/test.md): Simple distributed smoke test; explicitly,
      this will train a simple MLP on MNIST dataset using PyTorch + DDP.

        ```bash
        ezpz test
        ```

        - See
          \[[W\&B Report: `ezpz test`](https://api.wandb.ai/links/aurora_gpt/q56ai28l)\]
          for example output and demonstration of metric tracking with
          automatic `wandb` integration.

    ///

[^uvi]: If you don't have `uv` installed, you can install it via:

    ```bash
    pip install uv
    ```

    See the [uv documentation](https://uv.readthedocs.io/en/latest/) for more details.

[^bitly]: The <https://bit.ly/ezpz-utils> URL is just a short link for
    convenience that actually points to
    <https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh>

## Features

/// details | Core Features 1
    type: example

- Core features:
    - Automatic distributed initialization using
      [`ezpz.setup_torch()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.setup_torch)

        ```python
        import ezpz
        _ = ezpz.setup_torch()
        ```

    - Job launching utilities with automatic scheduler detection
      (PBS, Slurm), plus safe fallbacks when no scheduler is detected
    - Automatic accelerator and backend detection across 
    - Single-process logging with rank-aware filtering for distributed runs:

        ```python
        logger = ezpz.get_logger(__name__)
        ```

    - Metric tracking, aggregation, and recording via
      [`ezpz.History()`](https://saforem2.github.io/ezpz/python/Code-Reference/#ezpz.History):
        - Automatic distributed statistics (min, max, mean, std. dev.) across ranks[^distributed-history]
        - Weights & Biases integration
        - Plotting support:
            - Graphical plots (`svg`, `png`) via `matplotlib`
            - Terminal-based ASCII plots via
              [`plotext`](https://github.com/piccolomo/plotext#guide)  

              <details closed><summary><b>[Optional]</b></summary>

              ![`tplot` Split Dark](./assets/tplot-split-dark.png) ![`tplot` Dark](./assets/tplot-dark.png)

              ![`tplot` Split Light](./assets/tplot-split-light.png) ![`tplot` Light](./assets/tplot-light.png)

            </details>

        - Persistent storage of metrics in `.hdf5` format
///

/// details | Core Features 1
    type: example

- Core features:
    - Automatic distributed initialization using
      [`ezpz.setup_torch()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.setup_torch)
    - Automatic accelerator and backend detection across
      `{cuda, xpu, mps, cpu}` via
      [`ezpz.get_torch_device()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.get_torch_device),
      with correct backend selection (NCCL, XCCL, GLOO, …)

      ```python
      import ezpz
      _ = ezpz.setup_torch()
      ```

    - Single-process logging with rank-aware filtering for distributed runs:

      ```python
      logger = ezpz.get_logger(__name__)
      ```

    - Job launching utilities with automatic scheduler detection
      (PBS, Slurm), plus safe fallbacks when no scheduler is detected

      ```python
      logger = ezpz.get_logger(__name__)
      ```

    - Metric tracking, aggregation, and recording via
      [`ezpz.History()`](https://saforem2.github.io/ezpz/python/Code-Reference/#ezpz.History):
      - Automatic distributed statistics (min, max, mean, std. dev.) across ranks[^distributed-history]
      - Weights & Biases integration
      - Plotting support:
        - Graphical plots (`svg`, `png`) via `matplotlib`
        - Terminal-based ASCII plots via
          [`plotext`](https://github.com/piccolomo/plotext#guide)
      - Persistent storage of metrics in `.hdf5` format
///

<!-- - Core features include: -->
<!---->
<!--     ```python -->
<!--     import ezpz -->
<!--     ``` -->
<!---->
<!--     - Launching utilities -->
<!--         - **Including automatic job scheduler detection** (PBS, Slurm) -->
<!--     - Automatic Device / backend detection and distributed initialization -->
<!--       (e.g.: CUDA/NCCL, XPU/XCCL, MPS/GLOO, CPU/GLOO, ...) -->
<!--     - Single-process logging -->
<!---->
<!--         ```bash -->
<!--         logger = ezpz.get_logger(__name__) -->
<!--         ``` -->
<!---->
<!--     - Utilities for tracking and recording metrics, including automatic: -->
<!--         - Distributed statistics (min, max, mean, std. dev) across -->
<!--             ranks[^distributed-history] -->
<!--         - W&B integration -->
<!--         - Plotting, with support for both: -->
<!--             - Graphical {`svg`, `png`} plots with `matplotlib` -->
<!--             - Text based plots (ASCII in terminal) with [`plotext`](https://github.com/piccolomo/plotext#guide) -->
<!--         - Saving and recording metric data to `.hdf5` files -->
<!---->
<!-- - 🪄 _Automatic_: -->
<!--     - Accelerator detection: -->
<!--       [`ezpz.get_torch_device()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.get_torch_device),   -->
<!--       across {`cuda`, `xpu`, `mps`, `cpu`} -->
<!--     - Distributed initialization: -->
<!--       [`ezpz.setup_torch()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.setup_torch), -->
<!--       to pick the right device + backend combo -->
<!--     - Metric handling and utilities for {tracking, recording, plotting}: -->
<!--       [`ezpz.History()`](https://saforem2.github.io/ezpz/python/Code-Reference/#ezpz.History) -->
<!--       with Weights \& Biases support -->
<!--     - Integration with native job scheduler(s) (PBS, Slurm) -->
<!--         - with _safe fall-backs_ when no scheduler is detected -->
<!--     - Single-process logging with filtering for distributed runs -->
<!---->
<!-- - See [🚀 Quickstart](https://saforem2.github.io/ezpz/quickstart/) for an -->
<!--   in-depth walk-through of the various `ezpz` features. -->

## More Information

- Examples live under [`ezpz.examples.*`](https://saforem2.github.io/ezpz/examples/)—copy them or
  extend them for your workloads.
- Stuck? Check the [docs](https://saforem2.github.io/ezpz), or run `ezpz doctor` for actionable hints.
- See my (~ recent) talk on:
  [**_LLMs on Aurora_: Hands On with `ezpz`**](https://saforem2.github.io/ezpz/slides-2025-05-07/)
  for a detailed walk-through containing examples and use cases.
    - [🎥 YouTube](https://www.youtube.com/watch?v=15ZK9REQiBo)
    - [Slides (html)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/)
    - [Slides (reveal.js)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/slides)
- [Reach out](https://samforeman.me)!

### Environment Variables

Additional configuration can be done through environment variables, including:

1. The colorized logging output can be toggled via the `NO_COLOR` environment
   var, e.g. to turn off colors:

    ```bash
    NO_COLOR=1 ezpz launch python3 -m your_app.train
    ```

1. Forcing a specific torch device (useful on GPU hosts when you want CPU-only):

    ```bash
    TORCH_DEVICE=cpu ezpz test
    ```

1. Text Based Plots:

    1. Changing the plot marker used in the text-based plots:

        ```bash
        # highest resolution, may not be supported in all terminals
        EZPZ_TPLOT_MARKER="braille" ezpz launch python3 -m your_app.train
        # next-best resolution, more widely supported
        EZPZ_TPLOT_MARKER="fhd" ezpz launch python3 -m your_app.train
        ```

    1. Changing the plot size:

        The plots will automatically scale (up to a reasonable limit) with the
        dimensions of the terminal in which they're run.

        If desired, these can be specified explicitly by overriding the `LINES`
        and `COLUMNS` environment variables, e.g.:

        ```bash
        LINES=40 COLUMNS=100 ezpz test
        ```

<!--
        <details closed><summary>30x120</summary>

        ```bash
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        1.84┤ ++ loss/max                                                                                                      │
            │ -- loss/min                                                                                                      │
            │ ·· loss/mean                                                                                                     │
            │ ▞▞ loss                                                                                                          │
        1.54┤▐·                                                                                                                │
            │▐·                                                                                                                │
            │▝▖+                                                                                                               │
            │ ▝▖                                                                                                               │
        1.23┤ -▌                                                                                                               │
            │ -▌ +                                                                                                             │
            │  ▌+++                                                                                                            │
            │  ▝▖ +                                                                                                            │
        0.93┤  -▌ +                                                                                                            │
            │   ▚·+++▖                                                                                                         │
            │   ▐··▗▐▌  +                                                                                                      │
            │   ▐▞▖█▐▐++++ +                                                                                                   │
            │    ▘▐█▐▝▖  ++▗+ + +                                                                                              │
        0.62┤     -▘▜·▚▖▖ +█ + ++    +                                                                                         │
            │      --  █▚▞▟▐ ▗ ++  ++ +                                                                                        │
            │      - - ▜▝▌▝·▌█▗▚▗▄▌++▖+▗++++     +                                                                             │
            │        -----  █▝▞·▘·▝▟▐▌▗█  +++▄  +++ +  +    +                                                                  │
        0.31┤            ---▝- - · ▝▞▝▘▝▄▀▚▚▞▝▖++ +▖++++ ▖ ++++                                                                │
            │             - - ----------▝-· ··▙▀▀▌▞▚·▄ ▄▐▐+++ ▗▖+++ ▗▚+ +++ + + +   +     +         +     +  +                 │
            │                    -    -------·▝--▝▌▝▞ ▀·▜·▚▚▟·▌▌▗▀▞▄▐▐▗·▗·▄+++·+▞▖++ +++++++▗++++++++++++++++ ▖ ▗+++▗++++++++++│
            │                                -   -- ----·----▀▘▝▘---▜-▘▀▀▟·▚▞▚▞▚▌▝▚▄▚·▟·▗▚▚▖▌▚▄▄▖▄▄·▟·▖▄▚▄▗▚·▟▚·▌▜·▗▀▖·▟···▞▖··│
        0.01┤                                                                 --▘ -- ▀ ▀▘--▝▘---▝--▀-▀▝·--▘-▀--▀▘-▀▀-▝▀▌▀▀▀▘▝▀▀│
            └┬───────────────────────────┬────────────────────────────┬───────────────────────────┬───────────────────────────┬┘
            1.0                        49.2                         97.5                        145.8                     194.0
        text saved in /home/foremans/outputs/ezpz.test_dist/2026-01-07-222319/plots/tplot/loss_summary.txt
        ```

        </details>

-->

<!-- - 📝 *Ready-to-go Examples* that can be bootstrapped -->
<!--     for general use cases: -->
<!--     (ViT, FSDP, tensor-parallel, diffusion, HF Trainer).   -->
<!--     <br> -->
<!---->

<!-- /// note |  📓 Examples -->

<!-- 👀 See [Examples](#ready-to-go-examples) for ready-to-go examples -->
<!-- that can be used as templates or starting points for your own -->
<!-- distributed PyTorch workloads! -->

<!--
1. Using the `ezpz` python library (e.g. `import ezpz`) to write distributed
PyTorch code that runs anywhere
1. How to use the `ezpz` CLI (e.g. `ezpz launch`) to launch distributed PyTorch
modules
-->
<!-- /// -->
<!---->

<!--
`ezpz`: single distributed PyTorch script that can be
run at any scale, on any hardware, with **zero code changes**.

`ezpz`: Make iPyTorch launches portable across NVIDIA, AMD, Intel,
MPS, and CPU; with _zero-code changes_.
-->

<!-- `ezpz` makes distributed PyTorch launches portable across NVIDIA, AMD, Intel, -->
<!-- MPS, and CPU; with _zero-code changes_ and guardrails for HPC schedulers. -->
<!-- and guardrails for HPC schedulers. -->

<!--
## Python Library

At its core, `ezpz` is a Python library designed to make writing distributed
PyTorch code easy and portable across different hardware backends.

See [🐍 Python Library](https://saforem2.github.io/ezpz/python/Code-Reference/) for more information.
-->

<!--
Checkout the [docs](https://saforem2.github.io/ezpz) for more information on:

- [Quickstart](https://saforem2.github.io/ezpz/quickstart/):
    - [Writing Hardware Agnostic Distributed PyTorch Code](https://saforem2.github.io/ezpz/quickstart/#🌐-write-hardware-agnostic-distributed-pytorch-code)
        - Details on [Automatic Accelerator Detection and Setup](https://saforem2.github.io/ezpz/python/Code-Reference/dist/):
    - [Tracking Metrics with `ezpz.History`](https://saforem2.github.io/ezpz/quickstart/#📊-track-metrics-with-ezpzhistory)

    ```python
    >>> device = ezpz.get_device()
    'cuda'  # or 'xpu', 'mps', 'cpu' depending on available hardware
    ```

- [CLI Utilities] for:
    - [Diagnosing Environment Issues]: `ezpz doctor`
    - [Running distributed smoke tests]: `ezpz test`
    - [Launching _any_ executable]: `ezpz launch`, with support for:
        - [Automatic Job Scheduler Detection and Launching]
-->

<!--
#### 📝 Ready-to-go Examples

See [📝 **Examples**](https://saforem2.github.io/ezpz/examples/) for complete example scripts covering:

1. [Train MLP with DDP on MNIST](https://saforem2.github.io/ezpz/examples/test-dist/)
1. [Train CNN with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/fsdp/)
1. [Train ViT with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/vit/)
1. [Train Transformer with FSDP and TP on HF Datasets](https://saforem2.github.io/ezpz/examples/fsdp-tp/)
1. [Train Diffusion LLM with FSDP on HF Datasets](https://saforem2.github.io/ezpz/examples/diffusion/)
1. [Train or Fine-Tune an LLM with FSDP and HF Trainer on HF Datasets](https://saforem2.github.io/ezpz/examples/hf-trainer/)
-->

<!-- 1. [Use FSDP + MNIST to train a CNN](https://saforem2.github.io/ezpz/examples/fsdp/) -->
<!-- 1. [Use FSDP + MNIST to train a Vision Transformer](https://saforem2.github.io/ezpz/examples/vit/) -->
<!-- 1. [Use FSDP + HF Datasets to train a Diffusion Language Model](https://saforem2.github.io/ezpz/examples/diffusion/) -->
<!-- 1. [Use FSDP + HF Datasets + Tensor Parallelism to train a Llama style model](https://saforem2.github.io/ezpz/examples/fsdp-tp/) -->
<!-- 1. [Use FSDP + HF {Datasets + AutoModel + Trainer} to train / fine-tune an LLM](https://saforem2.github.io/ezpz/examples/hf-trainer/) -->
<!--     - [Comparison between Aurora/Polaris at ALCF](https://saforem2.github.io/ezpz/notes/hf-trainer-comparison/) -->

<!--
    - [\[docs\]](https://saforem2.github.io/ezpz/python/Code-Reference/examples/fsdp/), [\[source\]](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)
    - [\[docs\]](https://saforem2.github.io/ezpz/python/Code-Reference/examples/vit/), [\[source\]](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)
    - [\[docs\]](https://saforem2.github.io/ezpz/python/Code-Reference/examples/diffusion/), [\[source\]](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py)
    - [\[docs\]](https://saforem2.github.io/ezpz/python/Code-Reference/examples/fsdp_tp/), [\[source\]](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)
    - [\[docs\]](https://saforem2.github.io/ezpz/python/Code-Reference/examples/hf_trainer/), [\[source\]](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)
-->
