# 🍋 ezpz

> _Write once, run anywhere_.

`ezpz` makes distributed PyTorch code portable across any supported hardware
{NVIDIA, AMD, Intel, MPS, CPU} with **zero code changes**.

This lets us write Python applications that can be run anywhere, at any scale.

Built to run _at scale_, with native job scheduler (PBS, Slurm)[^lcfs]
integration, and graceful fallbacks for running locally (Mac, Linux) machines

[^lcfs]: With first class support for all of the major HPC Supercomputing
    centers (e.g. ALCF, OLCF, NERSC)


[^dev]: This is particularly useful if you'd like to run development /
    debugging experiments locally

## Overview

`ezpz` is, at its core, a Python library that provides a variety of utilities
for both _writing_ and _launching_ distributed PyTorch applications.

These can be broken down (~roughly) into three categories:

1. 🧰 [**CLI**](./cli/index.md): `ezpz <command>`  
   Utilities for launching distributed PyTorch applications:
    - [`ezpz doctor`](./cli/doctor.md): Health check your environment
    - [`ezpz test`](./cli/test.md): Run simple distributed smoke test
    - [`ezpz launch`](./cli/launch/index.md): Launch arbitrary distributed commands  
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

1. 📝 [**Complete Examples**](./examples/index.md): `ezpz.examples.*`  
    A collection of performant, scalable distributed training examples that can
    be run at _**any scale**_, on **_any hardware_**; or bootstrap them for
    your own applications!

    1. [`ezpz.examples.test_dist`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py):
       [Train MLP with DDP on MNIST](https://saforem2.github.io/ezpz/examples/test-dist/)
    1. [`ezpz.examples.fsdp`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py):
       [Train CNN with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/fsdp/)
    1. [`ezpz.examples.vit`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py):
       [Train ViT with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/vit/)
    1. [`ezpz.examples.fsdp_tp`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py):
       [Train Transformer with FSDP and TP on HF Datasets](https://saforem2.github.io/ezpz/examples/fsdp-tp/)
    1. [`ezpz.examples.diffusion`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py):
       [Train Diffusion LLM with FSDP on HF Datasets](https://saforem2.github.io/ezpz/examples/diffusion/)
    1. [`ezpz.examples.hf_trainer`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py):
       [Train LLM with FSDP and HF Trainer on HF Datasets](https://saforem2.github.io/ezpz/examples/hf-trainer/)
    1. /// details | `demo.py`
           type: example

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

        /// details | Output(s)
            type: abstract

        /// details | MacBook Pro
            type: success

        ```bash
        # from MacBook Pro
        $ ezpz launch python3 demo.py
        [2026-01-08 07:22:31,989741][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 /Users/samforeman/python/ezpz_demo.py
        Using [2 / 2] available "mps" devices !!
        Hello from rank 0 / 2 on mps!
        ```

        ///

        /// details | Aurora (2 nodes)
            type: success

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

        ///

    /// details | 🤗 HF Integration
        type: tip

    1. `ezpz.examples.{fsdp_tp,diffusion,hf_trainer}` all support
        arbitrary 🤗 Hugging Face
        [datasets](https://huggingface.co/docs/datasets/index) e.g.:

        ```bash
        # use any --dataset from HF Datasets hub
        ezpz launch python3 -m ezpz.examples.fsdp_tp --dataset stanfordnlp/imdb
        ```

    1. [`ezpz.examples.hf_trainer`](./examples/hf-trainer/index.md) supports
       arbitrary combinations of (compatible) `transformers.from_pretrained`
       models, and HF Datasets (with support for streaming!)

       ```bash
       ezpz launch python3 -m ezpz.examples.hf_trainer \
           --streaming \
           --dataset_name=eliplutchok/fineweb-small-sample \
           --tokenizer_name meta-llama/Llama-3.2-1B \
           --model_name_or_path meta-llama/Llama-3.2-1B \
           --bf16=true
           # ...etc.
       ```

    ///

[^distributed-history]: The `ezpz.History` class automatically computes
    distributed statistics (min, max, mean, std) across ranks for all
    recorded metrics.  
    **NOTE**: This is automatically disabled when
    `ezpz.get_world_size() >= 384` (e.g. >= {32, 96} {Aurora, Polaris} nodes)
    due to the additional overhead introduced (but can be manually enabled, if
    desired).



## Getting Started

To use `ezpz`, we first need:

1. A suitable MPI implementation (MPICH, OpenMPI), and
2. A Python environment; preferably _virtual_, ideally with {`torch`, `mpi4py`}
   installed

If you already have both of these things: skip directly to
[Install](#install-ezpz); otherwise, see the
details below:

/// details | [**Optional**]: Setup Python Environment
    type: tip

- We can use the provided
  [src/ezpz/bin/utils.sh](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)[^bitly]
  to set up our environment:

    ```bash
    source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
    ```

    /// details | [**Details**]
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

///

### Install `ezpz`

To install `ezpz`, we can use `uv`[^uvi] to install directly from GitHub:

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

/// details | Try _without installing_ via `uv run`
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

Core features:

- Job launching utilities with automatic scheduler detection
  (PBS, Slurm), plus safe fallbacks when no scheduler is detected

    ```bash
    ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'
    ```

    /// details | Output
        type: abstract

    /// details | MacBook Pro
        type: success

    ```bash
    #[01/08/26 @ 14:56:50][~/v/s/ezpz][dev][$✘!?] [󰔛  4s]
    ; ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'


    [2026-01-08 14:56:54,307030][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 -c 'import ezpz; print(ezpz.setup_torch())'
    Using [2 / 2] available "mps" devices !!
    0
    1
    [2025-12-23-162222] Execution time: 4s sec
    ```

    ///

    /// details | Aurora (2 Nodes)
        type: success

    ```bash
    #[aurora_frameworks-2025.2.0](torchtitan-aurora_frameworks-2025.2.0)[1m9s]
    #[01/08/26,14:56:42][x4418c6s1b0n0][/f/d/f/p/p/torchtitan][main][?]
    ; ezpz launch python3 -c 'import ezpz; print(ezpz.setup_torch())'


    [2026-01-08 14:58:01,994729][I][numexpr/utils:148:_init_num_threads] Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
    [2026-01-08 14:58:01,997067][I][numexpr/utils:151:_init_num_threads] Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
    [2026-01-08 14:58:01,997545][I][numexpr/utils:164:_init_num_threads] NumExpr defaulting to 16 threads.
    [2026-01-08 14:58:02,465850][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2026-01-08-145802]----
    [2026-01-08 14:58:04,765720][I][ezpz/launch:416:launch] Job ID: 8247203
    [2026-01-08 14:58:04,766527][I][ezpz/launch:417:launch] nodelist: ['x4418c6s1b0n0', 'x4717c0s6b0n0']
    [2026-01-08 14:58:04,766930][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    [2026-01-08 14:58:04,767616][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
    [2026-01-08 14:58:04,768399][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
    [2026-01-08 14:58:04,768802][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
    [2026-01-08 14:58:04,769517][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -c 'import ezpz; print(ezpz.setup_torch())'
    [2026-01-08 14:58:04,770278][I][ezpz/launch:433:launch] Took: 3.01 seconds to build command.
    [2026-01-08 14:58:04,770660][I][ezpz/launch:436:launch] Executing:
    mpiexec
    --envall
    --np=24
    --ppn=12
    --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    --no-vni
    --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
    python3
    -c
    import ezpz; print(ezpz.setup_torch())
    [2026-01-08 14:58:04,772125][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
    [2026-01-08 14:58:04,772651][I][ezpz/launch:443:launch] Execution started @ 2026-01-08-145804...
    [2026-01-08 14:58:04,773070][I][ezpz/launch:138:run_command] Caught 24 filters
    [2026-01-08 14:58:04,773429][I][ezpz/launch:139:run_command] Running command:
    mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8247203.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -c 'import ezpz; print(ezpz.setup_torch())'
    cpubind:list x4717c0s6b0n0 pid 118589 rank 12 0: mask 0x1c
    cpubind:list x4717c0s6b0n0 pid 118590 rank 13 1: mask 0x1c00
    cpubind:list x4717c0s6b0n0 pid 118591 rank 14 2: mask 0x1c0000
    cpubind:list x4717c0s6b0n0 pid 118592 rank 15 3: mask 0x1c000000
    cpubind:list x4717c0s6b0n0 pid 118593 rank 16 4: mask 0x1c00000000
    cpubind:list x4717c0s6b0n0 pid 118594 rank 17 5: mask 0x1c0000000000
    cpubind:list x4717c0s6b0n0 pid 118595 rank 18 6: mask 0x1c0000000000000
    cpubind:list x4717c0s6b0n0 pid 118596 rank 19 7: mask 0x1c000000000000000
    cpubind:list x4717c0s6b0n0 pid 118597 rank 20 8: mask 0x1c00000000000000000
    cpubind:list x4717c0s6b0n0 pid 118598 rank 21 9: mask 0x1c0000000000000000000
    cpubind:list x4717c0s6b0n0 pid 118599 rank 22 10: mask 0x1c000000000000000000000
    cpubind:list x4717c0s6b0n0 pid 118600 rank 23 11: mask 0x1c00000000000000000000000
    cpubind:list x4418c6s1b0n0 pid 66450 rank 0 0: mask 0x1c
    cpubind:list x4418c6s1b0n0 pid 66451 rank 1 1: mask 0x1c00
    cpubind:list x4418c6s1b0n0 pid 66452 rank 2 2: mask 0x1c0000
    cpubind:list x4418c6s1b0n0 pid 66453 rank 3 3: mask 0x1c000000
    cpubind:list x4418c6s1b0n0 pid 66454 rank 4 4: mask 0x1c00000000
    cpubind:list x4418c6s1b0n0 pid 66455 rank 5 5: mask 0x1c0000000000
    cpubind:list x4418c6s1b0n0 pid 66456 rank 6 6: mask 0x1c0000000000000
    cpubind:list x4418c6s1b0n0 pid 66457 rank 7 7: mask 0x1c000000000000000
    cpubind:list x4418c6s1b0n0 pid 66458 rank 8 8: mask 0x1c00000000000000000
    cpubind:list x4418c6s1b0n0 pid 66459 rank 9 9: mask 0x1c0000000000000000000
    cpubind:list x4418c6s1b0n0 pid 66460 rank 10 10: mask 0x1c000000000000000000000
    cpubind:list x4418c6s1b0n0 pid 66461 rank 11 11: mask 0x1c00000000000000000000000
    Using [24 / 24] available "xpu" devices !!
    8
    10
    0
    4
    3
    5
    7
    11
    6
    1
    9
    2
    14
    15
    12
    13
    16
    17
    19
    22
    20
    23
    18
    21
    [2026-01-08 14:58:14,252433][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2026-01-08-145814]----
    [2026-01-08 14:58:14,253726][I][ezpz/launch:448:launch] Execution finished with 0.
    [2026-01-08 14:58:14,254184][I][ezpz/launch:449:launch] Executing finished in 9.48 seconds.
    [2026-01-08 14:58:14,254555][I][ezpz/launch:450:launch] Took 9.48 seconds to run. Exiting.
    took: 18s
    ```

    ///

    ///

- Automatic distributed initialization using
  [`ezpz.setup_torch()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.setup_torch)
  with automatic {device, backend} selection

    ```python
    import ezpz
    _ = ezpz.setup_torch()

    device = ezpz.get_torch_device()
    # cuda, xpu, mps, cpu, ...
    ```

- Automatic single-process logging with rank-aware filtering for distributed
  runs:

    ```python
    logger = ezpz.get_logger(__name__)
    ```

- Metric tracking, aggregation, and recording via
  [`ezpz.History()`](https://saforem2.github.io/ezpz/python/Code-Reference/#ezpz.History):
    - Automatic distributed statistics (min, max, mean, stddev) across ranks[^distributed-history]
    - Weights & Biases integration
    - Persistent storage of metrics in `.h5` format
    - Plotting support:
        - Graphical plots (`svg`, `png`) via `matplotlib`
        - Terminal-based ASCII plots via
          [`plotext`](https://github.com/piccolomo/plotext#guide)  

          /// details | [Text Plot Example]
              type: example

          ![`tplot` Split Dark](./assets/tplot-split-dark.png) ![`tplot` Dark](./assets/tplot-dark.png)

          ![`tplot` Split Light](./assets/tplot-split-light.png) ![`tplot` Light](./assets/tplot-light.png)
          ///

          /// details | [Matplotlib Example(s)]
              type: example

          ![Accuracy](./assets/mplot/svgs/accuracy.svg)
          ![Loss](./assets/mplot/svgs/loss.svg)
          ![Forward time](./assets/mplot/svgs/dtf.svg)
          ![Backward time](./assets/mplot/svgs/dtb.svg)

          ///

## Environment Variables

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

/// details | Complete List
    type: info


| Environment Variable                         | Purpose / how it’s used                                                          |
| -------------------------------------------- | :------------------------------------------------------------------------------- |
| TORCH_DEVICE                                 | Force device selection (cpu, cuda, mps, xpu) when picking the torch device.      |
| TORCH_BACKEND                                | Override distributed backend (nccl, gloo, mpi, xla).                             |
| TORCH_DDP_TIMEOUT                            | Adjust DDP init timeout (seconds) for slow launches.                             |
| MASTER_ADDR                                  | Manually set rendezvous address if auto-detection is wrong/unreachable.          |
| MASTER_PORT                                  | Manually set rendezvous port for distributed init.                               |
| HOSTFILE                                     | Point ezpz at a specific hostfile when scheduler defaults are missing/incorrect. |
| NO_COLOR / NOCOLOR / COLOR / COLORTERM       | Enable/disable colored output to suit terminals or log sinks.                    |
| EZPZ_LOG_LEVEL                               | Set ezpz logging verbosity.                                                      |
| LOG_LEVEL                                    | General log level for various modules.                                           |
| LOG_FROM_ALL_RANKS                           | Allow logs from all ranks (not just rank 0).                                     |
| TENSORBOARD_DIR                              | Redirect TensorBoard logging output.                                             |
| PYTHONHASHSEED                               | Fix Python hash seed for reproducibility.                                        |
| WANDB_DISABLED                               | Disable Weights & Biases logging.                                                |
| WANDB_MODE                                   | Set W&B mode (online, offline, dryrun).                                          |
| WANDB_PROJECT / WB_PROJECT / WB_PROJECT_NAME | Set project name for W&B runs.                                                   |
| WANDB_API_KEY                                | Supply W&B API key for authentication.                                           |
| EZPZ_LOCAL_HISTORY                           | Control local history storage/enablement.                                        |
| EZPZ_NO_DISTRIBUTED_HISTORY                  | Disable distributed history aggregation.                                         |
| EZPZ_TPLOT_TYPE                              | Select timeline plot type.                                                       |
| EZPZ_TPLOT_MARKER                            | Marker style for timeline plots.                                                 |
| EZPZ_TPLOT_MAX_HEIGHT                        | Max height for timeline plots.                                                   |
| EZPZ_TPLOT_MAX_WIDTH                         | Max width for timeline plots.                                                    |
| EZPZ_TPLOT_RAW_MARKER                        | Marker for raw timeline data.                                                    |
| CPU_BIND                                     | Override default CPU binding for PBS launch commands (advanced).                 |

///


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

