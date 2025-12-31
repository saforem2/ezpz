# 🍋 ezpz

> Write once, run anywhere.

`ezpz` makes distributed PyTorch launches portable across NVIDIA, AMD, Intel,
MPS, and CPU—with zero-code changes and guardrails for HPC schedulers.

It provides a:

- 🧰 [CLI: `ezpz`](#ezpz-cli-toolbox) that provides utilities for launching
  distributed jobs
- 🐍 [Python library `ezpz`](#python-library) for writing
  hardware-agnostic, distributed PyTorch code
- 📝 [Pre-built examples](#ready-to-go-examples):  

    All of which:

    - Use modern distributed PyTorch features (FSDP, TP, HF Trainer)
    - Can be run anywhere (e.g. NVIDIA, AMD, Intel, MPS, CPU)

Checkout the [📘 **Docs**](https://saforem2.github.io/ezpz) for more information!

## 🐣 Getting Started

1. **Setup Python environment**:<br>
    To use `ezpz`, we first need a Python environment
    (preferably _virtual_) that has `torch` and `mpi4py` installed.
    - Already have one? Skip to (2.) below!
        <details closed><summary><b>[Optional]</b></summary>
        **Note**: This is _technically_ optional, but recommended.<br>
        Especially if you happen to be running behind a job scheduler (e.g.
        PBS/Slurm) at any of {ALCF, OLCF, NERSC}, this will automatically 
        load the appropriate modules and use these to bootstrap a virtual
        environment.<br>
        However, if you already have a Python environment with
        {`torch`, `mpi4py`} installed and would prefer to use that, skip
        directly to (2.) installing `ezpz` below
        </details>
    - _Otherwise_, we can use the provided
      [src/ezpz/bin/utils.sh](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)[^bitly]
      to setup our environment:

        ```bash
        source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
        ```

1. **Install `ezpz`[^uvi]**:

    ```bash
    uv pip install "git+https://github.com/saforem2/ezpz"
    ```

    - <details closed><summary>Need PyTorch or <code>mpi4py</code>?</summary>

        If you don't already have PyTorch or `mpi4py` installed,
        you can specify these as additional dependencies:

        ```bash
        uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz[torch,mpi]"
        ```

      </details>

    - <details closed><summary>... <i>or try without installing</i>!</summary>

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

    </details>

1. **Distributed Smoke Test**:

    Train simple MLP on MNIST with PyTorch + DDP:

    ```bash
    ezpz test
    ```

    > See:
    > \[[📑 ezpz test | W\&B Report](https://api.wandb.ai/links/aurora_gpt/q56ai28l)\]
    > for sample output and details of metric tracking.


[^uvi]: If you don't have `uv` installed, you can install it via:

    ```bash
    pip install uv
    ```

    See the [uv documentation](https://uv.readthedocs.io/en/latest/) for more details.

[^bitly]: The <https://bit.ly/ezpz-utils> URL is just a short link for
    convenience that actually points to
    <https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh>

## 🐍 Python Library

At its core, `ezpz` is a Python library designed to make writing distributed
PyTorch code easy and portable across different hardware backends.

See [🐍 Python Library](https://saforem2.github.io/ezpz/python/Code-Reference/) for more information.

## ✨ Features

- See [🚀 Quickstart](https://saforem2.github.io/ezpz/quickstart/) for a detailed
  walk-through of `ezpz` features.

- 🪄 _Automatic_:
    - Accelerator detection:
      [`ezpz.get_torch_device()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.get_torch_device),  
      across {`cuda`, `xpu`, `mps`, `cpu`}
    - Distributed initialization:
      [`ezpz.setup_torch()`](https://saforem2.github.io/ezpz/python/Code-Reference/dist/#ezpz.dist.setup_torch),
      to pick the right device + backend combo
    - Metric handling and utilities for {tracking, recording, plotting}:
      [`ezpz.History()`](https://saforem2.github.io/ezpz/python/Code-Reference/#ezpz.History)
      with Weights \& Biases support
    - Integration with native job scheduler(s) (PBS, Slurm)
        - with _safe fall-backs_ when no scheduler is detected
    - Single-process logging with filtering for distributed runs

<!-- - 📝 *Ready-to-go Examples* that can be bootstrapped -->
<!--     for general use cases: -->
<!--     (ViT, FSDP, tensor-parallel, diffusion, HF Trainer).   -->
<!--     <br> -->
<!---->

/// note |  📓 Examples

👀 See [Examples](#ready-to-go-examples) for ready-to-go examples
that can be used as templates or starting points for your own
distributed PyTorch workloads!

<!--
1. Using the `ezpz` python library (e.g. `import ezpz`) to write distributed
PyTorch code that runs anywhere
1. How to use the `ezpz` CLI (e.g. `ezpz launch`) to launch distributed PyTorch
modules
-->

///

## 🧰 `ezpz`: CLI Toolbox

Once installed, `ezpz` provides a CLI with a few useful utilities
to help with distributed launches and environment validation.

Explicitly, these are:

```bash
ezpz doctor  # environment validation and health-check
ezpz test    # distributed smoke test
ezpz launch  # general purpose, scheduler-aware launching
```

To see the list of available commands, run:

```bash
ezpz --help
```

/// note | 🧰 CLI Toolbox

Checkout [🧰 **CLI**](https://saforem2.github.io/ezpz/cli/) for additional information.

///

### 🩺 `ezpz doctor`

Health-check your environment and ensure that `ezpz` is installed correctly

```bash
ezpz doctor
ezpz doctor --json   # machine-friendly output for CI
```

Checks MPI, scheduler detection, Torch import + accelerators, and wandb
readiness, returning non-zero on errors.

See: [🩺 **Doctor**](https://saforem2.github.io/ezpz/cli/doctor/) for more information.


### ✅ `ezpz test`

Run the bundled test suite (great for first-time validation):

```bash
ezpz test
```

Or, try without installing:

```bash
TMPDIR=$(pwd) uv run \
    --python=$(which python3) \
    --with "git+https://github.com/saforem2/ezpz" \
    ezpz test
```

See [✅ **Test**](https://saforem2.github.io/ezpz/cli/test/) for more information.


### 🚀 `ezpz launch`

Single entry point for distributed jobs.

`ezpz` detects PBS/Slurm automatically and falls back to `mpirun`, forwarding
useful environment variables so your script behaves the same on laptops and
clusters.

Add your own args to any command (`--config`, `--batch-size`, etc.) and `ezpz`
will propagate them through the detected launcher.

Use the provided

```bash
ezpz launch <launch flags> -- <cmd> <cmd flags>
```

to automatically launch `<cmd>` across all available[^schedulers]
accelerators.

Use it to launch:

- Arbitrary command(s):

    ```bash
    ezpz launch hostname
    ```

- Arbitrary Python string:

    ```bash
    ezpz launch python3 -c 'import ezpz; ezpz.setup_torch()'
    ```

- One of the ready-to-go examples:

    ```bash
    ezpz launch python3 -m ezpz.test_dist --profile
    ezpz launch -n 8 -- python3 -m ezpz.examples.fsdp_tp --tp 4
    ```

- Your own distributed training script:

    ```bash
    ezpz launch -n 16 -ppn 8 -- python3 -m your_app.train --config configs/your_config.yaml
    ```

    to launch `your_app.train` across 16 processes, 8 per node.


See [🚀 **Launch**](https://saforem2.github.io/ezpz/cli/launch/) for more information.

[^schedulers]: By default, this will detect if we're running behind a job scheduler (e.g. PBS or Slurm).
    If so, we automatically determine the specifics of the currently active job; 
    explicitly, this will determine:

    1. The number of available nodes
    2. How many GPUs are present on each of these nodes
    3. How many GPUs we have _total_

    It will then use this information to automatically construct the appropriate 
    {`mpiexec`, `srun`} command to launch, and finally, execute the launch cmd.

#### 📝 Ready-to-go Examples

See [📝 **Examples**](https://saforem2.github.io/ezpz/examples/) for complete example scripts covering:

1. [Train MLP with DDP on MNIST](https://saforem2.github.io/ezpz/examples/test-dist/)
1. [Train CNN with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/fsdp/)
1. [Train ViT with FSDP on MNIST](https://saforem2.github.io/ezpz/examples/vit/)
1. [Train Transformer with FSDP and TP on HF Datasets](https://saforem2.github.io/ezpz/examples/fsdp-tp/)
1. [Train Diffusion LLM with FSDP on HF Datasets](https://saforem2.github.io/ezpz/examples/diffusion/)
1. [Train or Fine-Tune an LLM with FSDP and HF Trainer on HF Datasets](https://saforem2.github.io/ezpz/examples/hf-trainer/)

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

## ⚙️ Environment Variables

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

1. Changing the plot marker used in the text-based plots:

    ```bash
    # highest resolution, may not be supported in all terminals
    EZPZ_TPLOT_MARKER="braille" ezpz launch python3 -m your_app.train
    # next-best resolution, more widely supported
    EZPZ_TPLOT_MARKER="fhd" ezpz launch python3 -m your_app.train
    ```

## ➕ More Information

- Examples live under [`ezpz.examples.*`](https://saforem2.github.io/ezpz/examples/)—copy them or
  extend them for your workloads.
- Stuck? Check the [docs](https://saforem2.github.io/ezpz), or run `ezpz doctor` for actionable hints.
- See my recent talk on:
  [**_LLMs on Aurora_: Hands On with `ezpz`**](https://saforem2.github.io/ezpz/slides-2025-05-07/)
  for a detailed walk-through containing examples and use cases.
    - [🎥 YouTube](https://www.youtube.com/watch?v=15ZK9REQiBo)
    - [Slides (html)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/)
    - [Slides (reveal.js)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/slides)
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
