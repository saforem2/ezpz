# 🍋 ezpz

> Write once, run anywhere.

`ezpz` makes distributed PyTorch launches portable across NVIDIA, AMD, Intel,
MPS, and CPU—with zero-code changes and guardrails for HPC schedulers.

- Automatic accelerator detection + `ezpz.setup_torch()` to pick the right
  backend.
- PBS/Slurm-aware launcher with automatic fallback to `mpirun` when no
  scheduler is present.
- Batteries-included examples 
  (ViT, FSDP, tensor-parallel, diffusion, HF Trainer).
- Metrics that just work, with utilities for {tracking, recording, plotting},
  and support for syncing via Weights \&  Biases via `ezpz.History`.

See the [docs](https://saforem2.github.io/ezpz) and the
[Repository Guidelines](AGENTS.md) for more detail.

> [!NOTE]
> **Quickstart:**
>
> If you already have a Python environment with
> {`torch`, `mpi4py`} installed, you can:
>
> ```bash
> # pip install uv first, if needed
> uv run --active \
>     --no-build-isolation \
>     --python=$(which python3) \
>     --with="git+https://github.com/saforem2/ezpz" \
>     ezpz test
> ```

## 🐣 Getting Started

1. Setup Python environment:

   \[**Optional**\] Note: this is _technically_ optional (but recommended).

   If you already have a
   Python environment with {`torch`, `mpi4py`} installed and would prefer to use that, skip directly to (2.)

   Otherwise, we can use the provided 
   [src/ezpz/bin/utils.sh](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)[^bitly]
   to setup our environment:

   ```bash
   # 1) Load appropriate modules (@ LCFs*) and setup virtual environment
   source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
   ```

   - <details closed><summary><b>[Details]</b></summary>
     NOTE: This is optional but strongly recommended.  
     If running behind a job scheduler (e.g. PBS/Slurm) at any of {ALCF, OLCF,
     NERSC}, this will automatically load the appropriate modules and use these
     to bootstrap a virtual environment.
   </details>

2. Install `ezpz`:

   ```bash
   # 2) Install the latest ezpz
   uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
   ```

   - If you don't already have PyTorch or `mpi4py` installed,
     you can specify these as additional dependencies:

     ```bash
     uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz[torch,mpi]"
     ```

   - ... _or try without installing_!

     ```bash
     # ...or try without installing into your env:
     uv run --with "git+https://github.com/saforem2/ezpz" ezpz doctor
     ```

3. Run distributed smoke test:

   ```bash
   ezpz test
   ```

   Which will train a simple model on the MNIST dataset using PyTorch + DDP.

   > See
   > \[[ezpz test | W\&B Report](https://api.wandb.ai/links/aurora_gpt/q56ai28l)\]
   > for sample output and details of metric tracking.

[^bitly]: The <https://bit.ly/ezpz-utils> URL is just a short link for
    convenience that actually points to
    <https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh>

## 🏃‍♂️‍➡️ CLI Entry Points

### 🩺 Health check with `ezpz doctor`

Health-check your environment and ensure that `ezpz` is installed correctly

```bash
ezpz doctor
ezpz doctor --json   # machine-friendly output for CI
```

Checks MPI, scheduler detection, Torch import + accelerators, and wandb
readiness, returning non-zero on errors.

<details closed><summary>Output:</summary>

```bash
; ezpz doctor
== Runtime Context ==
User: foremans
Machine: x86_64
Hostname: x4712c1s0b0n0
PBS_JOBID: 8227686.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
PBS_NODEFILE: /var/spool/pbs/aux/8227686.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
ezpz: v0.9.0 (/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/src/ezpz/__init__.py)

Module list:
  gcc-runtime/13.3.0-ghotoln <H>
  gmp/6.3.0-mtokfaw <H>
  mpfr/4.2.1-gkcdl5w <H>
  mpc/1.3.1-rdrlvsl <H>
  gcc/13.3.0
  oneapi/release/2025.2.0
  libxml2/2.13.5
  hwloc/2.12.1-level-zero
  mpich/opt/develop-git.6037a7a
  libfabric/1.22.0
  cray-pals/1.8.0
  cray-libpals/1.8.0
  hdf5/1.14.6
  frameworks/2025.2.0

Base environment:
  Mamba root: /lus/flare/datascience/Aurora_deployment/foremans/micromamba

Active environment:
  Virtual env: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/venvs/aurora/ezpz-distributed-metrics-aurora_frameworks-2025.2.0
  Conda env: /opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0 (/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0)

Python: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/venvs/aurora/ezpz-distributed-metrics-aurora_frameworks-2025.2.0/bin/python3 (3.10.14)
PyTorch: 2.8.0a0+gitba56102 (/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/__init__.py)

Scheduler resources:
  NHOSTS: 2
  NGPU_PER_HOST: 12
  NGPUS: 24

[✅ OKAY] [mpi      ]: mpi4py import succeeded and an MPI launcher was found.
[✅ OKAY] [wandb    ]: wandb authentication provided in '~/.netrc' Should be all set.
[✅ OKAY] [torch    ]: PyTorch detected at least one accelerator.
[✅ OKAY] [hostfile ]: HOSTFILE=/var/spool/pbs/aux/8227686.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov (2 hosts).
[✅ OKAY] [scheduler]: Detected active scheduler: PBS.
```

</details>

### ✅ ezpz test

Run the bundled test suite (great for first-time validation).

```bash
ezpz test
uv run --with "git+https://github.com/saforem2/ezpz" ezpz test
```


### 🚀 `ezpz` Launch

Single entry point for distributed jobs.

`ezpz` detects PBS/Slurm automatically and falls back to `mpirun`, forwarding
useful environment variables so your script behaves the same on laptops and
clusters.

Add your own args to any command (`--config`, `--batch-size`, etc.) and `ezpz`
will propagate them through the detected launcher.

Use the provided

```bash
ezpz launch <cmd-to-launch>
```

to automatically launch `<cmd-to-launch>` across all available[^schedulers]
accelerators.

- Example(s):

  - arbitrary command(s):
 
    ```bash
    ezpz launch hostname
    ```

  - launch arbitrary Python string:
 
    ```bash
    ezpz launch python3 -c 'import ezpz; ezpz.setup_torch()'
    ```


[^schedulers]: By default, this will detect if we're running behind a job scheduler (e.g. PBS or Slurm).
    If so, we automatically determine the specifics of the currently active job; 
    explicitly, this will determine:

    1. The number of available nodes
    2. How many GPUs are present on each of these nodes
    3. How many GPUs we have _total_

    It will then use this information to automatically construct the appropriate 
    {`mpiexec`, `srun`} command to launch, and finally, execute the launch cmd.

#### 📝 Ready-to-go Examples:

1. **Simple fully connected (`torch.nn.Linear`) example with MNIST**:

   See [src/ezpz/test_dist.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)

    ```bash
    ezpz launch python3 -m ezpz.test_dist
    ```

1. **ViT + {DDP, FSDP}**:

   See [src/ezpz/examples/vit.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)

   ```bash
   ezpz launch python3 -m ezpz.examples.vit --compile # --fsdp
   ```

1. **FSDP Example**:

   See [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)

   ```bash
   ezpz launch python3 -m ezpz.examples.fsdp
   ```

1. **FSDP Example with Tensor Parallelism**:

   See [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)

   ```bash
   ezpz launch python3 -m ezpz.examples.fsdp_tp \
       --tp=2 \
       --epochs=5 \
       --batch-size=2 \
       --dataset=eliplutchok/fineweb-small-sample \
   ```

1. **Diffusion LLM Model**:

   See [src/ezpz/examples/diffusion.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py)

   ```bash
   ezpz launch python3 -m ezpz.examples.diffusion --batch_size 1 --hf_dataset stanfordnlp/imdb
   ```

1. **Finetuning an LLM with HF Trainer**:

   See [src/ezpz/examples/hf_trainer.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)

   ```bash
   ezpz launch python3 -m ezpz.examples.hf_trainer \
     --streaming \
     --dataset_name=eliplutchok/fineweb-small-sample \
     --tokenizer_name meta-llama/Llama-3.2-1B \
     --model_name_or_path meta-llama/Llama-3.2-1B \
     --bf16=true \
     --do_train=true \
     --do_eval=true \
     --report-to=wandb \
     --logging-steps=1 \
     --include-tokens-per-second=true \
     --max-steps=50000 \
     --include-num-input-tokens-seen=true \
     --optim=adamw_torch \
     --logging-first-step \
     --include-for-metrics='inputs,loss' \
     --max-eval-samples=50 \
     --per_device_train_batch_size=1 \
     --block-size=8192 \
     --gradient_checkpointing=true \
     --fsdp=shard_grad_op
   ```

## Bring ezpz to your application

- **Accelerator detection:** `ezpz.get_torch_device_type()` and 
  `ezpz.setup_torch()` normalize CUDA/XPU/MPS/CPU selection.
- **Scheduler smarts:** detects PBS/Slurm automatically;
  otherwise falls back to `mpirun` with sensible `--env` forwarding.
- **Observability:** structured logs, host/machine detection, 
  and launch-time filters to keep noisy scheduler output out of your logs.

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
  ezpz launch python3 -m your_app.train --additional-args ...
  ```


## Track metrics with `ezpz.History`

Capture metrics across all ranks, persist JSONL, generate text/PNG plots, and (when configured) log to Weights & Biases—no extra code on worker ranks.

```python
import ezpz
from ezpz import History

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
```


## More to explore

- Examples live under `ezpz.examples.*`—copy them or extend them for your workloads.
- Want richer visuals or profiling? 
  - Change the default plot marker used in the text based plots:

    ```bash
    # highest resolution, may not be supported in all terminals
    EZPZ_TPLOT_MARKER="braille" ezpz test
    # next-best resolution, more widely supported
    EZPZ_TPLOT_MARKER="fhd" ezpz test
    ```

- Stuck? Check the [docs](https://saforem2.github.io/ezpz), or run `ezpz doctor` for actionable hints.

- See my recent talk on:
  [**_LLMs on Aurora_: Hands On with `ezpz`**](https://saforem2.github.io/ezpz/slides-2025-05-07/)
  for a detailed walk-through containing examples and use cases.

  - [🎥 YouTube](https://www.youtube.com/watch?v=15ZK9REQiBo)
  - [Slides (html)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/)
  - [Slides (reveal.js)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/slides)
