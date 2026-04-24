---
# icon: lucide/citrus
hide:
  - navigation
  #  - toc
---

# 🍋 ezpz

> _Write once, run anywhere_.

`ezpz` makes distributed PyTorch code portable across any supported hardware
{NVIDIA, AMD, Intel, MPS, CPU} with **zero code changes**.
Built for researchers and engineers running distributed PyTorch on HPC
systems (ALCF, NERSC, OLCF) or local workstations.

This lets us write Python applications that can be run _anywhere_, _at any
scale_; with native job scheduler (PBS, Slurm)[^lcfs] integration and graceful
fallbacks for running locally[^dev] on Mac, Linux machines.

[^lcfs]: With first class support for all of the major HPC Supercomputing
    centers (e.g. ALCF, OLCF, NERSC)

[^dev]: This is particularly useful if you'd like to run development /
    debugging experiments locally

## 🤔 Why `ezpz`?

Distributed PyTorch requires boilerplate that varies by hardware, backend, and
job scheduler. `ezpz` replaces all of it.

=== "With ezpz"

    ``` python title="train.py" linenums='0'
    import ezpz
    import torch

    rank = ezpz.setup_torch()           # auto-detects device + backend
    device = ezpz.get_torch_device()
    model = torch.nn.Linear(128, 10).to(device)
    model = ezpz.wrap_model(model)       # FSDP (default)
    ```

    ```bash linenums='0' title='launch.sh'
    # Same command everywhere -- Mac laptop, NVIDIA cluster, Intel Aurora:
    ezpz launch python3 train.py
    ```

=== "Without ezpz"

    ```python title="train.py"
    import os, torch, torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    backend = "gloo"
    device_type = "cpu"
    if torch.cuda.is_available():
        backend = "nccl"
        device_type = "cuda"
    elif torch.xpu.is_available():
        backend = "xccl"
        device_type = "xpu"

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
    else:
        device = torch.device("cpu")

    model = torch.nn.Linear(128, 10).to(device)
    model = DDP(
        model,
        device_ids=[local_rank] if backend in ["nccl", "xccl"] else None
    )
    ```

    ```bash linenums='0'
    # Different launch per {scheduler, cluster}:
    mpiexec -np 8 --ppn 4 python3 train.py      # Polaris    @ ALCF  [NVIDIA / PBS]
    mpiexec -np 24 --ppn 12 python3 train.py    # Aurora     @ ALCF  [INTEL / PBS]
    srun -N 2 -n 8 python3 train.py             # Frontier   @ ORNL  [AMD / SLURM]
    srun -N 2 -n 8 python3 train.py             # Perlmutter @ NERSC [NVIDIA / SLURM]
    torchrun --nproc_per_node=4 train.py        # Generic    @ ???   [NVIDIA / ???]
    ```

## 🏃‍♂️ Try it out!

No cluster required — this runs on a laptop[^uv]:

```bash linenums='0'
uv pip install "git+https://github.com/saforem2/ezpz"
```

[^uv]: if you _still_ haven't installed uv:

    ```bash linenums="0"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

```python title="hello.py" linenums='0'
import ezpz
rank = ezpz.setup_torch()
print(f"Hello from rank {rank} on {ezpz.get_torch_device()}!")
ezpz.cleanup()
```

=== "Laptop (Mac)"

    ```bash linenums='0'
    $ python3 hello.py
    Hello from rank 0 on mps!

    $ ezpz launch python3 hello.py
    Hello from rank 0 on mps!
    Hello from rank 1 on mps!
    ```

=== "Aurora (ALCF)"

    ```bash linenums='0'
    $ ezpz launch python3 hello.py
    Using [24 / 24] available "xpu" devices !!
    Hello from rank 0 on xpu!
    ```

=== "Polaris (ALCF)"

    ```bash linenums='0'
    $ ezpz launch python3 hello.py
    Using [8 / 8] available "cuda" devices !!
    Hello from rank 0 on cuda!
    ```

=== "Perlmutter (NERSC)"

    ```bash linenums='0'
    $ ezpz launch python3 hello.py
    Using [8 / 8] available "cuda" devices !!
    Hello from rank 0 on cuda!
    ```

Ready to get started? See the [Quick Start](./quickstart.md).

## 👀 Overview

`ezpz` is, at its core, a Python library that provides a variety of utilities
for both _writing_ and _launching_ distributed PyTorch applications.

These can be broken down (~roughly) into:

1. 🐍 [**Python library**](./python/Code-Reference/index.md): `import ezpz`
   Python API for writing hardware-agnostic, distributed PyTorch code.

1. 🧰 [**CLI**](./cli/index.md): `ezpz <command>`
   Utilities for launching and managing distributed PyTorch jobs:
    - 🚀 [`ezpz launch`](./cli/launch/index.md): Launch commands with _automatic
      **job scheduler** detection_ (PBS, Slurm)
    - 📤 [`ezpz submit`](./cli/submit.md): Submit batch jobs to PBS or Slurm
    - 💯 [`ezpz test`](./cli/test.md): Run simple distributed smoke test
    - 📊 [`ezpz benchmark`](./cli/benchmark.md): Run and compare example benchmarks
    - 🩺 [`ezpz doctor`](./cli/doctor.md): Health check your environment

## ✨ Features

- **Automatic distributed initialization** — [`setup_torch()`](./python/Code-Reference/distributed.md#ezpz.distributed.setup_torch) detects device + backend
- **Universal launcher** — [`ezpz launch`](./cli/launch/index.md) auto-detects PBS, Slurm, or falls back to `mpirun`
- **Batch job submission** — [`ezpz submit`](./cli/submit.md) generates and submits PBS/Slurm job scripts
- **Model wrapping** — [`wrap_model()`](./python/Code-Reference/distributed.md#ezpz.distributed.wrap_model) for DDP, FSDP, or FSDP+TP with one call
- **Multi-backend experiment tracking** — [`History`](./history.md) with distributed statistics and automatic dispatch to W&B, MLflow, and CSV
- **Environment diagnostics** — [`ezpz doctor`](./cli/doctor.md) checks your setup
- **Cross-backend timing** — [`synchronize()`](./python/Code-Reference/distributed.md#ezpz.distributed.synchronize) works on CUDA, XPU, MPS, and CPU

## 🔗 Next Steps

- **[Quick Start](./quickstart.md)** — install, write a script, launch it
- **[Distributed Training Guide](./guides/distributed-training.md)** — progressive tutorial from hello world to production
- **[Recipes](./recipes.md)** — copy-pasteable patterns for common tasks
- **[End-to-End Walkthrough](./reference.md)** — full runnable example with real terminal output
- **[Experiment Tracking](./history.md)** — `History` guide: distributed stats, multi-backend logging, plots
- **[Examples](./examples/index.md)** — end-to-end training scripts (FSDP, ViT, Diffusion, etc.)
- **[FAQ](./notes/faq.md)** — common questions and troubleshooting
- **[Architecture](./architecture.md)** — how `ezpz` works under the hood
