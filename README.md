# 🍋 ezpz

> Write once, run anywhere.

`ezpz` makes distributed PyTorch launches portable across any supported
hardware {NVIDIA, AMD, Intel, MPS, CPU} with **zero code changes**.

## Features

- **Multi-hardware** — automatic device detection and backend selection
  (CUDA/NCCL, XPU/CCL, MPS, CPU/Gloo)
- **Zero code changes** — same script runs on a laptop, a single GPU, or a
  thousand-node supercomputer
- **HPC integration** — native PBS and SLURM support with automatic hostfile
  discovery and rank assignment
- **Metric tracking** — built-in `History` class for recording, plotting, and
  saving training metrics
- **CLI tools** — `ezpz launch`, `ezpz test`, `ezpz doctor` for launching
  jobs, smoke-testing, and diagnostics

## Quick Install

```bash
uv pip install git+https://github.com/saforem2/ezpz
```

## Quick Start

```python
import torch
import ezpz

rank = ezpz.setup_torch()           # auto-detects device + backend
device = ezpz.get_torch_device()
model = torch.nn.Linear(128, 10).to(device)
model = ezpz.wrap_model(model)       # FSDP (default)

# Multi-dim parallelism (TP/PP/CP) on XPU? Use ezpz.init_device_mesh_safe
# instead of torch's init_device_mesh — works around xccl's missing
# split_group on Aurora/Sunspot. See https://ezpz.cool/troubleshooting/.
```

```bash
# Same command everywhere -- Mac laptop, NVIDIA cluster, Intel Aurora:
ezpz launch python3 train.py
```

For a side-by-side diff against the equivalent raw-torch boilerplate, see
the [API Cheat Sheet](https://ezpz.cool/quickstart/#api-cheat-sheet).

## CLI

```bash
ezpz launch python3 train.py    # launch distributed training
ezpz submit -N 2 -q debug -- python3 train.py   # submit batch job to PBS/SLURM
ezpz test                       # smoke-test your setup
ezpz benchmark                  # run + compare example benchmarks
ezpz doctor                     # diagnose environment issues
```

## Why ezpz?

Compared to the alternatives:

- **vs raw `torchrun` / `mpirun` / `srun`**: one launcher that detects your
  scheduler, builds the right command, and works on a laptop too.
- **vs `accelerate`**: lower surface area, no config files, designed for
  HPC schedulers from the ground up rather than retrofitted.
- **vs `DeepSpeed`**: not an alternative — `ezpz` wraps your distributed
  init so you can still use DeepSpeed (or anything else) underneath.

See the [full comparison](https://ezpz.cool/compare/) for details.

## Documentation

Full documentation is available at [**ezpz.cool**](https://ezpz.cool).

Useful entry points:

- 🏃‍♂️ [Quickstart](https://ezpz.cool/quickstart/) — install → script → launch in 5 minutes
- 🎓 [Distributed Training Tutorial](https://ezpz.cool/guides/distributed-training/) — progressive hello-world → FSDP+TP
- 🍳 [Recipes](https://ezpz.cool/recipes/) — copy-pasteable patterns (checkpointing, gradient accumulation, MFU tracking)
- 🔧 [Troubleshooting](https://ezpz.cool/troubleshooting/) — XPU FSDP2 hangs, NCCL/CCL errors, scheduler issues
- 📝 [Examples](https://ezpz.cool/examples/) — runnable end-to-end (FSDP, ViT, Diffusion, HF Trainer)
