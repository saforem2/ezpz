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
model = ezpz.wrap_model(model)       # DDP by default
```

```bash
# Same command everywhere -- Mac laptop, NVIDIA cluster, Intel Aurora:
ezpz launch python3 train.py
```

## CLI

```bash
ezpz launch python3 train.py    # launch distributed training
ezpz test                       # smoke-test your setup
ezpz doctor                     # diagnose environment issues
```

## Documentation

Full documentation is available at [**ezpz.cool**](https://ezpz.cool).
