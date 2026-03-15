# 📖 Getting Started Guide

A lean walkthrough: install, write a script, launch it, track metrics.

For detailed examples, terminal output, and the full API cheat sheet, see the
[Quickstart](./quickstart.md).

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

## ⚡ Your First Distributed Script

```python title="train.py"
import torch
import ezpz

rank, world_size, local_rank = ezpz.setup_torch()
device = ezpz.get_torch_device()

model = torch.nn.Linear(128, 64).to(device)
model = ezpz.wrap_model(model)
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
group, wraps your model in DDP, and assigns each rank to the correct device.

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

On HPC systems, just allocate resources first:

=== "PBS"

    ```bash
    qsub -A my_project -q gpu -l select=2 -l walltime=00:30:00 -I
    ezpz launch python3 train.py
    ```

=== "SLURM"

    ```bash
    salloc --nodes=2 --ntasks-per-node=4 --gpus-per-node=4
    ezpz launch python3 train.py
    ```

For pass-through launcher flags, custom hostfiles, and advanced usage, see
the [Quickstart launcher section](./quickstart.md#scheduler-aware-launcher-ezpz-launch)
and the [CLI reference](./cli/launch/index.md).

## 📊 Track Metrics

Use the built-in `History` class to record, save, and plot training metrics:

```python
from ezpz.history import History

history = History()

for step in range(100):
    # ... training loop ...
    metrics = {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
    history.update(metrics)

history.finalize(outdir="./outputs")  # saves dataset + generates plots
```

!!! tip "What `finalize` produces"

    Calling `history.finalize()` writes a summary dataset and generates
    loss curves and other plots — ready for inspection or inclusion in
    reports. See the [Quickstart complete example](./quickstart.md#complete-example)
    for sample output with terminal plots.

## 🔗 Next Steps

- **[Quickstart](./quickstart.md)** — full walkthrough with diff cheat sheet, complete example, and terminal output
- **[Examples](./examples/index.md)** — end-to-end training scripts (FSDP, ViT, Diffusion, etc.)
- **[CLI Reference](./cli/index.md)** — full `ezpz launch` usage and flags
- **[Configuration](./configuration.md)** — environment variables and config dataclasses
- **[Architecture](./architecture.md)** — how `ezpz` works under the hood
