# Minimal Training (Synthetic Data)

The simplest ezpz example — trains an MLP to reconstruct random inputs
using env-var configuration. No dataset downloads required.

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`wrap_model_for_ddp()`][ezpz.distributed.wrap_model_for_ddp] — Wrap model for DDP
    - [`History`][ezpz.history.History] — Track and finalize metrics
    - [`get_logger()`][ezpz.log.config.get_logger] — Rank-aware logging

See:

- 📘 [examples/minimal](../python/Code-Reference/examples/minimal.md)
- 🐍 [src/ezpz/examples/minimal.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/minimal.py)

```bash
ezpz launch python3 -m ezpz.examples.minimal
```

## What to Expect

You'll see per-step loss and timing printed at regular intervals.
Loss should decrease as the model learns to reconstruct random vectors.
On completion, a `History` dataset is finalized with timing summaries.

## Code Walkthrough

### Setup

The `setup()` function initializes the distributed runtime, builds the model,
and wraps it for DDP when running on multiple GPUs:

```python
def setup():
    rank = ezpz.setup_torch(seed=int(os.environ.get("SEED", 0)))
    # Optional W&B initialization (rank 0 only)
    if rank == 0:
        ezpz.setup_wandb(project_name="ezpz.examples.minimal")

    model = SequentialLinearNet(
        input_dim=int(os.environ.get("INPUT_SIZE", 128)),
        output_dim=int(os.environ.get("OUTPUT_SIZE", 128)),
        sizes=[int(x) for x in os.environ.get(
            "LAYER_SIZES", "256,512,1024,2048,1024,512,256,128"
        ).split(",")],
    )
    model.to(device_type)
    optimizer = torch.optim.Adam(model.parameters())

    if ezpz.get_world_size() > 1:
        model = ezpz.distributed.wrap_model_for_ddp(model)

    return model, optimizer
```

All hyperparameters come from **environment variables** — no argparse needed.
This pattern keeps the script minimal while still being configurable.

### Training Loop

The `train()` function runs a synthetic reconstruction loop with
`history.update()` for metric tracking:

```python
for step in range(int(os.environ.get("TRAIN_ITERS", 500))):
    with torch.autocast(device_type=device_type, dtype=dtype):
        t0 = time.perf_counter()
        x = torch.rand((bsize, isize), dtype=dtype).to(device_type)
        y = model(x)
        loss = ((y - x) ** 2).sum()
        dtf = (t1 := time.perf_counter()) - t0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        dtb = time.perf_counter() - t1
        if step % log_freq == 0 and step > warmup:
            summary = history.update({
                "iter": step,
                "loss": loss.item(),
                "dt": dtf + dtb,
                "dtf": dtf,
                "dtb": dtb,
            })
```

The forward and backward timings (`dtf`, `dtb`) are tracked separately,
and `autocast` handles mixed-precision automatically.

### Finalization

After training, rank 0 calls `history.finalize()` to save metrics and
print a timing report:

```python
def main():
    model, optimizer = setup()
    history = train(model, optimizer, outdir)
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            outdir=outdir,
            run_name="ezpz.examples.minimal",
            dataset_fname="train",
        )
    logger.info("Timings: %s", timings)
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_ITERS` | `500` | Number of training iterations |
| `BATCH_SIZE` | `64` | Batch size |
| `INPUT_SIZE` | `128` | Input dimension |
| `OUTPUT_SIZE` | `128` | Output dimension |
| `LAYER_SIZES` | `256,512,...,128` | Comma-separated hidden layer sizes |
| `DTYPE` | `bfloat16` | Model dtype |
| `SEED` | `0` | Random seed |
| `LOG_FREQ` | `1` | Log metrics every N steps |
| `PRINT_FREQ` | `10` | Print summary every N steps |
| `WARMUP_ITERS` | `10` | Steps to skip before recording metrics |

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.minimal --help
Usage:
PRINT_ITERS=100 TRAIN_ITERS=1000 INPUT_SIZE=128 OUTPUT_SIZE=128 LAYER_SIZES="'128,256,128'" ezpz-launch -m ezpz.examples.minimal
```

</details>
