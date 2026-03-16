# Minimal Training (Synthetic Data)

!!! warning "Deprecated"

    This example is deprecated and may be removed in a future release.
    See [`ezpz.examples.test`](test.md) for the recommended smoke test.

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

## Source

<details closed><summary><code>src/ezpz/examples/minimal.py</code></summary>

```python title="src/ezpz/examples/minimal.py"
--8<-- "src/ezpz/examples/minimal.py"
```

</details>

## Code Walkthrough

### Setup

The `setup()` function initializes the distributed runtime, builds the model,
and wraps it for DDP when running on multiple GPUs.
All hyperparameters come from **environment variables** — no argparse needed.

```python title="src/ezpz/examples/minimal.py" linenums="94"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def setup():
    """Initialize distributed runtime, model, and optimizer."""
    rank = ezpz.setup_torch(seed=int(os.environ.get("SEED", 0)))
    if os.environ.get("WANDB_DISABLED", False):
        logger.info("WANDB_DISABLED is set, not initializing wandb")
    elif rank == 0:
        try:
            _ = ezpz.setup_wandb(
                project_name=os.environ.get("PROJECT_NAME", "ezpz.examples.minimal")
            )
        except Exception:
            logger.exception("Failed to initialize wandb, continuing without it")
    device_type = ezpz.get_torch_device_type()
    from ezpz.models.minimal import SequentialLinearNet

    model = SequentialLinearNet(
        input_dim=int((os.environ.get("INPUT_SIZE", 128))),
        output_dim=int(os.environ.get("OUTPUT_SIZE", 128)),
        sizes=[
            int(x)
            for x in os.environ.get(
                "LAYER_SIZES", "256,512,1024,2048,1024,512,256,128"
            ).split(",")
        ],
    )
    model.to(device_type)
    model.to((os.environ.get("DTYPE", torch.bfloat16)))
    try:
        from ezpz.utils import model_summary

        model_summary(model)
    except Exception:
        logger.exception("Failed to summarize model")
    logger.info(f"{model=}")
    optimizer = torch.optim.Adam(model.parameters())
    if ezpz.get_world_size() > 1:
        model = ezpz.distributed.wrap_model_for_ddp(model)

    return model, optimizer
```

### Training Loop

The `train()` function runs a synthetic reconstruction loop with
`history.update()` for metric tracking. Forward and backward timings
(`dtf`, `dtb`) are tracked separately.

```python title="src/ezpz/examples/minimal.py" linenums="64"
    for step in range(int(os.environ.get("TRAIN_ITERS", 500))):
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
        ):
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
                summary = history.update(
                    {
                        "iter": step,
                        "loss": loss.item(),
                        "dt": dtf + dtb,
                        "dtf": dtf,
                        "dtb": dtb,
                    }
                )
            if step % print_freq == 0 and step > warmup:
                logger.info(summary)
    return history
```

### Finalization

After training, rank 0 calls `history.finalize()` to save metrics and
print a timing report:

```python title="src/ezpz/examples/minimal.py" linenums="139"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def main():
    """Entrypoint for launching the minimal synthetic training example."""
    t0 = time.perf_counter()
    model, optimizer = setup()
    t_setup = time.perf_counter()
    module_name = "ezpz.examples.minimal"
    outdir = get_example_outdir(module_name)
    logger.info("Outputs will be saved to %s", outdir)
    train_start = time.perf_counter()
    history = train(model, optimizer, outdir)
    train_end = time.perf_counter()
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            outdir=outdir,
            run_name=module_name,
            dataset_fname="train",
            verbose=False,
        )
        logger.info(f"{dataset=}")
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
