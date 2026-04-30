# Minimal Training (Synthetic Data)

!!! danger "Deprecated — use `ezpz.examples.test` instead"

    This example is deprecated and will be removed in a future release.
    **New users should start with [`ezpz.examples.test`](test.md)**, which
    covers the same concepts with better defaults.

    The walkthrough below is preserved for reference only.

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


<details closed markdown><summary><strong>Imports and Logger</strong></summary>

Standard imports plus `ezpz` for distributed training utilities. The rank-aware logger ensures only rank 0 prints by default.

```python title="src/ezpz/examples/minimal.py:15:25"
--8<-- "src/ezpz/examples/minimal.py:15:25"
```

</details>

<details closed markdown><summary><strong><code>train()</code></strong></summary>

The `@ezpz.timeitlogit` decorator logs wall-clock time for the entire function. Inside, the model is unwrapped if DDP-wrapped, and an `ezpz.History` is created to track metrics to a JSONL file.

```python title="src/ezpz/examples/minimal.py:28:67"
--8<-- "src/ezpz/examples/minimal.py:28:67"
```

The training loop generates random input, computes a reconstruction loss, and records forward/backward timings separately. Metrics are logged via `history.update()` after a warmup period.

```python title="src/ezpz/examples/minimal.py:68:103"
--8<-- "src/ezpz/examples/minimal.py:68:103"
```

</details>

<details closed markdown><summary><strong><code>setup()</code></strong></summary>

Initializes the distributed backend via `ezpz.setup_torch()`, optionally sets up W&B on rank 0, builds a `SequentialLinearNet` with env-var-driven dimensions, and wraps the model for DDP when running multi-GPU.

```python title="src/ezpz/examples/minimal.py" linenums="93"
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
        # from torch.nn.parallel import DistributedDataParallel as DDP
        #
        # model = DDP(model, device_ids=[ezpz.get_local_rank()])

    return model, optimizer
```

</details>

<details closed markdown><summary><strong><code>main()</code></strong></summary>

Orchestrates the full run: calls `setup()`, runs `train()`, then finalizes the history on rank 0 to persist metrics. Timing breakdowns are logged and optionally sent to W&B.

```python title="src/ezpz/examples/minimal.py" linenums="138"
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
    timings = {
        "main/setup": t_setup - t0,
        "main/train": train_end - train_start,
        "main/total": train_end - t0,
        "timings/training_start": train_start - t0,
        "timings/train_duration": train_end - train_start,
        "timings/end-to-end": train_end - t0,
    }
    logger.info("Timings: %s", timings)
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    (f"timings/{k}" if not k.startswith("timings/") else k): v
                    for k, v in timings.items()
                }
            )
    except Exception:
        logger.debug("Skipping wandb timings log")
```

</details>

<details closed markdown><summary><strong><code>__main__</code> Guard</strong></summary>

Prints a usage message on `--help`, otherwise calls `main()`.

```python title="src/ezpz/examples/minimal.py:179:203"
--8<-- "src/ezpz/examples/minimal.py:179:203"
```

</details>

## MFU Tracking

This example reports per-step **TFLOPS** and **MFU** (Model FLOPS
Utilization) alongside loss/timing metrics. The model FLOPS are
counted once at startup via [`try_estimate`](../recipes.md#mfu-tracking),
and `compute_mfu` divides by the device's peak BF16 throughput
(see [`ezpz.flops`](../python/Code-Reference/flops.md) for details).

```python
from ezpz.flops import compute_mfu, try_estimate

_model_flops = try_estimate(model, (bsize, isize))
# ...
metrics["tflops"] = _model_flops / dt / 1e12
metrics["mfu"] = compute_mfu(_model_flops, dt)
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

## Example Output (Sunspot, 2 nodes × 12 ranks = 24 total)

```text
[2026-04-29 18:03:57][I][ezpz/distributed:1536:_setup_ddp] init_process_group: master_addr=x1921c6s0b0n0, master_port=53741, world_size=24, rank=0, backend=xccl, timeout=1:00:00
[2026-04-29 18:04:27][I][examples/minimal:150:main] Outputs will be saved to /tmp/outputs/ezpz.examples.minimal/2026-04-29-180355
[2026-04-29 18:04:37][I][examples/minimal:102:train] iter=20 loss=708.216064 dt=0.004312 dtf=0.000920 dtb=0.003391 tflops=0.496617 mfu=0.166545 ...
[2026-04-29 18:04:38][I][examples/minimal:102:train] iter=40 loss=676.927185 dt=0.004310 dtf=0.000922 dtb=0.003388 tflops=0.496832 mfu=0.166616 ...
[2026-04-29 18:04:38][I][examples/minimal:102:train] iter=60 loss=687.406372 dt=0.004286 dtf=0.000918 dtb=0.003368 tflops=0.499541 mfu=0.167525 ...
...
[2026-04-29 18:04:39][I][examples/minimal:102:train] iter=180 loss=676.073486 dt=0.004343 dtf=0.000921 dtb=0.003422 tflops=0.492990 mfu=0.165328 ...
```

Notice each step takes ~4 ms (forward `dtf=0.9 ms` + backward `dtb=3.4 ms`),
giving ~0.5 TFLOPS per device — about 0.17% MFU on PVC. This is a tiny
synthetic MLP, so MFU is dominated by collective overhead and kernel
launch latency rather than compute.

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.minimal --help
Usage:
PRINT_ITERS=100 TRAIN_ITERS=1000 INPUT_SIZE=128 OUTPUT_SIZE=128 LAYER_SIZES="'128,256,128'" ezpz launch python3 -m ezpz.examples.minimal
```

</details>
