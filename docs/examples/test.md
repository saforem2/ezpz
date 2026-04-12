# Train MLP with DDP on MNIST

Train a simple fully connected (`torch.nn.Linear`) network using DDP
on the MNIST dataset.

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`wrap_model()`][ezpz.distributed.wrap_model] — Wrap model for DDP
    - [`cleanup()`][ezpz.distributed.cleanup] — Tear down the process group

See: 📘 [docs](../python/Code-Reference/examples/test.md),
🐍 [source](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py)

```bash
# or, equivalently: ezpz test
ezpz launch python3 -m ezpz.examples.test
```

## Source

<details closed><summary><code>src/ezpz/examples/test.py</code></summary>

```python title="src/ezpz/examples/test.py"
--8<-- "src/ezpz/examples/test.py"
```

</details>

## Code Walkthrough


<details closed markdown><summary><strong>Imports and Constants</strong></summary>

`DDP` is imported directly because this example uses vanilla data-parallel
wrapping (no FSDP). `build_test_parser` from `ezpz.cli.flags` provides a
shared argument parser so CLI flags like `--epochs`, `--lr`, and `--compile`
are consistent across all built-in examples. `get_profiling_context` lets
the example optionally run under PyTorch Profiler or pyinstrument without
changing the training loop.

```python title="src/ezpz/examples/test.py" linenums="1"
#!/usr/bin/env python3
"""
examples/test.py
"""

import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from xarray import Dataset

import ezpz
import ezpz.distributed
from ezpz.configs import PathLike
from ezpz.cli.flags import build_test_parser
from ezpz.profile import get_profiling_context

START_TIME = time.perf_counter()  # start time

# noqa: E402
warnings.filterwarnings("ignore")

ModelOptimizerPair = tuple[torch.nn.Module, torch.optim.Optimizer]

logger = ezpz.get_logger(__name__)
```

`START_TIME` is captured at import time so end-to-end wall-clock timings
can be reported later.

</details>

<details closed markdown><summary><strong><code>MODEL_PRESETS</code></strong></summary>

Named presets that bundle batch size, iteration count, logging frequency,
and layer sizes into a single `--model` flag.

```python title="src/ezpz/examples/test.py" linenums="37"
MODEL_PRESETS = {
    "debug": {
        "batch_size": 16,
        "train_iters": 20,
        "log_freq": 1,
        "print_freq": 1,
        "layer_sizes": [128, 64],
    },
    "small": {
        "batch_size": 64,
        "train_iters": 200,
        "log_freq": 1,
        "print_freq": 10,
        "layer_sizes": [256, 128, 64],
    },
    "medium": {
        "batch_size": 128,
        "train_iters": 200,
        "log_freq": 1,
        "print_freq": 10,
        "layer_sizes": [512, 256, 128],
    },
    "large": {
        "batch_size": 256,
        "train_iters": 400,
        "log_freq": 1,
        "print_freq": 10,
        "layer_sizes": [1024, 512, 256],
    },
}
MODEL_PRESET_FLAGS = {
    "batch_size": ["--batch-size"],
    "train_iters": ["--train-iters", "--train_iters"],
    "log_freq": ["--log-freq", "--log_freq"],
    "print_freq": ["--print-freq", "--print_freq"],
    "layer_sizes": ["--layer-sizes"],
}
```

`MODEL_PRESET_FLAGS` maps each preset field to its CLI flags so that
explicitly provided flags take precedence over the preset defaults.

</details>

<details closed markdown><summary><strong><code>TrainConfig</code> Dataclass</strong></summary>

All runtime configuration lives in a single dataclass. `__post_init__`
creates the output directory, broadcasts the timestamp across ranks, and
builds the profiling context manager.

```python title="src/ezpz/examples/test.py" linenums="103"
@dataclass
class TrainConfig:
    """Runtime configuration for the `ezpz.examples.test` distributed smoke test."""

    warmup: int
    tp: int
    pp: int
    cp: int
    batch_size: int
    input_size: int
    output_size: int
    train_iters: int
    log_freq: int
    backend: str
    dtype: str
    print_freq: int
    pyinstrument_profiler: bool
    pytorch_profiler: bool
    pytorch_profiler_wait: int
    pytorch_profiler_warmup: int
    pytorch_profiler_active: int
    pytorch_profiler_repeat: int
    profile_memory: bool
    rank_zero_only: bool
    record_shapes: bool
    with_stack: bool
    with_flops: bool
    with_modules: bool
    acc_events: bool
    layer_sizes: list = field(default_factory=lambda: [512, 256, 128])
    dataset: str = "mnist"
    dataset_root: Optional[PathLike] = None
    num_workers: int = 0
    no_distributed_history: bool = False
    save_datasets: bool = False

    def __post_init__(self):
        """Initialise output paths and configure profiling context managers."""
        self._created_at = os.environ.get("EZPZ_LOG_TIMESTAMP")
        if self._created_at is None:
            self._created_at = (
                ezpz.get_timestamp() if ezpz.get_rank() == 0 else None
            )
        self._created_at = ezpz.distributed.broadcast(self._created_at, root=0)
        if self._created_at is not None:
            os.environ["EZPZ_LOG_TIMESTAMP"] = self._created_at
        self.outdir = Path(os.getcwd()).joinpath(
            "outputs", "ezpz.examples.test", f"{self._created_at}"
        )
        self.outdir.mkdir(parents=True, exist_ok=True)
        dataset_root = (
            Path(self.dataset_root).expanduser()
            if self.dataset_root is not None
            else self.outdir.parent.joinpath("datasets", self.dataset)
        )
        dataset_root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = dataset_root
        profiler_type = "torch" if self.pytorch_profiler else "pyinstrument"
        self.ctx = get_profiling_context(
            profiler_type=profiler_type,
            rank_zero_only=self.rank_zero_only,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
            acc_events=self.acc_events,
            profile_memory=self.profile_memory,
            wait=self.pytorch_profiler_wait,
            warmup=self.pytorch_profiler_warmup,
            active=self.pytorch_profiler_active,
            repeat=self.pytorch_profiler_repeat,
            outdir=self.outdir,
        )
        logger.info(f"Outputs will be saved to {self.outdir}")
```

`get_torch_dtype()` converts the string `dtype` field into the
corresponding `torch.dtype`:

```python title="src/ezpz/examples/test.py" linenums="178"
    def get_torch_dtype(self) -> torch.dtype:
        """Return the torch dtype requested by this configuration."""
        if self.dtype is None:
            return torch.get_default_dtype()
        if self.dtype in {
            "fp16",
            "half",
            "float16",
        }:
            return torch.float16
        if self.dtype in {
            "bfloat16",
            "bf16",
        }:
            return torch.bfloat16
        if self.dtype in {
            "float32",
            "fp32",
            "float",
            "single",
        }:
            return torch.float32
        logger.warning(f"Unknown dtype: {self.dtype=}, using float32")
        return torch.float32
```

</details>

<details closed markdown><summary><strong><code>Trainer</code> Class</strong></summary>

Coordinates the training loop, metric tracking, and profiling.

#### `__init__` and `__post_init__`

Class-level attributes capture distributed topology at definition time.
`__post_init__` moves the model to the device, initializes the
`History` tracker, optionally sets up W&B, and builds the dataloader.

```python title="src/ezpz/examples/test.py" linenums="204"
@dataclass
class Trainer:
    """Co-ordinate training loops, logging, and profiling for the test model."""

    config: TrainConfig
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    # history: ezpz.History = field(init=False)
    train_iter: int = 0
    rank: int = ezpz.get_rank()
    # device_type: str = ezpz.get_torch_device_type()
    device_type = os.environ.get("TORCH_DEVICE", ezpz.get_torch_device())
    world_size = ezpz.get_world_size()
    local_rank = ezpz.get_local_rank()
    device_id = f"{device_type}:{local_rank}"
    _train_loader: Optional[DataLoader] = field(init=False, default=None)
    _train_iterator: Optional[Iterator[tuple[torch.Tensor, torch.Tensor]]] = (
        field(init=False, default=None)
    )
    _feature_dim: int = field(init=False, default=0)

    def __post_init__(self):
        """Move the model to the target device and register logging hooks."""
        self.device_id = f"{self.device_type}:{self.local_rank}"
        self.dtype = self.config.get_torch_dtype()
        self.model.to(self.device_id)
        self.model.to(self.dtype)
        metrics_path = self.config.outdir.joinpath("metrics.jsonl")
        self.history: ezpz.history.History = ezpz.history.History(
            report_dir=self.config.outdir,
            report_enabled=True,
            jsonl_path=metrics_path,
            jsonl_overwrite=True,
            distributed_history=(
                1 < self.world_size <= 384 and not self.config.pytorch_profiler
            ),
        )

        if self.config.tp > 1 or self.config.pp > 1 or self.config.cp > 1:
            ezpz.distributed.barrier(group=ezpz.tp.get_tensor_parallel_group())
            ezpz.distributed.barrier(group=ezpz.tp.get_data_parallel_group())
            ezpz.distributed.barrier(group=ezpz.tp.get_pipeline_parallel_group())
            ezpz.distributed.barrier(group=ezpz.tp.get_context_parallel_group())

        if self.rank == 0 and not WANDB_DISABLED:
            logger.debug("Setting up wandb")
            wbconfig = {}
            wbconfig |= asdict(self.config)
            wbconfig |= ezpz.get_dist_info()
            _ = ezpz.setup_wandb(
                project_name="ezpz.examples.test",
                config=wbconfig,
            )
            if (wbrun := getattr(wandb, "run", None)) is not None and callable(
                wbrun.watch
            ):
                wbrun.watch(self.model, log="all")

        if self.world_size > 1:
            logger.debug("Hit torch.distributed.barrier()")
            ezpz.distributed.barrier()
        self._train_loader, self._train_iterator = self._build_dataloader()
        self._feature_dim = self.config.input_size
```

#### `_build_dataloader`

Constructs the training `DataLoader`. Currently supports MNIST via
`ezpz.data.vision.get_mnist`; only rank 0 downloads the data.

```python title="src/ezpz/examples/test.py" linenums="268"
    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def _build_dataloader(
        self,
    ) -> tuple[DataLoader, Iterator[tuple[torch.Tensor, torch.Tensor]]]:
        """Construct a training dataloader for the requested dataset."""
        dataset_name = self.config.dataset.lower()
        if dataset_name == "mnist":
            try:
                from ezpz.data.vision import get_mnist
            except (
                ModuleNotFoundError
            ) as exc:  # pragma: no cover - optional dep
                msg = (
                    "torchvision is required to use the MNIST dataset. "
                    "Install it via `pip install torchvision`."
                )
                raise RuntimeError(msg) from exc
            assert self.config.dataset_root is not None
            dset_root = Path(self.config.dataset_root).expanduser().resolve()
            dset_root.mkdir(parents=True, exist_ok=True)
            bundle = get_mnist(
                train_batch_size=self.config.batch_size,
                test_batch_size=self.config.batch_size,
                outdir=dset_root,
                num_workers=self.config.num_workers,
                download=self.rank == 0,
                pin_memory=str(self.device_type).startswith(("cuda", "mps")),
            )
            train_loader = bundle["train"]["loader"]
            return train_loader, iter(train_loader)
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
```

#### `_forward_step`

Fetches a batch, reshapes/pads inputs to match the model's expected
feature dimension, runs the forward pass, and computes cross-entropy
loss and accuracy. Returns both a metrics dict and the raw loss tensor.

```python title="src/ezpz/examples/test.py" linenums="323"
    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def _forward_step(self) -> tuple[dict, torch.Tensor]:
        """Execute a forward pass returning metrics and the loss tensor."""
        t0 = time.perf_counter()
        batch_inputs, targets = self._next_batch()
        inputs = self._prepare_inputs(batch_inputs)
        targets = targets.to(self.device_id)
        logits = self.model(inputs)
        loss = calc_loss(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        ezpz.distributed.synchronize()
        metrics = {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "dtf": time.perf_counter() - t0,
        }
        return metrics, loss
```

#### `_backward_step`

Runs backpropagation and the optimizer step. Branches on whether the
backend is DeepSpeed (which manages its own `.backward()` / `.step()`)
or standard PyTorch.

```python title="src/ezpz/examples/test.py" linenums="341"
    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def _backward_step(self, loss: torch.Tensor) -> float:
        """Perform the backwards/optimiser step and return elapsed seconds."""
        t0 = time.perf_counter()
        if self.config.backend == "deepspeed":
            self.model.backward(loss)
            self.model.step(loss)
        else:
            loss.backward()
            self.optimizer.step()
        ezpz.distributed.synchronize()
        return time.perf_counter() - t0
```

#### `train_step`

One full optimizer step: forward, backward, zero-grad. Logs metrics at
`log_freq` intervals and prints summaries at `print_freq` intervals.

```python title="src/ezpz/examples/test.py" linenums="354"
    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def train_step(self) -> dict:
        """Run one optimiser step, emitting periodic logs/metrics."""
        self.train_iter += 1
        metrics, loss = self._forward_step()
        metrics["dtb"] = self._backward_step(loss)
        self.optimizer.zero_grad()
        if self.train_iter == self.config.train_iters:
            return metrics
        if (
            self.train_iter % self.config.log_freq == 0
            or self.train_iter % self.config.print_freq == 0
        ):
            summary = self.history.update({"iter": self.train_iter, **metrics})
            if self.train_iter % self.config.print_freq == 0:
                logger.info(f"{summary}")
        return metrics
```

#### `train`

The outer training loop. Iterates for `config.train_iters` steps and
optionally advances the PyTorch profiler each step.

```python title="src/ezpz/examples/test.py" linenums="406"
    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def train(self, profiler: Optional[torch.profiler.profile] = None) -> None:
        """Loop over all training iterations and return the final dataset."""
        for step in range(self.config.train_iters):
            if step == self.config.warmup:
                logger.info(f"Warmup complete at step {step}")
            _ = self.train_step()
            if profiler is not None:
                profiler.step()
```

#### `finalize`

Flushes the `History` tracker, generates plots (rank 0 only), and
optionally logs the final dataset to W&B.

```python title="src/ezpz/examples/test.py" linenums="372"
    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def finalize(
        self, outdir: Optional[str | Path | os.PathLike] = None
    ) -> Dataset:
        """Flush profilers and return the aggregated training dataset."""
        import ambivalent
        import matplotlib.pyplot as plt

        plt.style.use(ambivalent.STYLES["ambivalent"])
        outdir = Path(outdir) if outdir is not None else self.config.outdir
        env_info = self._gather_environment_snapshot()
        dataset = self.history.finalize(
            run_name="ezpz.examples.test",
            dataset_fname="train",
            warmup=self.config.warmup,
            save=False,  # XXX: don't bother saving test data
            plot=(self.rank == 0),
            outdir=outdir,
            env_info=env_info,
        )
        logger.info(f"{dataset=}")
        if wandb is not None and ezpz.verify_wandb() and not WANDB_DISABLED:
            try:
                wandb.log(
                    {
                        "train_metrics": wandb.Table(
                            dataframe=dataset.to_dataframe()
                        )
                    }
                )
            except Exception:
                logger.warning("Failed to log final dataset to wandb")
        return dataset
```

</details>

<details closed markdown><summary><strong><code>calc_loss</code></strong></summary>

Standalone loss function using cross-entropy. Cast to `float32` before
computing to avoid issues with reduced-precision dtypes.

```python title="src/ezpz/examples/test.py" linenums="662"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def calc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the cross entropy loss for the classification dataset."""
    return torch.nn.functional.cross_entropy(logits.float(), targets)
```

</details>

<details closed markdown><summary><strong><code>build_model_and_optimizer</code></strong></summary>

Moves the model to the appropriate device, creates an Adam optimizer,
and wraps the model in `DDP` when running multi-rank. Falls back to CPU
on MPS since MPS does not support `torch.distributed` collectives.

```python title="src/ezpz/examples/test.py" linenums="668"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def build_model_and_optimizer(
    model: torch.nn.Module,
    dtype: Optional[str] = None,
) -> ModelOptimizerPair:
    """Prepare the model and optimiser for the requested backend."""
    device_override = os.environ.get("TORCH_DEVICE")
    device_type = device_override or ezpz.get_torch_device()
    if isinstance(device_type, str) and device_type.startswith("mps"):
        logger.warning(
            "MPS does not support torch.distributed collectives; falling back to CPU"
        )
        device_type = "cpu"
    world_size = ezpz.get_world_size()
    local_rank = ezpz.get_local_rank()
    if isinstance(device_type, str) and device_type in {"cuda", "xpu"}:
        device_type = f"{device_type}:{local_rank}"
    model.to(device_type)
    if isinstance(device_type, str) and device_type.startswith("cuda"):
        model.to(local_rank)
    logger.info(f"model=\n{model}")
    optimizer = torch.optim.Adam(model.parameters())
    if world_size > 1:
        model.to(device_type)
        model = ezpz.distributed.wrap_model(
            model=model, use_fsdp=False, dtype=dtype
        )
        try:
            if isinstance(device_type, str) and device_type.startswith(
                "cuda"
            ):
                model = DDP(model, device_ids=[local_rank])
            else:
                model = DDP(model)
        except Exception:
            model = DDP(model)
    return model, optimizer
```

</details>

<details closed markdown><summary><strong><code>train</code> (module-level)</strong></summary>

Orchestrates the full training run: builds the `SequentialLinearNet`
model, wraps it with `build_model_and_optimizer`, constructs a `Trainer`,
runs the training loop, finalizes history, and logs timings.

```python title="src/ezpz/examples/test.py" linenums="481"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    config: TrainConfig, profiler: Optional[torch.profiler.profile] = None
) -> Trainer:
    """Instantiate the model/optimiser and run the training loop."""
    from ezpz.models.minimal import SequentialLinearNet
    from ezpz.utils import model_summary

    timings = {}
    t0m = time.perf_counter()
    model = SequentialLinearNet(
        input_dim=config.input_size,
        output_dim=config.output_size,
        sizes=config.layer_sizes,
    )
    logger.info(
        f"Model size: {sum(p.numel() for p in model.parameters())} parameters"
    )
    try:
        logger.info(f"\n{model_summary(model)}")
    except Exception as e:
        logger.warning(
            f"Failed to summarize model: {e}, using default summary"
        )
        logger.info(model)
    t1m = time.perf_counter()
    dt_model = t1m - t0m
    logger.info(f"Took: {dt_model} seconds to build model")
    model, optimizer = build_model_and_optimizer(
        model, dtype=config.dtype
    )
    t2m = time.perf_counter()
    dt_optimizer = time.perf_counter() - t1m
    logger.info(f"Took: {dt_optimizer:.2f} seconds to build optimizer")
    trainer = Trainer(config=config, model=model, optimizer=optimizer)
    t1tr = time.perf_counter()
    logger.info(
        f"Took: {(dt_trainer := t1tr - t2m):.2f} seconds to build trainer"
    )
    jstr = json.dumps(asdict(config), indent=2, sort_keys=True, default=str)
    logger.info(f"config:\n{jstr}")
    t1s = time.perf_counter()
    logger.info(
        f"Took: {(dt_train_start := t1s - START_TIME):.2f} to get here."
    )

    # -------------------------------------------
    # Main training loop
    t0t = time.perf_counter()
    _ = trainer.train(profiler=profiler)
    t1t = time.perf_counter()
    logger.info(
        f"Took: {(dt_train_duration := t1t - t0t):.2f} seconds to finish training"
    )

    rank = ezpz.get_rank()
    history: ezpz.history.History = trainer.history

    dataset = None
    # Record timings and return trainer
    if rank == 0:
        dataset = history.finalize(
            run_name="ezpz.examples.test",
            dataset_fname="train",
            warmup=config.warmup,
            save=(rank == 0 and config.save_datasets),
            plot=(rank == 0),
            outdir=config.outdir,
            env_info=trainer._gather_environment_snapshot(),
        )
        logger.info(f"{dataset=}")
    timings = {
        "timings/model": dt_model,
        "timings/optimizer": dt_optimizer,
        "timings/trainer": dt_trainer,
        "timings/training_start": dt_train_start,
        "timings/train_duration": dt_train_duration,
        "timings/end-to-end": time.perf_counter() - START_TIME,
    }
    if wandb is not None and ezpz.verify_wandb() and not WANDB_DISABLED:
        try:
            wandb.log(
                {
                    (f"timings/{k}" if not k.startswith("timings/") else k): v
                    for k, v in timings.items()
                },
                commit=False,
            )
        except Exception as e:
            logger.exception(e)
            logger.warning("Unable to 'wandb.log(timings)', skipping!")
        if dataset is not None:
            try:
                wandb.log(
                    {
                        "train_metrics": wandb.Table(
                            dataframe=dataset.to_dataframe()
                        )
                    }
                )
            except Exception as e:
                logger.exception(e)
                logger.warning("Failed to log final dataset to wandb")

    if ezpz.get_world_size() > 1:
        ezpz.barrier()
    return trainer
```

</details>

<details closed markdown><summary><strong><code>main</code></strong></summary>

Entry point for `python -m ezpz.examples.test`. Parses CLI args, calls
`setup_torch` (with optional tensor/pipeline/context parallelism),
enters the profiling context, and kicks off `train()`.

```python title="src/ezpz/examples/test.py" linenums="749"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def main() -> Trainer:
    """Entry point used by ``python -m ezpz.examples.test``."""
    t0 = time.perf_counter()
    args = parse_args()
    config = get_config_from_args(args)
    timings = {}
    _ = ezpz.distributed.setup_torch(
        tensor_parallel_size=config.tp,
        pipeline_parallel_size=config.pp,
        context_parallel_size=config.cp,
    )
    t_setup = time.perf_counter()
    logger.info(f"Took: {(t_setup - t0):.2f} seconds to setup torch")
    with config.ctx as c:
        trainer = train(config, profiler=c)
    t_train = time.perf_counter()
    if trainer.config.backend.lower() in ["ds", "deepspeed"]:
        try:
            import deepspeed.comm

            deepspeed.comm.log_summary()
        except ImportError as e:
            logger.exception(e)
            logger.exception(
                "Deepspeed not available. "
                "Install via `python3 -m pip install deepspeed`"
            )
            logger.info("Continuing without deepspeed summary...")

    logger.info(f"Took: {time.perf_counter() - START_TIME:.2f} seconds")
    t1 = time.perf_counter()
    timings = {
        "main/setup_torch": (t_setup - t0),
        "main/train": (t_train - t_setup),
        "main/total": (t1 - t0),
    }
    if wandb is not None and (run := getattr(wandb, "run")) is not None:
        try:
            wandb.log(
                {
                    (f"timings/{k}" if not k.startswith("timings/") else k): v
                    for k, v in timings.items()
                }
            )
        except Exception:
            logger.warning("Failed to log timings to wandb")
        logger.info(f"wandb.run=[{run.name}]({run.url})")
    return trainer
```

</details>

<details closed markdown><summary><strong><code>__main__</code> guard</strong></summary>

Calls `main()`, tears down the distributed process group, and exits.

```python title="src/ezpz/examples/test.py" linenums="800"
if __name__ == "__main__":
    import sys

    trainer = main()
    ezpz.cleanup()
    sys.exit(0)
```

</details>

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.test --help
usage: test.py [-h] [--warmup WARMUP] [--tp TP] [--pp PP] [--deepspeed_config DEEPSPEED_CONFIG] [--cp CP] [--backend BACKEND]
                    [--pyinstrument-profiler] [-p] [--rank-zero-only] [--pytorch-profiler-wait PYTORCH_PROFILER_WAIT]
                    [--pytorch-profiler-warmup PYTORCH_PROFILER_WARMUP] [--pytorch-profiler-active PYTORCH_PROFILER_ACTIVE]
                    [--pytorch-profiler-repeat PYTORCH_PROFILER_REPEAT] [--profile-memory] [--record-shapes] [--with-stack]
                    [--with-flops] [--with-modules] [--acc-events] [--train-iters TRAIN_ITERS] [--log-freq LOG_FREQ]
                    [--print-freq PRINT_FREQ] [--batch-size BATCH_SIZE] [--input-size INPUT_SIZE] [--output-size OUTPUT_SIZE]
                    [--layer-sizes LAYER_SIZES] [--dtype DTYPE] [--dataset DATASET] [--dataset-root DATASET_ROOT]
                    [--num-workers NUM_WORKERS] [--no-distributed-history]

ezpz test: A simple PyTorch distributed smoke test Trains a simple MLP on MNIST dataset using DDP. NOTE: `ezpz test` is a lightweight
wrapper around: `ezpz launch python3 -m ezpz.examples.test`

options:
    -h, --help            show this help message and exit
    --warmup WARMUP       Warmup iterations
    --tp TP               Tensor parallel size
    --pp PP               Pipeline length
    --deepspeed_config DEEPSPEED_CONFIG
                        Deepspeed config file
    --cp CP               Context parallel size
    --backend BACKEND     Backend (DDP, DeepSpeed, etc.)
    --pyinstrument-profiler
                        Profile the training loop
    -p, --profile         Use PyTorch profiler
    --rank-zero-only      Run profiler only on rank 0
    --pytorch-profiler-wait PYTORCH_PROFILER_WAIT
                        Wait time before starting the PyTorch profiler
    --pytorch-profiler-warmup PYTORCH_PROFILER_WARMUP
                        Warmup iterations for the PyTorch profiler
    --pytorch-profiler-active PYTORCH_PROFILER_ACTIVE
                        Active iterations for the PyTorch profiler
    --pytorch-profiler-repeat PYTORCH_PROFILER_REPEAT
                        Repeat iterations for the PyTorch profiler
    --profile-memory      Profile memory usage
    --record-shapes       Record shapes in the profiler
    --with-stack          Include stack traces in the profiler
    --with-flops          Include FLOPs in the profiler
    --with-modules        Include module information in the profiler
    --acc-events          Accumulate events in the profiler
    --train-iters TRAIN_ITERS, --train_iters TRAIN_ITERS
                        Number of training iterations
    --log-freq LOG_FREQ, --log_freq LOG_FREQ
                        Logging frequency
    --print-freq PRINT_FREQ, --print_freq PRINT_FREQ
                        Printing frequency
    --batch-size BATCH_SIZE
                        Batch size
    --input-size INPUT_SIZE
                        Input size
    --output-size OUTPUT_SIZE
                        Output size
    --layer-sizes LAYER_SIZES
                        Comma-separated list of layer sizes
    --dtype DTYPE         Data type (fp16, float16, bfloat16, bf16, float32, etc.)
    --dataset DATASET     Dataset to use for training (e.g., mnist).
    --dataset-root DATASET_ROOT
                        Directory to cache dataset downloads.
    --num-workers NUM_WORKERS
                        Number of dataloader workers to use.
    --no-distributed-history
                        Disable distributed history aggregation
```

</details>

## Output

<details closed><summary>Output</summary>

```bash
$ ezpz test
[2025-12-31 12:42:16,253799][I][ezpz/examples/test:132:__post_init__] Outputs will be saved to /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216
[2025-12-31 12:42:16,255418][I][ezpz/dist:1501:setup_torch_distributed] Using device=mps with backend=gloo
[2025-12-31 12:42:16,269406][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=64609 from environment!
[2025-12-31 12:42:16,269926][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='Sams-MacBook-Pro-2.local'
- master_port='64609'
- world_size=2
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='gloo'
[2025-12-31 12:42:16,270684][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=2 backend=gloo
[2025-12-31 12:42:16,357662][I][ezpz/dist:1727:setup_torch] Using device='mps' with backend='gloo' + 'gloo' for distributed training.
[2025-12-31 12:42:16,384815][I][ezpz/dist:1774:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=1/1][local_rank=1/1]
[2025-12-31 12:42:16,424260][W][ezpz/dist:544:print_dist_setup] Using [2 / 2] available "mps" devices !!
[2025-12-31 12:42:16,424719][I][ezpz/dist:1774:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=0/1][local_rank=0/1]
[2025-12-31 12:42:16,425119][I][ezpz/examples/test:678:main] Took: 0.18 seconds to setup torch
[2025-12-31 12:42:16,434549][I][ezpz/examples/test:461:train] Model size: 567434 parameters
[2025-12-31 12:42:16,435638][I][ezpz/examples/test:465:train]
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
SequentialLinearNet                      --
├─Sequential: 1-1                        567,434
=================================================================
Total params: 567,434
Trainable params: 567,434
Non-trainable params: 0
=================================================================
[2025-12-31 12:42:16,436420][I][ezpz/examples/test:473:train] Took: 0.009592041955329478 seconds to build model
[2025-12-31 12:42:16,436687][W][ezpz/examples/test:590:build_model_and_optimizer] MPS does not support torch.distributed collectives; falling back to CPU
[2025-12-31 12:42:16,437061][I][ezpz/examples/test:601:build_model_and_optimizer] model=
SequentialLinearNet(
  (layers): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ReLU()
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2025-12-31 12:42:17,072997][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
[2025-12-31 12:42:17,087695][I][ezpz/examples/test:479:train] Took: 0.65 seconds to build optimizer
[2025-12-31 12:42:17,330686][I][ezpz/history:220:__init__] Using History with distributed_history=True
[2025-12-31 12:42:17,420667][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 12:42:17,421071][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.examples.test
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run de0ra7dh
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /Users/samforeman/vibes/saforem2/ezpz/wandb/run-20251231_124217-de0ra7dh
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run glad-fire-6862
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.examples.test
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.examples.test/runs/de0ra7dh
[2025-12-31 12:42:19,114502][I][ezpz/dist:2069:setup_wandb] wandb.run=[glad-fire-6862](https://wandb.ai/aurora_gpt/ezpz.examples.test/runs/de0ra7dh)
[2025-12-31 12:42:19,195945][I][ezpz/dist:2112:setup_wandb] Running on machine='localhost'
[2025-12-31 12:42:19,575441][I][ezpz/examples/test:482:train] Took: 2.49 seconds to build trainer
[2025-12-31 12:42:19,576180][I][ezpz/examples/test:486:train] config:
{
  "acc_events": false,
  "backend": "DDP",
  "batch_size": 128,
  "cp": 1,
  "dataset": "mnist",
  "dataset_root": "/Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples/test/datasets/mnist",
  "dtype": "bf16",
  "input_size": 784,
  "layer_sizes": [
    512,
    256,
    128
  ],
  "log_freq": 1,
  "no_distributed_history": false,
  "num_workers": 0,
  "output_size": 10,
  "pp": 1,
  "print_freq": 10,
  "profile_memory": true,
  "pyinstrument_profiler": false,
  "pytorch_profiler": false,
  "pytorch_profiler_active": 3,
  "pytorch_profiler_repeat": 5,
  "pytorch_profiler_wait": 1,
  "pytorch_profiler_warmup": 2,
  "rank_zero_only": false,
  "record_shapes": true,
  "tp": 1,
  "train_iters": 200,
  "warmup": 5,
  "with_flops": true,
  "with_modules": true,
  "with_stack": true
}
[2025-12-31 12:42:19,577931][I][ezpz/examples/test:488:train] Took: 4.05 to get here.
[2025-12-31 12:42:20,031182][I][ezpz/examples/test:369:train] Warmup complete at step 5
[2025-12-31 12:42:20,084940][I][ezpz/examples/test:325:train_step] iter=10 loss=1.156702 accuracy=0.617188 dtf=0.006740 dtb=0.002168 loss/mean=1.162131 loss/max=1.167561 loss/min=1.156702 loss/std=0.005437 accuracy/mean=0.679688 accuracy/max=0.742188 accuracy/min=0.617188 accuracy/std=0.062500 dtf/mean=0.006928 dtf/max=0.007116 dtf/min=0.006740 dtf/std=0.000188 dtb/mean=0.002139 dtb/max=0.002168 dtb/min=0.002110 dtb/std=0.000029
[2025-12-31 12:42:20,239358][I][ezpz/examples/test:325:train_step] iter=20 loss=0.689231 accuracy=0.773438 dtf=0.007689 dtb=0.004232 loss/mean=0.729648 loss/max=0.770065 loss/min=0.689231 loss/std=0.040417 accuracy/mean=0.777344 accuracy/max=0.781250 accuracy/min=0.773438 accuracy/std=0.003906 dtf/mean=0.007831 dtf/max=0.007973 dtf/min=0.007689 dtf/std=0.000142 dtb/mean=0.004175 dtb/max=0.004232 dtb/min=0.004118 dtb/std=0.000057
[2025-12-31 12:42:20,390147][I][ezpz/examples/test:325:train_step] iter=30 loss=0.530459 accuracy=0.851562 dtf=0.006860 dtb=0.004432 loss/mean=0.479719 loss/max=0.530459 loss/min=0.428980 loss/std=0.050740 accuracy/mean=0.867188 accuracy/max=0.882812 accuracy/min=0.851562 accuracy/std=0.015625 dtf/mean=0.006832 dtf/max=0.006860 dtf/min=0.006804 dtf/std=0.000028 dtb/mean=0.004483 dtb/max=0.004534 dtb/min=0.004432 dtb/std=0.000051
[2025-12-31 12:42:20,619267][I][ezpz/examples/test:325:train_step] iter=40 loss=0.340644 accuracy=0.898438 dtf=0.010063 dtb=0.001984 loss/mean=0.357486 loss/max=0.374328 loss/min=0.340644 loss/std=0.016842 accuracy/mean=0.886719 accuracy/max=0.898438 accuracy/min=0.875000 accuracy/std=0.011719 dtf/mean=0.010164 dtf/max=0.010265 dtf/min=0.010063 dtf/std=0.000101 dtb/mean=0.001927 dtb/max=0.001984 dtb/min=0.001870 dtb/std=0.000057
[2025-12-31 12:42:20,774938][I][ezpz/examples/test:325:train_step] iter=50 loss=0.361376 accuracy=0.882812 dtf=0.008326 dtb=0.005981 loss/mean=0.337681 loss/max=0.361376 loss/min=0.313987 loss/std=0.023694 accuracy/mean=0.890625 accuracy/max=0.898438 accuracy/min=0.882812 accuracy/std=0.007812 dtf/mean=0.008406 dtf/max=0.008487 dtf/min=0.008326 dtf/std=0.000081 dtb/mean=0.006001 dtb/max=0.006022 dtb/min=0.005981 dtb/std=0.000021
[2025-12-31 12:42:20,944339][I][ezpz/examples/test:325:train_step] iter=60 loss=0.377222 accuracy=0.906250 dtf=0.006605 dtb=0.001605 loss/mean=0.328784 loss/max=0.377222 loss/min=0.280345 loss/std=0.048439 accuracy/mean=0.906250 accuracy/max=0.906250 accuracy/min=0.906250 accuracy/std=0.000000 dtf/mean=0.006642 dtf/max=0.006680 dtf/min=0.006605 dtf/std=0.000037 dtb/mean=0.001830 dtb/max=0.002055 dtb/min=0.001605 dtb/std=0.000225
[2025-12-31 12:42:21,088867][I][ezpz/examples/test:325:train_step] iter=70 loss=0.576832 accuracy=0.851562 dtf=0.006629 dtb=0.001574 loss/mean=0.491650 loss/max=0.576832 loss/min=0.406469 loss/std=0.085181 accuracy/mean=0.871094 accuracy/max=0.890625 accuracy/min=0.851562 accuracy/std=0.019531 dtf/mean=0.006417 dtf/max=0.006629 dtf/min=0.006206 dtf/std=0.000212 dtb/mean=0.001499 dtb/max=0.001574 dtb/min=0.001424 dtb/std=0.000075
[2025-12-31 12:42:21,253870][I][ezpz/examples/test:325:train_step] iter=80 loss=0.190064 accuracy=0.945312 dtf=0.010226 dtb=0.002672 loss/mean=0.248800 loss/max=0.307535 loss/min=0.190064 loss/std=0.058736 accuracy/mean=0.929688 accuracy/max=0.945312 accuracy/min=0.914062 accuracy/std=0.015625 dtf/mean=0.010215 dtf/max=0.010226 dtf/min=0.010203 dtf/std=0.000011 dtb/mean=0.004028 dtb/max=0.005383 dtb/min=0.002672 dtb/std=0.001355
[2025-12-31 12:42:21,421837][I][ezpz/examples/test:325:train_step] iter=90 loss=0.347430 accuracy=0.906250 dtf=0.007348 dtb=0.005348 loss/mean=0.338818 loss/max=0.347430 loss/min=0.330205 loss/std=0.008612 accuracy/mean=0.910156 accuracy/max=0.914062 accuracy/min=0.906250 accuracy/std=0.003906 dtf/mean=0.007332 dtf/max=0.007348 dtf/min=0.007316 dtf/std=0.000016 dtb/mean=0.005451 dtb/max=0.005554 dtb/min=0.005348 dtb/std=0.000103
[2025-12-31 12:42:21,583712][I][ezpz/examples/test:325:train_step] iter=100 loss=0.205180 accuracy=0.937500 dtf=0.006650 dtb=0.001697 loss/mean=0.186145 loss/max=0.205180 loss/min=0.167109 loss/std=0.019036 accuracy/mean=0.945312 accuracy/max=0.953125 accuracy/min=0.937500 accuracy/std=0.007812 dtf/mean=0.006642 dtf/max=0.006650 dtf/min=0.006634 dtf/std=0.000008 dtb/mean=0.001716 dtb/max=0.001736 dtb/min=0.001697 dtb/std=0.000019
[2025-12-31 12:42:21,747500][I][ezpz/examples/test:325:train_step] iter=110 loss=0.280337 accuracy=0.890625 dtf=0.007608 dtb=0.001616 loss/mean=0.403753 loss/max=0.527169 loss/min=0.280337 loss/std=0.123416 accuracy/mean=0.871094 accuracy/max=0.890625 accuracy/min=0.851562 accuracy/std=0.019531 dtf/mean=0.007645 dtf/max=0.007683 dtf/min=0.007608 dtf/std=0.000038 dtb/mean=0.001591 dtb/max=0.001616 dtb/min=0.001565 dtb/std=0.000026
[2025-12-31 12:42:21,911124][I][ezpz/examples/test:325:train_step] iter=120 loss=0.193751 accuracy=0.945312 dtf=0.007935 dtb=0.003730 loss/mean=0.222549 loss/max=0.251347 loss/min=0.193751 loss/std=0.028798 accuracy/mean=0.941406 accuracy/max=0.945312 accuracy/min=0.937500 accuracy/std=0.003906 dtf/mean=0.007733 dtf/max=0.007935 dtf/min=0.007531 dtf/std=0.000202 dtb/mean=0.003998 dtb/max=0.004267 dtb/min=0.003730 dtb/std=0.000269
[2025-12-31 12:42:22,062758][I][ezpz/examples/test:325:train_step] iter=130 loss=0.141411 accuracy=0.968750 dtf=0.008744 dtb=0.001594 loss/mean=0.129989 loss/max=0.141411 loss/min=0.118568 loss/std=0.011421 accuracy/mean=0.972656 accuracy/max=0.976562 accuracy/min=0.968750 accuracy/std=0.003906 dtf/mean=0.008772 dtf/max=0.008801 dtf/min=0.008744 dtf/std=0.000028 dtb/mean=0.001590 dtb/max=0.001594 dtb/min=0.001585 dtb/std=0.000005
[2025-12-31 12:42:22,209100][I][ezpz/examples/test:325:train_step] iter=140 loss=0.211549 accuracy=0.921875 dtf=0.010127 dtb=0.001660 loss/mean=0.199538 loss/max=0.211549 loss/min=0.187527 loss/std=0.012011 accuracy/mean=0.929688 accuracy/max=0.937500 accuracy/min=0.921875 accuracy/std=0.007812 dtf/mean=0.010193 dtf/max=0.010259 dtf/min=0.010127 dtf/std=0.000066 dtb/mean=0.001661 dtb/max=0.001661 dtb/min=0.001660 dtb/std=0.000000
[2025-12-31 12:42:22,373980][I][ezpz/examples/test:325:train_step] iter=150 loss=0.388431 accuracy=0.898438 dtf=0.007742 dtb=0.004872 loss/mean=0.371302 loss/max=0.388431 loss/min=0.354173 loss/std=0.017129 accuracy/mean=0.890625 accuracy/max=0.898438 accuracy/min=0.882812 accuracy/std=0.007812 dtf/mean=0.007812 dtf/max=0.007882 dtf/min=0.007742 dtf/std=0.000070 dtb/mean=0.004859 dtb/max=0.004872 dtb/min=0.004847 dtb/std=0.000012
[2025-12-31 12:42:22,532635][I][ezpz/examples/test:325:train_step] iter=160 loss=0.194098 accuracy=0.945312 dtf=0.009664 dtb=0.002261 loss/mean=0.237500 loss/max=0.280903 loss/min=0.194098 loss/std=0.043402 accuracy/mean=0.925781 accuracy/max=0.945312 accuracy/min=0.906250 accuracy/std=0.019531 dtf/mean=0.009647 dtf/max=0.009664 dtf/min=0.009630 dtf/std=0.000017 dtb/mean=0.002264 dtb/max=0.002267 dtb/min=0.002261 dtb/std=0.000003
[2025-12-31 12:42:22,698409][I][ezpz/examples/test:325:train_step] iter=170 loss=0.310664 accuracy=0.859375 dtf=0.008939 dtb=0.001697 loss/mean=0.293060 loss/max=0.310664 loss/min=0.275457 loss/std=0.017604 accuracy/mean=0.886719 accuracy/max=0.914062 accuracy/min=0.859375 accuracy/std=0.027344 dtf/mean=0.008993 dtf/max=0.009047 dtf/min=0.008939 dtf/std=0.000054 dtb/mean=0.001674 dtb/max=0.001697 dtb/min=0.001652 dtb/std=0.000022
[2025-12-31 12:42:22,867578][I][ezpz/examples/test:325:train_step] iter=180 loss=0.144313 accuracy=0.953125 dtf=0.008492 dtb=0.002695 loss/mean=0.154619 loss/max=0.164924 loss/min=0.144313 loss/std=0.010305 accuracy/mean=0.949219 accuracy/max=0.953125 accuracy/min=0.945312 accuracy/std=0.003906 dtf/mean=0.008534 dtf/max=0.008576 dtf/min=0.008492 dtf/std=0.000042 dtb/mean=0.002694 dtb/max=0.002695 dtb/min=0.002692 dtb/std=0.000002
[2025-12-31 12:42:23,009535][I][ezpz/examples/test:325:train_step] iter=190 loss=0.143928 accuracy=0.968750 dtf=0.008190 dtb=0.001721 loss/mean=0.186032 loss/max=0.228136 loss/min=0.143928 loss/std=0.042104 accuracy/mean=0.949219 accuracy/max=0.968750 accuracy/min=0.929688 accuracy/std=0.019531 dtf/mean=0.008230 dtf/max=0.008270 dtf/min=0.008190 dtf/std=0.000040 dtb/mean=0.001679 dtb/max=0.001721 dtb/min=0.001637 dtb/std=0.000042
[2025-12-31 12:42:23,797136][I][ezpz/history:2385:finalize] Saving plots to /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples/test/2025-12-31-124216/plots/mplot (matplotlib) and /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples/test/2025-12-31-124216/plots/tplot (tplot)
                            accuracy                                                  accuracy/min
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
0.992┤                ▗   ▖    ▖    ▗▄▗▄ ▗▖ ▟▖▟▄▄▗▖▖▄▗▌▗▟▄▄│0.969┤      -----------------------------------------------│
0.930┤       ▗▄▌▙▙▙▟█▄██▚▄▙▙▛▞▚▜▟▚▄▛▀▝▘▀▞▜▀▟▀▛▝▀▝▀█▝ ▛▝▘▐█▌│0.734┤------ -   -                                         │
0.867┤    ▗█▞█▝▝▝▝   ▘ ▝ ▝▝                       ▝      ▝ │     └┬────────────┬────────────┬────────────┬────────────┬┘
0.742┤ ▖▟▟▀█ ▘                                             │     1.0         49.2         97.5         145.8      194.0
0.680┤█▛▛  ▝                                               │accuracy/min                  iter
0.617┤▜▌                                                   │                          accuracy/std
     └┬────────────┬────────────┬────────────┬────────────┬┘     ┌─────────────────────────────────────────────────────┐
     1.0         49.2         97.5         145.8      194.0 0.066┤ *** * * * **   * *                                  │
accuracy                      iter                          0.022┤*****************************************************│
                          accuracy/mean                          └┬────────────┬────────────┬────────────┬────────────┬┘
     ┌─────────────────────────────────────────────────────┐     1.0         49.2         97.5         145.8      194.0
0.973┤                  ·      ·     ···  · · ···· ········│accuracy/std                  iter
0.915┤       ··································· ··· ······│                          accuracy/max
0.857┤   ······· ··· ·                                     │     ┌─────────────────────────────────────────────────────┐
0.741┤ ·····                                               │0.992┤       + ++++++++++++++++++++++++++++++++++++++++++++│
0.683┤···                                                  │0.870┤ +++++++++++++ +++++ ++     +          +           + │
0.625┤··                                                   │0.686┤++                                                   │
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
     1.0         49.2         97.5         145.8      194.0      1.0         49.2         97.5         145.8      194.0
accuracy/mean                 iter                          accuracy/max                  iter
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/accuracy.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.992┤ ++ accuracy/max                                                                      ▗               ▟    +▖    │
     │ -- accuracy/min                                                  ▗▌ +▖ ▟        ▗▌  +█  +▗+▗+▖  +▟▗  █   ▖▞▌▟ + │
     │ ·· accuracy/mean              +   ▖+ ▗   ▗▖    ▗    ▟ ▖       +▗▟▐▌+▞▙▙▜▗+▖▞▖▗  ▐▐▗ ▐▐▗▄·█▗▛▟▌▖▄▚▌▘▚▞▌▌▗▐▝·▌█·▟▗│
0.930┤ ▞▞ accuracy         ▗       ▖▗·  ▐▌·▗▘▌  ▌▌+   █ ▗▗▌█▞▌▗+▟+  ▄▀▌▝▌▚·▌▜▜▐▌▀▌▌▌▛▖+▐·▜▞▌▐▌▐▞▝▛▌▜██··--▐▌-▐█▌ -█▐▐▐▌│
     │               ▟+ ▗▚▖█+▞▙▙▚▄█▝▘▜··▟▝▟▐·▙▌ ▌▌▞▀▄▀ ▙▘▀▝▘▘▙▜▞▐▗▌▞█··  -▀  - ▘ ▐▌▝ ▝▚▘ ▝▌··▘-▘   -▜█- - ▝▌ ·▘▘  ▝▐▞▐▌│
     │              +█· ▐·▌█▗▘█▜·▜▜ ·▐▞▀▝·▝▟-▜▝▖▌▙▘ ▜  ▝--   ▝ ▘▝▌▝▘▝            -▘       ·          █             ▐▌-▘│
0.867┤             +·▛▄▀█·▚▘▜·▝···---▝▌ -· █-  ▝▌▝-            -- ·                                  ▜              ▘  │
     │      +  + ▖▖▗▜▌█-▝·  ···---  -    - ▝                      -                                                    │
     │   +  + ++▐█▚▌▐▌▜     -·- -                                                                                      │
0.805┤  ++ ▖· +▗██·▘▐▌      --- -                                                                                      │
     │  ++▐▌▗▌▞▟▜█·  ▘       -                                                                                         │
     │ ▗+·▐▚▐▝▘·-█-                                                                                                    │
0.742┤ █·▗▐▐▞ -· █-                                                                                                    │
     │ █·█▐▝▌  - █-                                                                                                    │
     │▖█▗▀▟-     ▝                                                                                                     │
0.680┤▌▌█ ▜-                                                                                                           │
     │▌▌█                                                                                                              │
     │▌▌█                                                                                                              │
0.617┤▝▘▜                                                                                                              │
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                        49.2                        97.5                        145.8                     194.0
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/accuracy_summary.txt
                       accuracy/mean hist                                           accuracy/max hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
80.0┤                                           ██████     │85.0┤                                           ██████     │
66.7┤                                           ██████     │70.8┤                                           ██████     │
53.3┤                                           ██████     │56.7┤                                      ███████████     │
40.0┤                                      ████████████████│42.5┤                                      ███████████     │
26.7┤                                      ████████████████│28.3┤                                      ████████████████│
13.3┤                      █████     ██████████████████████│14.2┤                           ███████████████████████████│
 0.0┤██████████████████████████████████████████████████████│ 0.0┤█████      ███████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
   0.61         0.70          0.80         0.89        0.99    0.61         0.71          0.81         0.91        1.01
                        accuracy/min hist                                           accuracy/std hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
77.0┤                                           ██████     │61.0┤█████                                                 │
64.2┤                                           ██████     │50.8┤███████████                                           │
51.3┤                                           ██████     │40.7┤████████████████                                      │
38.5┤                                      ███████████     │30.5┤████████████████                                      │
25.7┤                                      ████████████████│20.3┤████████████████                                      │
12.8┤                      █████     ██████████████████████│10.2┤███████████████████████████                           │
 0.0┤██████████████████████████████████████████████████████│ 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
   0.60         0.70          0.79         0.89        0.98   -0.003        0.015         0.033        0.051      0.069
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/accuracy_hist.txt
                                dtb                                                       dtb/min
       ┌───────────────────────────────────────────────────┐       ┌───────────────────────────────────────────────────┐
0.00626┤       ▖   ▗ ▗               ▐     ▖  ▖            │0.00606┤  ------- -- ----   - -  --  ---  ------ - ---    -│
0.00547┤  ▄▙  ▐▙ ▗ ▐ █▖     ▄▗▌▖  ▖  ▐    ▐▌ ▐▌     ▗     ▞│0.00297┤--- --------------------------------- -------------│
0.00469┤  ██▗ ▞▀▖▐ ▐ ██ ▖▗  █▐█▌ █▌  ▐ ▖  ▐▙▗▌▌▖ ▗ ▗█▌    ▌│       └┬────────────┬───────────┬────────────┬───────────┬┘
0.00311┤  █▐▐ ▌ ▌▐ █ ██▐▌▐ ▐▐▐█▌▐█▌  ▐█▌  ▐██ █▌ █▟▐▛▌ ▌  ▌│       1.0         49.2        97.5         145.8     194.0
0.00233┤  █▐█▖▌ ▌▞▖▛▖██▐▙█▗▟▐▐█▙█▜▌  ▐██  ▐█▌ █▌▙██▟▌█▌▌  ▌│dtb/min                        iter
0.00154┤▄▀▀▝▛▛▌ ▐▌▀▌▀▀▌▜▜▝▀▀▝▀▝▜▜ ▜▛▚█▝█▄▄▀▛▘ ▛▀▀▜▀█▘▜▙▚▟▙▌│                              dtb/std
       └┬────────────┬───────────┬────────────┬───────────┬┘       ┌───────────────────────────────────────────────────┐
       1.0         49.2        97.5         145.8     194.0 0.00187┤    *  *      *  * ** ****    *         *          │
dtb                            iter                         0.00062┤***************************************************│
                             dtb/mean                              └┬────────────┬───────────┬────────────┬───────────┬┘
       ┌───────────────────────────────────────────────────┐       1.0         49.2        97.5         145.8     194.0
0.00616┤           ·                 ·                     │dtb/std                        iter
0.00538┤  ··   ·· ·· ··       ·      ·    ·  ··     ·     ·│                              dtb/max
0.00461┤  ·········· ····  ·· ·  ··  ···  ······ · ···    ·│       ┌───────────────────────────────────────────────────┐
0.00305┤  · ········ ····· ·· ·····  ···  ··· ········ ·  ·│0.00626┤   +   +   + ++    ++ ++     +    +   + +   +     +│
0.00228┤ ·· ··· ·········· ·············  ··  ··········  ·│0.00469┤  +++++++ ++ +++++ ++ +++++  +++  ++++++++++++ +  +│
0.00150┤··· ··· ··········· ····· ··········  ············ │0.00233┤+++ +++ ++++++++++++++++++++++++++++  +++++++++++++│
       └┬────────────┬───────────┬────────────┬───────────┬┘       └┬────────────┬───────────┬────────────┬───────────┬┘
       1.0         49.2        97.5         145.8     194.0        1.0         49.2        97.5         145.8     194.0
dtb/mean                       iter                         dtb/max                        iter
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/dtb.txt
       ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.00626┤ ++ dtb/max                                                     ▟                                              │
       │ -- dtb/min             ▗▌   ▖                                  █                                              │
       │ ·· dtb/mean   ▖        ▐▌  ▐▌                                  █           ▖     ▗▌                           │
0.00545┤ ▞▞ dtb       ▐▌        ▐▌  ▐▌            +     ▖               █          ▐▌     ▐▌     +                    ▞│
       │     ▗▜       ▐▌▖       ▐▌  ▐▌▗           + ▖  ▐▌ ▖     ▗       █          ▐▌     ▐▌    ++       +           ·▌│
       │    ▗▞▐▗      ▐█▚·   ▖  ▐▌  ▐▌█           +▐▐  ▐▌▐▌     █       ▌▌         ▐▌    ▗▜▌    ++      +▟           ·▌│
0.00465┤    █▌▐█  +   ▐▜▐·  ▐▌  ▐▌  ▐▌█           +▐▐  ▐▌▐▌   ▗▌█       ▌▌ +       ▐▌    ▐ ▌    ++      ·█▗          ·▌│
       │    █▌▝█ +▗  ▞▘·▐·  ▐▌  ▐▌  ▐▌█▞▌         +▐▐  ▐▌▐▌  +▐█▐       ▌▌++       ▐▌    ▐ ▌    +++     ·██          ·▌│
       │    █▌  ▌+█  ▌ ·▝▄+ ▐▌  ▞▚  ▐▌█▌▌  ▖  ▗   ·▌▐  ▐▌▐▌ ++▌█▐       ▌▌+▗       ▐▌▟ ▟ ▞ ▌    ++▟    ▟·██          ·▌│
0.00384┤    █▌  ▌+█  ▌ ··▐· ▐▌  ▌▐  ▐▐█▌▌ ▐▌  █   ·▌▐  ▐▌▐▌ ++▌█▐       ▌▚▌█       ▐▌█ █▞  ▌ ▟  +·█ +  █·██    ·     ·▌│
       │    █▌  ▌+█  ▌ -·▐· ▐▚  ▌▐  ▐▐█▌▌ ▐▌  █   ·▌▐  ▐▌▐▌ ++▌█▝▖      ▌ ▌█       ▐▌█▗▜▌  ▌▞▐  +·█ ▗▌ █·▌▜   ▗▌     ·▌│
       │    █▌  ▌+█  ▌ --▐· ▐▐  ▌▐  ▐▐█▌▌ ▐▌  █   ·▌▐  ▐▌▌▌ +·▌█·▌      ▌ ▌█       ▐▌█▐▝▌  ▌▌▐  ··█ ▐▌ █·▌▐   ▐▌     ·▌│
0.00304┤    █▌  ▌▗▜  ▌ --▐· ▐▐  ▌▐  ▐▐█▌▌ ▐▌  █   ·▌▐  ▐▚▌▌ ▖·▌█·▌      ▌ ▌█       ▐▌█▐-   ▌▌▐  ··█ ▞▌ █·▌▐   ▐▌     ·▌│
       │    █▌  ▙█▐  ▌ -- ▌ ▐▐  ▌▐  ▐▐█▌▌ ▐▌▗▌█   ▗▌▐  ▐▐▌▌▐▌·▌█·▌      ▌ ▌█▟      ▐█▐▐    ▌▌▐ ▗··█ ▌▌ █·▌▐▗+ ▐▌     ·▌│
       │    █▌  ██▝▖ ▌ -- ▌ ▌▐· ▌▐  ▐▐█▌▌ ▐▌▐▌█  +█▌▐  ▟▐▌▌▐▌·▌█·▌      ▌ ▌██      ▐█▐▐    ▌▌▐ █··█ ▌▌▗█▗▘▐▌▌+▐▌     ·▌│
0.00223┤  ▗ █▌  ██-▌ ▌  - ▌+▌ ▌ ▌▝▖ ▌▐█▌▌ ▐▐▐▐█ ▗▗▀▌▐ ▐█▐▌▌▟▌·▌▝·▌     +▌ ▌█▛▖     ▐█▝▟    ▌▌▐ ▛▖·█▐·▌██▞ ▐▌▌+▐▌     ·▌│
       │ ▗▜▗█▌  ██-▌ ▌    ▌▗▌ ▌ ▌-▌▞▌▝█▌▌·▐▐▐▐█▗▌▘ -▐+▞▝▐▌██▙▌▌--▚▗▌·▗+·▌ ▚▜▌▌· + ▗▐█ ▜    ▌▌▐▗▘▚·█▞ ▌█▌▘ ▐▌▐·▐▌   · ▗▘│
       │▄▌ ▘ ▘  ▜▝ ▙▚▌    ▝▛▌ ▝▀▌ ▝▘· ▐▌▝▚▟▐▐ ▘▘-   -▀▘  ▘▝▜▝▝▌- -▜▙▀▀▄▌▌ ·▐▌▚▗▗▖▗▘▀▜      ▜  ▀ -▀▜▌ ▝█▌- ▐▌▝▄▘▚▄▄▞▚·▟ │
0.00142┤           ▝           -       ▘  ▝ ▀                      ▝   ▝    ▘ ▘▘▝▘                 ▘   ▘   ▘ ▝   -▘ ▀▝ │
       └┬───────────────────────────┬──────────────────────────┬───────────────────────────┬──────────────────────────┬┘
       1.0                        49.2                       97.5                        145.8                    194.0
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/dtb_summary.txt
                          dtb/mean hist                                               dtb/max hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
95.0┤█████                                                 │88.0┤█████                                                 │
79.2┤█████                                                 │73.3┤█████                                                 │
63.3┤█████                                                 │58.7┤█████                                                 │
47.5┤█████                                                 │44.0┤█████                                                 │
31.7┤███████████                                           │29.3┤███████████                                           │
15.8┤███████████████████████████████████████████           │14.7┤████████████████           ██████████████████████     │
 0.0┤██████████████████████████████████████████████████████│ 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.0013       0.0026        0.0038       0.0051     0.0064   0.0013       0.0026        0.0039       0.0052     0.0065
                          dtb/min hist                                                dtb/std hist
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
105.0┤█████                                                │166.0┤█████                                                │
 87.5┤█████                                                │138.3┤█████                                                │
 70.0┤█████                                                │110.7┤█████                                                │
 52.5┤█████                                                │ 83.0┤█████                                                │
 35.0┤█████                                                │ 55.3┤█████                                                │
 17.5┤████████████████           ██████████                │ 27.7┤█████                                                │
  0.0┤██████████████████████████ ██████████████████████████│  0.0┤█████████████████████      ██████████████████████████│
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
   0.0012       0.0025       0.0037       0.0050     0.0063    -0.00008     0.00043      0.00094      0.00144   0.00195
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/dtb_hist.txt
                                dtf                                                       dtf/min
      ┌────────────────────────────────────────────────────┐      ┌────────────────────────────────────────────────────┐
0.0122┤                    ▟              ▄    ▖           │0.0122┤  -      -- - ---  -- - - - -  - - -    -- ---   ---│
0.0112┤              ▖▐   ▗█              █    ▌    ▟  ▖ ▖▗│0.0080┤-------------------- -------------- ----------------│
0.0101┤ ▗▌   ▖  █ ▗▌ ▌▐▖  ▞█▗▙ ▐▟▌ ▙  ▌ ▐▗▘▌   ▙▌ ▖▙█ ▐▌▗▌▌│      └┬────────────┬────────────┬───────────┬────────────┬┘
0.0080┤ ▟▙▟▟█▌▖ █▙█▌▌▌▌▌▄ ▌▐▐█▌▌██▟█▜▗▌▌█▐ ▌▗ ▖█▌▞▚▜█▐▐▌▟█▌│      1.0         49.2         97.5        145.8      194.0
0.0070┤▖██▀▜█▜▙▟█▛██▌▙▌██ ▌▝▟██▌██▘▘▐█████ ▜▐█▛▀▚▌ ▐ █▐▌██▌│dtf/min                        iter
0.0059┤▐▛▝ ▐▀    ▘▝▐▜▝▘▝▘▀  ▜▘▀▘▀▀   ▀ ▘▜▀  ▘▀▌ ▐▘ ▐  ▀▀▛▜▘│                              dtf/std
      └┬────────────┬────────────┬───────────┬────────────┬┘      ┌────────────────────────────────────────────────────┐
      1.0         49.2         97.5        145.8      194.0 0.0329┤       *                                            │
dtf                            iter                         0.0110┤****************************************************│
                             dtf/mean                             └┬────────────┬────────────┬───────────┬────────────┬┘
      ┌────────────────────────────────────────────────────┐      1.0         49.2         97.5        145.8      194.0
0.0400┤       ·                                            │dtf/std                        iter
0.0343┤       ·                                            │                             dtf/max
0.0287┤       ·                                            │     ┌─────────────────────────────────────────────────────┐
0.0173┤       ·                                            │0.073┤        +                                            │
0.0116┤  ·    · ·· · ···  ·· · ··· ·  · · ·    ··  ·· · ···│0.051┤       ++                                            │
0.0059┤···················· ·············· ················│0.017┤+++++++++++++++++++++++++++++++++++++++++++++++++++++│
      └┬────────────┬────────────┬───────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
      1.0         49.2         97.5        145.8      194.0      1.0         49.2         97.5         145.8      194.0
dtf/mean                       iter                         dtf/max                       iter
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/dtf.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.073┤ ++ dtf/max     +                                                                                                │
     │ -- dtf/min     +                                                                                                │
     │ ·· dtf/mean    +                                                                                                │
0.062┤ ▞▞ dtf         +                                                                                                │
     │                +                                                                                                │
     │                +                                                                                                │
0.051┤                +                                                                                                │
     │                +                                                                                                │
     │                +                                                                                                │
0.039┤                ·                                                                                                │
     │                ·                                                                                                │
     │                ·                                                                                                │
0.028┤                ·                                                                                                │
     │                ·                                                                                                │
     │                ·                                                                                                │
0.017┤                ·                                                                                                │
     │                ·                           ·                                ·         +                         │
     │   ·▖··▗·▗ ▖·▖ ··   ▄▖··▖ ▖ ·  ▖  ▞▖ ···  ▗▄▞▌  ▖▗·+ ▄▖▗▗ ▗·▖▄▖▄▄ + ▖ ▖ ▗▗· ▗▀▖ + · ···▞▖·▖ ·▗▄▗▖·▄▚ ·▖ ▟  · ▄·▖▗│
0.006┤▚▄▄▜▝▚▞▘▀▀▟▚▀▝▀▀▀▀▀▀ ▚▀▀▝▞▝▄▀▄▞▚▞▞ ▚▞▄▀▄▄▞▘  ▝▄▞▚▘▚▞▟ ▝▛▛▄▘▀▝ ▝  ▚▀▞▝▜▝▀▀▌▚▄▌ ▝▀▚▞▀▚▚▞▀▘▝▀▝▞▞▘ ▘▝▜▝ ▀▀▚▄▛▄▄▞▞▝▞▚▘│
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                        49.2                        97.5                        145.8                     194.0
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/dtf_summary.txt
                          dtf/mean hist                                               dtf/max hist
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
157.0┤█████                                                │193.0┤█████                                                │
130.8┤█████                                                │160.8┤█████                                                │
104.7┤█████                                                │128.7┤█████                                                │
 78.5┤█████                                                │ 96.5┤█████                                                │
 52.3┤█████                                                │ 64.3┤█████                                                │
 26.2┤███████████                                          │ 32.2┤█████                                                │
  0.0┤███████████                                     █████│  0.0┤█████                                           █████│
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
    0.004        0.014        0.023        0.032      0.042     0.003        0.021        0.039        0.058      0.076
                         dtf/min hist                                                 dtf/std hist
  ┌────────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
48┤█████                                                   │193.0┤█████                                                │
40┤█████ █████                                             │160.8┤█████                                                │
32┤█████ █████                                             │128.7┤█████                                                │
24┤█████ ██████████████████████                            │ 96.5┤█████                                                │
16┤█████ █████████████████████████████████                 │ 64.3┤█████                                                │
 8┤█████ ███████████████████████████████████████           │ 32.2┤█████                                                │
 0┤█████ ████████████████████████████████████████████ █████│  0.0┤█████                                           █████│
  └┬─────────────┬─────────────┬────────────┬─────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
 0.0056       0.0074        0.0091       0.0108      0.0125    -0.0015      0.0075       0.0164       0.0254     0.0343
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/dtf_hist.txt
                              loss                                                      loss/min
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
1.69┤▚                                                     │1.69┤---                                                   │
1.42┤▐                                                     │0.61┤ -----------------------------------------------------│
1.15┤ ▚▖                                                   │    └┬────────────┬─────────────┬────────────┬────────────┬┘
0.61┤  ▚▙▄▟▖                                               │    1.0         49.2          97.5         145.8      194.0
0.34┤     ▝▀▚▜█▄▙▄▄▖▄▄▟ ▄▖▄▗▐▖▖▄▖▄▄▗    ▖▖  ▖▖    ▗  ▖ ▖ ▟ │loss/min                      iter
0.07┤              ▀ ▝▝▀▘▀▝▘▘▝▝▀▀▝▀▘▀▛▀▜▀▀▀▀▀▀▞▀▛▀▀▞▛▀▛▚▜▛▀│                            loss/std
    └┬────────────┬─────────────┬────────────┬────────────┬┘     ┌─────────────────────────────────────────────────────┐
    1.0         49.2          97.5         145.8      194.0 0.146┤  ** * * * **    **    *  * * *   **      *   * *    │
loss                          iter                          0.049┤*****************************************************│
                            loss/mean                            └┬────────────┬────────────┬────────────┬────────────┬┘
    ┌──────────────────────────────────────────────────────┐     1.0         49.2         97.5         145.8      194.0
1.72┤·                                                     │loss/std                      iter
1.45┤ ·                                                    │                            loss/max
1.18┤ ··                                                   │    ┌──────────────────────────────────────────────────────┐
0.65┤ ·····                                                │1.74┤++                                                    │
0.38┤   ····························   · ·   ·    ·  ·   · │1.20┤ +++++++++ ++     +          +                        │
0.11┤               ·· · ·· ····· ·························│0.39┤      ++++++++++++++++++++++++++++++++++++++++++++++++│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
    1.0         49.2          97.5         145.8      194.0     1.0         49.2          97.5         145.8      194.0
loss/mean                     iter                          loss/max                      iter
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/loss.txt
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
1.74┤ ++ loss/max                                                                                                      │
    │ -- loss/min                                                                                                      │
    │ ·· loss/mean                                                                                                     │
1.46┤ ▞▞ loss                                                                                                          │
    │ ▌                                                                                                                │
    │ ▌                                                                                                                │
1.18┤ ▐▗                                                                                                               │
    │ ▝█                                                                                                               │
    │  ▝▖+                                                                                                             │
0.91┤  -▚▄+                                                                                                            │
    │   -▐▗▌ +  ▗▌                                                                                                     │
    │     █▚ ▖ +▐▌                                                                                                     │
0.63┤     ▜▐▞▝▞▟▞▚+            +                                                                                       │
    │      -▘-·▐▌▝▖▄▟▟▗▌    ▖+ ++         ▟                       +                                                    │
    │      -  -▝▌ ▝·▐▛▟▝▖▟ ▐▌····· ·  ▖  +█  +▖     ▗  ▖ +    ▖  +·                                                ▗▌  │
0.35┤               ▝▌  ▚▛▖▞▚▞▀▄▚▚▖▖▗▜▝▟·▗█ ▗█▝▚▗▀▄·█+▐▌+·▄▗ ▐▌ ▗▖▗ ▖▟    +    ▖ ▗       ▟+          +    ▟   +▖   ▐▌  │
    │                     ▝▘ ▘    ▜▚▌-  ▚▘▀▖▐▜ ▝▟ ▝▄▘▚▌▝▀▜▐▛▖▞▝▖▌▚▀▟▌▛▖▗· ···+▞▌·█+▖▗▟▗▟+▌▚▖▟···▗▗▌▖▞▟++▖·▛▖ ▗▐▚ +·▐▌· │
    │                                      ▝▘   ▝    --   ▝▌▝  ▝▘  ▜▝-▚▘▙▀▀▚▚▞▌▝▀▘▀▝▘-▘ ▀▌ ▚▌▀▀▜▌▜▜▜ ▝▄▀▌▄▌▚▄▘▜▐·▖▞█▚▄▚│
0.07┤                                                                   ▝                  ▝▌   ▘     ▝ ▝   ▜   ▀▝▘▝   │
    └┬───────────────────────────┬────────────────────────────┬───────────────────────────┬───────────────────────────┬┘
    1.0                        49.2                         97.5                        145.8                     194.0
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/loss_summary.txt
                         loss/mean hist                                              loss/max hist
    ┌──────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
95.0┤█████                                                 │84┤█████                                                   │
79.2┤█████                                                 │70┤█████ █████                                             │
63.3┤███████████                                           │56┤█████ █████                                             │
47.5┤███████████                                           │42┤█████ █████                                             │
31.7┤███████████                                           │28┤█████ █████                                             │
15.8┤██████████████████████                                │14┤█████ ████████████████                                  │
 0.0┤██████████████████████████████████████     ███████████│ 0┤█████ ███████████████████████████████████████      █████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
   0.04         0.48          0.91         1.35        1.79  0.05          0.49          0.93         1.37         1.81
                         loss/min hist                                                loss/std hist
  ┌────────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
84┤█████                                                   │51.0┤█████                                                 │
70┤█████ █████                                             │42.5┤███████████                                           │
56┤█████ █████                                             │34.0┤██████████████████████                                │
42┤█████ █████                                             │25.5┤██████████████████████                                │
28┤█████ █████                                             │17.0┤███████████████████████████                           │
14┤█████ ████████████████                                  │ 8.5┤██████████████████████████████████████                │
 0┤█████ █████████████████████████████████      █████ █████│ 0.0┤██████████████████████████████████████████████████████│
  └┬─────────────┬─────────────┬────────────┬─────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
 -0.00         0.44          0.88         1.32         1.76   -0.006        0.034         0.073        0.113      0.153
text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/plots/tplot/loss_hist.txt
[2025-12-31 12:42:27,454023][I][ezpz/history:2433:finalize] Saving history report to /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-124216/report.md
[2025-12-31 12:42:27,457202][I][ezpz/examples/test:348:finalize] dataset=<xarray.Dataset> Size: 39kB
Dimensions:        (draw: 194)
Coordinates:
  * draw           (draw) int64 2kB 0 1 2 3 4 5 6 ... 188 189 190 191 192 193
Data variables: (12/25)
    iter           (draw) int64 2kB 6 7 8 9 10 11 12 ... 194 195 196 197 198 199
    loss           (draw) float32 776B 1.689 1.526 1.385 ... 0.2018 0.1583
    accuracy       (draw) float32 776B 0.7031 0.625 0.625 ... 0.8906 0.9453
    dtf            (draw) float64 2kB 0.006963 0.006409 ... 0.007307 0.01037
    dtb            (draw) float64 2kB 0.001635 0.001716 ... 0.002136 0.005472
    iter_mean      (draw) float64 2kB 6.0 7.0 8.0 9.0 ... 197.0 198.0 199.0
    ...             ...
    dtf_min        (draw) float64 2kB 0.006873 0.006409 ... 0.007263 0.01029
    dtf_std        (draw) float64 2kB 4.481e-05 0.0001108 ... 4.349e-05
    dtb_mean       (draw) float64 2kB 0.001713 0.001719 ... 0.002266 0.005477
    dtb_max        (draw) float64 2kB 0.001791 0.001722 ... 0.002395 0.005482
    dtb_min        (draw) float64 2kB 0.001635 0.001716 ... 0.002136 0.005472
    dtb_std        (draw) float64 2kB 7.823e-05 3.127e-06 ... 5.046e-06
[2025-12-31 12:42:28,064428][I][ezpz/examples/test:500:train] Took: 8.49 seconds to finish training
[2025-12-31 12:42:28,065364][I][ezpz/examples/test:695:main] Took: 12.54 seconds
wandb:
wandb: 🚀 View run glad-fire-6862 at:
wandb: Find logs at: wandb/run-20251231_124217-de0ra7dh/logs
[2025-12-23-162222] Execution time: 19s sec
```

</details>


<details closed><summary>Output on Sunspot</summary>

```bash
#[aurora_frameworks-2025.2.0](ezpz-aurora_frameworks-2025.2.0)
#[/t/d/f/p/s/ezpz][dev][2s]
#[12/31/25 @ 11:30:29][x1921c0s7b0n0]
; ezpz test


[2025-12-31 11:30:42,775692][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-113042]----
[2025-12-31 11:30:43,644334][I][ezpz/launch:416:launch] Job ID: 12458339
[2025-12-31 11:30:43,645146][I][ezpz/launch:417:launch] nodelist: ['x1921c0s3b0n0', 'x1921c0s7b0n0']
[2025-12-31 11:30:43,645554][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 11:30:43,646228][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 11:30:43,647063][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 11:30:43,647473][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 11:30:43,648172][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/venvs/sunspot/ezpz-aurora_frameworks-2025.2.0/bin/python3 -m ezpz.examples.test
[2025-12-31 11:30:43,648999][I][ezpz/launch:433:launch] Took: 0.87 seconds to build command.
[2025-12-31 11:30:43,649375][I][ezpz/launch:436:launch] Executing:
mpiexec
  --envall
  --np=24
  --ppn=12
  --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
  --no-vni
  --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
  /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/venvs/sunspot/ezpz-aurora_frameworks-2025.2.0/bin/python3
  -m
  ezpz.examples.test
[2025-12-31 11:30:43,650731][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-113043...
[2025-12-31 11:30:43,651220][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/venvs/sunspot/ezpz-aurora_frameworks-2025.2.0/bin/python3 -m ezpz.examples.test
cpubind:list x1921c0s7b0n0 pid 82824 rank 12 0: mask 0x1c
cpubind:list x1921c0s7b0n0 pid 82825 rank 13 1: mask 0x1c00
cpubind:list x1921c0s7b0n0 pid 82826 rank 14 2: mask 0x1c0000
cpubind:list x1921c0s7b0n0 pid 82827 rank 15 3: mask 0x1c000000
cpubind:list x1921c0s7b0n0 pid 82828 rank 16 4: mask 0x1c00000000
cpubind:list x1921c0s7b0n0 pid 82829 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c0s7b0n0 pid 82830 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c0s7b0n0 pid 82831 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c0s7b0n0 pid 82832 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c0s7b0n0 pid 82833 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s7b0n0 pid 82834 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s7b0n0 pid 82835 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c0s3b0n0 pid 92001 rank 0 0: mask 0x1c
cpubind:list x1921c0s3b0n0 pid 92002 rank 1 1: mask 0x1c00
cpubind:list x1921c0s3b0n0 pid 92003 rank 2 2: mask 0x1c0000
cpubind:list x1921c0s3b0n0 pid 92004 rank 3 3: mask 0x1c000000
cpubind:list x1921c0s3b0n0 pid 92005 rank 4 4: mask 0x1c00000000
cpubind:list x1921c0s3b0n0 pid 92006 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c0s3b0n0 pid 92007 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c0s3b0n0 pid 92008 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c0s3b0n0 pid 92009 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c0s3b0n0 pid 92010 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s3b0n0 pid 92011 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s3b0n0 pid 92012 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 11:30:49,869292][I][ezpz/examples/test:132:__post_init__] Outputs will be saved to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049
[2025-12-31 11:30:49,871638][I][ezpz/dist:1501:setup_torch_distributed] Using torch_{device,backend}= {xpu, xccl}
[2025-12-31 11:30:49,872308][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=32935 from environment!
[2025-12-31 11:30:49,872846][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s3b0n0'
- master_port='32935'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 11:30:49,873745][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 11:30:50,606235][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 11:30:50,607054][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 11:30:50,607503][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 11:30:50,606736][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 11:30:50,606796][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 11:30:50,606796][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 11:30:50,606807][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 11:30:50,606817][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 11:30:50,606806][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 11:30:50,606747][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 11:30:50,606787][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 11:30:50,606737][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 11:30:50,606783][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 11:30:50,606793][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 11:30:50,610395][I][ezpz/examples/test:678:main] Took: 0.76 seconds to setup torch
[2025-12-31 11:30:50,606862][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 11:30:50,606858][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 11:30:50,606933][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 11:30:50,606960][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 11:30:50,606980][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 11:30:50,606972][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 11:30:50,606946][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 11:30:50,606949][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 11:30:50,606925][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 11:30:50,606960][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 11:30:50,606970][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 11:30:50,606949][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
[2025-12-31 11:30:50,617739][I][ezpz/examples/test:461:train] Model size: 567434 parameters
[2025-12-31 11:30:50,619069][I][ezpz/examples/test:465:train]
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
SequentialLinearNet                      --
├─Sequential: 1-1                        567,434
=================================================================
Total params: 567,434
Trainable params: 567,434
Non-trainable params: 0
=================================================================
[2025-12-31 11:30:50,620391][I][ezpz/examples/test:473:train] Took: 0.006357558071613312 seconds to build model
[2025-12-31 11:30:50,622656][I][ezpz/examples/test:601:build_model_and_optimizer] model=
SequentialLinearNet(
  (layers): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ReLU()
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2025-12-31 11:30:50,624614][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
2025:12:31-11:30:50:(92001) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-11:30:50:(92001) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-12-31 11:31:03,431773][I][ezpz/examples/test:479:train] Took: 12.81 seconds to build optimizer
[2025-12-31 11:31:03,484655][I][ezpz/history:220:__init__] Using History with distributed_history=True
[2025-12-31 11:31:03,487406][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 11:31:03,487916][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.examples.test
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_113103-cppqal9m
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run polar-surf-6861
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.examples.test
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.examples.test/runs/cppqal9m
[2025-12-31 11:31:05,498721][I][ezpz/dist:2069:setup_wandb] wandb.run=[polar-surf-6861](https://wandb.ai/aurora_gpt/ezpz.examples.test/runs/cppqal9m)
[2025-12-31 11:31:05,504592][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 11:31:05,852704][I][ezpz/examples/test:482:train] Took: 2.42 seconds to build trainer
[2025-12-31 11:31:05,853930][I][ezpz/examples/test:486:train] config:
{
  "acc_events": false,
  "backend": "DDP",
  "batch_size": 128,
  "cp": 1,
  "dataset": "mnist",
  "dataset_root": "/lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/datasets/mnist",
  "dtype": "bf16",
  "input_size": 784,
  "layer_sizes": [
    512,
    256,
    128
  ],
  "log_freq": 1,
  "no_distributed_history": false,
  "num_workers": 0,
  "output_size": 10,
  "pp": 1,
  "print_freq": 10,
  "profile_memory": true,
  "pyinstrument_profiler": false,
  "pytorch_profiler": false,
  "pytorch_profiler_active": 3,
  "pytorch_profiler_repeat": 5,
  "pytorch_profiler_wait": 1,
  "pytorch_profiler_warmup": 2,
  "rank_zero_only": false,
  "record_shapes": true,
  "tp": 1,
  "train_iters": 200,
  "warmup": 5,
  "with_flops": true,
  "with_modules": true,
  "with_stack": true
}
[2025-12-31 11:31:05,856304][I][ezpz/examples/test:488:train] Took: 19.73 to get here.
[2025-12-31 11:31:20,012644][I][ezpz/examples/test:369:train] Warmup complete at step 5
[2025-12-31 11:31:20,317949][I][ezpz/examples/test:325:train_step] iter=10 loss=1.046174 accuracy=0.726562 dtf=0.010201 dtb=0.001751 loss/mean=1.094917 loss/max=1.271076 loss/min=0.961706 loss/std=0.078834 accuracy/mean=0.704427 accuracy/max=0.765625 accuracy/min=0.593750 accuracy/std=0.042472 dtf/mean=0.010742 dtf/max=0.012226 dtf/min=0.010026 dtf/std=0.000653 dtb/mean=0.001594 dtb/max=0.001852 dtb/min=0.001306 dtb/std=0.000177
[2025-12-31 11:31:21,122273][I][ezpz/examples/test:325:train_step] iter=20 loss=0.931834 accuracy=0.779412 dtf=0.005889 dtb=0.178909 loss/mean=0.592798 loss/max=0.931834 loss/min=0.390783 loss/std=0.136802 accuracy/mean=0.817402 accuracy/max=0.897059 accuracy/min=0.691176 accuracy/std=0.050930 dtf/mean=0.006413 dtf/max=0.006798 dtf/min=0.005839 dtf/std=0.000321 dtb/mean=0.204767 dtb/max=0.234498 dtb/min=0.178215 dtb/std=0.020612
[2025-12-31 11:31:21,659906][I][ezpz/examples/test:325:train_step] iter=30 loss=0.500784 accuracy=0.851562 dtf=0.009988 dtb=0.001570 loss/mean=0.459434 loss/max=0.755573 loss/min=0.280539 loss/std=0.115654 accuracy/mean=0.861003 accuracy/max=0.937500 accuracy/min=0.773438 accuracy/std=0.038405 dtf/mean=0.010492 dtf/max=0.011835 dtf/min=0.009957 dtf/std=0.000546 dtb/mean=0.001607 dtb/max=0.001853 dtb/min=0.001314 dtb/std=0.000157
[2025-12-31 11:31:22,283836][I][ezpz/examples/test:325:train_step] iter=40 loss=0.478971 accuracy=0.867647 dtf=0.005750 dtb=0.001340 loss/mean=0.319425 loss/max=0.549011 loss/min=0.172847 loss/std=0.095734 accuracy/mean=0.903799 accuracy/max=0.970588 accuracy/min=0.823529 accuracy/std=0.040494 dtf/mean=0.006246 dtf/max=0.007002 dtf/min=0.005576 dtf/std=0.000431 dtb/mean=0.001377 dtb/max=0.001588 dtb/min=0.001080 dtb/std=0.000155
[2025-12-31 11:31:22,834823][I][ezpz/examples/test:325:train_step] iter=50 loss=0.349907 accuracy=0.875000 dtf=0.010030 dtb=0.001433 loss/mean=0.298854 loss/max=0.401245 loss/min=0.211164 loss/std=0.060323 accuracy/mean=0.910807 accuracy/max=0.953125 accuracy/min=0.867188 accuracy/std=0.024594 dtf/mean=0.010629 dtf/max=0.011482 dtf/min=0.009914 dtf/std=0.000466 dtb/mean=0.001609 dtb/max=0.001820 dtb/min=0.001307 dtb/std=0.000167
[2025-12-31 11:31:23,512006][I][ezpz/examples/test:325:train_step] iter=60 loss=0.313367 accuracy=0.926471 dtf=0.005733 dtb=0.001325 loss/mean=0.182718 loss/max=0.313367 loss/min=0.097909 loss/std=0.047081 accuracy/mean=0.952206 accuracy/max=0.985294 accuracy/min=0.911765 accuracy/std=0.018628 dtf/mean=0.006096 dtf/max=0.006441 dtf/min=0.005576 dtf/std=0.000266 dtb/mean=0.001393 dtb/max=0.001545 dtb/min=0.001141 dtb/std=0.000121
[2025-12-31 11:31:24,034381][I][ezpz/examples/test:325:train_step] iter=70 loss=0.262147 accuracy=0.929688 dtf=0.009596 dtb=0.001584 loss/mean=0.216635 loss/max=0.327601 loss/min=0.138058 loss/std=0.060615 accuracy/mean=0.938477 accuracy/max=0.968750 accuracy/min=0.890625 accuracy/std=0.025447 dtf/mean=0.009859 dtf/max=0.011898 dtf/min=0.009397 dtf/std=0.000751 dtb/mean=0.001568 dtb/max=0.001724 dtb/min=0.001312 dtb/std=0.000131
[2025-12-31 11:31:24,636021][I][ezpz/examples/test:325:train_step] iter=80 loss=0.156417 accuracy=0.955882 dtf=0.005665 dtb=0.001253 loss/mean=0.111892 loss/max=0.192739 loss/min=0.060052 loss/std=0.033597 accuracy/mean=0.971814 accuracy/max=1.000000 accuracy/min=0.941176 accuracy/std=0.012681 dtf/mean=0.006088 dtf/max=0.006498 dtf/min=0.005522 dtf/std=0.000325 dtb/mean=0.001360 dtb/max=0.001627 dtb/min=0.001064 dtb/std=0.000163
[2025-12-31 11:31:25,102420][I][ezpz/examples/test:325:train_step] iter=90 loss=0.195402 accuracy=0.937500 dtf=0.009428 dtb=0.001697 loss/mean=0.154431 loss/max=0.232959 loss/min=0.095484 loss/std=0.043264 accuracy/mean=0.954102 accuracy/max=0.976562 accuracy/min=0.921875 accuracy/std=0.017222 dtf/mean=0.010213 dtf/max=0.012565 dtf/min=0.009396 dtf/std=0.000995 dtb/mean=0.001599 dtb/max=0.001942 dtb/min=0.001136 dtb/std=0.000188
[2025-12-31 11:31:25,595109][I][ezpz/examples/test:325:train_step] iter=100 loss=0.079676 accuracy=1.000000 dtf=0.005687 dtb=0.001371 loss/mean=0.076284 loss/max=0.154268 loss/min=0.046152 loss/std=0.028206 accuracy/mean=0.987132 accuracy/max=1.000000 accuracy/min=0.955882 accuracy/std=0.012246 dtf/mean=0.006011 dtf/max=0.007985 dtf/min=0.005620 dtf/std=0.000624 dtb/mean=0.001366 dtb/max=0.001765 dtb/min=0.001072 dtb/std=0.000164
[2025-12-31 11:31:26,171868][I][ezpz/examples/test:325:train_step] iter=110 loss=0.154822 accuracy=0.960938 dtf=0.009538 dtb=0.001698 loss/mean=0.115471 loss/max=0.210122 loss/min=0.066416 loss/std=0.037737 accuracy/mean=0.969401 accuracy/max=0.984375 accuracy/min=0.937500 accuracy/std=0.013327 dtf/mean=0.009814 dtf/max=0.012341 dtf/min=0.009225 dtf/std=0.000823 dtb/mean=0.001554 dtb/max=0.001722 dtb/min=0.001096 dtb/std=0.000190
[2025-12-31 11:31:26,733522][I][ezpz/examples/test:325:train_step] iter=120 loss=0.053370 accuracy=0.985294 dtf=0.005611 dtb=0.001238 loss/mean=0.056127 loss/max=0.126032 loss/min=0.025829 loss/std=0.023853 accuracy/mean=0.990196 accuracy/max=1.000000 accuracy/min=0.955882 accuracy/std=0.013197 dtf/mean=0.006132 dtf/max=0.006825 dtf/min=0.005512 dtf/std=0.000383 dtb/mean=0.001366 dtb/max=0.001595 dtb/min=0.001062 dtb/std=0.000149
[2025-12-31 11:31:27,304428][I][ezpz/examples/test:325:train_step] iter=130 loss=0.106341 accuracy=0.976562 dtf=0.009662 dtb=0.001578 loss/mean=0.087535 loss/max=0.152122 loss/min=0.041659 loss/std=0.031179 accuracy/mean=0.979818 accuracy/max=1.000000 accuracy/min=0.953125 accuracy/std=0.010792 dtf/mean=0.010319 dtf/max=0.010932 dtf/min=0.009599 dtf/std=0.000411 dtb/mean=0.001625 dtb/max=0.001806 dtb/min=0.001306 dtb/std=0.000155
[2025-12-31 11:31:27,837406][I][ezpz/examples/test:325:train_step] iter=140 loss=0.038179 accuracy=1.000000 dtf=0.005624 dtb=0.001283 loss/mean=0.031276 loss/max=0.058039 loss/min=0.018561 loss/std=0.009414 accuracy/mean=1.000000 accuracy/max=1.000000 accuracy/min=1.000000 accuracy/std=0.000000 dtf/mean=0.006149 dtf/max=0.006750 dtf/min=0.005484 dtf/std=0.000363 dtb/mean=0.001352 dtb/max=0.001624 dtb/min=0.001071 dtb/std=0.000156
[2025-12-31 11:31:28,429546][I][ezpz/examples/test:325:train_step] iter=150 loss=0.075107 accuracy=0.976562 dtf=0.009395 dtb=0.001523 loss/mean=0.078655 loss/max=0.160285 loss/min=0.043498 loss/std=0.028539 accuracy/mean=0.978516 accuracy/max=1.000000 accuracy/min=0.945312 accuracy/std=0.014131 dtf/mean=0.010171 dtf/max=0.010841 dtf/min=0.009276 dtf/std=0.000518 dtb/mean=0.001609 dtb/max=0.001818 dtb/min=0.001068 dtb/std=0.000180
[2025-12-31 11:31:28,948358][I][ezpz/examples/test:325:train_step] iter=160 loss=0.039511 accuracy=0.985294 dtf=0.005642 dtb=0.001399 loss/mean=0.038084 loss/max=0.072799 loss/min=0.011212 loss/std=0.016665 accuracy/mean=0.993873 accuracy/max=1.000000 accuracy/min=0.970588 accuracy/std=0.008404 dtf/mean=0.006147 dtf/max=0.006805 dtf/min=0.005509 dtf/std=0.000402 dtb/mean=0.001383 dtb/max=0.001682 dtb/min=0.001094 dtb/std=0.000159
[2025-12-31 11:31:29,524806][I][ezpz/examples/test:325:train_step] iter=170 loss=0.090782 accuracy=0.968750 dtf=0.009549 dtb=0.001483 loss/mean=0.063093 loss/max=0.135812 loss/min=0.029736 loss/std=0.026772 accuracy/mean=0.984375 accuracy/max=1.000000 accuracy/min=0.960938 accuracy/std=0.011935 dtf/mean=0.010131 dtf/max=0.010809 dtf/min=0.009328 dtf/std=0.000468 dtb/mean=0.001589 dtb/max=0.001801 dtb/min=0.001083 dtb/std=0.000189
[2025-12-31 11:31:30,100256][I][ezpz/examples/test:325:train_step] iter=180 loss=0.028730 accuracy=1.000000 dtf=0.005630 dtb=0.001255 loss/mean=0.031807 loss/max=0.089583 loss/min=0.009972 loss/std=0.017174 accuracy/mean=0.995098 accuracy/max=1.000000 accuracy/min=0.970588 accuracy/std=0.008130 dtf/mean=0.006067 dtf/max=0.006434 dtf/min=0.005511 dtf/std=0.000292 dtb/mean=0.001388 dtb/max=0.001594 dtb/min=0.001066 dtb/std=0.000147
[2025-12-31 11:31:30,617119][I][ezpz/examples/test:325:train_step] iter=190 loss=0.044844 accuracy=0.984375 dtf=0.009522 dtb=0.001602 loss/mean=0.051969 loss/max=0.151458 loss/min=0.025844 loss/std=0.027686 accuracy/mean=0.985677 accuracy/max=1.000000 accuracy/min=0.953125 accuracy/std=0.011423 dtf/mean=0.010067 dtf/max=0.012187 dtf/min=0.009460 dtf/std=0.000770 dtb/mean=0.001618 dtb/max=0.001843 dtb/min=0.001297 dtb/std=0.000141
[2025-12-31 11:31:32,515303][I][ezpz/history:2385:finalize] Saving plots to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/mplot (matplotlib) and /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot (tplot)
                  accuracy                              accuracy/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
1.000┤              ▟▗▙▟▄▄▙▟▟▟▄▙▄▟▄███▛│1.000┤            ---------------------│
0.931┤       ▖▄▄▟▟██▛▛▛▀▝▐▀▘▀▀ ▝ ▝ ▘   │0.826┤   -------------      -  -       │
     │   ▖▗▄██▜▀▀▀ ▘                   │0.651┤----                             │
0.862┤   ▌█▘▝▘                         │0.477┤-                                │
0.793┤ ▌▗█▝                            │     └┬───────┬───────┬───────┬───────┬┘
0.724┤▗█▌                              │     1.0    49.2    97.5    145.8 194.0
     │▐█▌                              │accuracy/min        iter
0.655┤█▌▘                              │                accuracy/std
0.586┤▀▌                               │     ┌─────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.067┤**                               │
     1.0    49.2    97.5    145.8 194.0 0.056┤****                             │
accuracy            iter                0.034┤ ************         *          │
                accuracy/mean           0.022┤       **************************│
     ┌─────────────────────────────────┐0.000┤                      *   *  *   │
1.000┤                ·  ··············│     └┬───────┬───────┬───────┬───────┬┘
0.928┤         ············ ····       │     1.0    49.2    97.5    145.8 194.0
     │      ·····                      │accuracy/std        iter
0.856┤   ···                           │                accuracy/max
0.784┤  ··                             │     ┌─────────────────────────────────┐
     │ ··                              │1.000┤      + +++++++++++++++++++++++++│
0.712┤ ·                               │0.952┤  +++++++++                      │
0.640┤·                                │0.855┤ +++                             │
     │·                                │0.807┤ +                               │
0.568┤·                                │0.711┤+                                │
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
accuracy/mean       iter                accuracy/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/accuracy.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
1.000┤ ++ accuracy/max        +  ++  +▗++▟++▗++++▖▗++▐++▟+▗++▖+·▖+▗▌++▗▟+▟▗▗▌▗▀│
     │ -- accuracy/min  ++++++++++·++▌▟+▙█▞▖▟▗▀▖▟▙▜▗▜▞▟▌█▗▛▞▀▀▜▞▚▀▀▙▞▀▌▜▞·▀▀▀▘·│
     │ ·· accuracy/mean ++·++▗ ▌ ▄▙▌▄▌█·▛█▌▚▌▜-▜-█▝▟▝▌-▚▀▌▌----▘---▌--▘--- --- │
     │ ▞▞ accuracy      +▖▗·▙▛▄▌▐██▜ ▝-▀▘▐▌------▜------ - ----- ---- -  --  --│
0.913┤        +++ +  ▗▌▐▖▙▀▖█▌▜▝▞▝▝------ ▘-   - -  -  -       -      -        │
     │    ++++  ▖  ▙▟▟▌▞▚▀-▜-▘--- --    -              -       -               │
     │    + ▗▌ ▐▙▟▄▌▐▝▛-------  -                                              │
     │  +++ ▐▌▗▀▌▘  ▝-▘- -      -                                              │
0.826┤  +++ ▐▌▐ ▌- ---                                                         │
     │ + + ·▐▌▞-▌-----                                                         │
     │ +▟ ·▗▟▚▌-- -  -                                                         │
     │ +█··▌ - --                                                              │
0.738┤ +█▗▌▌--                                                                 │
     │+▗▌▘█ -                                                                  │
     │+█▌ ▌-                                                                   │
     │ █▌ ▌-                                                                   │
     │▐█▌ ▌-                                                                   │
0.651┤▐█▌-                                                                     │
     │▐▐▌-                                                                     │
     │▘▐▌                                                                      │
     │·-▘                                                                      │
0.564┤·-                                                                       │
     │ -                                                                       │
     │ -                                                                       │
     │ -                                                                       │
0.477┤--                                                                       │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              49.2              97.5              145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/accuracy_summary.txt
            accuracy/mean hist                       accuracy/max hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
108┤                               ████│132┤                               ████│
 90┤                               ████│110┤                               ████│
   │                               ████│   │                               ████│
 72┤                               ████│ 88┤                               ████│
 54┤                               ████│ 66┤                               ████│
   │                               ████│   │                               ████│
 36┤                            ███████│ 44┤                               ████│
 18┤                        ███████████│ 22┤                            ███████│
   │              █████████████████████│   │                        ███████████│
  0┤███████████████████████████████████│  0┤███████   █████████████████████████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
  0.55     0.67    0.78     0.90   1.02   0.698    0.777   0.855    0.934 1.013
              accuracy/min hist                       accuracy/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
85.0┤                           ████   │73.0┤       ███                        │
    │                           ████   │    │       ███                        │
70.8┤                           ████   │60.8┤       ███                        │
56.7┤                           ████   │48.7┤   ███████                        │
    │                           ████   │    │   ███████                        │
42.5┤                           ███████│36.5┤   ███████                        │
    │                           ███████│    │   ███████████                    │
28.3┤                        ██████████│24.3┤   ██████████████                 │
14.2┤                    ██████████████│12.2┤   ██████████████                 │
    │          ████   █████████████████│    │   ████████████████████████       │
 0.0┤██████████████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
   0.45    0.60     0.74    0.88   1.02   -0.003   0.015    0.034   0.052 0.070
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/accuracy_hist.txt
                     dtb                                   dtb/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
0.179┤  ▟                              │0.178┤  -                              │
0.149┤  █                              │0.119┤  -                              │
     │  █                              │0.060┤  -                              │
0.120┤  █                              │0.001┤---------------------------------│
0.090┤  █                              │     └┬───────┬───────┬───────┬───────┬┘
0.060┤  █                              │     1.0    49.2    97.5    145.8 194.0
     │  █                              │dtb/min             iter
0.031┤  █                              │                    dtb/std
0.001┤▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│      ┌────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0206┤  *                             │
     1.0    49.2    97.5    145.8 194.0 0.0172┤  *                             │
dtb                 iter                0.0103┤  *                             │
                  dtb/mean              0.0069┤  *                             │
     ┌─────────────────────────────────┐0.0001┤********************************│
0.205┤  ·                              │      └┬───────┬───────┬──────┬───────┬┘
0.171┤  ·                              │      1.0    49.2    97.5   145.8 194.0
     │  ·                              │dtb/std              iter
0.137┤  ·                              │                   dtb/max
0.103┤  ·                              │     ┌─────────────────────────────────┐
     │  ·                              │0.234┤  +                              │
0.069┤  ·                              │0.196┤  +                              │
0.035┤  ·                              │0.118┤  +                              │
     │  ·                              │0.079┤  +                              │
0.001┤·································│0.002┤+++++++++++++++++++++++++++++++++│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
dtb/mean            iter                dtb/max             iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/dtb.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
0.234┤ ++ dtb/max                                                              │
     │ -- dtb/min                                                              │
     │ ·· dtb/mean                                                             │
     │ ▞▞ dtb                                                                  │
0.196┤     ·                                                                   │
     │     ·                                                                   │
     │     ▟                                                                   │
     │     █                                                                   │
0.157┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.118┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.079┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.040┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.001┤▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              49.2              97.5              145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/dtb_summary.txt
                dtb/mean hist                           dtb/max hist
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
193.0┤████                             │193.0┤████                             │
160.8┤████                             │160.8┤████                             │
     │████                             │     │████                             │
128.7┤████                             │128.7┤████                             │
 96.5┤████                             │ 96.5┤████                             │
     │████                             │     │████                             │
 64.3┤████                             │ 64.3┤████                             │
 32.2┤████                             │ 32.2┤████                             │
     │████                             │     │████                             │
  0.0┤███                          ████│  0.0┤███                          ████│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
   -0.008   0.048   0.103   0.158 0.214    -0.009   0.055   0.118   0.181 0.245
                dtb/min hist                            dtb/std hist
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
193.0┤████                             │193.0┤████                             │
     │████                             │     │████                             │
160.8┤████                             │160.8┤████                             │
128.7┤████                             │128.7┤████                             │
     │████                             │     │████                             │
 96.5┤████                             │ 96.5┤████                             │
     │████                             │     │████                             │
 64.3┤████                             │ 64.3┤████                             │
 32.2┤████                             │ 32.2┤████                             │
     │████                             │     │████                             │
  0.0┤███                          ████│  0.0┤███                          ████│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬────────┘
   -0.007   0.041   0.090   0.138 0.186    -0.0008  0.0048  0.0103  0.0159
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/dtb_hist.txt
                      dtf                                   dtf/min
      ┌────────────────────────────────┐      ┌────────────────────────────────┐
0.0122┤  ▐                             │0.0115┤  -   -  -  -  -  -   -  -  -   │
0.0111┤  ▐  ▐         ▐  ▗▌  ▖  ▟  ▗   │0.0095┤--------------------------------│
      │▙ ▐ ▖▐   ▌  ▌▖ █  ▐▌  ▌  █▖▖▐ ▗▖│0.0075┤  -  -   -  -  -  -  --  -  -   │
0.0100┤▛▙█▙▙▟▟▙▄▙█▟▙▙▟█▄▙▟▙▄▟▙▄▙█▙▙▟▄█▙│0.0055┤  -  -   -  -  -  -   -  -  -   │
0.0089┤  █▝ ▜ ▝▐▌  ▌  █  ▐  ▐▌▘▝▌  █  ▀│      └┬───────┬───────┬──────┬───────┬┘
0.0078┤  █  ▐  ▐▌  ▌  █  ▐  ▐▌  ▌  █   │      1.0    49.2    97.5   145.8 194.0
      │  █  ▐  ▐▌  ▌  █  ▐  ▐▌  ▌  █   │dtf/min              iter
0.0067┤  █  ▐  ▐▌  ▌  █  ▐  ▐▌  ▌  █   │                    dtf/std
0.0056┤  ▝  ▐  ▝▌  ▌  ▜  ▐  ▝▌  ▌  ▜   │       ┌───────────────────────────────┐
      └┬───────┬───────┬──────┬───────┬┘0.00104┤     * **  * * *       **** * *│
      1.0    49.2    97.5   145.8 194.0 0.00091┤ ** ****** ********************│
dtf                  iter               0.00065┤*******************************│
                   dtf/mean             0.00052┤*******************************│
      ┌────────────────────────────────┐0.00027┤ ** *   **  *   *    * *** *** │
0.0123┤  ·                      ·      │       └┬───────┬──────┬───────┬───────┘
0.0113┤  ·   ·  ·  ·  ·  ·   ·  ·  ·   │       1.0    49.2   97.5    145.8
      │···· ··· · ·····  ·  ··  ·· · · │dtf/std              iter
0.0102┤································│                    dtf/max
0.0092┤······  ·····  · ·· ···· ·  ··· │      ┌────────────────────────────────┐
      │  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0144┤  +   +  +        +   +     +   │
0.0081┤  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0131┤++++++++++++++++++++++++++++++++│
0.0071┤  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0104┤ +++++++++++++++++++++++++++++++│
      │  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0091┤  +  +   +  +  +  +  ++  +  +   │
0.0060┤     ·   ·  ·  ·  ·   ·  ·  ·   │0.0064┤  +  +   +  +     +   +  +  +   │
      └┬───────┬───────┬──────┬───────┬┘      └┬───────┬───────┬──────┬───────┬┘
      1.0    49.2    97.5   145.8 194.0       1.0    49.2    97.5   145.8 194.0
dtf/mean             iter               dtf/max              iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/dtf.txt
      ┌────────────────────────────────────────────────────────────────────────┐
0.0144┤ ++ dtf/max  +                            +                             │
      │ -- dtf/min  +      +                     +       +                     │
      │ ·· dtf/mean +      +                     +      ++             +       │
      │ ▞▞ dtf      +      +                     +      ++      +      +       │
0.0129┤     ++      + +    +       +   ++ +   ++ +  +   ++     ++   +  +       │
      │     ++    + +++ + +++    ++++ +++ ++  ++ + ++   ++    +++ +++ ++  +  + │
      │ ++++▗·+++++++++ +++++ +++++++++++ +++++++++++++++++++++++++++++++++++++│
      │ ++++█·+++++++++++++++++++++++++++ ++++++++++++++++++++++·++++++++++++++│
0.0115┤+ +++█·++++++▖++++++++++++++++++++ ·++++++▗+++++++·++++++▟++++++++++++++│
      │  +++█·+++++▐▌ +++++·+++++++·++++++▌++++++█+ ++++▗▌++++++█++++++·▖++++++│
      │▖·+++█··+ + ▐▌  ++++▗++ ++·▗▌+·+ ·+▌+ ++ +█+  + +▐▌ +++++█+·++++▐▌+++++ │
      │▌·   █·· ▗  ▐▌· ·  +█ · ···▐▌··  ·▟▌   + ·█+     ▐▌   ·++█·▗  + ▐▌++ · ▖│
0.0100┤█▜···█···█··▐▌▗·▗···█······▐▌▗▌··▗█▌···▖·▗█······▐▌······█·█·▌··▐▌···▐▐▌│
      │█▐▖▖▐▌▙▌▐▐··▟▌█▖█···█▗▟▗▌▗▌▐▌▐▌▗ ███· ▐▌▗██···▄·▗█▌··▗▌▖·█▖█▖▌  ▐▌··▗▐▐▌│
      │▝-▀▚▞▌▝▝█▝▟▟█▐▝▙▜▞▞▜▛▛▛▟▚█▐█▚▜▐▛▞▌▜▛▞▀▞▝▛▌▌▜▄█▝▀▞▐▝▚▚▟▜▛▚▛▛▀▚▀▄▜▟▚▟▄▛▛▟▙│
      │     ▌  -  -█·     ▐▌  -   █      ▐▌  --  ▌   ---▐-- -  ▐▌  -   █  -   -│
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
0.0085┤     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
0.0070┤     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      ▜-     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
0.0055┤             -      ▘      ▝       ▘      ▘      ▝       ▘      ▝       │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
      1.0              49.2              97.5             145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/dtf_summary.txt
                dtf/mean hist                           dtf/max hist
     ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
152.0┤                    ███          │75.0┤                        ███       │
126.7┤                    ███          │62.5┤                 ████   ███       │
     │                    ███          │    │                 ████   ███       │
101.3┤                    ███          │50.0┤                 ████   ███       │
 76.0┤                    ███          │37.5┤                 ████   ███       │
     │                    ███          │    │                 ████   ███       │
 50.7┤                    ███          │25.0┤                 ██████████       │
 25.3┤                    ███          │12.5┤                 ██████████████   │
     │████                ██████       │    │████             █████████████████│
  0.0┤███             █████████████████│ 0.0┤███████          █████████████████│
     └┬───────┬───────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
   0.0057  0.0074  0.0092  0.0109         0.0061  0.0083   0.0104  0.0126
               dtf/min hist                             dtf/std hist
   ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
156┤                     ████          │31.0┤       ███                        │
   │                     ████          │    │       ███                        │
130┤                     ████          │25.8┤       ███              ███       │
104┤                     ████          │20.7┤   ███████              ███████   │
   │                     ████          │    │   ██████████████       ███████   │
 78┤                     ████          │15.5┤   ███████████████████████████████│
   │                     ████          │    │██████████████████████████████████│
 52┤                     ████          │10.3┤██████████████████████████████████│
 26┤                     ████          │ 5.2┤██████████████████████████████████│
   │████                 ███████       │    │██████████████████████████████████│
  0┤███                  ██████████████│ 0.0┤██████████████████████████████████│
   └┬────────┬───────┬────────┬────────┘    └┬───────┬────────┬───────┬────────┘
  0.0052  0.0069  0.0085   0.0101         0.00023  0.00044  0.00065 0.00086
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/dtf_hist.txt
                    loss                                  loss/min
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
1.75┤▌                                 │1.66┤-                                 │
1.46┤▌                                 │1.11┤--                                │
    │▐                                 │0.56┤ -----                            │
1.17┤▐▌                                │0.01┤    ------------------------------│
0.89┤ ▙█                               │    └┬───────┬────────┬───────┬───────┬┘
0.60┤ ▝▜▄                              │    1.0    49.2     97.5    145.8 194.0
    │   ▀█▄▖▖▖                         │loss/min            iter
0.31┤    ▝▘▝▀▛▀▙▟▄▙▄▄▗   ▖  ▗          │                  loss/std
0.02┤           ▝ ▀▀▀▀▀▀▀▛▜▛▜▜█▜▜▟█▟▄▄▄│     ┌─────────────────────────────────┐
    └┬───────┬────────┬───────┬───────┬┘0.137┤  *                              │
    1.0    49.2     97.5    145.8 194.0 0.116┤ ******                          │
loss                iter                0.073┤** **********                    │
                  loss/mean             0.052┤*      **************************│
    ┌──────────────────────────────────┐0.009┤                  *** * ******** │
1.74┤·                                 │     └┬───────┬───────┬───────┬───────┬┘
1.46┤·                                 │     1.0    49.2    97.5    145.8 194.0
    │·                                 │loss/std            iter
1.17┤ ·                                │                  loss/max
0.89┤ ·                                │    ┌──────────────────────────────────┐
    │ ··                               │1.83┤+                                 │
0.60┤  ···                             │1.53┤++                                │
0.32┤    ····                          │0.94┤ ++++                             │
    │       ···········                │0.65┤    +++++++++++                   │
0.03┤             ·  ··················│0.06┤           +++++++++++++++++++++++│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
    1.0    49.2     97.5    145.8 194.0     1.0    49.2     97.5    145.8 194.0
loss/mean           iter                loss/max            iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/loss.txt
    ┌──────────────────────────────────────────────────────────────────────────┐
1.83┤ ++ loss/max                                                              │
    │ -- loss/min                                                              │
    │ ·· loss/mean                                                             │
    │ ▞▞ loss                                                                  │
1.53┤▐+                                                                        │
    │▐+                                                                        │
    │▐·                                                                        │
    │ ▌                                                                        │
1.22┤ ▌+                                                                       │
    │ ▌+                                                                       │
    │ ▙▌                                                                       │
    │ ▝▌+                                                                      │
0.92┤  ▌+ ▗                                                                    │
    │  ▌ ▌█ +                                                                  │
    │  ▐▌▌█+++                                                                 │
    │  -▝██+ ++                                                                │
    │   -▝█▖  ++                                                               │
0.62┤   --·▚▗▌ +++   +                                                         │
    │    --▐▞▚▟ ++++++                                                         │
    │    ---▘▝█▗▗▙▚+++▌     +                                                  │
    │     ----▝▌▜▝▝▄▄ ▌+++▖++++ +                                              │
0.31┤        ---- ··▀▞▚▞▚▛▐▖++▖++▗ ++  +                                       │
    │         -  -----▝▌··▝▚▙▞▙▚▞▌▌++▗+++▄+++++++▖ +   +     + + +             │
    │             - -----·---▘█··▌▜▀▙▜▞▚▖█▗▄▗▖ ▗▗▌+▖▗▗++▗▌++++++▗+++++▖  ++++++│
    │                    - ---▝--·----▌-▝▌▀·▌▚▀▀█▝▟▚▛▟▀▟▌▚▟▗▙▗▄▄▛▄▄▚▖▟▌▄▌·▄▗···│
0.01┤                                    -     -▝---▘---▘--▘-▘-▝--▝-▝▝▝▝▝▀▝▘▀▀▞│
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
    1.0              49.2               97.5              145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/loss_summary.txt
               loss/mean hist                           loss/max hist
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
127.0┤████                             │107.0┤████                             │
105.8┤████                             │ 89.2┤████                             │
     │████                             │     │████                             │
 84.7┤████                             │ 71.3┤████                             │
 63.5┤████                             │ 53.5┤████                             │
     │████                             │     │████                             │
 42.3┤████                             │ 35.7┤███████                          │
 21.2┤███████                          │ 17.8┤███████                          │
     │█████████████                    │     │█████████████████                │
  0.0┤█████████████████████████████████│  0.0┤█████████████████████████████████│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
    -0.04   0.42    0.89    1.35   1.82     -0.02   0.46    0.94    1.43   1.91
                loss/min hist                           loss/std hist
     ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
146.0┤████                             │73.0┤   ████                           │
     │████                             │    │   ████                           │
121.7┤████                             │60.8┤   ████                           │
 97.3┤████                             │48.7┤   ████                           │
     │████                             │    │   ███████                        │
 73.0┤████                             │36.5┤   ███████                        │
     │████                             │    │   ███████                        │
 48.7┤████                             │24.3┤   ███████████                    │
 24.3┤███████                          │12.2┤   █████████████████████          │
     │██████████                       │    │███████████████████████████       │
  0.0┤█████████████████████████████████│ 0.0┤██████████████████████████████████│
     └┬───────┬───────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
    -0.06   0.39    0.83    1.28   1.73    0.004   0.038    0.073   0.108 0.142
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/plots/tplot/loss_hist.txt
[2025-12-31 11:31:37,843701][I][ezpz/history:2433:finalize] Saving history report to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.examples.test/2025-12-31-113049/report.md
[2025-12-31 11:31:37,850659][I][ezpz/examples/test:348:finalize] dataset=<xarray.Dataset> Size: 39kB
Dimensions:        (draw: 194)
Coordinates:
  * draw           (draw) int64 2kB 0 1 2 3 4 5 6 ... 188 189 190 191 192 193
Data variables: (12/25)
    iter           (draw) int64 2kB 6 7 8 9 10 11 12 ... 194 195 196 197 198 199
    loss           (draw) float32 776B 1.749 1.581 1.343 ... 0.02264 0.03851
    accuracy       (draw) float32 776B 0.6094 0.625 0.6797 ... 0.9922 1.0 1.0
    dtf            (draw) float64 2kB 0.01076 0.009483 ... 0.009295 0.009377
    dtb            (draw) float64 2kB 0.001477 0.001752 ... 0.00169 0.001521
    iter_mean      (draw) float64 2kB 6.0 7.0 8.0 9.0 ... 197.0 198.0 199.0
    ...             ...
    dtf_min        (draw) float64 2kB 0.00996 0.009416 ... 0.009295 0.009235
    dtf_std        (draw) float64 2kB 0.0004828 0.0005322 ... 0.0005547
    dtb_mean       (draw) float64 2kB 0.001618 0.001605 ... 0.001609 0.001615
    dtb_max        (draw) float64 2kB 0.001847 0.001759 ... 0.001745 0.001847
    dtb_min        (draw) float64 2kB 0.001336 0.00131 ... 0.001343 0.00108
    dtb_std        (draw) float64 2kB 0.0001653 0.0001423 ... 0.0001303 0.000167
[2025-12-31 11:31:38,752472][I][ezpz/examples/test:500:train] Took: 32.90 seconds to finish training
[2025-12-31 11:31:38,753362][I][ezpz/examples/test:695:main] Took: 52.63 seconds
wandb:
wandb: 🚀 View run polar-surf-6861 at:
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_113103-cppqal9m/logs
[2025-12-31 11:31:40,781540][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-113140]----
[2025-12-31 11:31:40,782309][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 11:31:40,782738][I][ezpz/launch:449:launch] Executing finished in 57.13 seconds.
[2025-12-31 11:31:40,783099][I][ezpz/launch:450:launch] Took 57.13 seconds to run. Exiting.
took: 1m 5s
```

</details>



