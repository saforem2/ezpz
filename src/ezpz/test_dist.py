#!/usr/bin/env python3
"""
test_dist.py

- to launch:

  ```bash
  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
  ```
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
from typing import Iterator, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from xarray import Dataset

import ezpz
from ezpz.configs import PathLike
from ezpz.cli.flags import build_test_parser
from ezpz.profile import get_profiling_context

START_TIME = time.perf_counter()  # start time

# noqa: E402
warnings.filterwarnings("ignore")

ModelOptimizerPair = tuple[torch.nn.Module, torch.optim.Optimizer]

logger = ezpz.get_logger(__name__)

WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
WANDB_MODE = os.environ.get("WANDB_MODE", "").lower()
if not WANDB_DISABLED and WANDB_MODE != "disabled":
    try:
        wandb = ezpz.lazy.lazy_import("wandb")
        if not ezpz.dist.verify_wandb():
            logger.warning("W&B API key not found, skipping wandb setup!")
            logger.info(
                "To enable W&B logging, run `wandb login` or set the WANDB_API_KEY"
            )
    except Exception as e:
        wandb = None
        WANDB_DISABLED = True
        logger.exception(e)
        logger.warning("W&B not available, skipping wandb setup!")
        logger.info("Continue without W&B logging...")
else:
    wandb = None
    WANDB_DISABLED = True


@dataclass
class TrainConfig:
    """Runtime configuration for the ``ezpz.test_dist`` distributed smoke test."""

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

    def __post_init__(self):
        """Initialise output paths and configure profiling context managers."""
        self._created_at = (
            ezpz.get_timestamp() if ezpz.get_rank() == 0 else None
        )
        self._created_at = ezpz.dist.broadcast(self._created_at, root=0)
        self.outdir = Path(os.getcwd()).joinpath(
            "outputs", "ezpz.test_dist", f"{self._created_at}"
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
        self.history = ezpz.history.History(
            report_dir=self.config.outdir,
            report_enabled=True,
            jsonl_path=metrics_path,
            jsonl_overwrite=True,
            distributed_history=(
                1 < self.world_size <= 384 and not self.config.pytorch_profiler
            ),
        )

        if self.config.tp > 1 or self.config.pp > 1 or self.config.cp > 1:
            ezpz.dist.barrier(group=ezpz.tp.get_tensor_parallel_group())
            ezpz.dist.barrier(group=ezpz.tp.get_data_parallel_group())
            ezpz.dist.barrier(group=ezpz.tp.get_pipeline_parallel_group())
            ezpz.dist.barrier(group=ezpz.tp.get_context_parallel_group())

        if self.rank == 0 and not WANDB_DISABLED:
            logger.debug("Setting up wandb")
            wbconfig = {}
            wbconfig |= asdict(self.config)
            wbconfig |= ezpz.get_dist_info()
            _ = ezpz.setup_wandb(
                project_name="ezpz.test_dist",
                config=wbconfig,
            )
            if (wbrun := getattr(wandb, "run", None)) is not None and callable(
                wbrun.watch
            ):
                wbrun.watch(self.model, log="all")

        if self.world_size > 1:
            logger.debug("Hit torch.distributed.barrier()")
            ezpz.dist.barrier()
        self._train_loader, self._train_iterator = self._build_dataloader()
        self._feature_dim = self.config.input_size

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

    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch from the training dataloader."""
        assert self._train_loader is not None
        assert self._train_iterator is not None
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self._train_loader)
            return next(self._train_iterator)

    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Move inputs to the configured device and coerce feature dimensions."""
        inputs = inputs.to(self.device_id)
        inputs = inputs.reshape(inputs.size(0), -1)
        if inputs.size(1) < self._feature_dim:
            pad = self._feature_dim - inputs.size(1)
            inputs = torch.nn.functional.pad(inputs, (0, pad))
        elif inputs.size(1) > self._feature_dim:
            inputs = inputs[:, : self._feature_dim]
        return inputs.to(self.dtype)

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
        ezpz.dist.synchronize()
        metrics = {
            "loss": loss.detach(),
            "accuracy": accuracy.detach(),
            "dtf": time.perf_counter() - t0,
        }
        return metrics, loss

    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def _backward_step(self, loss: torch.Tensor) -> float:
        """Perform the backwards/optimiser step and return elapsed seconds."""
        t0 = time.perf_counter()
        if self.config.backend == "deepspeed":
            self.model.backward(loss)  # type:ignore
            self.model.step(loss)  # type:ignore
        else:
            loss.backward()
            self.optimizer.step()
        ezpz.dist.synchronize()
        return time.perf_counter() - t0

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
            run_name="ezpz.test_dist",
            dataset_fname="train",
            warmup=self.config.warmup,
            save=False,  # XXX: don't bother saving test data
            plot=(self.rank == 0),
            outdir=outdir,
            env_info=env_info,
        )
        logger.info(f"{dataset=}")
        if wandb is not None and not WANDB_DISABLED:
            try:
                wandb.log(
                    {
                        "train_metrics": wandb.Table(
                            dataframe=dataset.to_dataframe()
                        )
                    }
                )
            except Exception:
                pass
        return dataset

    @ezpz.timeitlogit(rank=ezpz.get_rank())
    def train(
        self, profiler: Optional[torch.profiler.profile] = None
    ) -> Dataset:
        """Loop over all training iterations and return the final dataset."""
        for step in range(self.config.train_iters):
            if step == self.config.warmup:
                logger.info(f"Warmup complete at step {step}")
            _ = self.train_step()
            if profiler is not None:
                profiler.step()

        return (
            self.finalize()
            if self.rank == 0
            else self.history.get_dataset(warmup=self.config.warmup)
        )

    def _gather_environment_snapshot(self) -> dict[str, dict[str, str]]:
        """Collect key runtime environment details for reporting."""

        python_details = {
            "Version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "Implementation": sys.implementation.name,
            "Executable": sys.executable,
        }

        torch_details = {
            "Version": torch.__version__,
            "Device": str(self.device_id),
            "Backend": (
                ezpz.dist.get_backend()
                if hasattr(ezpz.dist, "get_backend")
                else "unknown"
            ),
        }

        host_name = (
            platform.uname().node
            if hasattr(platform, "uname")
            else os.environ.get("HOSTNAME", "unknown")
        )
        path_details = {
            "Working directory": str(Path.cwd()),
            "Output directory": str(self.config.outdir),
            "Dataset root": str(self.config.dataset_root),
            "Hostname": host_name,
        }

        dist_details = {
            "Rank": str(self.rank),
            "Local rank": str(self.local_rank),
            "World size": str(self.world_size),
        }

        env_vars: dict[str, str] = {}
        for key in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "NODE_RANK",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
        ):
            value = os.environ.get(key)
            if value is not None:
                env_vars[key] = value

        snapshot: dict[str, dict[str, str]] = {
            "Paths": path_details,
            "Python": python_details,
            "Torch": torch_details,
            "Distributed": dist_details,
        }
        if env_vars:
            snapshot["Environment Variables"] = env_vars
        return snapshot


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    config: TrainConfig, profiler: Optional[torch.profiler.profile] = None
) -> Trainer:
    """Instantiate the model/optimiser and run the training loop."""
    from ezpz.models.minimal import SequentialLinearNet
    from ezpz.utils import model_summary

    # logger.info(f"Setting up torch with {config.backend=}...")
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
        model, backend=config.backend, dtype=config.dtype
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
    # -------------------------------------------

    # Record timings and return trainer
    logger.info(
        f"Took: {(dt_train_duration := t1t - t0t):.2f} seconds to finish training"
    )
    timings = {
        "timings/model": dt_model,
        "timings/optimizer": dt_optimizer,
        "timings/trainer": dt_trainer,
        "timings/training_start": dt_train_start,
        "timings/train_duration": dt_train_duration,
    }
    try:
        wandb.log(timings)  # type:ignore
    except Exception:
        pass

    return trainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ``ezpz.test_dist``."""
    parser = build_test_parser()
    args = parser.parse_args()
    if args.backend.lower() in {"ds", "deepspeed"}:
        try:
            import deepspeed  # type:ignore  # noqa: F401

            args.deepspeed = True
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(
                "Deepspeed not available. "
                "Install via `python3 -m pip install deepspeed`"
            )
            raise e
    return args


def get_config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Translate CLI arguments into a :class:`TrainConfig`."""
    config = TrainConfig(
        acc_events=args.acc_events,
        batch_size=args.batch_size,
        profile_memory=args.profile_memory,
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
        with_modules=args.with_modules,
        rank_zero_only=args.rank_zero_only,
        backend=args.backend,
        dtype=args.dtype,
        log_freq=args.log_freq,
        print_freq=args.print_freq,
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        input_size=args.input_size,
        output_size=args.output_size,
        train_iters=args.train_iters,
        layer_sizes=args.layer_sizes,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        pyinstrument_profiler=args.pyinstrument_profiler,
        pytorch_profiler=args.pytorch_profiler,
        pytorch_profiler_wait=args.pytorch_profiler_wait,
        pytorch_profiler_warmup=args.pytorch_profiler_warmup,
        pytorch_profiler_active=args.pytorch_profiler_active,
        pytorch_profiler_repeat=args.pytorch_profiler_repeat,
        warmup=args.warmup,
    )
    return config


@ezpz.timeitlogit(rank=ezpz.get_rank())
def calc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the cross entropy loss for the classification dataset."""
    return torch.nn.functional.cross_entropy(logits.float(), targets)


@ezpz.timeitlogit(rank=ezpz.get_rank())
def build_model_and_optimizer(
    model: torch.nn.Module,
    dtype: Optional[str] = None,
    backend: str = "DDP",
) -> ModelOptimizerPair:
    """Prepare the model and optimiser for the requested backend."""
    if backend is not None:
        assert backend.lower() in {"ddp", "deepspeed", "ds"}
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
    if backend.lower() == "ddp":
        if world_size > 1:
            model.to(device_type)
            model = ezpz.dist.wrap_model(
                model=model, use_fsdp=False, dtype=dtype
            )
            # model = DDP(model)
            try:
                if isinstance(device_type, str) and device_type.startswith(
                    "cuda"
                ):
                    model = DDP(model, device_ids=[local_rank])
                else:
                    model = DDP(model)
            except Exception:
                model = DDP(model)

    elif backend.lower() in ("ds", "deepspeed"):
        parser = argparse.ArgumentParser(
            prog="deepspeed", description="My training script."
        )
        parser.add_argument(
            "--local_rank",
            required=False,
            type=int,
            default=-1,
            help="local rank passed from distributed launcher",
        )
        # parser.add_argument(
        #     '--deepspeed',
        #     action='store_true',
        #     default=True,
        #     help='Use deepspeed',
        # )
        # parser.add_argument(
        #     '--deepspeed_config',
        #     type=str,
        #     default='deepspeed_config.json',
        #     help='Deepspeed config file',
        # )
        try:
            import deepspeed  # type:ignore
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(
                "Deepspeed not available. "
                "Install via `python3 -m pip install deepspeed`"
            )
            raise e

        # Include DeepSpeed configuration arguments
        parser = deepspeed.add_config_arguments(parser)
        cmd_args = parser.parse_args()
        model, optimizer, *_ = deepspeed.initialize(
            args=cmd_args,
            model=model,
            optimizer=optimizer,
        )
        logger.info(f"{cmd_args=}")
    return model, optimizer


@ezpz.timeitlogit(rank=ezpz.get_rank())
def main() -> Trainer:
    """Entry point used by ``python -m ezpz.test_dist``."""
    t0 = time.perf_counter()
    args = parse_args()
    config = get_config_from_args(args)
    timings = {}
    _ = ezpz.setup_torch(  # noqa
        backend=config.backend,
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
    if wandb is not None and (wbrun := getattr("wandb", "run", None)) is not None:
        try:
            wandb.log(data=timings)
        except Exception:
            logger.warning("Failed to log timings to wandb")
        logger.info(f"{wbrun.url=}")
    return trainer


if __name__ == "__main__":
    import sys

    trainer = main()
    ezpz.cleanup()
    sys.exit(0)
