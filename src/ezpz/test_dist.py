"""
test_dist.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Optional
import warnings

# from ezpz.lazy import lazy_import
# ezpz = lazy_import('ezpz')
import ezpz

# import ezpz.tp

import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from xarray import Dataset


START_TIME = time.perf_counter()  # start time

# noqa: E402

warnings.filterwarnings("ignore")

try:
    import wandb
    WANDB_DISABLED = (
        os.environ.get("WANDB_DISABLED", False)
        or os.environ.get("WANDB_MODE", "online").lower() == "disabled"
    )

except Exception:
    wandb = None
    WANDB_DISABLED = True


ModelOptimizerPair = tuple[torch.nn.Module, torch.optim.Optimizer]

logger = ezpz.get_logger(__name__)


@dataclass
class TrainConfig:
    warmup: int
    tp: int
    pp: int
    cp: int
    batch_size: int
    input_size: int
    output_size: int
    train_iters: int
    log_freq: int
    print_freq: int
    backend: str = "DDP"
    dtype: Optional[str] = None
    pyinstrument_profiler: Optional[bool] = None
    layer_sizes: list = field(default_factory=lambda: [1024, 512, 256, 128])

    def __post_init__(self):
        from contextlib import nullcontext
        from ezpz.profile import get_context_manager

        self.ctx = (
            get_context_manager(rank=ezpz.get_rank(), strict=False)
            if self.pyinstrument_profiler
            else nullcontext()
        )

    def get_torch_dtype(self) -> torch.dtype:
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
        logger.warning(f"Unknown dtype: {self.dtype=}, using float32")
        return torch.float32


@dataclass
class Trainer:
    config: TrainConfig
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    history: ezpz.History = field(default_factory=ezpz.History)
    train_iter: int = 0
    rank: int = ezpz.get_rank()
    device_type: str = ezpz.get_torch_device_type()
    world_size = ezpz.get_world_size()
    local_rank = ezpz.get_local_rank()
    device_id = f"{device_type}:{local_rank}"

    def __post_init__(self):
        self.device_id = f"{self.device_type}:{self.local_rank}"
        self.dtype = self.config.get_torch_dtype()
        self.model.to(self.device_id)
        self.model.to(self.dtype)

        if self.config.tp > 1 or self.config.pp > 1 or self.config.cp > 1:
            tpgroup = ezpz.tp.get_tensor_parallel_group()
            tdist.barrier(group=tpgroup)
            dpgroup = ezpz.tp.get_data_parallel_group()
            tdist.barrier(group=dpgroup)
            ppgroup = ezpz.tp.get_pipeline_parallel_group()
            tdist.barrier(group=ppgroup)
            cpgroup = ezpz.tp.get_context_parallel_group()
            tdist.barrier(group=cpgroup)

        if wandb is not None and not WANDB_DISABLED and self.rank == 0:
            logger.debug("Setting up wandb")
            run = ezpz.setup_wandb(project_name="ezpz.test_dist")
            assert wandb is not None and run is wandb.run and wandb.run is not None
            wandb.run.config.update(ezpz.get_dist_info())
            wandb.run.config.update(asdict(self.config))
            wandb.run.watch(self.model, log="all")

        if self.world_size > 1:
            logger.debug("Hit torch.distributed.barrier()")
            tdist.barrier()

    def _forward_step(self) -> dict:
        t0 = time.perf_counter()
        x = torch.rand(
            *(self.config.batch_size, self.config.input_size),
            device=self.device_type,
            dtype=self.config.get_torch_dtype(),
        )
        y = self.model(x)
        return {"loss": calc_loss(x, y), "dtf": (time.perf_counter() - t0)}

    def _backward_step(self, loss: torch.Tensor) -> float:
        t0 = time.perf_counter()
        if self.config.backend == "deepspeed":
            self.model.backward(loss)
            self.model.step(loss)
        else:
            loss.backward()
            self.optimizer.step()
        return time.perf_counter() - t0

    def train_step(self) -> dict:
        self.train_iter += 1
        metrics = self._forward_step()
        metrics["dtb"] = self._backward_step(metrics["loss"])
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

    def finalize(self) -> Dataset:
        import matplotlib.pyplot as plt
        import ambivalent

        plt.style.use(ambivalent.STYLES["ambivalent"])
        dataset = self.history.finalize(
            run_name="ezpz.test_dist",
            dataset_fname="train",
            warmup=self.config.warmup,
            save=False,  # XXX: don't bother saving test data
            plot=(self.rank == 0),
            outdir=Path(os.getcwd()).joinpath("outputs", "ezpz.test_dist"),
        )
        logger.info(f"{dataset=}")
        return dataset

    def train(self) -> Dataset:
        for step in range(self.config.train_iters):
            if step == self.config.warmup:
                logger.info(f"Warmup complete at step {step}")
            _ = self.train_step()

        return (
            self.finalize()
            if self.rank == 0
            else self.history.get_dataset(warmup=self.config.warmup)
        )


def train(config: TrainConfig) -> Trainer:
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
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    try:
        logger.info(f"\n{model_summary(model)}")
    except Exception as e:
        logger.warning(f"Failed to summarize model: {e}, using default summary")
        logger.info(model)
    t1m = time.perf_counter()
    dt_model = t1m - t0m
    logger.info(f"Took: {dt_model} seconds to build model")
    model, optimizer = build_model_and_optimizer(model, backend=config.backend)
    t2m = time.perf_counter()
    dt_optimizer = time.perf_counter() - t1m
    logger.info(f"Took: {dt_optimizer:.2f} seconds to build optimizer")
    trainer = Trainer(config=config, model=model, optimizer=optimizer)
    t1tr = time.perf_counter()
    logger.info(f"Took: {(dt_trainer := t1tr - t2m):.2f} seconds to build trainer")
    jstr = json.dumps(asdict(config), indent=2, sort_keys=True)
    logger.info(f"config:\n{jstr}")
    t1s = time.perf_counter()
    logger.info(f"Took: {(dt_train_start := t1s - START_TIME):.2f} to get here.")
    t0t = time.perf_counter()
    _ = trainer.train()
    t1t = time.perf_counter()
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
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(timings)

    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training configuration parameters")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline length",
    )

    # parser.add_argument(
    #     '--deepspeed',
    #     action='store_true',
    #     default=True,
    #     help='Use deepspeed',
    # )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="deepspeed_config.json",
        help="Deepspeed config file",
    )
    parser.add_argument(
        "--cp",
        type=int,
        default=1,
        help="Context parallel size",
    )
    parser.add_argument(
        "--backend",
        required=False,
        type=str,
        default="DDP",
        help="Backend (DDP, DeepSpeed, etc.)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the training loop",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1,
        help="Logging frequency",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Printing frequency",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Input size",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=128,
        help="Output size",
    )
    parser.add_argument(
        "--layer-sizes",
        help="Comma-separated list of layer sizes",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[1024, 512, 256, 128],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type (fp16, float16, bfloat16, bf16, float32, etc.)",
    )

    args = parser.parse_args()
    if args.backend.lower() in {"ds", "deepspeed"}:
        try:
            import deepspeed  # type:ignore
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(
                "Deepspeed not available. "
                "Install via `python3 -m pip install deepspeed`"
            )
            raise e
        args.deepspeed = True
    return args


def get_config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(
        warmup=args.warmup,
        log_freq=args.log_freq,
        print_freq=args.print_freq,
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        batch_size=args.batch_size,
        input_size=args.input_size,
        output_size=args.output_size,
        dtype=args.dtype,
        train_iters=args.train_iters,
        backend=args.backend,
        pyinstrument_profiler=args.profile,
        layer_sizes=args.layer_sizes,
    )
    return config


def calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y - x).pow(2).sum()


def build_model_and_optimizer(
    model: torch.nn.Module, backend: str = "DDP"
) -> ModelOptimizerPair:
    if backend is not None:
        assert backend.lower() in {"ddp", "deepspeed", "ds"}
    device_type = ezpz.get_torch_device()
    # device_id = f"{device_type}:{ezpz.get_local_rank()}"
    world_size = ezpz.get_world_size()
    local_rank = ezpz.get_local_rank()
    model.to(device_type)
    model.to(local_rank)
    logger.info(f"model=\n{model}")
    optimizer = torch.optim.Adam(model.parameters())
    if backend.lower() == "ddp":
        if world_size > 1:
            # model = DDP(model)
            model = DDP(model, device_ids=[local_rank])

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


def main() -> Trainer:
    t0 = time.perf_counter()
    args = parse_args()
    config = get_config_from_args(args)
    with config.ctx:
        _ = ezpz.setup_torch(
            backend=config.backend,
            tensor_parallel_size=config.tp,
            pipeline_parallel_size=config.pp,
            context_parallel_size=config.cp,
        )
        logger.info(f"Took: {time.perf_counter() - t0:.2f} seconds to setup torch")
        trainer = train(config)

    if trainer.config.backend.lower() in ["ds", "deepspeed"]:
        import deepspeed.comm as dscomm  # type:ignore

        dscomm.log_summary()

    logger.info(f"Took: {time.perf_counter() - START_TIME:.2f} seconds")
    return trainer


if __name__ == "__main__":
    import sys
    trainer = main()
    ezpz.cleanup()
    sys.exit(0)
