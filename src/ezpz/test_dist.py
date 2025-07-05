"""
test_dist.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""

import argparse
from contextlib import nullcontext
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
from ezpz.profile import (
    get_pytorch_profiler,
    get_context_manager,
    # get_torch_profiler_context_manager,
)

import torch
import torch.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
from xarray import Dataset


START_TIME = time.perf_counter()  # start time

# noqa: E402
warnings.filterwarnings("ignore")

WANDB_DISABLED = False
try:
    import wandb

except Exception:
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
    pytorch_profiler_wait: int
    pytorch_profiler_warmup: int
    pytorch_profiler_active: int
    pytorch_profiler_repeat: int
    backend: str = "DDP"
    dtype: Optional[str] = None
    pyinstrument_profiler: Optional[bool] = None
    pytorch_profiler: Optional[bool] = None
    profile_memory: bool = True
    record_shapes: bool = True
    with_stack: bool = True
    with_flops: bool = True
    with_modules: bool = True
    acc_events: bool = False
    layer_sizes: list = field(default_factory=lambda: [1024, 512, 256, 128])

    def __post_init__(self):
        self._created_at = (
            ezpz.get_timestamp() if ezpz.get_rank() == 0 else None
        )
        ezpz.dist.broadcast(self._created_at, root=0)
        self.outdir = Path(os.getcwd()).joinpath(
            "outputs", "ezpz.test_dist", f"{self._created_at}"
        )
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.ctx = self.get_profiling_context(
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
        )
        logger.info(f"Outputs will be saved to {self.outdir}")

    def get_profiling_context(
        self,
        wait: int,
        warmup: int,
        active: int,
        repeat: int,
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
        acc_events: bool = False,
        profile_memory: bool = True,
    ):
        """
        Returns a context manager for profiling based on the configuration.
        """
        if self.pytorch_profiler and self.pyinstrument_profiler:
            raise ValueError(
                "Cannot use both PyTorch profiler and pyinstrument profiler at the same time."
            )
        elif self.pytorch_profiler and not self.pyinstrument_profiler:
            logger.info("Using PyTorch profiler")

            # Non-default profiler schedule allows user to turn profiler on and off
            # on different iterations of the training loop;
            # trace_handler is called every time a new trace becomes available
            def trace_handler(prof: torch.profiler.profile):
                logger.info(
                    "\n"
                    + prof.key_averages().table(
                        sort_by=(
                            f"self_{ezpz.get_torch_device_type()}_time_total"
                        ),
                        row_limit=-1,
                    )
                )
                fname: str = "-".join(
                    [
                        "torch-profile",
                        f"{ezpz.get_rank()}",
                        f"{self._created_at}",
                        f"{str(prof.step_num)}",
                    ]
                )
                trace_output: Path = Path(self.outdir).joinpath(
                    f"{fname}.json"
                )
                logger.info(
                    f"Saving trace at step {prof.step_num} to: "
                    f"{trace_output.as_posix()}"
                )
                prof.export_chrome_trace(trace_output.as_posix())

            schedule = torch.profiler.schedule(
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
            )

            # torch.profiler.profile(,)
            ctx = get_pytorch_profiler(
                rank=ezpz.get_rank(),
                schedule=schedule,
                on_trace_ready=trace_handler,
                profile_memory=profile_memory,
                record_shapes=record_shapes,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
                acc_events=acc_events,
            )
        elif not self.pytorch_profiler and self.pyinstrument_profiler:
            ctx = get_context_manager(rank=ezpz.get_rank(), strict=False)
        else:
            ctx = nullcontext()
        return ctx

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
            torch.distributed.barrier(group=tpgroup)
            dpgroup = ezpz.tp.get_data_parallel_group()
            torch.distributed.barrier(group=dpgroup)
            ppgroup = ezpz.tp.get_pipeline_parallel_group()
            torch.distributed.barrier(group=ppgroup)
            cpgroup = ezpz.tp.get_context_parallel_group()
            torch.distributed.barrier(group=cpgroup)

        if self.rank == 0 and not WANDB_DISABLED:
            import wandb

            logger.debug("Setting up wandb")
            wbconfig = {}
            wbconfig.update(asdict(self.config))
            wbconfig.update(ezpz.get_dist_info())
            run = ezpz.setup_wandb(
                project_name="ezpz.test_dist",
                config=wbconfig,
            )
            if (wbrun := getattr(wandb, "run", None)) is not None and callable(
                wbrun.watch
            ):
                wbrun.watch(self.model, log="all")

        if self.world_size > 1:
            logger.debug("Hit torch.distributed.barrier()")
            torch.distributed.barrier()

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
            self.model.backward(loss)  # type:ignore
            self.model.step(loss)  # type:ignore
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

    def finalize(
        self, outdir: Optional[str | Path | os.PathLike] = None
    ) -> Dataset:
        import matplotlib.pyplot as plt
        import ambivalent

        plt.style.use(ambivalent.STYLES["ambivalent"])
        outdir = Path(outdir) if outdir is not None else self.config.outdir
        dataset = self.history.finalize(
            run_name="ezpz.test_dist",
            dataset_fname="train",
            warmup=self.config.warmup,
            save=False,  # XXX: don't bother saving test data
            plot=(self.rank == 0),
            outdir=outdir,
        )
        logger.info(f"{dataset=}")
        return dataset

    def train(
        self, profiler: Optional[torch.profiler.profile] = None
    ) -> Dataset:
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


def train(
    config: TrainConfig, profiler: Optional[torch.profiler.profile] = None
) -> Trainer:
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
    model, optimizer = build_model_and_optimizer(model, backend=config.backend)
    t2m = time.perf_counter()
    dt_optimizer = time.perf_counter() - t1m
    logger.info(f"Took: {dt_optimizer:.2f} seconds to build optimizer")
    trainer = Trainer(config=config, model=model, optimizer=optimizer)
    t1tr = time.perf_counter()
    logger.info(
        f"Took: {(dt_trainer := t1tr - t2m):.2f} seconds to build trainer"
    )
    jstr = json.dumps(asdict(config), indent=2, sort_keys=True)
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
    # if not WANDB_DISABLED:
    #     try:
    # if wandb is not None and getattr(wandb, "run", None) is not None:
    #     wandb.log(timings)

    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training configuration parameters"
    )
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
        "--pyinstrument-profiler",
        action="store_true",
        help="Profile the training loop",
    )
    parser.add_argument(
        "--pytorch-profiler",
        action="store_true",
        help="Use PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-wait",
        type=int,
        default=1,
        help="Wait time before starting the PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-warmup",
        type=int,
        default=2,
        help="Warmup iterations for the PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-active",
        type=int,
        default=3,
        help="Active iterations for the PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-repeat",
        type=int,
        default=5,
        help="Repeat iterations for the PyTorch profiler",
    )
    parser.add_argument(
        "--profile-memory",
        default=True,
        action="store_true",
        help="Profile memory usage",
    )
    parser.add_argument(
        "--record-shapes",
        default=True,
        action="store_true",
        help="Record shapes in the profiler",
    )
    parser.add_argument(
        "--with-stack",
        default=True,
        action="store_true",
        help="Include stack traces in the profiler",
    )
    parser.add_argument(
        "--with-flops",
        default=True,
        action="store_true",
        help="Include FLOPs in the profiler",
    )
    parser.add_argument(
        "--with-modules",
        default=True,
        action="store_true",
        help="Include module information in the profiler",
    )
    parser.add_argument(
        "--acc-events",
        default=False,
        action="store_true",
        help="Accumulate events in the profiler",
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
        backend=args.backend,
        dtype=args.dtype,
        log_freq=args.log_freq,
        print_freq=args.print_freq,
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        batch_size=args.batch_size,
        input_size=args.input_size,
        output_size=args.output_size,
        train_iters=args.train_iters,
        layer_sizes=args.layer_sizes,
        pyinstrument_profiler=args.pyinstrument_profiler,
        pytorch_profiler=args.pytorch_profiler,
        pytorch_profiler_wait=args.pytorch_profiler_wait,
        pytorch_profiler_warmup=args.pytorch_profiler_warmup,
        pytorch_profiler_active=args.pytorch_profiler_active,
        pytorch_profiler_repeat=args.pytorch_profiler_repeat,
        warmup=args.warmup,
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
    with config.ctx as c:
        _ = ezpz.setup_torch(
            backend=config.backend,
            tensor_parallel_size=config.tp,
            pipeline_parallel_size=config.pp,
            context_parallel_size=config.cp,
        )
        logger.info(
            f"Took: {time.perf_counter() - t0:.2f} seconds to setup torch"
        )
        trainer = train(config, profiler=c)
    if trainer.config.backend.lower() in ["ds", "deepspeed"]:
        try:
            import deepspeed
            import deepspeed.comm

            deepspeed.comm.log_summary()
        except ImportError as e:
            logger.exception(
                "Deepspeed not available. "
                "Install via `python3 -m pip install deepspeed`"
            )
            logger.info("Continuing without deepspeed summary...")

    logger.info(f"Took: {time.perf_counter() - START_TIME:.2f} seconds")
    return trainer


if __name__ == "__main__":
    import sys

    trainer = main()
    ezpz.cleanup()
    sys.exit(0)
