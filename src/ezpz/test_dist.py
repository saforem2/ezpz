"""
ezpz_ddp.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""
import time

from ezpz.history import History, summarize_dict
T0 = time.perf_counter()  # start time
timers_import = {}

import os  # noqa E402
t_os = time.perf_counter()

import logging  # noqa: E402
t_logging = time.perf_counter()

from typing import Optional  # noqa: E402
t_typing = time.perf_counter()

from pathlib import Path     # noqa: E402
t_pathlib = time.perf_counter()

import ezpz as ez  # noqa: E402
t_ezpz = time.perf_counter()

import torch  # noqa: E402
import torch.distributed as tdist  # noqa: E402
t_torch = time.perf_counter()

from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
t_torch_ddp = time.perf_counter()

try:
    import wandb
    wandb.require("core")
    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
except Exception:
    wandb = None
    WANDB_DISABLED = True
t_wandb = time.perf_counter()

timers_import = {
    'os': t_os - T0,
    'logging': t_logging - t_os,
    'typing': t_typing - t_logging,
    'pathlib': t_pathlib - t_typing,
    'ezpz': t_ezpz - t_pathlib,
    'torch': t_torch - t_ezpz,
    'torch_ddp': t_torch_ddp - t_torch,
    'wandb': t_wandb - t_torch_ddp,
    'total': t_wandb - T0,
}

logger = logging.getLogger(__name__)

# Model = Callable[[torch.Tensor], torch.Tensor]
ModelOptimizerPair = tuple[torch.nn.Module, torch.optim.Optimizer]

T1 = time.perf_counter()  # import time = (T1 - T0)
# backend can be any of DDP, deespepeed, horovod
RANK = ez.setup_torch(
    backend=(
        backend := os.environ.get('BACKEND', 'DDP')
    ),
    port=(
        port := os.environ.get("MASTER_PORT", "29500")
    )
)
T2 = time.perf_counter()  # torch_setup_time = (T2 - T1)
TIMERS = {
    'timers/ezpz.setup_torch': T2 - T1,
    'timers/imports': T1 - T0,
}
DEVICE_TYPE = ez.get_torch_device()
WORLD_SIZE = ez.get_world_size()
LOCAL_RANK = ez.get_local_rank()
DEVICE_ID = f"{DEVICE_TYPE}:{LOCAL_RANK}"

# log only from RANK == 0
logger = logging.getLogger(__name__)
logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")

WARMUP = 0
LOG_FREQ = int(os.environ.get("LOG_FREQ", 1))
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))  # 64
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 128))  # 128
OUTPUT_SIZE = int(os.environ.get("OUTPUT_SIZE", 128))  # 128
PYINSTRUMENT_PROFILER = os.environ.get(
    "PYINSTRUMENT_PROFILER",
    None
)
sizes = os.environ.get(
    "LAYER_SIZES",
    os.environ.get(
        "SIZES",
        os.environ.get(
            "LAYERS",
            None,   # [1024, 512, 256, 128]
        )
    )
)
if sizes is not None:
    LAYER_SIZES = [int(i) for i in sizes.split(',')]
    logger.info(f'Caught: {LAYER_SIZES=}')
else:
    LAYER_SIZES = [1024, 512, 256, 128]

# LAYER_SIZES = ()
# [1024, 512, 256, 128]

DTYPE: torch.dtype  = torch.get_default_dtype()
if (dtype := os.environ.get("DTYPE", None)) is not None:
    if dtype.startswith('fp16'):
        DTYPE = torch.half
    elif dtype.startswith('bf16'):
        DTYPE = torch.bfloat16

CONFIG = {
    'warmup': WARMUP,
    'log_freq': LOG_FREQ,
    'batch_size': BATCH_SIZE,
    'input_size': INPUT_SIZE,
    'output_size': OUTPUT_SIZE,
    'dtype': DTYPE,
    'device': DEVICE_TYPE,
    'world_size': WORLD_SIZE,
    'train_iters': TRAIN_ITERS,
}

run = None
if wandb is not None and not WANDB_DISABLED and RANK == 0:
    run = ez.setup_wandb(project_name='ezpz.test_dist')
    assert wandb is not None and run is wandb.run and wandb.run is not None
    wandb.run.config.update(CONFIG)

if RANK == 0:
    ez.dist.log_dict_as_bulleted_list(timers_import, name='timers_import')
    ez.dist.log_dict_as_bulleted_list(CONFIG, name='CONFIG')

tdist.barrier()

class Network(torch.nn.Module):
    def __init__(
            self,
            input_dim: int = 128,
            output_dim: int = 128,
            sizes: Optional[list[int]] = None,
    ):
        super(Network, self).__init__()
        if sizes is None:
            self.layers = torch.nn.Linear(input_dim, output_dim)
        elif len(sizes) > 0:
            layers = [torch.nn.Linear(input_dim, sizes[0])]
            for idx, size in enumerate(sizes[1:]):
                layers.append(
                    torch.nn.Linear(sizes[idx], size)
                )
            layers.append(torch.nn.Linear(sizes[-1], output_dim))
            self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y - x).pow(2).sum()

def build_model_and_optimizer() -> ModelOptimizerPair:
    # :   #  -> tuple[Union[torch.nn.Module, DDP, deepspeed.DeepSpeedEngine], torch:
    model = Network(
        input_dim=INPUT_SIZE,
        output_dim=OUTPUT_SIZE,
        sizes=LAYER_SIZES,  # [1024, 512, 256, 128]
    )
    if RANK == 0 and not WANDB_DISABLED and wandb is not None:
        assert wandb.run is not None
        wandb.run.watch(model, log='all')
    model.to(DEVICE_TYPE)
    model.to(DEVICE_ID)
    logger.info(f'{model=}')
    optimizer = torch.optim.Adam(model.parameters())
    # with profiler:
    if backend.lower() == 'ddp':
        if WORLD_SIZE > 1:
            model = DDP(
                model,
                device_ids=[]
            )
    elif backend.lower() in ('ds', 'deepspeed'):
        import deepspeed
        import argparse
        parser = argparse.ArgumentParser(
            description='My training script.'
        )
        parser.add_argument(
            '--local_rank',
            required=False,
            type=int,
            default=-1,
            help='local rank passed from distributed launcher',
        )
        # Include DeepSpeed configuration arguments
        parser = deepspeed.add_config_arguments(parser)
        cmd_args = parser.parse_args()
        logger.info(f'{cmd_args=}')
        model, optimizer, *_ = deepspeed.initialize(
            args=cmd_args,
            model=model,
            optimizer=optimizer,
        )
    return model, optimizer


def main():
    metrics = {
        'train/dt': [],    # time per iteration
        'train/dtf': [],   # time in forward pass
        'train/dtb': [],   # time in backward pass
        'train/loss': [],  # loss
        'train/iter': [],  # iteration
        'train/sps': [],   # samples per second = (BATCH_SIZE / dt)
    }
    history = History(
        keys=[
            'train/dt',
            'train/dtf',
            'train/dtb',
            'train/loss',
            'train/iter',
            'train/sps',
        ],
    )
    T3 = time.perf_counter()
    TIMERS['timers/init_to_first_step'] = T3 - T0

    model, optimizer = build_model_and_optimizer()

    def _forward_step() -> torch.Tensor:
        x = torch.rand(*(BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE_TYPE)
        y = model(x)
        return calc_loss(x, y)

    def _backward_step(loss: torch.Tensor) -> None:
        if backend == 'deepspeed':
            model.backward(loss)
            model.step(loss)
        else:
            loss.backward()
            optimizer.step()

    # with profiler:
    #     loss = _forward_step()
    # with profiler:
    #     _ = _backward_step(loss)

    for iter in range(TRAIN_ITERS):
        t0 = time.perf_counter()
        loss = _forward_step()
        t1 = time.perf_counter()
        _ = _backward_step(loss)
        t2 = time.perf_counter()
        optimizer.zero_grad()
        if iter > WARMUP and iter % LOG_FREQ == 0:
            dt = t2 - t0
            dtf = t1 - t0
            dtb = t2 - t1
            sps = BATCH_SIZE / dt
            _metrics = {
                'train/iter': iter,
                'train/dt': dt,
                'train/dtf': dtf,
                'train/dtb': dtb,
                'train/loss': loss,
                'train/sps': sps,
            }
            _ = history.update(_metrics)
            summary = summarize_dict(_metrics)
            logger.info(summary)
            # for k, v in _metrics.items():
            #     try:
            #         metrics[k].append(v)
            #     except KeyError:
            #         metrics[k] = [v]
            # logger.info(
            #     ', '.join([
            #         f'{iter=}',
            #         f'loss={loss.item():<.6g}',
            #         f'{sps=:<.4f}',
            #         # f'sps={sps:<.6f}',
            #         # f'dt={dtf+dtb:<3.6f}',
            #         f'{dt=:<.4f}',
            #         f'{dtf=:<3.4g}',
            #         f'{dtb=:<3.4g}'
            #     ])
            if not WANDB_DISABLED and RANK == 0 and wandb is not None:
                wandb.log(_metrics)
    if RANK == 0:
        import numpy as np
        from ezpz.plot import tplot, tplot_dict, plot_metric, plot_array, plot_arr
        outdir = Path(os.getcwd()).joinpath('test-dist-plots')
        tplotdir = outdir.joinpath('tplot')
        mplotdir = outdir.joinpath('mplot')
        outdir.mkdir(parents=True, exist_ok=True)
        tplotdir.mkdir(exist_ok=True, parents=True)
        mplotdir.mkdir(exist_ok=True, parents=True)
        import matplotlib.pyplot as plt
        import ambivalent
        plt.style.use(ambivalent.STYLES['ambivalent'])
        import plotext as pltx
        for key, val in metrics.items():
            if key == 'iter':
                continue
            try:
                arr = np.array(val)
            except Exception:
                arr = torch.Tensor(val).cpu().numpy()
            tplot(
                y=arr,
                x=np.array(metrics['train/iter']),
                # label=f"{key} vs. train iter",
                xlabel="iter",
                ylabel=key,
                append=True,
                title=f"{key} [{ez.get_timestamp()}]",
                outfile=tplotdir.joinpath(f"{key}_line.txt").as_posix(),
            )
            # tplot(
            #     y=arr,
            #     x=np.array(metrics['train/iter']),
            #     # label=f"{key} vs. train iter",
            #     xlabel="iter",
            #     ylabel=key,
            #     type="scatter",
            #     append=True,
            #     title=f"{key} [{ez.get_timestamp()}]",
            #     outfile=tplotdir.joinpath(f"{key}_scatter.txt").as_posix(),
            # )
            # plot_array(val=arr, key=key, xlabel='iter', outdir=mplotdir)
            # plt.show()
            # plot_metric(
            #     val=arr,
            #     outdir=mplotdir.joinpath(f'{key}'),
            #     key=key,
            # )
    tdist.barrier()


if __name__ == '__main__':
    # from ezpz.profile import get_context_manager
    # NOTE: if rank is passed to get_context_manager,
    # it will ONLY be instantiated if rank == 0,
    # otherwise, it will return a contextlib.nullcontext() instance.
    profiler = ez.profile.get_context_manager(rank=RANK, strict=False)
    with profiler:
        main()
    T4 = time.perf_counter()
    runtime = torch.tensor(T4 - T0)
    # tdist.all_reduce(runtime)
    TIMERS['timers/runtime'] = runtime.item()
    logger.info(f'[{RANK}] {runtime=:.6f}s')
    if not WANDB_DISABLED and RANK == 0 and wandb is not None:
        if (
            (run := getattr(wandb, 'run', None)) is not None
            and run is wandb.run
        ):
            wandb.log(TIMERS)
        # wandb.finish()
    tdist.barrier()
    import sys
    sys.exit(0)
