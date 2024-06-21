"""
ezpz_ddp.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""
import time
T0 = time.perf_counter()

# noqa: E402
import os  # noqa:E402
import logging  # noqa:E402
from typing import Optional, Union  # noqa:E402
import torch  # noqa: E402
import ezpz as ez  # noqa: E402
from pathlib import Path  # noqa: E402
try:
    import wandb
    wandb.require("core")
    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
except Exception:
    wandb = None
    WANDB_DISABLED = True

# log only from RANK == 0
logger = logging.getLogger(__name__)

# backend can be any of DDP, deespepeed, horovod
T1 = time.perf_counter()
RANK = ez.setup_torch(
    backend=(
        backend := os.environ.get('BACKEND', 'DDP')
    ),
    port=(
        port := os.environ.get("MASTER_PORT", "29500")
    )
)
T2 = time.perf_counter()
TIMERS = {
    'timers/ezpz.setup_torch': T2 - T1,
    'timers/imports': T1 - T0,
}
logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")
DEVICE = ez.get_torch_device()
WORLD_SIZE = ez.get_world_size()
LOCAL_RANK = ez.get_local_rank()
DEVICE_ID = f"{DEVICE}:{LOCAL_RANK}"

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

# dtype = os.environ.get("DTYPE", None)
# torch.get_num_interop_threads
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
    'device': DEVICE,
    'world_size': WORLD_SIZE,
    'train_iters': TRAIN_ITERS,
}

run = None
if not WANDB_DISABLED and RANK == 0 and wandb is not None:
    run = ez.setup_wandb(project_name='ezpz.test_dist')
    assert wandb.run is not None
    wandb.run.config.update(CONFIG)

# logger.info(f"{DIST_INIT=}")
# logger.info(f'


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


def main():
    model = Network(
        input_dim=INPUT_SIZE,
        output_dim=OUTPUT_SIZE,
        sizes=[1024, 512, 256, 128]
    )
    if RANK == 0 and not WANDB_DISABLED and wandb is not None:
        assert wandb.run is not None
        wandb.run.watch(model, log='all')
    model.to(DEVICE)
    model.to(DEVICE_ID)
    logger.info(f'{model=}')
    optimizer = torch.optim.Adam(model.parameters())
    if backend.lower() == 'ddp':
        if WORLD_SIZE > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
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

    metrics = {
        'train/dt': [],    # time per iteration
        'train/dtf': [],   # time in forward pass
        'train/dtb': [],   # time in backward pass
        'train/loss': [],  # loss
        'train/iter': [],  # iteration
    }
    T3 = time.perf_counter()
    TIMERS['timers/init_to_first_step'] = T3 - T0
    for iter in range(TRAIN_ITERS):
        t0 = time.perf_counter()
        x = torch.rand(*(BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE)
        y = model(x)
        loss = calc_loss(x, y)
        t1 = time.perf_counter()
        if backend == 'deepspeed':
            model.backward(loss)
            model.step(loss)
        else:
            loss.backward()
            optimizer.step()
        t2 = time.perf_counter()
        optimizer.zero_grad()
        if iter > WARMUP and iter % LOG_FREQ == 0:
            dt = t2 - t0
            dtf = t1 - t0
            dtb = t2 - t1
            _metrics = {
                'train/iter': iter,
                'train/dt': dt,
                'train/dtf': dtf,
                'train/dtb': dtb,
                'train/loss': loss,
            }
            for k, v in _metrics.items():
                try:
                    metrics[k].append(v)
                except KeyError:
                    metrics[k] = [v]
            logger.info(
                ', '.join([
                    f'{iter=}',
                    f'loss={loss.item():.4f}',
                    f'dt={dtf+dtb:.4f}',
                    f'{dtf=:.6g}',
                    f'{dtb=:.6g}'
                ])
            )
            if not WANDB_DISABLED and RANK == 0 and wandb is not None:
                wandb.log(_metrics)
    if RANK == 0:
        outdir = Path(os.getcwd()).joinpath('test-dist-plots')
        outdir.mkdir(parents=True, exist_ok=True)
        for key, val in metrics.items():
            if key == 'iter':
                continue
            ez.plot.tplot_dict(
                data=dict(zip(metrics['train/iter'], val)),
                xlabel="iter",
                ylabel=key,
                append=True,
                title=f"{key} [{ez.get_timestamp()}]",
                outfile=outdir.joinpath(f"{key}.txt").as_posix(),
            )


if __name__ == '__main__':
    from ezpz.profile import get_context_manager
    # NOTE: if rank is passed to get_context_manager,
    # it will ONLY be instantiated if rank == 0,
    # otherwise, it will return a contextlib.nullcontext() instance.
    cm = get_context_manager(rank=RANK, strict=False)
    with cm:
        main()
    T4 = time.perf_counter()
    TIMERS['timers/runtime'] = T4 - T0
    if not WANDB_DISABLED and RANK == 0 and wandb is not None:
        wandb.log(TIMERS)
        wandb.finish()
