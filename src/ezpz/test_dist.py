"""
ezpz_ddp.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""
import time
T0 = time.perf_counter()

import os
import logging
from typing import Optional, Union
import torch
import ezpz as ez
from pathlib import Path
try:
    import wandb
    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
except Exception:
    WANDB_DISABLED = True

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
# RANK = DIST_INIT['rank']
# WORLD_SIZE = DIST_INIT['world_size']
# LOCAL_RANK = DIST_INIT['local_rank']
# if DEVICE == "cuda" and torch.cuda.is_available():
#     torch.cuda.set_device(LOCAL_RANK)
DEVICE = ez.get_torch_device()
WORLD_SIZE = ez.get_world_size()
LOCAL_RANK = ez.get_local_rank()
DEVICE_ID = f"{DEVICE}:{LOCAL_RANK}"


# log only from RANK == 0
logger = logging.getLogger(__name__)
logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")

WARMUP = 0
LOG_FREQ = int(os.environ.get("LOG_FREQ", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))  # 64
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 128))  # 128
OUTPUT_SIZE = int(os.environ.get("OUTPUT_SIZE", 128))  # 128
# dtype = os.environ.get("DTYPE", None)
# torch.get_num_interop_threads
# DTYPE = torch.dtype(dtype) if dtype is not None else torch.get_default_dtype()
DTYPE: torch.dtype  = torch.get_default_dtype()
if (dtype := os.environ.get("DTYPE", None)) is not None:
    if dtype.startswith('fp16'):
        DTYPE = torch.half
    elif dtype.startswith('bf16'):
        DTYPE = torch.bfloat16

TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 100))

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
if not WANDB_DISABLED and RANK == 0:
    run = ez.setup_wandb(project_name='ezpz.test_dist')
    assert run is wandb.run
    run.config.update(CONFIG)

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


def tplot_dict(
        data: dict,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        outfile: Optional[Union[str, Path]] = None,
        append: bool = True,
) -> None:
    import plotext as pltx
    pltx.clear_figure()
    pltx.theme('clear')
    pltx.scatter(list(data.values()))
    if ylabel is not None:
        pltx.ylabel(ylabel)
    if xlabel is not None:
        pltx.xlabel(xlabel)
    if title is not None:
        pltx.title(title)
    pltx.show()
    if outfile is not None:
        logger.info(f'Appending plot to: {outfile}')
        pltx.save_fig(outfile, append=append)


def main():
    model = Network(
        input_dim=INPUT_SIZE,
        output_dim=OUTPUT_SIZE,
        sizes=[1024, 512, 256, 128]
    )
    if RANK == 0 and not WANDB_DISABLED:
        assert run is wandb.run
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
        # config = ez.load_ds_config().update(
        #     {"train_micro_batch_size_per_gpu": BATCH_SIZE}
        # )
        import argparse
        parser = argparse.ArgumentParser(
            description='My training script.'
        )
        parser.add_argument(
            '--local_rank',
            required=False,
            type=int,
            default=-1,
            # default=ez.get_local_rank()),
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
        'dt': [],    # time per iteration
        'dtf': [],   # time in forward pass
        'dtb': [],   # time in backward pass
        'loss': [],  # loss
        'iter': [],  # iteration
    }
    T3 = time.perf_counter()
    TIMERS['timers/init_to_first_step'] = T3 - T0
    for iter in range(TRAIN_ITERS):
        t0 = time.perf_counter()
        x = torch.rand(*(BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE)
        y = model(x)
        loss = calc_loss(x, y)
        t1 = time.perf_counter()
        # dtf = ((t1 := time.perf_counter()) - t0)
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
                'iter': iter,
                'dt': dt,
                'dtf': dtf,
                'dtb': dtb,
                'loss': loss,
            }
            for k, v in _metrics.items():
                try:
                    metrics[k].append(v)
                except KeyError:
                    metrics[k] = [v]
            # metrics = {k: v.append(_v) for k, v in metrics.items()}
            # metrics['iter'].append(iter)
            # metrics['dt'].append(dt)
            # metrics['dtf'].append(dtf)
            # metrics['dtb'].append(dtb)
            # metrics['loss'].append(loss)
            logger.info(
                ', '.join([
                    f'{iter=}',
                    f'loss={loss.item():.4f}',
                    f'dt={dtf+dtb:.4f}',
                    f'{dtf=:.6g}',
                    f'{dtb=:.6g}'
                ])
            )
            if not WANDB_DISABLED and RANK == 0:
                wandb.log(_metrics)
                # wandb.log({'iteration': iter, **_metrics})
    if RANK == 0:
        outdir = Path(os.getcwd()).joinpath('test-dist-plots')
        outdir.mkdir(parents=True, exist_ok=True)
        for key, val in metrics.items():
            if key == 'iter':
                continue
            tplot_dict(
                data=dict(zip(metrics['iter'], val)),
                xlabel="iter",
                ylabel=key,
                append=True,
                title=f"{key} [{ez.get_timestamp()}]",
                outfile=outdir.joinpath(f"{key}.txt").as_posix(),
            )
        # tplot_dict(
        #     data=dict(zip(metrics['iter'], metrics['dtf'])),
        #     xlabel="iter",
        #     title="loss",
        #     outfile="./test_dist_loss.txt",
        # )
        # tplot_dict(
        #     data=dict(zip(metrics['iter'], metrics['dtb'])),
        #     xlabel="iter",
        #     title="loss",
        #     outfile="./test_dist_loss.txt",
        # )
        # tplot_dict(
        #     data=dict(zip(metrics['iter'], metrics['dt'])),
        #     xlabel="iter",
        #     title="loss",
        #     outfile="./test_dist_loss.txt",
        # )


if __name__ == '__main__':
    main()
    T4 = time.perf_counter()
    TIMERS['timers/runtime'] = T4 - T0
    if not WANDB_DISABLED and RANK == 0:
        wandb.log(TIMERS)
        wandb.finish()
