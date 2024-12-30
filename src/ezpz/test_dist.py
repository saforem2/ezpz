"""
test_dist.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""

import time
import argparse

T0 = time.perf_counter()  # start time

import os
from typing import Optional

import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP

import ezpz
from ezpz.history import History, summarize_dict

try:
    import wandb

    WANDB_DISABLED = os.environ.get('WANDB_DISABLED', False)
except Exception:
    wandb = None
    WANDB_DISABLED = True

ModelOptimizerPair = tuple[torch.nn.Module, torch.optim.Optimizer]

T1 = time.perf_counter()  # import time = (T1 - T0)
# backend can be any of DDP, deespepeed, horovod
RANK = ezpz.setup_torch(
    backend=(BACKEND := os.environ.get('BACKEND', 'DDP')),
    port=(port := os.environ.get('MASTER_PORT', '29500')),
)


from ezpz import mp

_ = mp.initialize_model_parallel(model_parallel_size=1)
DP_SIZE = mp.get_data_parallel_world_size()
MP_SIZE = mp.get_model_parallel_world_size()


T2 = time.perf_counter()  # torch_setup_time = (T2 - T1)
TIMERS = {
    'timers/ezpz.setup_torch': T2 - T1,
    'timers/imports': T1 - T0,
}
DEVICE_TYPE = ezpz.get_torch_device()
WORLD_SIZE = ezpz.get_world_size()
LOCAL_RANK = ezpz.get_local_rank()
DEVICE_ID = f'{DEVICE_TYPE}:{LOCAL_RANK}'

logger = ezpz.get_logger(__name__)
# logger = logging.getLogger(__name__)
# # log only from RANK == 0
# logger.setLevel('INFO') if RANK == 0 else logger.setLevel('CRITICAL')

WARMUP = 0
LOG_FREQ = int(os.environ.get('LOG_FREQ', 1))
TRAIN_ITERS = int(os.environ.get('TRAIN_ITERS', 100))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))  # 64
INPUT_SIZE = int(os.environ.get('INPUT_SIZE', 128))  # 128
OUTPUT_SIZE = int(os.environ.get('OUTPUT_SIZE', 128))  # 128
PYINSTRUMENT_PROFILER = os.environ.get('PYINSTRUMENT_PROFILER', None)
sizes = os.environ.get(
    'LAYER_SIZES',
    os.environ.get(
        'SIZES',
        os.environ.get(
            'LAYERS',
            None,  # [1024, 512, 256, 128]
        ),
    ),
)
if sizes is not None:
    LAYER_SIZES = [int(i) for i in sizes.split(',')]
    logger.info(f'Caught: {LAYER_SIZES=}')
else:
    LAYER_SIZES = [1024, 512, 256, 128]

DTYPE: torch.dtype = torch.get_default_dtype()
if (dtype := os.environ.get('DTYPE', None)) is not None:
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
    run = ezpz.setup_wandb(project_name='ezpz.test_dist')
    assert wandb is not None and run is wandb.run and wandb.run is not None
    wandb.run.config.update(CONFIG)

if RANK == 0:
    # ezpz.dist.log_dict_as_bulleted_list(timers_import, name="timers_import")
    ezpz.dist.log_dict_as_bulleted_list(CONFIG, name='CONFIG')

if WORLD_SIZE > 1:
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
                layers.append(torch.nn.Linear(sizes[idx], size))
            layers.append(torch.nn.Linear(sizes[-1], output_dim))
            self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y - x).pow(2).sum()


def build_model_and_optimizer() -> ModelOptimizerPair:
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
    if BACKEND.lower() == 'ddp':
        if WORLD_SIZE > 1:
            model = DDP(model, device_ids=[])
    elif BACKEND.lower() in ('ds', 'deepspeed'):
        try:
            import deepspeed  # type:ignore
        except (ImportError, ModuleNotFoundError):
            logger.error('deepspeed not installed')
            raise

        parser = argparse.ArgumentParser(description='My training script.')
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
        model, optimizer, *_ = deepspeed.initialize(
            args=cmd_args,
            model=model,
            optimizer=optimizer,
        )
        logger.info(f'{cmd_args=}')
    return model, optimizer


def main():
    history = History()
    T3 = time.perf_counter()
    TIMERS['timers/init_to_first_step'] = T3 - T0

    model, optimizer = build_model_and_optimizer()

    def _forward_step() -> torch.Tensor:
        x = torch.rand(*(BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE_TYPE)
        y = model(x)
        return calc_loss(x, y)

    def _backward_step(loss: torch.Tensor) -> None:
        if BACKEND == 'deepspeed':
            model.backward(loss)
            model.step(loss)
        else:
            loss.backward()
            optimizer.step()

    for iter in range(TRAIN_ITERS):
        t0 = time.perf_counter()
        loss = _forward_step()
        t1 = time.perf_counter()
        _ = _backward_step(loss)
        t2 = time.perf_counter()
        optimizer.zero_grad()
        if iter > WARMUP and iter % LOG_FREQ == 0:
            _metrics = {
                'train/iter': iter,
                'train/dt': (dt := (t2 - t0)),
                'train/dtf': (t1 - t0),
                'train/dtb': (t2 - t1),
                'train/loss': loss,
                'train/sps': (BATCH_SIZE / dt),
            }
            _ = history.update(_metrics)
            summary = summarize_dict(_metrics)
            logger.info(summary.replace('train/', ''))
    if RANK == 0:
        import matplotlib.pyplot as plt
        import ambivalent

        plt.style.use(ambivalent.STYLES['ambivalent'])
        dataset = history.finalize()
        logger.info(f'{dataset=}')
    if WORLD_SIZE > 1:
        tdist.barrier()


if __name__ == '__main__':
    import sys

    # Wrap training loop in pyinstrument profiler context block
    with ezpz.profile.get_context_manager(rank=RANK, strict=False):
        main()
    T4 = time.perf_counter()
    runtime = torch.tensor(T4 - T0)
    # tdist.all_reduce(runtime)
    if BACKEND.lower() in ['ds', 'deepspeed']:
        import deepspeed.comm as dscomm  # type:ignore

        dscomm.log_summary()
    if not WANDB_DISABLED and RANK == 0 and wandb is not None:
        if (
            run := getattr(wandb, 'run', None)
        ) is not None and run is wandb.run:
            wandb.log(TIMERS)
        # wandb.finish()
    if WORLD_SIZE > 1:
        tdist.barrier()
    TIMERS['timers/runtime'] = runtime.item()
    logger.info(f'[{RANK}] {runtime=:.6f}s')
    sys.exit(0)
