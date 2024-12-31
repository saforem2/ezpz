"""
test_dist.py

- to launch:

  $ source ezpz/src/ezpz/bin/savejobenv
  $ BACKEND=DDP launch python3 ezpz_ddp.py
"""

import argparse
from dataclasses import dataclass, field, asdict
import os
import time
from typing import Optional

import ezpz

import warnings

# from ezpz import summarize_dict
from ezpz.history import History
import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP

T0 = time.perf_counter()  # start time

# noqa: E402

# from ezpz.history import History, summarize_dict
#
warnings.filterwarnings('ignore')

try:
    import wandb

    WANDB_DISABLED = os.environ.get('WANDB_DISABLED', False)
except Exception:
    wandb = None
    WANDB_DISABLED = True

ModelOptimizerPair = tuple[torch.nn.Module, torch.optim.Optimizer]


logger = ezpz.get_logger(__name__)
#
# WARMUP = 0
# LOG_FREQ = int(os.environ.get('LOG_FREQ', 1))
# TRAIN_ITERS = int(os.environ.get('TRAIN_ITERS', 100))
# BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))  # 64
# INPUT_SIZE = int(os.environ.get('INPUT_SIZE', 128))  # 128
# OUTPUT_SIZE = int(os.environ.get('OUTPUT_SIZE', 128))  # 128
# PYINSTRUMENT_PROFILER = os.environ.get('PYINSTRUMENT_PROFILER', None)
# sizes = os.environ.get(
#     'LAYER_SIZES',
#     os.environ.get(
#         'SIZES',
#         os.environ.get(
#             'LAYERS',
#             None,  # [1024, 512, 256, 128]
#         ),
#     ),
# )
# if sizes is not None:
#     LAYER_SIZES = [int(i) for i in sizes.split(',')]
#     logger.info(f'Caught: {LAYER_SIZES=}')
# else:
#     LAYER_SIZES = [1024, 512, 256, 128]
#
# DTYPE: torch.dtype = torch.get_default_dtype()
# if (dtype := os.environ.get('DTYPE', None)) is not None:
#     if dtype.startswith('fp16'):
#         DTYPE = torch.half
#     elif dtype.startswith('bf16'):
#         DTYPE = torch.bfloat16
#
# CONFIG = {
#     'warmup': WARMUP,
#     'log_freq': LOG_FREQ,
#     'batch_size': BATCH_SIZE,
#     'input_size': INPUT_SIZE,
#     'output_size': OUTPUT_SIZE,
#     'dtype': DTYPE,
#     'device': DEVICE_TYPE,
#     'world_size': WORLD_SIZE,
#     'train_iters': TRAIN_ITERS,
# }
#


@dataclass
class TrainConfig:
    warmup: int
    log_freq: int
    tp: int
    pp: int
    cp: int
    batch_size: int
    input_size: int
    output_size: int
    train_iters: int
    backend: str = 'DDP'
    dtype: Optional[str] = None
    pyinstrument_profiler: Optional[bool] = None
    layer_sizes: list = field(default_factory=lambda: [1024, 512, 256, 128])

    def get_torch_dtype(self) -> torch.dtype:
        if self.dtype is None:
            return torch.get_default_dtype()
        if self.dtype in {
            'fp16',
            'half',
            'float16',
        }:
            return torch.float16
        if self.dtype in {
            'bfloat16',
            'bf16',
        }:
            return torch.bfloat16
        logger.warning(f'Unknown dtype: {self.dtype=}, using float32')
        return torch.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training configuration parameters'
    )
    parser.add_argument(
        '--log_freq',
        type=int,
        default=1,
        help='Logging frequency',
    )
    parser.add_argument(
        '--tp',
        type=int,
        default=1,
        help='Tensor parallel size',
    )
    parser.add_argument(
        '--pp',
        type=int,
        default=1,
        help='Pipeline length',
    )
    parser.add_argument(
        '--cp',
        type=int,
        default=1,
        help='Context parallel size',
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='DDP',
        help='Backend (DDP, DeepSpeed, etc.)',
    )
    parser.add_argument(
        '--train_iters',
        type=int,
        default=100,
        help='Number of training iterations',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=128,
        help='Input size',
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=128,
        help='Output size',
    )
    parser.add_argument(
        '--pyinstrument_profiler',
        action='store_true',
        help='PyInstrument profiler',
    )
    parser.add_argument(
        '--layer_sizes',
        help='Comma-separated list of layer sizes',
        # parser.add_argument('-l', '--list', help='delimited list input',
        type=lambda s: [int(item) for item in s.split(',')],
        default=[1024, 512, 256, 128],
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        help='Data type (fp16, float16, bfloat16, bf16, float32, etc.)',
    )

    args = parser.parse_args()
    return args


def get_config_from_args(args: argparse.Namespace) -> TrainConfig:
    # layer_sizes = [1024, 512, 256, 128]
    # if args.layer_sizes is not None:
    #     layer_sizes = [int(i) for i in args.layer_sizes.split(',')]
    #     logger.info(f'Caught: {layer_sizes=}')

    config = TrainConfig(
        warmup=0,
        log_freq=args.log_freq,
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        batch_size=args.batch_size,
        input_size=args.input_size,
        output_size=args.output_size,
        dtype=args.dtype,
        train_iters=args.train_iters,
        pyinstrument_profiler=args.pyinstrument_profiler,
        layer_sizes=args.layer_sizes,
    )

    return config


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


def build_model_and_optimizer(
    model: torch.nn.Module, backend: str = 'DDP'
) -> ModelOptimizerPair:
    if backend is not None:
        assert backend.lower() in {'ddp', 'deepspeed', 'ds'}
    rank = ezpz.get_rank()
    device_type = ezpz.get_torch_device()
    device_id = f'{device_type}:{ezpz.get_local_rank()}'
    world_size = ezpz.get_world_size()
    if rank == 0 and not WANDB_DISABLED and wandb is not None:
        assert wandb.run is not None
        wandb.run.watch(model, log='all')
    model.to(device_type)
    model.to(device_id)
    logger.info(f'{model=}')
    optimizer = torch.optim.Adam(model.parameters())
    # with profiler:
    if backend.lower() == 'ddp':
        if world_size > 1:
            model = DDP(model, device_ids=[])
    elif backend.lower() in ('ds', 'deepspeed'):
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


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.history = History()
        self.train_iter = 0

    def _forward_step(self) -> dict:
        t0 = time.perf_counter()
        x = torch.rand(
            *(self.config.batch_size, self.config.input_size),
            device=self.model.device,
            dtype=self.config.get_torch_dtype(),
        )
        # with torch.autocast(device_type=str(DEVICE_TYPE), dtype=dtype):
        y = self.model(x)
        return {'loss': calc_loss(x, y), 'dtf': (time.perf_counter() - t0)}

    def _backward_step(self, loss: torch.Tensor) -> float:
        t0 = time.perf_counter()
        if self.config.backend == 'deepspeed':
            self.model.backward(loss)
            self.model.step(loss)
        else:
            loss.backward()
            self.optimizer.step()
        return time.perf_counter() - t0

    def train_step(self) -> dict:
        metrics = self._forward_step()
        metrics['dtb'] = self._backward_step(metrics['loss'])
        self.optimizer.zero_grad()
        summary = self.history.update(
            {
                'iter': self.train_iter,
                **metrics,
            }
        )
        logger.info(summary)
        self.train_iter += 1
        return metrics


def train(config: TrainConfig) -> Trainer:
    config_dict = asdict(config)
    logger.info(f'{config=}')

    import ezpz

    # T1 = time.perf_counter()  # import time = (T1 - T0)
    # backend can be any of DDP, deespepeed, horovod
    rank = ezpz.setup_torch(
        backend=config.backend,
        # port=(),
        tensor_parallel_size=config.tp,
        pipeline_length=config.pp,
        context_parallel_size=config.cp,
    )

    # T2 = time.perf_counter()  # torch_setup_time = (T2 - T1)
    # timers = {
    #     'timers/ezpz.setup_torch': T2 - T1,
    #     'timers/imports': T1 - T0,
    # }
    device_type = ezpz.get_torch_device()
    world_size = ezpz.get_world_size()
    local_rank = ezpz.get_local_rank()
    device_id = f'{device_type}:{local_rank}'

    import ezpz.tp

    tpgroup = ezpz.tp.get_tensor_parallel_group()
    dpgroup = ezpz.tp.get_data_parallel_group()
    ppgroup = ezpz.tp.get_pipeline_parallel_group()
    cpgroup = ezpz.tp.get_context_parallel_group()

    # tprank = ezpz.tp.get_tensor_parallel_rank()
    # tpranks = ezpz.tp.get_tensor_parallel_ranks()
    #
    # dprank = ezpz.tp.get_data_parallel_rank()
    # dpranks = ezpz.tp.get_data_parallel_ranks()
    #
    # pprank = ezpz.tp.get_pipeline_parallel_rank()
    # ppranks = ezpz.tp.get_pipeline_parallel_ranks()
    #
    # cpranks = ezpz.tp.get_context_parallel_ranks()
    # cprank = ezpz.tp.get_context_parallel_rank()

    # if dprank == 0 or tprank == 0 or cprank == 0 or pprank == 0:
    #     print(f'[{tprank}]: {tpranks=}')
    #     # print(f'[{tprank}]: {tpgroup=}')
    #
    #     print(f'[{dprank}]: {dpranks=}')
    #     # print(f'[{dprank}]: {dpgroup=}')
    #
    #     print(f'[{pprank}]: {ppranks=}')
    #     # print(f'[{pprank}]: {ppgroup=}')
    #
    #     print(f'[{cprank}]: {cpranks=}')
    #     # logger.info(f'[{cprank}]: {cpgroup=}')
    #
    # # from rich import print
    # #
    # # print(f'[{tprank}]: {tpranks=}')
    # # print(f'[{dprank}]: {dpranks=}')
    # # print(f'[{pprank}]: {ppranks=}')
    # # print(f'[{cprank}]: {cpranks=}')

    tdist.barrier(group=tpgroup)
    tdist.barrier(group=dpgroup)
    tdist.barrier(group=cpgroup)
    tdist.barrier(group=ppgroup)
    # tdist.barrier(group=tpgroup)
    run = None
    if wandb is not None and not WANDB_DISABLED and rank == 0:
        run = ezpz.setup_wandb(project_name='ezpz.test_dist')
        assert wandb is not None and run is wandb.run and wandb.run is not None
        wandb.run.config.update(config_dict)

    if rank == 0:
        ezpz.dist.log_dict_as_bulleted_list(config_dict, name='TrainConfig')

    if world_size > 1:
        tdist.barrier()

    model = Network(
        input_dim=config.input_size,
        output_dim=config.output_size,
        sizes=config.layer_sizes,  # [1024, 512, 256, 128]
    )
    # T3 = time.perf_counter()
    # timers['timers/init_to_first_step'] = T3 - T0

    model, optimizer = build_model_and_optimizer(model, backend=config.backend)
    dtype = config.get_torch_dtype()
    model.to(device_id)
    model.to(dtype)
    trainer = Trainer(config=config, model=model, optimizer=optimizer)
    # device_str = ezpz.get_torch_device(as_torch_device=False)

    for _ in range(config.train_iters):
        _ = trainer.train_step()
        # t0 = time.perf_counter()
        # loss = _forward_step()
        # t1 = time.perf_counter()
        # _ = _backward_step(loss)
        # t2 = time.perf_counter()
        # optimizer.zero_grad()
        # if iter > config.warmup and iter % config.log_freq == 0:
        #     _metrics = {
        #         'train/iter': iter,
        #         'train/dt': (dt := (t2 - t0)),
        #         'train/dtf': (t1 - t0),
        #         'train/dtb': (t2 - t1),
        #         'train/loss': loss,
        #         'train/sps': (config.batch_size / dt),
        #     }
        #     summary = history.update(_metrics)
        #     logger.info(summary.replace('train/', ''))
    if rank == 0:
        import matplotlib.pyplot as plt
        import ambivalent

        plt.style.use(ambivalent.STYLES['ambivalent'])
        dataset = trainer.history.finalize(
            run_name='ezpz.test_dist', dataset_fname='train'
        )
        logger.info(f'{dataset=}')
    if world_size > 1:
        tdist.barrier()

    return trainer


def main():
    args = parse_args()
    config = get_config_from_args(args)
    # with ezpz.profile.get_context_manager(rank=RANK, strict=False):
    # main()
    return train(config)


if __name__ == '__main__':
    import sys

    # T1 = time.perf_counter()  # import time = (T1 - T0)
    # # backend can be any of DDP, deespepeed, horovod
    # RANK = ezpz.setup_torch(
    #     backend=(BACKEND := os.environ.get('BACKEND', 'DDP')),
    #     port=(port := os.environ.get('MASTER_PORT', '29500')),
    #     tensor_parallel_size=(mpsize := os.environ.get('MPSIZE', '1')),
    #     pipeline_length=(plength := os.environ.get('PPSIZE', '1')),
    #     context_parallel_size=(cpsize := os.environ.get('CPSIZE', '1')),
    # )
    #
    # T2 = time.perf_counter()  # torch_setup_time = (T2 - T1)
    # TIMERS = {
    #     'timers/ezpz.setup_torch': T2 - T1,
    #     'timers/imports': T1 - T0,
    # }
    # DEVICE_TYPE = ezpz.get_torch_device()
    # WORLD_SIZE = ezpz.get_world_size()
    # LOCAL_RANK = ezpz.get_local_rank()
    # DEVICE_ID = f'{DEVICE_TYPE}:{LOCAL_RANK}'
    # Wrap training loop in pyinstrument profiler context block
    trainer = main()
    # T4 = time.perf_counter()
    # runtime = torch.tensor(T4 - T0)
    # tdist.all_reduce(runtime)
    if trainer.config.backend.lower() in ['ds', 'deepspeed']:
        import deepspeed.comm as dscomm  # type:ignore

        dscomm.log_summary()
    # if not WANDB_DISABLED and ezpz.get_rank() == 0 and wandb is not None:
    #     if (
    #         run := getattr(wandb, 'run', None)
    #     ) is not None and run is wandb.run:
    #         wandb.log()
    # wandb.finish()
    if ezpz.get_world_size() > 1:
        tdist.barrier()
    # TIMERS['timers/runtime'] = runtime.item()
    # logger.info(f'[{ezpz.get_rank()}] {runtime=:.6f}s')
    sys.exit(0)
