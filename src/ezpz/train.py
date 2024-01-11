# -*- coding: utf-8 -*-
r"""
ezpz/train.py

Simple training smoke test.
"""
from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)
import logging
import os
import time

import hydra
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from ezpz import (
    TrainConfig,
    get_gpus_per_node,
    get_local_rank,
    get_rank,
    get_torch_backend,
    get_torch_device,
    get_world_size,
    setup,
    setup_wandb,
    timeitlogit,
)
from ezpz.dist import get_dist_info, get_node_index
from ezpz.model import SimpleCNN

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


log = logging.getLogger(__name__)

RANK = get_rank()
NODE_ID = get_node_index()
DEVICE = get_torch_device()
BACKEND = get_torch_backend()
WORLD_SIZE = get_world_size()
LOCAL_RANK = get_local_rank()
GPUS_PER_NODE = get_gpus_per_node()
NUM_NODES = WORLD_SIZE // GPUS_PER_NODE


def train_model() -> float:
    model = SimpleCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    devid = f'{DEVICE}:{get_local_rank()}'
    # log.warning(
    #     f'[{RANK} / {WORLD_SIZE}]'
    #     f'Current device: '
    #     f'{DEVICE}:{LOCAL_RANK} (/ {GPUS_PER_NODE})'
    #     f'on {NODE_ID=} (/ {NUM_NODES})'
    # )
    # if WORLD_SIZE > 1:
    model.to(devid)
    model = DistributedDataParallel(model, device_ids=[devid])
    inputs = torch.rand((1, 1, 5, 5), device=f'{devid}')
    outputs = model(inputs)
    if RANK == 0:
        log.warning(f'{devid=}')
        log.warning(f'{inputs=}')
        log.warning(f'{model=}')
        log.warning(f'{outputs=}')
    label = torch.full((1,), 1.0, dtype=torch.float, device=f'{devid}')
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
    return loss.item()


@timeitlogit(verbose=True)
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> int:
    config: TrainConfig = instantiate(cfg)
    assert isinstance(config, TrainConfig)
    rank = setup(
        framework=config.framework,
        backend=config.backend,
        seed=config.seed
    )
    t1 = time.perf_counter()
    t0 = os.environ.get('START_TIME', None)
    # if t0 is not None:
    assert t0 is not None
    startup_time = t1 - float(t0)
    run = None
    if rank != 0:
        log.setLevel("CRITICAL")
    else:
        if config.use_wandb and wandb is not None:
            run = setup_wandb(
                config=cfg,
                project_name=config.wandb_project_name,
            )
        log.info(f"{config=}")
        log.info(f'Output dir: {os.getcwd()}')
    if (
            wandb is not None
            and run is not None
            and run is wandb.run
            and config.use_wandb
    ):
        assert run is not None
        from rich.emoji import Emoji
        from rich.text import Text
        log.warning(f'Startup time: {startup_time}')
        wandb.log({'startup_time': startup_time})
        log.warning(
            Text(f'{Emoji("rocket")} [{run.name}]({run.url})')
        )
        _ = get_dist_info(config.framework, verbose=True)
    # ---- TRAIN MODEL ---------------------------------
    _ = train_model()
    # --------------------------------------------------
    return rank


if __name__ == '__main__':
    t0 = time.perf_counter()
    os.environ['START_TIME'] = f'{t0}'
    _ = main()
