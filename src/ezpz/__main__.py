# -*- coding: utf-8 -*-
"""
ezpz/__main__.py

Contains main entry point for training.
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

import hydra
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz import (
    TrainConfig,
    setup,
    setup_wandb,
    timeitlogit,
    get_rank,
    get_torch_device,
    get_torch_backend,
    get_world_size,
    get_local_rank,
    get_gpus_per_node,
)

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


log = logging.getLogger(__name__)

RANK = get_rank()
DEVICE = get_torch_device()
BACKEND = get_torch_backend()
WORLD_SIZE = get_world_size()
LOCAL_RANK = get_local_rank()
GPUS_PER_NODE = get_gpus_per_node()
NUM_NODES = WORLD_SIZE // GPUS_PER_NODE


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
    run = None
    if rank != 0:
        log.setLevel("CRITICAL")
    # log.info(f'[GLOBAL]: {RANK=} / {WORLD_SIZE-1=}, {NUM_NODES=}')
    # log.info(f'[LOCAL]: {LOCAL_RANK=} / {GPUS_PER_NODE=}')
    # if rank == 0:
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
        log.warning(
            Text(f'{Emoji("rocket")} [{run.name}]({run.url})')
        )
    return rank


if __name__ == '__main__':
    rank = main()
