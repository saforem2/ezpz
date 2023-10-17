# -*- coding: utf-8 -*-
"""
ezpz/train.py

Contains main entry point for training.
"""
from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals
)
import os
import sys

import hydra
import logging

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.dist import setup
from ezpz.configs import TrainConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> int:
    config = instantiate(cfg)
    assert isinstance(config, TrainConfig)
    rank = setup(
        framework=config.framework,
        backend=config.backend,
        seed=config.seed
    )
    if rank != 0:
        log.setLevel("CRITICAL")
    log.info(f'config: {config}')
    log.info(f'Output dir: {os.getcwd()}')
    return rank



if __name__ == '__main__':
    # import wandb
    # wandb.require(experiment='service')
    rank = main()
    # if wandb.run is not None:
    #     wandb.finish(0)
    sys.exit(0)
