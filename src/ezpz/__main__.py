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
import sys

import hydra
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.configs import TrainConfig, git_ds_info
from ezpz.dist import setup, setup_wandb

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
    else:
        try:
            from omegaconf import OmegaConf
            import json
            log.info(json.dumps(OmegaConf.to_container(cfg), indent=4))
        except (ImportError, ModuleNotFoundError):
            log.info(config)
        if config.use_wandb:
            setup_wandb(
                project_name=config.wandb_project_name,
                config=cfg,
            )
    log.info(f'Output dir: {os.getcwd()}')
    if rank == 0 and config.backend.lower() in ['ds', 'dspeed', 'deepspeed']:
        git_ds_info()
    return rank


if __name__ == '__main__':
    rank = main()
    try:
        import wandb
    except (ImportError, ModuleNotFoundError):
        wandb = None
    if wandb is not None and wandb.run is not None:
        wandb.run.finish()
    sys.exit(0)
