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
    unicode_literals
)
import os
import sys

import hydra
import logging
# from enrich import get_logger

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.dist import setup, setup_wandb
from ezpz.configs import TrainConfig, git_ds_info

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> int:
    # log = get_logger(__name__, "INFO")
    config = instantiate(cfg)
    assert isinstance(config, TrainConfig)
    rank = setup(
        framework=config.framework,
        backend=config.backend,
        seed=config.seed
    )
    if rank != 0:
        log.setLevel("CRITICAL")
    # log.info(f'log.handlers: {log.handlers}')
    else:
        from rich import print_json
        print_json(config.to_json())
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
    # import wandb
    # wandb.require(experiment='service')
    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     record_shapes=True,
    #     with_stack=True
    # ) as prof:
    rank = main()
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
    # prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")
    # prof.export_chrome_trace("trace.json")
    # if wandb.run is not None:
    #     wandb.finish(0)
    import logging
    logging.shutdown()
    # sys.exit(0)
