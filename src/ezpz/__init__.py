"""
ezpz/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
# from typing import Optional
# from enrich import get_logger
# from enrich.console import is_interactive, get_console
# import warnings

from mpi4py import MPI

# from enrich.handler import RichHandler
# from rich import print
# from enrich import get_logger
# from pathlib import Path
# pyright: ignore 
# noqa:E402
from ezpz import dist    # noqa:E402
from ezpz.dist import (  # noqa:E402
    setup_wandb,
    seed_everything,
    setup_tensorflow,
    setup_torch,
    cleanup,
    get_world_size,
    get_rank,
    get_local_rank,
    query_environment,
    check
)
# from ezpz.logging import get_logger, get_rich_logger, get_file_logger, get_console

from ezpz.configs import (  # noqa:E402
    HERE,
    PROJECT_DIR,
    PROJECT_ROOT,
    CONF_DIR,
    LOGS_DIR,
    BIN_DIR,
    SAVEJOBENV,
    GETJOBENV,
    OUTPUTS_DIR,
    QUARTO_OUTPUTS_DIR,
    FRAMEWORKS,
    BACKENDS,
    load_ds_config,
    TrainConfig
)

__all__ = [
    'dist',
    'setup_wandb',
    'setup_tensorflow',
    'setup_torch',
    'cleanup',
    'seed_everything',
    # 'get_console',
    # 'get_logger',
    # 'get_rich_logger',
    # 'get_file_logger',
    'get_rank',
    'get_local_rank',
    'get_world_size',
    'query_environment',
    'check',
    'HERE',
    'PROJECT_DIR',
    'PROJECT_ROOT',
    'CONF_DIR',
    'LOGS_DIR',
    'OUTPUTS_DIR',
    'QUARTO_OUTPUTS_DIR',
    'FRAMEWORKS',
    'BACKENDS',
    'BIN_DIR',
    'GETJOBENV',
    'SAVEJOBENV',
    'load_ds_config',
    'TrainConfig',
]


os.environ['PYTHONIOENCODING'] = 'utf-8'
RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())
log = logging.getLogger(__name__)
# LEVEL = "INFO" if RANK == 0 else "CRITICAL"
# log = get_logger(__name__, "INFO")
# if RANK != 0:
#     log.setLevel("CRITICAL")



# # Check that MPS is available
# if (
#         torch.backends.mps.is_available()
#         and torch.get_default_dtype() != torch.float64
# ):
#     DEVICE = torch.device("mps")
# elif not torch.backends.mps.is_built():
#     DEVICE = 'cpu'
#     print(
#         "MPS not available because the current PyTorch install was not "
#         "built with MPS enabled."
#     )
# else:
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")

def savejobenv():
    return SAVEJOBENV

def get_jobenv():
    return GETJOBENV
