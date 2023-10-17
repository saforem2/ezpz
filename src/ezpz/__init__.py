"""
ezpz/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from typing import Optional
from enrich.console import is_interactive, get_console
# import warnings

from mpi4py import MPI
from enrich.handler import RichHandler
import tqdm
# from rich import print
# from enrich import get_logger
from pathlib import Path
from ezpz.dist import (
    setup_wandb,
    seed_everything,
    setup_tensorflow,
    setup_torch,
    cleanup,
    get_world_size,
    get_rank,
    query_environment,
    check
)

from ezpz.configs import (
    HERE,
    PROJECT_DIR,
    PROJECT_ROOT,
    CONF_DIR,
    LOGS_DIR,
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
    'get_rank',
    'get_world_size',
    'query_environment',
    'check'
    'HERE',
    'PROJECT_DIR',
    'PROJECT_ROOT',
    'CONF_DIR',
    'LOGS_DIR',
    'OUTPUTS_DIR',
    'QUARTO_OUTPUTS_DIR',
    'FRAMEWORKS',
    'BACKENDS',
    'load_ds_config',
    'TrainConfig',
]


os.environ['PYTHONIOENCODING'] = 'utf-8'
RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())

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


class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        # if len(x.rstrip()) > 0:
        tqdm.tqdm.write(x, file=self.file, end='\n')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


def get_rich_logger(
        name: Optional[str] = None,
        level: str = 'INFO'
) -> logging.Logger:
    # log: logging.Logger = get_logger(name=name, level=level)
    log = logging.getLogger(name)
    log.handlers = []
    console = get_console(
        markup=True,
        redirect=(WORLD_SIZE > 1),
    )
    handler = RichHandler(
        level,
        rich_tracebacks=False,
        console=console,
        show_path=False,
        enable_link_path=False
    )
    log.handlers = [handler]
    log.setLevel(level)
    return log


def get_file_logger(
        name: Optional[str] = None,
        level: str = 'INFO',
        rank_zero_only: bool = True,
        fname: Optional[str] = None,
        # rich_stdout: bool = True,
) -> logging.Logger:
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    import logging
    fname = 'ezpz' if fname is None else fname
    log = logging.getLogger(name)
    if rank_zero_only:
        fh = logging.FileHandler(f"{fname}.log")
        if RANK == 0:
            log.setLevel(level)
            fh.setLevel(level)
        else:
            log.setLevel('CRITICAL')
            fh.setLevel('CRITICAL')
    else:
        fh = logging.FileHandler(f"{fname}-{RANK}.log")
        log.setLevel(level)
        fh.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_logger(
        name: Optional[str] = None,
        level: str = 'INFO',
        rank_zero_only: bool = True,
        **kwargs,
) -> logging.Logger:
    log = logging.getLogger(name)
    # log.handlers = []
    # from rich.logging import RichHandler
    # from l2hmc.utils.rich import get_console, is_interactive
    from enrich.console import get_console
    # from enrich import is_interactive
    # format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    if rank_zero_only:
        if RANK != 0:
            log.setLevel('CRITICAL')
        else:
            log.setLevel(level)
    if RANK == 0:
        console = get_console(
            markup=True,  # (WORLD_SIZE == 1),
            redirect=(WORLD_SIZE > 1),
            **kwargs
        )
        if console.is_jupyter:
            console.is_jupyter = False
        # log.propagate = True
        # log.handlers = []
        use_markup = (
            WORLD_SIZE == 1
            and not is_interactive()
        )
        log.addHandler(
            RichHandler(
                omit_repeated_times=False,
                level=level,
                console=console,
                show_time=True,
                show_level=True,
                show_path=True,
                markup=use_markup,
                enable_link_path=use_markup,
            )
        )
        log.setLevel(level)
    if (
            len(log.handlers) > 1
            and all([i == log.handlers[0] for i in log.handlers])
    ):
        log.handlers = [log.handlers[0]]
    return log

#
# if __name__ == '__main__':
#     import sys
#     try:
#         framework = sys.argv[1]
#     except IndexError:
#         framework = 'pytorch'
#     try:
#         backend = sys.argv[2]
#     except IndexError:
#         backend = 'deepspeed'
#     try:
#         port = sys.argv[3]
#     except IndexError:
#         port = '5432'
#     check(framework=framework, backend=backend, port=port)
