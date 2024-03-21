"""
ezpz/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import logging.config
import os
import re
from typing import Any, Optional
from typing import Union

from enrich.console import get_console, is_interactive
from mpi4py import MPI
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
import torch
import tqdm

from ezpz import dist
from ezpz.configs import (
    BACKENDS,
    BIN_DIR,
    CONF_DIR,
    FRAMEWORKS,
    GETJOBENV,
    HERE,
    LOGS_DIR,
    OUTPUTS_DIR,
    PROJECT_DIR,
    PROJECT_ROOT,
    QUARTO_OUTPUTS_DIR,
    SAVEJOBENV,
    SCHEDULERS,
    TrainConfig,
    command_exists,
    get_logging_config,
    get_scheduler,
    git_ds_info,
    load_ds_config,
    print_config_tree,
)
from ezpz.dist import (
    build_mpiexec_thetagpu,
    check,
    cleanup,
    get_cobalt_nodefile,
    get_cobalt_resources,
    get_cpus_per_node,
    get_dist_info,
    get_gpus_per_node,
    get_hostname,
    get_hosts_from_hostfile,
    get_local_rank,
    get_machine,
    get_node_index,
    get_nodes_from_hostfile,
    get_num_nodes,
    get_rank,
    get_torch_backend,
    get_torch_device,
    get_world_size,
    include_file,
    init_deepspeed,
    init_process_group,
    inspect_cobalt_running_job,
    print_dist_setup,
    query_environment,
    run_bash_command,
    seed_everything,
    setup,
    setup_tensorflow,
    setup_torch,
    setup_torch_DDP,
    setup_torch_distributed,
    setup_wandb,
    timeit,
    timeitlogit,
)
from ezpz.jobs import loadjobenv, savejobenv
try:
    import wandb  # type:ignore noqa
except Exception:
    wandb = None

# import importlib.util
# import sys

#
# def lazy_import(name: str):
#     spec = importlib.util.find_spec(name)
#     loader = importlib.util.LazyLoader(spec.loader)
#     spec.loader = loader
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[name] = module
#     loader.exec_module(module)
#     return module


log_config = logging.config.dictConfig(get_logging_config())
log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.getLogger('sh').setLevel('WARNING')

os.environ['PYTHONIOENCODING'] = 'utf-8'
RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())

ScalarLike = Union[int, float, bool, np.floating]


__all__ = [
    'BACKENDS',
    'BIN_DIR',
    'CONF_DIR',
    'FRAMEWORKS',
    'GETJOBENV', 'HERE',
    'LOGS_DIR',
    'OUTPUTS_DIR',
    'PROJECT_DIR',
    'PROJECT_DIR',
    'PROJECT_ROOT',
    'QUARTO_OUTPUTS_DIR',
    'SAVEJOBENV',
    'SCHEDULERS',
    'TrainConfig',
    'build_mpiexec_thetagpu',
    'check',
    'cleanup',
    'command_exists',
    'dist',
    'get_cobalt_nodefile',
    'get_cobalt_resources',
    'get_cpus_per_node',
    'get_dist_info',
    'get_gpus_per_node',
    'get_hostname',
    'get_hosts_from_hostfile',
    'get_local_rank',
    'get_logging_config',
    'get_machine',
    'get_node_index',
    'get_nodes_from_hostfile',
    'get_num_nodes',
    'get_rank',
    'get_scheduler',
    'get_scheduler',
    'get_torch_backend',
    'get_torch_device',
    'get_world_size',
    'git_ds_info',
    'grab_tensor',
    'include_file',
    'init_deepspeed',
    'init_process_group',
    'inspect_cobalt_running_job',
    'load_ds_config',
    'loadjobenv',
    'print_config_tree',
    'print_dist_setup',
    'query_environment',
    'run_bash_command',
    'savejobenv',
    'seed_everything',
    'setup',
    'setup_tensorflow',
    'setup_torch',
    'setup_torch_DDP',
    'setup_torch_distributed',
    'setup_wandb',
    'timeit',
    'timeitlogit'
]

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def get_console_from_logger(logger: logging.Logger) -> Console:
    from enrich.handler import RichHandler as EnrichHandler
    for handler in logger.handlers:
        if isinstance(handler, (RichHandler, EnrichHandler)):
            return handler.console
    from enrich.console import get_console
    return get_console()


def grab_tensor(x: Any) -> np.ndarray | ScalarLike | None:
    if x is None:
        return None
    if isinstance(x, (int, float, bool, np.floating)):
        return x
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        elif isinstance(x[0], np.ndarray):
            return np.stack(x)
        else:
            try:
                import tensorflow as tf  # type:ignore
            except (ImportError, ModuleNotFoundError) as exc:
                raise exc
            if isinstance(x[0], tf.Tensor):
                return grab_tensor(tf.stack(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif callable(getattr(x, 'numpy', None)):
        assert callable(getattr(x, 'numpy'))
        return x.numpy()
    raise ValueError


def get_rich_logger(
        name: Optional[str] = None,
        level: str = 'INFO'
) -> logging.Logger:
    from enrich.handler import RichHandler
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
    from enrich.handler import RichHandler
    # log.handlers = []
    # from rich.logging import RichHandler
    # from l2hmc.utils.rich import get_console, is_interactive
    from enrich.console import get_console
    # from enrich import is_interactive
    # format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    _ = (
            log.setLevel("CRITICAL") if (RANK == 0 and rank_zero_only)
            else log.setLevel(level)
    )
    # if rank_zero_only:
    #     if RANK != 0:
    #         log.setLevel('CRITICAL')
    #     else:
    #         log.setLevel(level)
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
