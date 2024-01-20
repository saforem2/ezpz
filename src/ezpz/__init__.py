"""
ezpz/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from typing import Any, Optional
from typing import Union

from enrich.console import get_console, is_interactive
from rich.console import Console
from rich.logging import RichHandler
from ezpz.configs import (
    # getjobenv,
    # savejobenv,
    load_ds_config,
    command_exists,
    git_ds_info,
    BACKENDS,
    get_scheduler,
    print_config_tree,
    BIN_DIR,
    CONF_DIR,
    FRAMEWORKS,
    HERE,
    LOGS_DIR,
    OUTPUTS_DIR,
    PROJECT_DIR,
    PROJECT_ROOT,
    QUARTO_OUTPUTS_DIR,
    SCHEDULERS,
    SAVEJOBENV,
    GETJOBENV,
    TrainConfig,
)

from mpi4py import MPI
import numpy as np
import torch
import tqdm
try:
    import wandb  # type:ignore noqa
except Exception:
    wandb = None

from ezpz import dist
from ezpz.dist import (
    build_mpiexec_thetagpu,
    check,
    timeit,
    timeitlogit,
    setup,
    cleanup,
    get_dist_info,
    get_torch_device,
    get_torch_backend,
    get_hosts_from_hostfile,
    get_machine,
    get_hostname,
    get_local_rank,
    get_rank,
    get_cobalt_nodefile,
    get_nodes_from_hostfile,
    get_num_nodes,
    get_gpus_per_node,
    get_cpus_per_node,
    get_node_index,
    get_cobalt_resources,
    get_world_size,
    init_process_group,
    init_deepspeed,     # ✔︎
    include_file,
    query_environment,
    print_dist_setup,    # ✔︎
    run_bash_command,
    setup_torch_DDP,    # ✔︎
    seed_everything,     # ✔︎
    setup_tensorflow,    # ✔︎
    setup_torch,         # ✔︎
    setup_torch_distributed,  # ✔︎
    setup_wandb,         # ✔︎
    inspect_cobalt_running_job,
)

from ezpz.jobs import (
    savejobenv,
    loadjobenv,
)

log = logging.getLogger(__name__)

__all__ = [
    'build_mpiexec_thetagpu',
    'dist',
    'setup',
    'setup_wandb',
    'print_config_tree',
    'setup_tensorflow',
    'loadjobenv',
    'savejobenv',
    'command_exists',
    'print_dist_setup',
    'get_torch_device',
    'get_scheduler',
    'get_hostname',
    'get_hosts_from_hostfile',
    'get_torch_backend',
    'get_scheduler',
    'get_cpus_per_node',
    'get_num_nodes',
    'git_ds_info',
    'init_process_group',
    'get_dist_info',
    'get_machine',
    'setup_torch',
    'timeit',
    'timeitlogit',
    'setup_torch_DDP',
    'include_file',
    'init_deepspeed',
    'setup_torch_distributed',
    'inspect_cobalt_running_job',
    'get_cobalt_nodefile',
    'get_cobalt_resources',
    'get_nodes_from_hostfile',
    'get_gpus_per_node',
    'cleanup',
    'seed_everything',
    'run_bash_command',
    'get_rank',
    'get_local_rank',
    'get_node_index',
    'get_world_size',
    'query_environment',
    'check',
    'HERE',
    'PROJECT_DIR',
    'PROJECT_ROOT',
    'CONF_DIR',
    'LOGS_DIR',
    'SCHEDULERS',
    'BIN_DIR',
    "SAVEJOBENV",
    "GETJOBENV",
    'OUTPUTS_DIR',
    'QUARTO_OUTPUTS_DIR',
    'FRAMEWORKS',
    'BACKENDS',
    'load_ds_config',
    'TrainConfig',
    'grab_tensor',
]


os.environ['PYTHONIOENCODING'] = 'utf-8'
RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())

ScalarLike = Union[int, float, bool, np.floating]


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
            import tensorflow as tf
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
