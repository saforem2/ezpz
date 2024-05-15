"""
ezpz/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
from mpi4py import MPI
import logging
import logging.config
import os
import re
from typing import Any, Optional
from typing import Union

from ezpz.log.console import get_console, is_interactive
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
import torch
# import tqdm

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

from ezpz.log import get_logger, get_file_logger

from ezpz.log.style import (
    make_layout,
    build_layout,
    add_columns,
    flatten_dict,
    nested_dict_to_df,
    print_config,
    CustomLogging,
    printarr,
    BEAT_TIME,
    COLORS
)

from ezpz.log.handler import (
    RichHandler,
    FluidLogRender,
)

from ezpz.log.console import (
    get_theme,
    is_interactive,
    get_width,
    Console,
    to_bool,
    should_do_markup,
    get_console,
)

from ezpz.log.config import (
    STYLES,
    NO_COLOR,
    DEFAULT_STYLES,
)

try:
    import wandb  # type: ignore
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
    "BACKENDS",
    "BIN_DIR",
    "CONF_DIR",
    "FRAMEWORKS",
    "GETJOBENV", "HERE",
    "LOGS_DIR",
    "OUTPUTS_DIR",
    "PROJECT_DIR",
    "PROJECT_DIR",
    "PROJECT_ROOT",
    "QUARTO_OUTPUTS_DIR",
    "SAVEJOBENV",
    "SCHEDULERS",
    "TrainConfig",
    "build_mpiexec_thetagpu",
    "check",
    "cleanup",
    "command_exists",
    "dist",
    "get_cobalt_nodefile",
    "get_cobalt_resources",
    "get_cpus_per_node",
    "get_dist_info",
    "get_gpus_per_node",
    "get_hostname",
    "get_hosts_from_hostfile",
    "get_local_rank",
    "get_logging_config",
    "get_machine",
    "get_node_index",
    "get_nodes_from_hostfile",
    "get_num_nodes",
    "get_rank",
    "get_scheduler",
    "get_scheduler",
    "get_torch_backend",
    "get_torch_device",
    "get_world_size",
    "git_ds_info",
    "grab_tensor",
    "include_file",
    "init_deepspeed",
    "init_process_group",
    "inspect_cobalt_running_job",
    "load_ds_config",
    "loadjobenv",
    "print_config_tree",
    "print_dist_setup",
    "query_environment",
    "run_bash_command",
    "savejobenv",
    "seed_everything",
    "setup",
    "setup_tensorflow",
    "setup_torch",
    "setup_torch_DDP",
    "setup_torch_distributed",
    "setup_wandb",
    "timeit",
    "make_layout",
    "build_layout",
    "add_columns",
    "flatten_dict",
    "nested_dict_to_df",
    "print_config",
    "CustomLogging",
    "printarr",
    "BEAT_TIME",
    "COLORS",
    "RichHandler",
    "FluidLogRender",
    "get_console",
    "get_theme",
    "is_interactive",
    "get_width",
    "should_do_markup",
    "to_bool",
    "Console",
    "STYLES",
    "NO_COLOR",
    "DEFAULT_STYLES",
    'timeitlogit',
    "get_logger",
    "get_file_logger",
]


def get_timestamp(fstr=None) -> str:
    """Get formatted timestamp."""
    import datetime
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def get_console_from_logger(logger: logging.Logger) -> Console:
    from ezpz.log.handler import RichHandler as EnrichHandler
    for handler in logger.handlers:
        if isinstance(handler, (RichHandler, EnrichHandler)):
            return handler.console
    from ezpz.log.console import get_console
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
    from ezpz.log.handler import RichHandler
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


def _get_file_logger_old(
        name: Optional[str] = None,
        level: str = 'INFO',
        rank_zero_only: bool = True,
        fname: Optional[str] = None,
) -> logging.Logger:
    import logging
    fname = 'output' if fname is None else fname
    log = logging.getLogger(name)
    fh = logging.FileHandler(f"{fname}.log")
    log.setLevel(level)
    fh.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_enrich_logging_config_as_yaml(
        name: str = 'enrich',
        level: str = 'INFO') -> str:
    return fr"""
    ---
    # version: 1
    handlers:
      {name}:
        (): ezpz.log.handler.RichHandler
        show_time: true
        show_level: true
        enable_link_path: false
        level: {level.upper()}
    root:
      handlers: [{name}]
    disable_existing_loggers: false
    ...
    """


# def get_logging_config_as_yaml(level: str = 'DEBUG') -> str:
#     # >>> import yaml
#     # >>>
#     # >>> names_yaml = """
#     # ... - 'eric'
#     # ... - 'justin'
#     # ... - 'mary-kate'
#     # ... """
#     return fr"""
#     handlers:
#       term:
#         class: ezpz.log.handler.RichHandler
#         show_time: true
#         show_level: true
#         enable_link_path: false
#         level: {level}
#     root:
#       handlers: [term]
#     disable_existing_loggers: false
#     """
#

# def get_logging_config(level: str = 'INFO') -> logging.config.dictConfig:
#     config = yaml.safe_load(get_logging_config_as_yaml(level=level))
#     # with Path('logconf.yaml').open('r') as stream:
#     #     config = yaml.load(stream, Loader=yaml.FullLoader)
#     return logging.config.dictConfig(config)


def get_logger_new(
        name: str,
        level: str = 'INFO',
):
    import yaml
    config = yaml.safe_load(
        get_enrich_logging_config_as_yaml(
            name=name,
            level=level
        ),
    )
    #     #     config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)
    log = logging.getLogger(name=name)
    log.setLevel(level)
    return log
