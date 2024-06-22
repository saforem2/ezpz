"""
ezpz/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function
from mpi4py import MPI
import logging
import logging.config
import os
import re

# import socket
from typing import Any, Optional
from typing import Union

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
import torch

from ezpz import dist
from ezpz import plot
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
    check,
    cleanup,
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
from ezpz.log import get_file_logger, get_logger
from ezpz.log.config import DEFAULT_STYLES, NO_COLOR, STYLES
from ezpz.log.console import (
    Console,
    get_console,
    get_theme,
    get_width,
    is_interactive,
    should_do_markup,
    to_bool,
)
from ezpz.log.console import get_console, is_interactive
from ezpz.log.handler import FluidLogRender, RichHandler
from ezpz.log.style import (
    BEAT_TIME,
    COLORS,
    CustomLogging,
    add_columns,
    build_layout,
    flatten_dict,
    make_layout,
    nested_dict_to_df,
    print_config,
    printarr,
)
from ezpz.profile import PyInstrumentProfiler, get_context_manager

try:
    import wandb  # pyright: ignore
except Exception:
    wandb = None


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
# log.setLevel('INFO')
logging.getLogger("sh").setLevel("WARNING")


ScalarLike = Union[int, float, bool, np.floating]
os.environ["PYTHONIOENCODING"] = "utf-8"
# noqa: E402
RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())

log.setLevel("INFO") if RANK == 0 else log.setLevel("CRITICAL")
log.info("Setting logging level to 'INFO' on 'RANK == 0'")
log.info("Setting logging level to 'CRITICAL' on 'RANK != 0'")
log.info(
    ' ' .join(
        [
            "To disable this behavior,",
            "and log from ALL ranks (not recommended), set:",
            "'export LOG_FROM_ALL_RANKS=1' in your environment, and re-run."
        ]
    )
)
LOG_FROM_ALL_RANKS = os.environ.get(
    "LOG_FROM_ALL_RANKS",
    os.environ.get(
        "LOG_FROM_ALL_RANK",
        False
    )
)
if LOG_FROM_ALL_RANKS:
    log.setLevel("INFO")
else:
    log.setLevel("INFO") if RANK == 0 else log.setLevel("CRITICAL")


__all__ = [
    "BACKENDS",
    "BEAT_TIME",
    "BIN_DIR",
    "COLORS",
    "CONF_DIR",
    "Console",
    "CustomLogging",
    "DEFAULT_STYLES",
    "FRAMEWORKS",
    "FluidLogRender",
    "GETJOBENV",
    "HERE",
    "LOGS_DIR",
    "NO_COLOR",
    "OUTPUTS_DIR",
    "PROJECT_DIR",
    "PROJECT_DIR",
    "PROJECT_ROOT",
    "QUARTO_OUTPUTS_DIR",
    "RichHandler",
    "SAVEJOBENV",
    "SCHEDULERS",
    "STYLES",
    "PyInstrumentProfiler",
    "TrainConfig",
    "add_columns",
    "build_layout",
    "check",
    "cleanup",
    "command_exists",
    "dist",
    "flatten_dict",
    "get_console",
    "get_context_manager",
    "get_cpus_per_node",
    "get_dist_info",
    "get_file_logger",
    "get_gpus_per_node",
    "get_hostname",
    "get_hosts_from_hostfile",
    "get_local_rank",
    "get_logger",
    "get_logging_config",
    "get_machine",
    "get_node_index",
    "get_nodes_from_hostfile",
    "get_num_nodes",
    "get_rank",
    "get_scheduler",
    "get_scheduler",
    "get_theme",
    "get_torch_backend",
    "get_torch_device",
    "get_width",
    "get_world_size",
    "git_ds_info",
    "grab_tensor",
    "include_file",
    "init_deepspeed",
    "init_process_group",
    "is_interactive",
    "load_ds_config",
    "loadjobenv",
    "make_layout",
    "nested_dict_to_df",
    "plot",
    "print_config",
    "print_config_tree",
    "print_dist_setup",
    "printarr",
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
    "should_do_markup",
    "timeit",
    "timeitlogit",
    "to_bool",
]


def get_timestamp(fstr: Optional[str] = None) -> str:
    """Get formatted timestamp."""
    import datetime

    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime("%Y-%m-%d-%H%M%S")
    return now.strftime(fstr)


def normalize(name: str) -> str:
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
    elif callable(getattr(x, "numpy", None)):
        assert callable(getattr(x, "numpy"))
        return x.numpy()
    raise ValueError


def get_rich_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
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
        enable_link_path=False,
    )
    log.handlers = [handler]
    log.setLevel(level)
    return log


# def _get_file_logger_old(
#         name: Optional[str] = None,
#         level: str = 'INFO',
#         rank_zero_only: bool = True,
#         fname: Optional[str] = None,
# ) -> logging.Logger:
#     import logging
#     fname = 'output' if fname is None else fname
#     log = logging.getLogger(name)
#     fh = logging.FileHandler(f"{fname}.log")
#     log.setLevel(level)
#     fh.setLevel(level)
#     # create formatter and add it to the handlers
#     formatter = logging.Formatter(
#         "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
#     )
#     fh.setFormatter(formatter)
#     log.addHandler(fh)
#     return log


def get_enrich_logging_config_as_yaml(name: str = "enrich", level: str = "INFO") -> str:
    return rf"""
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


def get_logger_new(
    name: str,
    level: str = "INFO",
):
    import yaml
    config = yaml.safe_load(
        get_enrich_logging_config_as_yaml(name=name, level=level),
    )
    logging.config.dictConfig(config)
    log = logging.getLogger(name=name)
    log.setLevel(level)
    return log
