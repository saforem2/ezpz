"""
ezpz/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function
import logging
import logging.config
import os
import re
from typing import Optional
import warnings

from ezpz import dist
from ezpz import log
from ezpz import profile
from ezpz import configs
from ezpz.configs import (
    BACKENDS,
    BIN_DIR,
    CONF_DIR,
    DS_CONFIG_JSON,
    DS_CONFIG_PATH,
    DS_CONFIG_YAML,
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
    UTILS,
    command_exists,
    get_logging_config,
    get_scheduler,
    get_timestamp,
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
    get_torch_device_type,
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
from ezpz.history import History, StopWatch
import ezpz.log
from ezpz.log import (
    get_file_logger,
    get_logger,
    COLORS,
    Console,
    RichHandler,
    STYLES,
    use_colored_logs,
    add_columns,
    build_layout,
    flatten_dict,
    get_console,
    get_theme,
    get_width,
    is_interactive,
    make_layout,
    nested_dict_to_df,
    print_config,
    printarr,
    should_do_markup,
    to_bool,
)


# from ezpz.log import get_file_logger, get_logger
# from ezpz.log.config import STYLES
# from ezpz.log.console import (
#     Console,
#     get_console,
#     get_theme,
#     get_width,
#     is_interactive,
#     should_do_markup,
#     to_bool,
# )
# from ezpz.log.handler import FluidLogRender, RichHandler
# from ezpz.log.style import (
#     BEAT_TIME,
#     COLORS,
#     CustomLogging,
#     add_columns,
#     build_layout,
#     flatten_dict,
#     make_layout,
#     nested_dict_to_df,
#     print_config,
#     printarr,
# )
import ezpz.tp

from ezpz.tp import (
    destroy_tensor_parallel,
    ensure_divisibility,
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_ranks,
    get_context_parallel_world_size,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_src_rank,
    get_tensor_parallel_world_size,
    get_pipeline_parallel_group,
    get_pipeline_parallel_ranks,
    initialize_tensor_parallel,
    tensor_parallel_is_initialized,
)
from ezpz.plot import tplot, tplot_dict
from ezpz.profile import PyInstrumentProfiler, get_context_manager
from ezpz.utils import grab_tensor
from jaxtyping import ScalarLike
from mpi4py import MPI
import numpy as np
import yaml

# try:
#     import wandb  # pyright: ignore
# except Exception:
#     wandb = None


# def lazy_import(name: str):
#     spec = importlib.util.find_spec(name)
#     loader = importlib.util.LazyLoader(spec.loader)
#     spec.loader = loader
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[name] = module
#     loader.exec_module(module)
#     return module

# TERM = os.environ.get('TERM', None)
# PLAIN = os.environ.get(
#     'NO_COLOR',
#     os.environ.get(
#         'NOCOLOR',
#         os.environ.get(
#             'COLOR', os.environ.get('COLORS', os.environ.get('DUMB', False))
#         ),
#     ),
# )
# if (not PLAIN) and TERM not in ['dumb', 'unknown']:

if use_colored_logs():
    from ezpz.log.config import use_colored_logs

    try:
        log_config = logging.config.dictConfig(get_logging_config())
    except (ValueError, TypeError, AttributeError) as e:
        warnings.warn(f'Failed to configure logging: {e}')
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
        )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    )


logger = logging.getLogger(__name__)
# logger.setLevel("INFO")
logging.getLogger('sh').setLevel('WARNING')


os.environ['PYTHONIOENCODING'] = 'utf-8'
# noqa: E402

RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())

LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FROM_ALL_RANKS = os.environ.get(
    'LOG_FROM_ALL_RANKS', os.environ.get('LOG_FROM_ALL_RANK', False)
)
if LOG_FROM_ALL_RANKS:
    if RANK == 0:
        logger.info('LOGGING FROM ALL RANKS! BE SURE YOU WANT TO DO THIS !!!')
    logger.setLevel(LOG_LEVEL)
# logger.info("Setting logging level to 'INFO' on 'RANK == 0'")
# logger.info("Setting logging level to 'CRITICAL' on all others 'RANK != 0'")
# logger.info(
#     " ".join(
#         [
#             "To disable this behavior,",
#             "and log from ALL ranks (not recommended),",
#             "set: 'export LOG_FROM_ALL_RANKS=1' ",
#             "in your environment, and re-run.",
#         ]
#     )
# )
else:
    logger.setLevel(LOG_LEVEL) if RANK == 0 else logger.setLevel('CRITICAL')


__all__ = [
    'BACKENDS',
    # 'BEAT_TIME',
    'BIN_DIR',
    'COLORS',
    'CONF_DIR',
    'Console',
    # 'CustomLogging',
    'DS_CONFIG_JSON',
    'DS_CONFIG_PATH',
    'DS_CONFIG_YAML',
    'FRAMEWORKS',
    # 'FluidLogRender',
    'GETJOBENV',
    'HERE',
    'History',
    'LOGS_DIR',
    'NO_COLOR',
    'OUTPUTS_DIR',
    'PROJECT_DIR',
    'PROJECT_DIR',
    'PROJECT_ROOT',
    'PyInstrumentProfiler',
    'QUARTO_OUTPUTS_DIR',
    'RichHandler',
    'SAVEJOBENV',
    'SCHEDULERS',
    'STYLES',
    'StopWatch',
    'TrainConfig',
    'UTILS',
    'add_columns',
    'build_layout',
    'check',
    'cleanup',
    'command_exists',
    'configs',
    # 'initialize_tensor_parallel',
    # 'tensor_parallel_is_initialized',
    'destroy_tensor_parallel',
    'dist',
    'ensure_divisibility',
    'flatten_dict',
    'format_pair',
    'get_console',
    'get_context_manager',
    'get_context_parallel_group',
    'get_context_parallel_rank',
    'get_context_parallel_ranks',
    'get_context_parallel_world_size',
    'get_cpus_per_node',
    'get_data_parallel_group',
    'get_data_parallel_rank',
    'get_data_parallel_world_size',
    'get_dist_info',
    'get_file_logger',
    'get_gpus_per_node',
    'get_hostname',
    'get_hosts_from_hostfile',
    'get_local_rank',
    'get_logger',
    'get_logging_config',
    'get_machine',
    'get_tensor_parallel_group',
    'get_tensor_parallel_rank',
    'get_tensor_parallel_src_rank',
    'get_tensor_parallel_world_size',
    'get_node_index',
    'get_nodes_from_hostfile',
    'get_num_nodes',
    'get_pipeline_parallel_group',
    'get_pipeline_parallel_ranks',
    'get_rank',
    'get_scheduler',
    'get_scheduler',
    'get_theme',
    'get_timestamp',
    'get_torch_backend',
    'get_torch_device',
    'get_torch_device_type',
    'get_width',
    'get_world_size',
    'git_ds_info',
    'grab_tensor',
    'include_file',
    'init_deepspeed',
    'init_process_group',
    'initialize_tensor_parallel',
    'is_interactive',
    'load_ds_config',
    'log',
    'make_layout',
    'tensor_parallel_is_initialized',
    'nested_dict_to_df',
    'print_config',
    'print_config_tree',
    'print_dist_setup',
    'printarr',
    'profile',
    'query_environment',
    'run_bash_command',
    'seed_everything',
    'setup',
    'setup_tensorflow',
    'setup_torch',
    'setup_torch_DDP',
    'setup_torch_distributed',
    'setup_wandb',
    'should_do_markup',
    'summarize_dict',
    'timeit',
    'timeitlogit',
    'to_bool',
    'tplot',
    'tplot_dict',
]


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f'{k}={v}'
    # return f'{k}={v:<3.4f}'
    return f'{k}={v:<.6f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


def normalize(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


if __name__ == '__main__':
    pass
