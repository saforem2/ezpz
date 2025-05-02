"""
ezpz/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function

# import importlib.util
# from importlib import import_module
import logging
import logging.config
import os

# import sys
import warnings
import socket

# sys.flags.lazy_imports = 1
#
# NOTE: Need to swap import order on Polaris (hostname: [x3...])
if socket.gethostname().startswith("x3"):
    from mpi4py import MPI  # type:ignore  # noqa: F401
    import torch  # type:ignore
else:
    import torch  # type: ignore
    from mpi4py import MPI  # type:ignore  # noqa: F401

# try:
#     import deepspeed  # type:ignore
# except Exception:
#     pass


from ezpz.__about__ import __version__

from ezpz import configs
from ezpz import tp
from ezpz import log
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
    OUTPUTS_DIR,
    PROJECT_DIR,
    PROJECT_ROOT,
    SAVEJOBENV,
    SCHEDULERS,
    TrainConfig,
    UTILS,
    command_exists,
    get_logging_config,
    get_scheduler,
    git_ds_info,
    load_ds_config,
    print_config_tree,
)
# from ezpz import dist
# dist_imports = [
#     'check',
#     'cleanup',
#     'get_cpus_per_node',
#     'get_dist_info',
#     'get_gpus_per_node',
#     'get_hostname',
#     'get_hosts_from_hostfile',
#     'get_local_rank',
#     'get_machine',
#     'get_node_index',
#     'get_nodes_from_hostfile',
#     'get_num_nodes',
#     'get_rank',
#     'get_torch_backend',
#     'get_torch_device',
#     'get_torch_device_type',
#     'get_world_size',
#     'include_file',
#     'init_deepspeed',
#     'init_process_group',
#     'print_dist_setup',
#     'query_environment',
#     'run_bash_command',
#     'seed_everything',
#     'setup',
#     # setup_tensorflow,
#     'setup_torch',
#     'setup_torch_DDP',
#     'setup_torch_distributed',
#     'setup_wandb',
#     'synchronize',
#     'timeit',
#     'timeitlogit',
# ]
# for name in dist_imports:
#     module = lazy_import(f"ezpz.dist.{name}")
#     globals()[name] = module

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
    # setup_tensorflow,
    setup_torch,
    setup_torch_DDP,
    setup_torch_distributed,
    setup_wandb,
    synchronize,
    timeit,
    timeitlogit,
)
from ezpz.history import History, StopWatch

# import ezpz.jobs
from ezpz.jobs import (
    add_to_jobslog,
    check_scheduler,
    get_jobdir_from_env,
    get_jobdir_from_jobslog,
    get_jobdirs_from_jobslog,
    get_jobenv,
    get_jobfile_ext,
    get_jobfile_json,
    get_jobfile_sh,
    get_jobfile_yaml,
    get_jobid,
    get_jobslog_file,
    loadjobenv,
    loadjobenv_from_yaml,
    save_to_dotenv_file,
    savejobenv,
    savejobenv_json,
    savejobenv_sh,
    savejobenv_yaml,
    write_launch_shell_script,
)

from ezpz.log import (
    #     # COLORS,
    #     # Console,
    #     # RichHandler,
    #     # STYLES,
    #     # add_columns,
    #     # build_layout,
    #     flatten_dict,
    #     get_console,
    #     # get_file_logger,
    get_logger,
    #     # get_theme,
    #     # get_width,
    #     is_interactive,
    #     # make_layout,
    #     nested_dict_to_df,
    #     print_config,
    #     printarr,
    #     should_do_markup,
    #     # to_bool,
    use_colored_logs,
)
from ezpz import pbs

# from ezpz.plot import tplot, tplot_dict
from ezpz.tplot import tplot, tplot_dict
from ezpz import profile
from ezpz.profile import PyInstrumentProfiler, get_context_manager
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
    get_pipeline_parallel_group,
    get_pipeline_parallel_ranks,
    get_tensor_parallel_group,
    get_tensor_parallel_rank,
    get_tensor_parallel_src_rank,
    get_tensor_parallel_world_size,
    initialize_tensor_parallel,
    tensor_parallel_is_initialized,
)
from ezpz.utils import (
    breakpoint,
    summarize_dict,
    format_pair,
    dataset_from_h5pyfile,
    dataset_to_h5pyfile,
    dict_from_h5pyfile,
    get_max_memory_allocated,
    get_max_memory_reserved,
    get_timestamp,
    grab_tensor,
    save_dataset,
)


# def lazy_import(name: str):
#     spec = importlib.util.find_spec(name)
#     loader = importlib.util.LazyLoader(spec.loader)
#     spec.loader = loader
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[name] = module
#     loader.exec_module(module)
#     return module

if use_colored_logs():
    from ezpz.log.config import use_colored_logs

    try:
        log_config = logging.config.dictConfig(get_logging_config())
    except (ValueError, TypeError, AttributeError) as e:
        warnings.warn(f"Failed to configure logging: {e}")
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    )


logger = logging.getLogger(__name__)
# logger.setLevel("INFO")

RANK = int(MPI.COMM_WORLD.Get_rank())
WORLD_SIZE = int(MPI.COMM_WORLD.Get_size())

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FROM_ALL_RANKS = os.environ.get(
    "LOG_FROM_ALL_RANKS", os.environ.get("LOG_FROM_ALL_RANK", False)
)
if LOG_FROM_ALL_RANKS:
    if RANK == 0:
        logger.info("LOGGING FROM ALL RANKS! BE SURE YOU WANT TO DO THIS !!!")
    logger.setLevel(LOG_LEVEL)
else:
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
    logger.setLevel(LOG_LEVEL) if RANK == 0 else logger.setLevel("CRITICAL")


__all__ = [
    "__version__",
    "BACKENDS",
    # 'BEAT_TIME',
    "BIN_DIR",
    # "COLORS",
    "CONF_DIR",
    # "Console",
    # 'CustomLogging',
    "DS_CONFIG_JSON",
    "DS_CONFIG_PATH",
    "DS_CONFIG_YAML",
    "FRAMEWORKS",
    # 'FluidLogRender',
    "GETJOBENV",
    "HERE",
    "History",
    # 'LOGS_DIR',
    # 'NO_COLOR',
    "OUTPUTS_DIR",
    "PROJECT_DIR",
    "PROJECT_DIR",
    "PROJECT_ROOT",
    "PyInstrumentProfiler",
    # 'QUARTO_OUTPUTS_DIR',
    # "RichHandler",
    "SAVEJOBENV",
    "SCHEDULERS",
    # "STYLES",
    "StopWatch",
    "TrainConfig",
    "UTILS",
    "add_to_jobslog",
    # "add_columns",
    # "build_layout",
    "breakpoint",
    "check",
    "cleanup",
    "command_exists",
    "configs",
    "check_scheduler",
    "get_jobdir_from_env",
    "get_jobdir_from_jobslog",
    "get_jobdirs_from_jobslog",
    "get_jobenv",
    "get_jobfile_ext",
    "get_jobfile_json",
    "get_jobfile_sh",
    "get_jobfile_yaml",
    "get_jobid",
    "get_jobslog_file",
    "loadjobenv",
    "loadjobenv_from_yaml",
    "save_to_dotenv_file",
    "savejobenv",
    "savejobenv_json",
    "savejobenv_sh",
    "savejobenv_yaml",
    "write_launch_shell_script",
    # 'initialize_tensor_parallel',
    # 'tensor_parallel_is_initialized',
    "dataset_from_h5pyfile",
    "dataset_to_h5pyfile",
    "destroy_tensor_parallel",
    "dict_from_h5pyfile",
    # "dist",
    "ensure_divisibility",
    # "flatten_dict",
    "format_pair",
    # "get_console",
    "get_context_manager",
    "get_context_parallel_group",
    "get_context_parallel_rank",
    "get_context_parallel_ranks",
    "get_context_parallel_world_size",
    "get_cpus_per_node",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_dist_info",
    # "get_file_logger",
    "get_gpus_per_node",
    "get_hostname",
    "get_hosts_from_hostfile",
    "get_local_rank",
    "get_logger",
    "get_logging_config",
    "get_machine",
    "get_max_memory_allocated",
    "get_max_memory_reserved",
    "get_tensor_parallel_group",
    "get_tensor_parallel_rank",
    "get_tensor_parallel_src_rank",
    "get_tensor_parallel_world_size",
    "get_node_index",
    "get_nodes_from_hostfile",
    "get_num_nodes",
    "get_pipeline_parallel_group",
    "get_pipeline_parallel_ranks",
    "get_rank",
    "get_scheduler",
    "get_scheduler",
    # "get_theme",
    "get_timestamp",
    "get_torch_backend",
    "get_torch_device",
    "get_torch_device_type",
    # "get_width",
    "get_world_size",
    "git_ds_info",
    "grab_tensor",
    "include_file",
    "init_deepspeed",
    "init_process_group",
    "initialize_tensor_parallel",
    # "is_interactive",
    "load_ds_config",
    "log",
    # "make_layout",
    "pbs",
    # 'print_job_env',
    "tp",
    "tensor_parallel_is_initialized",
    # "nested_dict_to_df",
    # "print_config",
    "print_config_tree",
    "print_dist_setup",
    # "printarr",
    "profile",
    "query_environment",
    "run_bash_command",
    "save_dataset",
    "seed_everything",
    "setup",
    # "setup_tensorflow",
    "setup_torch",
    "setup_torch_DDP",
    "setup_torch_distributed",
    "setup_wandb",
    # "should_do_markup",
    "summarize_dict",
    "synchronize",
    "timeit",
    "timeitlogit",
    # "to_bool",
    "tplot",
    "tplot_dict",
]


os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["ITEX_VERBOSE"] = os.environ.get("ITEX_VERBOSE", "0")
os.environ["LOG_LEVEL_ALL"] = os.environ.get("LOG_LEVEL_ALL", "5")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "5")
os.environ["ITEX_CPP_MIN_LOG_LEVEL"] = os.environ.get(
    "ITEX_CPP_MIN_LOG_LEVEL", "5"
)
os.environ["CCL_LOG_LEVEL"] = os.environ.get("CCL_LOG_LEVEL", "ERROR")
# noqa: E402
warnings.filterwarnings("ignore")

logging.getLogger("deepseed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("sh").setLevel("WARNING")
logging.getLogger("jax").setLevel(logging.ERROR)

try:
    import deepspeed  # noqa type:ignore
except (ImportError, ModuleNotFoundError):
    logger.warning(
        "Unable to import deepspeed. Please install it to use DeepSpeed features."
    )
    pass


if __name__ == "__main__":
    pass
