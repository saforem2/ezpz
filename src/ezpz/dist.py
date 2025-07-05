"""
dist.py

Contains methods for initializing distributed communication.
"""

from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
import socket
from pathlib import Path
import time
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Union

import ezpz.tp

from mpi4py import MPI

import torch
import torch.distributed
from datetime import timedelta
from omegaconf import DictConfig, OmegaConf

from ezpz.lazy import lazy_import

try:
    wandb = lazy_import("wandb")
except Exception:
    wandb = None

try:
    ipex = lazy_import("intel_extension_for_pytorch")
except (ImportError, ModuleNotFoundError):
    ipex = None


if not os.environ.get(
    "DUMB", os.environ.get("NOCOLOR", os.environ.get("NO_COLOR", False))
):
    os.environ["COLORTERM"] = "truecolor"

PathLike = Union[str, os.PathLike, Path]

EZPZ_LOG_LEVEL = str(os.environ.get("EZPZ_LOG_LEVEL", "INFO")).upper()
logger = logging.getLogger(__name__)
logger.setLevel(EZPZ_LOG_LEVEL)
logging.getLogger("sh").setLevel("WARNING")


# def try_import(module_name: str):
#     try:
#         return __import__(module_name)
#     except Exception:
#         logger.info(f"Unable to import '{module_name}', trying to continue")


ACCELERATOR_TYPE = (
    "IntelGPU"
    if torch.xpu.is_available() and torch.xpu.device_count() > 0
    else (
        "NvidiaGPU"
        if (torch.cuda.is_available() and torch.cuda.device_count() > 0)
        else ("MPS" if torch.backends.mps.is_available() else "CPU")
    )
)


# @dataclass
# class TorchDistributedInfo:
#     backend: str  # [DDP, deepspeed, horovod]
#     rank: int    # [0, ..., world_size - 1]
#     local_rank: int  # [0, ..., ]
#     world_size: int


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed to set.
    """
    import torch
    import numpy as np
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed)


def log_dict_as_bulleted_list(d: dict, name: Optional[str] = None):
    """Print a dict as bullets."""
    tag = name or getattr(d, "__qualname__", "dict")
    lines = [f"[{tag}]:"] + [f"  • {k}={v}" for k, v in d.items()]
    logger.info("\n\n" + "\n".join(lines) + "\n")


def timeitlogit(rank: Optional[int] = None, verbose: bool = True):
    """Decorator to time a function and log the time taken.

    Args:
        rank (int, optional): Rank of the process. Defaults to None.
        verbose (bool, optional): Whether to log the time taken. Defaults to True.

    Example:
        @timeitlogit(rank=0, verbose=True)
        def my_function(arg1, arg2):
            # Function implementation
            pass
    """
    rank = get_rank() if rank is None else rank
    try:
        import wandb
    except Exception:
        wandb = None  # type:ignore

    def decorator(func: Callable):
        """Decorator to time a function and log the time taken.

        Args:
            func (Callable): Function to be timed.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            assert isinstance(rank, int)
            result = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            fname = getattr(
                func, "__qualname__", getattr(func, "__name__", "unknown")
            )
            if verbose and rank == 0:
                arg_str = ", ".join(map(str, args))
                kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                inner = ", ".join(filter(None, [arg_str, kw_str]))
                logger.info(f"{fname}({inner}) took {dt:.4f} s")
                if wandb is not None and wandb.run is not None:
                    wandb.log({f"timeit/{fname}": dt}, commit=False)
            # if verbose:
            #     if rank == 0:
            #         astr = []
            #         if len(args) > 0:
            #             astr.append(f"({args}")
            #         _ = (
            #             astr.append(f", {kwargs})")
            #             if len(kwargs) > 0
            #             else (astr.append(")") if len(args) > 0 else "")
            #         )
            #         zstr = [f"Called: '{fname}' with arguments:"]
            #         if len(astr) > 0:
            #             zstr.append(f"{''.join(astr)}")
            #         zstr.append(f"'{fname}' took: {dt=:.4f} s")
            #         logger.info("\n".join(zstr))
            return result

        return wrapper

    return decorator


def timeit(func: Callable):
    """
    Decorator to time a function and log the time taken.

    Args:
        func (Callable): Function to be timed.

    Example:
        @timeit
        def my_function(arg1, arg2):
            # Function implementation
            pass
    """
    try:
        import wandb
    except Exception:
        wandb = None  # type:ignore

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        fname = getattr(
            func, "__qualname__", getattr(func, "__name__", "unknown")
        )
        logger.info(f"{fname}({args}, {kwargs}) took: {dt=:.4f}s")
        if wandb is not None and wandb.run is not None:
            wandb.log({f"timeit/{fname}": dt})
        return result

    return wrapper


def get_hosts_from_hostfile(
    hostfile: Optional[str | Path] = None,  # type:ignore[reportDeprecated]
) -> tuple[str, list[str]]:
    """
    Get hosts from the hostfile or environment variables.

    Args:
        hostfile (str | Path, optional): Path to the hostfile. Defaults to None.

    Returns:
        tuple[str, list[str]]: Tuple containing the hostfile path and a list of hosts.

    Example:
        >>> get_hosts_from_hostfile("/path/to/hostfile")
        ('/path/to/hostfile', ['host1', 'host2', ...])
    """
    # hostfile = '' if hostfile is None else hostfile
    hostname = get_hostname()
    hostfile = os.environ.get(
        "HOSTFILE",
        os.environ.get(
            "PBS_NODEFILE",
            os.environ.get(
                "COBALT_NODEFILE",
                None,
            ),
        ),
    )
    hosts: list[str] = []
    assert hostfile is not None
    if Path(hostfile).is_file():
        if get_rank() == 0:
            logger.debug(f"Reading hosts from {hostfile}")
        hpath = Path(hostfile).resolve().absolute()
        with hpath.open("r") as f:
            hosts.extend([h.rstrip("\n") for h in f.readlines()])
    else:
        hosts.append(hostname)
    return Path(hostfile).as_posix(), hosts


def get_hostname() -> str:
    """Get the hostname of the current machine.

    Returns:
        str: The hostname of the current machine.
    """
    import socket

    try:
        hostname = socket.gethostbyaddr(socket.gethostname())[0].lower()
    # except socket.herror as exc:
    except Exception:
        from sh import hostname as sh_hostname  # type:ignore[missingTypeStubs]

        hostname = sh_hostname()
        # if get_rank() == 0:
        #     logger.debug('Unable to determine hostname with `socket`.')
        #     logger.debug(f'hostname from`sh`: {hostname}')
        #     # logger.exception(exc)
    return hostname.rstrip("\n")


def _get_dist_info(
    hostfile: Optional[PathLike] = None,
    framework: Optional[str] = None,
    # max_hosts_to_print: Optional[int] = None,  # truncate in logs
) -> dict:
    """
    Get distributed info from the hostfile or environment variables.

    Args:
        hostfile (PathLike, optional): Path to the hostfile. Defaults to None.
        framework (str, optional): Framework to use. Defaults to None.
        max_hosts_to_print (int, optional): Maximum number of hosts to print in logs.
            Defaults to None.

    Returns:
        dict: Dictionary containing the distributed info.
            Includes keys like 'DEVICE', 'DEVICE_ID', 'DISTRIBUTED_BACKEND', etc.
    """
    from ezpz.configs import get_scheduler

    hf = get_hostfile_with_fallback(hostfile) if hostfile is None else hostfile
    hfp = Path(hf)
    assert hfp is not None and hfp.is_file()
    hosts = get_nodes_from_hostfile(hfp.as_posix())
    num_nodes = len(hosts)
    num_gpus_per_node = get_gpus_per_node()
    num_gpus = num_nodes * num_gpus_per_node
    dist_info = {}
    if framework is not None:
        dist_info |= {"FRAMEWORK": framework}
    dist_info |= {
        "DEVICE": get_torch_device(),
        "DEVICE_ID": f"{get_torch_device()}:{get_local_rank()}",
        "DISTRIBUTED_BACKEND": get_torch_backend(),
        "GPUS_PER_NODE": num_gpus_per_node,
        "HOSTS": f"{hosts}",
        "HOSTFILE": hfp.absolute().resolve().as_posix(),
        "HOSTNAME": get_hostname(),
        "LOCAL_RANK": get_local_rank(),
        "MACHINE": get_machine(),
        "NUM_NODES": num_nodes,
        "NGPUS": num_gpus,
        "NGPUS_AVAILABLE": get_world_size_total(),
        # 'NGPUS': get_world_size_total(),
        "NODE_ID": get_node_index(),
        "RANK": get_rank(),
        "SCHEDULER": (scheduler := get_scheduler()),
        # 'WORLD_SIZE': get_world_size(),
        "WORLD_SIZE_TOTAL": get_world_size_total(),
        "WORLD_SIZE_IN_USE": get_world_size_in_use(),
        "LAUNCH_CMD": (
            ezpz.pbs.get_pbs_launch_cmd(
                hostfile=hfp,
            )
            if scheduler.lower() == "pbs"
            else None
        ),
    }
    # ws = os.environ.get("WORLD_SIZE", None)
    # if ws is not None:
    #     logger.info(f'Caught "WORLD_SIZE"={ws} from environment!')
    #     dist_info |= {"WORLD_SIZE": int(ws)}
    # hostfile = (
    #     Path(get_hostfile_with_fallback(hostfile)).as_posix()
    #     if hostfile is None else hostfile
    # )
    # assert hostfile is not None and Path(hostfile).is_file(), (
    #     f'{hostfile=} not None and {Path(hostfile).is_file()=}'
    # )
    # if max_hosts_to_print is not None and len(hosts) > max_hosts_to_print:
    #     # if len(hosts) > max_hosts_to_print:
    #     logger.warning(f'{len(hosts)=} > {max_hosts_to_print=} in dist.get_dist_info')
    #     logger.warning(f'Truncating `hosts: [addr1, addr2, ...] at {max_hosts_to_print}')
    # hosts = (
    #     [h.split('.')[0] for h in hosts] if (
    #                 max_hosts_to_print is not None
    #                 and len(hosts) < max_hosts_to_print
    #     )
    #     else (
    #         [h.split('.')[0] for h in hosts[:max_hosts_to_print]].extend(
    #             [
    #                 f'[(...) truncated ({len(hosts)} > {max_hosts_to_print})]'
    #             ]
    #         )
    #     )
    # )
    return dist_info


def get_dist_info(
    framework: Optional[str] = None,
    verbose: Optional[bool] = None,
    hostfile: Optional[PathLike] = None,
) -> dict[str, str | int | list]:
    """Get distributed info.

    Args:
        framework (str, optional): Framework to use. Defaults to None.
        verbose (bool, optional): Whether to print the info. Defaults to None.
        hostfile (PathLike, optional): Path to the hostfile. Defaults to None.

    Returns:
        dict: Dictionary containing the distributed info.
    """
    dist_info = _get_dist_info(
        hostfile=hostfile,
        framework=framework,
    )
    if verbose:
        import json

        logger.info(
            f"DistInfo={json.dumps(dist_info, indent=4, sort_keys=True)}"
        )
    return dist_info


def print_dist_setup(
    framework: Optional[str] = None,
    hostfile: Optional[PathLike] = None,
) -> str:
    """Print distributed setup.

    Args:
        framework (str, optional): Framework to use. Defaults to None.
        hostfile (PathLike, optional): Path to the hostfile. Defaults to None.

    Returns:
        str: String containing the distributed setup.
    """
    rank = get_rank()
    wst = get_world_size(total=True)
    wsa = get_world_size(in_use=True)
    # world_size = get_world_size()
    local_rank = get_local_rank()
    gpus_per_node = get_gpus_per_node()
    hostfile = get_hostfile_with_fallback(hostfile)
    # NOTE:
    # We ensure that num_nodes is AT LEAST 1
    # since if gpus_per_node > wsa, wsa // gpus_per_node = 0
    # if gpus_per_node > wsa, wsa // gpus_per_node = 0
    num_nodes = max((wsa // gpus_per_node, 1))
    num_nodes_from_hostfile = get_num_nodes()
    # assert num_nodes_from_hostfile == num_nodes
    # if num_nodes != num_nodes_from_hostfile:
    #     logger.critical(f'{num_nodes=} vs. {num_nodes_from_hostfile=} ??')
    node = get_node_index()
    device = None
    # if framework.lower() in {'pt', 'torch', 'pytorch'}:
    device = get_torch_device_type()
    rank_len = len(str(rank))
    ws_len = len(str(wsa))
    lr_len = len(str(local_rank))
    gpn_len = len(str(gpus_per_node))
    node_len = len(str(node))
    num_nodes_len = len(str(num_nodes))
    dist_list = [
        f"[{device=}]",
        f"[{rank=:>{rank_len}}/{(wsa - 1):<{ws_len}}]",
        f"[{local_rank=:>{lr_len}}/{gpus_per_node - 1:<{gpn_len}}]",
        f"[{node=:>{node_len}}/{(num_nodes - 1):<{num_nodes_len}}]",
    ]
    if framework is not None:
        dist_list.append(f"[{framework=}]")
    dist_str = "".join(dist_list)
    logger.info(f"{dist_str}")
    if rank == 0:
        if wsa > 1000:
            logger.warning(
                f"WORLD_SIZE={wsa} > 1000, only printing on RANK={rank}"
            )
        logger.warning(
            f'Using [{wsa} / {wst}] available "{device}" devices !!'
        )
        if num_nodes_from_hostfile != num_nodes:
            logger.critical(
                f"num_nodes_from_hostfile = [{num_nodes_from_hostfile=}]"
                f"vs."
                f"[{wsa=} // {gpus_per_node=}] = {num_nodes}"
                r"¯\_(ツ)_/¯ ??"
            )
    return dist_str


def synchronize(device: Optional[torch.device | int | str] = None):
    """
    Synchronize the given device.

    Args:
        device (torch.device | int | str, optional): The device to synchronize.
            If None, the default device will be used. Defaults to None.

    Returns:
        None
    """
    return (
        torch.cuda.synchronize(device)
        if torch.cuda.is_available()
        else (
            torch.xpu.synchronize(device)
            if torch.xpu.is_available()
            else (
                torch.mps.synchronize()
                if torch.backends.mps.is_available()
                else torch.cpu.synchronize(device)
            )
        )
    )


def setup(
    framework: str = "pytorch",
    backend: str = "DDP",
    port: str = "5432",
    seed: Optional[int] = None,
    precision: Optional[str] = None,
    ngpus: Optional[int] = None,
):
    """
    Setup distributed environment for the specified framework.

    Args:
        framework (str): The framework to use for distributed training.
            Defaults to "pytorch".
        backend (str): The backend to use for distributed training.
            Defaults to "DDP".
        port (str): The port to use for distributed communication.
            Defaults to "5432".
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        precision (str, optional): Precision to use for training. Defaults to None.
        ngpus (int, optional): Number of GPUs to use. Defaults to None.

    Returns:
        None
    """
    return (
        setup_tensorflow(precision=precision, ngpus=ngpus)
        if framework in {"tensorflow", "tf", "t"}
        else setup_torch(backend=backend, port=port, seed=seed)
    )


def init_deepspeed(
    dist_backend: Optional[str] = None,
    auto_mpi_discovery: bool = True,
    distributed_port: int | str = 29500,
    verbose: bool = True,
    timeout: Optional[int] = None,
    init_method: Optional[str] = None,
    dist_init_required: Optional[bool] = None,
    config: Optional[dict] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """
    Initialize DeepSpeed distributed environment.

    Args:
        dist_backend (str, optional): The distributed backend to use.
            Defaults to None.
        auto_mpi_discovery (bool, optional): Whether to automatically discover MPI.
            Defaults to True.
        distributed_port (int | str, optional): The port for distributed communication.
            Defaults to 29500.
        verbose (bool, optional): Whether to print verbose logs. Defaults to True.
        timeout (int | None, optional): Timeout in seconds for distributed initialization.
            Defaults to None.
        init_method (str, optional): Initialization method for distributed training.
            Defaults to None.
        dist_init_required (bool, optional): Whether distributed initialization is required.
            Defaults to None.
        config (dict, optional): DeepSpeed configuration dictionary. Defaults to None.
        rank (int | None, optional): Rank of the current process. Defaults to None.
        world_size (int | None, optional): Total number of processes. Defaults to None.

    Raises:
        ImportError: If DeepSpeed is not installed.
        Exception: If there is an error during DeepSpeed initialization.

    Example:
        >>> init_deepspeed(
        ...     dist_backend="nccl",
        ...     distributed_port=29500,
        ...     verbose=True,
        ...     timeout=3600,
        ...     rank=0,
        ...     world_size=4,
        ... )
    """
    try:
        import deepspeed  # noqa type:ignore

        os.environ["DEEPSPEED_VERSION"] = deepspeed.__version__
    except ImportError as e:
        logger.warning(
            "Unable to import deepspeed. Please install it to use DeepSpeed features."
        )
        raise ImportError(
            "DeepSpeed is not installed. Install with 'pip install deepspeed'"
        ) from e

    rank = get_rank() if rank is None else rank
    world_size = get_world_size() if world_size is None else world_size
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        import deepspeed  # type:ignore

        # logger.warning(f'Setting {timeout=}')
        dt = 3600 if timeout is None else timeout
        deepspeed.init_distributed(
            dist_backend=dist_backend,
            auto_mpi_discovery=auto_mpi_discovery,
            distributed_port=int(distributed_port),
            verbose=verbose,
            timeout=datetime.timedelta(seconds=dt),
            init_method=init_method,
            dist_init_required=dist_init_required,
            config=config,
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:
        logger.warning("Unable to `import deepspeed`. Exiting!")
        logger.exception(exc)
        raise exc


def get_torch_device_type(device_type: Optional[str] = None) -> str:
    """Get the current PyTorch device type.

    Args:
        device_type (str, optional): The type of device to return.
            If None, it will be determined automatically.
            Defaults to None.

    Returns:
        str: The current PyTorch device type.
            Possible values are "cpu", "mps", "xpu", or "cuda".

    Example:
        >>> get_torch_device_type()  # Returns the current device type as a string
    """
    if device_type is not None:
        assert device_type in (
            "cpu",
            "mps",
            "xpu",
            "cuda",
        )
        logger.warning(
            " ".join(
                [
                    f"device_type: {device_type} passed to",
                    "ezpz.dist.get_torch_device_type",
                ]
            )
        )
        return device_type
    if (tdevice := os.environ.get("TORCH_DEVICE")) is not None:
        if get_rank() == 0:
            logger.warning(f"Caught TORCH_DEVICE={tdevice} from environment!")
        tdevice = tdevice.lower()
        assert tdevice is not None and tdevice in (
            "cpu",
            "mps",
            "xpu",
            "cuda",
        )
        return tdevice.lower()
    return (
        "xpu"
        if torch.xpu.is_available()
        else (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if (
                    torch.backends.mps.is_available()
                    and torch.get_default_dtype() != torch.float64
                )
                else "cpu"
            )
        )
    )


def get_torch_device(
    *,
    device_type: Optional[str] = None,
    as_torch_device: Optional[bool] = None,
) -> str | torch.device:
    """Get the current PyTorch device.

    Args:
        device_type (str, optional): The type of device to return.
            If None, it will be determined automatically.
            Defaults to None.
        as_torch_device (bool, optional): If True, return a torch.device object.
            If False, return a string representing the device type.
            Defaults to False.

    Returns:
        str | torch.device: The current PyTorch device.
            If as_torch_device is True, returns a torch.device object.
            If as_torch_device is False, returns a string representing the device type.

    Example:
        >>> get_torch_device()  # Returns the current device type as a string
    """
    device_type = get_torch_device_type(device_type)
    return torch.device(device_type) if as_torch_device else device_type


def get_torch_version_as_float() -> float:
    """Get the PyTorch version as a float.

    Returns:
        float: The PyTorch version as a float, e.g., 1.10.0 -> 1.10
    """
    return float(".".join(torch.__version__.split(".")[:2]))


def get_torch_backend_on_xpu() -> str:
    """Deal with breaking change introduced in torch 2.6:

    See: https://github.com/pytorch/pytorch/pull/141856

    Example:

        ```python
        >>> torch_version = float('.'join(torch.__version__.split('.')[:2]))
        >>> if torch_version > 2.5:
        >>>     backend = 'xccl'
        >>> else:
        >>>     backend = 'ccl'
        ```
    """
    torch_version = get_torch_version_as_float()
    assert torch.xpu.is_available()
    return "xccl" if torch_version > 2.5 else "ccl"


def get_torch_backend() -> str:
    """
    Get the current PyTorch backend.

    Returns:
        str: The current PyTorch backend.
    """
    backend_from_env = os.environ.get("TORCH_BACKEND", None)
    if backend_from_env is not None:
        return backend_from_env
    return (
        "nccl"
        if torch.cuda.is_available()
        else (
            get_torch_backend_on_xpu() if torch.xpu.is_available() else "gloo"
        )
    )


def init_process_group(
    rank: int | str,
    world_size: int | str,
    timeout: str | int | timedelta,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize the PyTorch distributed process group.

    Args:
        rank (int | str): The rank of the current process.
        world_size (int | str): The total number of processes.
        timeout (str | int | timedelta): Timeout for the process group initialization.
        backend (str, optional): The backend to use for distributed training.
            Defaults to None, which will use the default backend based on the device.
    """
    backend = get_torch_backend() if backend is None else backend
    if get_rank() == 0:
        logger.info(
            " ".join(
                [
                    "Calling torch.distributed.init_process_group_with:",
                    f"rank={rank}",
                    f"world_size={world_size}",
                    f"backend={backend}",
                ]
            )
        )
    if not isinstance(timeout, timedelta):
        timeout = timedelta(
            seconds=int(timeout),
        )
    if not torch.distributed.is_initialized():  # type:ignore
        torch.distributed.init_process_group(  # type:ignore
            backend=backend,
            timeout=timeout,
            rank=int(rank),
            world_size=int(world_size),
            init_method="env://",
        )


def run_ddp(fn: Callable, world_size: int) -> None:
    """
    Run a function in a distributed data parallel (DDP) setup.

    Args:
        fn (Callable): The function to run in DDP.
        world_size (int): The total number of processes to run.
    """
    import torch.multiprocessing as mp

    mp.spawn(  # type:ignore
        fn, args=(world_size,), nprocs=world_size, join=True
    )


def get_rank() -> int:
    """Get current MPI rank.

    Returns:
        int: The rank of the current process in the MPI world.

    Example:
        >>> rank = get_rank()
        >>> print(f"Current MPI rank: {rank}")
    """
    return int(MPI.COMM_WORLD.Get_rank())


def get_world_size_in_use() -> int:
    """Get number of currently in use MPI ranks

    Returns:
        int: The number of currently in use MPI ranks.
            This is the size of the MPI communicator.
            It is the number of processes that are currently running
            and participating in the distributed computation.

    Example:
        >>> world_size_in_use = get_world_size_in_use()
        >>> print(f"Number of currently in use MPI ranks: {world_size_in_use}")
    """
    return int(MPI.COMM_WORLD.Get_size())


def get_world_size_total() -> int:
    """Calculate total AVAILABLE *PUs as:

    total = [num_hosts] * [num_*pu_per_host]

    Returns:
        int: The total number of available *PUs across all nodes.
            This is the product of the number of nodes and the number of *PUs per node.

    Example:
        >>> total_pus = get_world_size_total()
        >>> print(f"Total available *PUs: {total_pus}")
    """
    # nhosts = get_num_nodes()
    # ngpu_per_host = get_gpus_per_node()
    # return ngpu_per_host * nhosts
    return get_gpus_per_node() * get_num_nodes()


def get_world_size(
    total: Optional[bool] = None,
    in_use: Optional[bool] = None,
) -> int:
    """
    Get the total number of *PUs available or currently in use.
    Args:
        total (bool, optional): If True, return the total number of *PUs available.
            Defaults to None.
        in_use (bool, optional): If True, return the number of *PUs currently in use.
            Defaults to None.

    Returns:
        int: The total number of *PUs available or currently in use.
            If both `total` and `in_use` are None, it returns the number of *PUs
            currently in use by the MPI communicator.

    Example:
        >>> world_size = get_world_size(total=True)
        >>> print(f"Total number of *PUs available: {world_size}")
        >>> world_size_in_use = get_world_size(in_use=True)
        >>> print(f"Number of *PUs currently in use: {world_size_in_use}")
    """
    if total:
        return get_world_size_total()
    if in_use:
        return get_world_size_in_use()
    # TODO: Deal with subtlety between:
    # 1. 'world_size' == total AVAILABLE gpus (for record keeping)
    # 2. 'world_size' == number of gpus CURRENTLY IN USE (from {`mpi`, ...})
    # ¯\_(ツ)_/¯
    try:
        world_size = int(MPI.COMM_WORLD.Get_size())
    except Exception:
        num_nodes = get_num_nodes()
        gpus_per_node = get_gpus_per_node()
        world_size = num_nodes * gpus_per_node
        logger.warning(
            "MPI not initialized !!"
            "Calculating (and using!! ??) "
            "[world_size]=[(num_nodes) x (num_*pus_per_node)]=[num_*pus_total]"
            f"[{world_size}]=[({num_nodes}) x ({gpus_per_node})]"
        )
    # if world_size == 1:
    #     gpus_per_node = get_gpus_per_node()
    #     num_nodes = get_num_nodes()
    #     world_size = num_nodes * gpus_per_node
    return world_size


def get_local_rank() -> int:
    """Return `get_rank() % get_gpus_per_node()`

    Returns:
        int: The local rank of the current process within its node.
            This is calculated as the current rank modulo the number of GPUs per node.

    Example:
        >>> local_rank = get_local_rank()
        >>> print(f"Local rank of the current process: {local_rank}")
    """
    return int(get_rank() % get_gpus_per_node()) if get_world_size() > 1 else 0


def get_free_port():
    """
    Get a free port on the local machine.

    Returns:
        int: A free port number that can be used for communication.
    """
    sock = socket.socket()
    sock.bind(
        ("127.0.0.1", 0)
    )  # Bind to an available port on the loopback interface
    port = sock.getsockname()[1]
    sock.close()
    return port


def query_environment() -> dict[str, int]:
    """Query environment variables for info about distributed setup

    Returns:
        dict[str, int]: A dictionary containing the distributed setup information.
            Includes keys like 'world_size', 'rank', and 'local_rank'.
            If the environment variables are not set, it falls back to using
            `get_world_size()`, `get_rank()`, and `get_local_rank()` functions.

    Example:
        >>> env_info = query_environment()
        >>> print(env_info)
        {'world_size': 4, 'rank': 0, 'local_rank': 0}
    """
    ws = os.environ.get("WORLD_SIZE", None)
    r = os.environ.get("RANK", None)
    lr = os.environ.get("LOCAL_RANK", None)
    if ws is not None and r is not None and lr is not None:
        return {
            "world_size": int(ws),
            "rank": int(r),
            "local_rank": int(lr),
            # 'machine': machine,
        }
    return {
        "world_size": int(get_world_size()),
        "rank": int(get_rank()),
        "local_rank": int(get_local_rank()),
    }


def broadcast(
    obj: Any,
    root: int = 0,
) -> Any:
    MPI.COMM_WORLD.bcast(obj, root=root)


def setup_torch_DDP(
    port: str = "2345",
    timeout: int | str | timedelta = 3600,
    backend: Optional[str] = None,
) -> dict[str, int]:
    """
    Setup PyTorch Distributed Data Parallel (DDP) environment.
    Args:
        port (str, optional): The port to use for distributed communication.
            Defaults to "2345".
        timeout (int | str | timedelta, optional): Timeout for the process group initialization.
            Defaults to 3600 seconds.
        backend (str, optional): The backend to use for distributed training.
            Defaults to None, which will use the default backend based on the device.

    Returns:
        dict[str, int]: A dictionary containing the distributed setup information.
            Includes keys like 'world_size', 'rank', and 'local_rank'.
    """
    if not isinstance(timeout, timedelta):
        timeout = timedelta(seconds=int(timeout))
    os_rank = os.environ.get("RANK", None)
    os_world_size = os.environ.get("WORLD_SIZE", None)
    os_local_rank = os.environ.get("LOCAL_RANK", None)
    world_size = int(get_world_size())
    rank = int(get_rank())
    local_rank = int(get_local_rank())
    # ensure there is no funny business going on
    if os_rank and int(os_rank) != int(rank):
        logger.warning(f"Mismatch between {os_rank=} and {rank=}")
    if os_world_size and int(os_world_size) != int(world_size):
        logger.warning(f"Mismatch between {os_world_size=} and {world_size=}")
    if os_local_rank and int(os_local_rank) != int(local_rank):
        logger.warning(f"Mismatch between {os_local_rank=} and {local_rank=}")
    # now, set these variables explicitly in the process' environment
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # get `hostname` ONLY from rank 0
    master_addr = socket.gethostname() if rank == 0 else None
    if (mn := ezpz.get_machine().lower()) in {
        "aurora",
        "polaris",
        "sirius",
    }:
        master_addr = f"{master_addr}.hsn.cm.{mn}.alcf.anl.gov"
    elif mn == "sophia":
        master_addr = f"{master_addr}.lab.alcf.anl.gov"
    # check if we have specified a 'MASTER_PORT' explicitly, if so, use this
    free_port = str(get_free_port()) if rank == 0 else None
    eport = os.environ.get("MASTER_PORT", free_port)
    if eport is not None:
        _ = (
            logger.info(f"Caught MASTER_PORT={eport} from environment!")
            if rank == 0
            else None
        )
    else:
        eport = port
    # grab it from rank 0
    master_port = eport if rank == 0 else None
    # broadcast it to make sure everyones tapped in
    master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    # set it explicitly in each process' environment
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    # now, torch is ready for us
    if rank == 0:
        logger.info(
            "\n".join(
                [
                    "Using torch.distributed.init_process_group with",
                    f"- {master_addr=}",
                    f"- {master_port=}",
                    f"- {world_size=}",
                    f"- {rank=}",
                    f"- {local_rank=}",
                    f"- {timeout=}",
                    f"- {backend=}",
                ]
            )
        )
    init_process_group(
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        backend=backend,
    )
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


def setup_torch_distributed(
    framework: Optional[str] = None,
    backend: Optional[str] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
    tensor_parallel_backend: Optional[str] = None,
    pipeline_parallel_backend: Optional[str] = None,
    context_parallel_backend: Optional[str] = None,
    data_parallel_backend: Optional[str] = None,
    port: Optional[str | int] = None,
    timeout: Optional[str | int] = None,
) -> dict[str, int]:
    """
    Setup distributed environment for PyTorch.

    Args:
        framework (str, optional): The framework to use for distributed training.
            Defaults to None, which will use "ddp".
        backend (str, optional): The backend to use for distributed training.
            Defaults to None, which will use the default backend based on the device.
        tensor_parallel_size (int, optional): Size of tensor parallelism. Defaults to 1.
        pipeline_parallel_size (int, optional): Size of pipeline parallelism. Defaults to 1.
        context_parallel_size (int, optional): Size of context parallelism. Defaults to 1.
        tensor_parallel_backend (str, optional): Backend for tensor parallelism. Defaults to None.
        pipeline_parallel_backend (str, optional): Backend for pipeline parallelism. Defaults to None.
        context_parallel_backend (str, optional): Backend for context parallelism. Defaults to None.
        data_parallel_backend (str, optional): Backend for data parallelism. Defaults to None.
        port (str | int, optional): Port for distributed communication. Defaults to "1234".
        timeout (str | int, optional): Timeout for distributed initialization. Defaults to 3600 seconds.

    Returns:
        dict[str, int]: A dictionary containing the distributed setup information.
            Includes keys like 'world_size', 'rank', and 'local_rank'.

    Raises:
        AssertionError: If the framework is not one of the supported frameworks.
            Supported frameworks are "ddp", "ds", "deepspeed", "horovod", and "hvd".
        ValueError: If the backend is not one of the supported backends.
            Supported backends are "ddp", "ds", "deepspeed", "horovod", and "hvd".

    Example:
        >>> setup_torch_distributed(
        ...     framework="ddp",
        ...     backend="nccl",
        ...     tensor_parallel_size=2,
        ...     pipeline_parallel_size=1,
        ...     context_parallel_size=1,
        ...     port=1234,
        ...     timeout=3600,
        ... )
    """
    framework = "ddp" if framework is None else framework
    # if str(framework).lower() not in {"ddp", "ds", "deepspeed", "horovod", "hvd"}:
    assert str(framework).lower() in {
        "ddp",
        "ds",
        "deepspeed",
        "horovod",
        "hvd",
    }, (
        f"Invalid framework: {framework=}, expected one of "
        f"{'ddp', 'ds', 'deepspeed', 'horovod', 'hvd'}"
    )

    timeout = (
        3600
        if timeout is None
        else int(timeout)
        if isinstance(timeout, str)
        else timeout
    )
    port = (
        "1234"
        if port is None
        else str(port)
        if isinstance(port, int)
        else port
    )
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    fw = str(framework).lower()
    be = (
        str(get_torch_backend()).lower()
        if backend is None
        else str(backend).lower()
    )
    # be = str(framework).lower()
    # assert fw in {"ds", "deepspeed", "ddp", "horovod", "hvd"}, (
    #     f"Invalid backend: {framework=}, expected one of "
    #     f"{'ds', 'deepspeed', 'ddp', 'horovod', 'hvd'}"
    # )
    # assert be in BACKENDS['pytorch']
    if rank == 0:
        logger.info(
            " ".join(
                [
                    f"Using {fw=} with",
                    "torch_{device,backend}={",
                    f"{get_torch_device_type()}, {be}" + "}",
                ]
            )
        )
    if fw == "ddp":
        dsetup = setup_torch_DDP(port, timeout, backend=be)
        world_size = dsetup["world_size"]
        rank = dsetup["rank"]
        local_rank = dsetup["local_rank"]
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    elif fw in {"deepspeed", "ds"}:
        init_deepspeed(timeout=timeout)
        world_size = get_world_size()
        rank = get_rank()
        local_rank = get_local_rank()
    elif fw in {"horovod", "hvd"}:
        import horovod.torch as hvd  # type:ignore noqa

        _ = None if hvd.is_initialized() else hvd.init()  # type:ignore
        # hvd.init() if not hvd.is_initialized() else None
        rank = hvd.rank()  # type:ignore
        world_size = hvd.size()  # type:ignore
        local_rank = hvd.local_rank()  # type:ignore
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())  # type:ignore
    else:
        raise ValueError(f"Unable to parse backend: {be=}")

    if (
        tensor_parallel_size > 1
        or context_parallel_size > 1
        or pipeline_parallel_size > 1
    ):
        ezpz.tp.initialize_tensor_parallel(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            tensor_parallel_backend=tensor_parallel_backend,
            pipeline_parallel_backend=pipeline_parallel_backend,
            context_parallel_backend=context_parallel_backend,
            data_parallel_backend=data_parallel_backend,
            timeout=timedelta(seconds=timeout),
        )

    os.environ["world_size"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


def barrier(
    device: Optional[torch.device | int | str] = None,
    group: (
        torch.distributed.ProcessGroup | None  # type:ignore
    ) = torch.distributed.GroupMember.WORLD,  # type:ignore
    async_op: bool = False,
    device_ids: str | Iterable | None = None,
) -> torch.distributed.Work | None:  # type:ignore
    """
    Barrier for all processes in the group.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Args:
        device (torch.device | int | str, optional): The device to synchronize.
            If None, the default device will be used. Defaults to None.
        group (torch.distributed.ProcessGroup | None, optional): The process group to work on.
            If None, the default process group (WORLD) will be used.
            Defaults to torch.distributed.GroupMember.WORLD.
        async_op (bool, optional): If True, the barrier will be asynchronous.
        device_ids (str | Iterable | None, optional): The device IDs to synchronize.

    Returns:
        torch.distributed.Work | None: If async_op is True, returns a work handle.
            If async_op is False, returns None.
    """
    try:
        torch.distributed.barrier(
            group=group, async_op=async_op, device_ids=device_ids
        )
    except Exception:
        logger.warning(
            "Unable to use `torch.distributed.barrier` "
            "for this process group. "
            "Falling back to `mpi4py` barrier."
        )
        MPI.COMM_WORLD.barrier()


def setup_torch(
    framework: Optional[str] = None,
    backend: Optional[str] = None,
    port: Optional[str | int] = None,
    seed: Optional[int] = None,
    timeout: Optional[str | int] = None,
    verbose: Optional[bool] = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
    tensor_parallel_backend: Optional[str] = None,
    pipeline_parallel_backend: Optional[str] = None,
    context_parallel_backend: Optional[str] = None,
    data_parallel_backend: Optional[str] = None,
) -> int:
    """Setup torch.

    Args:
        backend (str, optional): Backend to use. Defaults to None.
        port (str | int, optional): Port to use. Defaults to None.
        seed (int, optional): Seed to use. Defaults to None.
        timeout (str | int, optional): Timeout to use. Defaults to None.
        verbose (bool, optional): Whether to print the info. Defaults to False.
        tensor_parallel_size (int, optional): Tensor parallel size. Defaults to 1.
        pipeline_parallel_size (int, optional): Pipeline parallel size. Defaults to 1.
        context_parallel_size (int, optional): Context parallel size. Defaults to 1.
        tensor_parallel_backend (str, optional): Tensor parallel backend. Defaults to None.
        pipeline_parallel_backend (str, optional): Pipeline parallel backend. Defaults to None.
        context_parallel_backend (str, optional): Context parallel backend. Defaults to None.
        data_parallel_backend (str, optional): Data parallel backend. Defaults to None.

    Returns:
        int: Rank of the process.
    """
    device = get_torch_device()
    # if ACCELERATOR_TYPE == 'NvidiaGPU' and device == 'cuda':
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #     torch.backends.cudnn.deterministic = True     # type:ignore
    #     torch.backends.cudnn.benchmark = True         # type:ignore
    #     torch.backends.cudnn.allow_tf32 = True        # type:ignore
    #     torch.backends.cuda.matmul.allow_tf32 = True  # type:ignore
    # torch.use_deterministic_algorithms(True)
    ws_from_env = os.environ.get("WORLD_SIZE", None)
    framework = "DDP" if framework is None else framework
    framework = framework.lower()
    backend = str(get_torch_backend()).lower()
    if ws_from_env is not None and ws_from_env == "1":
        logger.info(
            f"Running on a single {device}, not initializing torch.distributed!"
        )
        rank = 0
        world_size = 1
        local_rank = 0
        local_size = 1
        num_nodes = 1
    else:
        dsetup = setup_torch_distributed(
            framework=framework,
            backend=backend,
            port=port,
            timeout=timeout,
            tensor_parallel_size=int(tensor_parallel_size),
            pipeline_parallel_size=int(pipeline_parallel_size),
            context_parallel_size=int(context_parallel_size),
            tensor_parallel_backend=tensor_parallel_backend,
            pipeline_parallel_backend=pipeline_parallel_backend,
            context_parallel_backend=context_parallel_backend,
            data_parallel_backend=data_parallel_backend,
        )
        rank = dsetup["rank"]
        world_size = dsetup["world_size"]
        local_rank = dsetup["local_rank"]
        try:
            local_size = get_gpus_per_node()
        except Exception:
            local_size = 1

        try:
            num_nodes = get_num_nodes()
        except Exception:
            num_nodes = 1
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NUM_NODES"] = str(num_nodes)
    os.environ["LOCAL_SIZE"] = str(local_size)
    os.environ["WORLD_SIZE"] = str(world_size)
    # nthreads = os.environ.get('OMP_NUM_THREADS', None)
    # if ACCELERATOR_TYPE == "IntelGPU" and device == "xpu":
    if torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        # try:
        #     import intel_extension_for_pytorch as ipex  # type:ignore[missingTypeStubs]
        # except Exception:
        #     ipex = None
        # if ipex is not None:
        #     logger.debug(f"Using ipex from: {ipex.__file__}")
        #
        # try:
        #     import oneccl_bindings_for_pytorch as oneccl_bpt  # type:ignore[missingTypeStubs]
        # except Exception:
        #     oneccl_bpt = None
        # if oneccl_bpt is not None:
        #     logger.debug(f"Using oneccl_bindings from: {oneccl_bpt.__file__}")
        #
        #     # logger.warning(f'Using {get_torch_device()}:{get_local_rank()}')
        #     # os.environ['CCL_LOCAL_RANK'] = str(local_rank)
        #     # os.environ['CCL_LOCAL_SIZE'] = str(local_size)
    if seed is not None:
        seed_everything(seed * (rank + 1) * (local_rank + 1))
    if rank == 0:
        if backend in {"ds", "deepspeed", "dspeed"}:
            from ezpz.configs import git_ds_info

            git_ds_info()
        _ = get_dist_info(verbose=verbose)
        if verbose:
            _ = print_dist_setup()
    if world_size > 1:
        barrier()

    if rank == 0:
        logger.info(
            f"Using {device=} with {backend=} "
            f"+ '{get_torch_backend()}' "
            "for distributed training."
        )
    lrank = len(str(world_size - 1))
    # nz = lrank - len(str(rank))
    hn = socket.gethostname()
    psizes = [f"['{hn}']" + f"[{rank:>{lrank}}/{world_size - 1:<{lrank}}] "]
    if (
        tensor_parallel_size > 1
        or context_parallel_size > 1
        or pipeline_parallel_size > 1
    ):
        import ezpz.tp

        tpsize = ezpz.tp.get_tensor_parallel_world_size()
        cpsize = ezpz.tp.get_context_parallel_world_size()
        ppsize = ezpz.tp.get_pipeline_parallel_world_size()
        dpsize = ezpz.tp.get_data_parallel_world_size()
        if cpsize > 1 or ppsize > 1 or tpsize > 1:
            if cpsize > 1:
                lcp = len(str(cpsize - 1))
                cprank = ezpz.tp.get_context_parallel_rank()
                # cpranks = ezpz.tp.get_context_parallel_ranks()
                psizes.append(f"[cp:{cprank:>{lcp}}/{cpsize - 1:<{lcp}}]")
                barrier(group=ezpz.tp.get_context_parallel_group())
            if ppsize > 1:
                pprank = ezpz.tp.get_pipeline_parallel_rank()
                # ppranks = ezpz.tp.get_pipeline_parallel_ranks()
                lpp = len(str(ppsize - 1))
                psizes.append(f"[pp:{pprank:>{lpp}}/{ppsize - 1:<{lpp}}]")
                barrier(group=ezpz.tp.get_pipeline_parallel_group())
            if tpsize > 1:
                ltp = len(str(tpsize - 1))
                tprank = ezpz.tp.get_tensor_parallel_rank()
                # tpranks = ezpz.tp.get_tensor_parallel_ranks()
                psizes.append(f"[tp:{tprank:>{ltp}}/{tpsize - 1:<{ltp}}]")
                barrier(group=ezpz.tp.get_tensor_parallel_group())
            if dpsize > 1:
                ldp = len(str(dpsize - 1))
                dprank = ezpz.tp.get_data_parallel_rank()
                # dpranks = ezpz.tp.get_data_parallel_ranks()
                psizes.append(f"[dp:{dprank:>{ldp}}/{dpsize - 1:<{ldp}}]")
                barrier(group=ezpz.tp.get_data_parallel_group())
    logger.info("".join(psizes))
    barrier()
    return rank


def cleanup() -> None:
    """
    Cleanup the distributed environment.
    This function destroys the process group if it is initialized.

    Example:
        >>> cleanup()
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def setup_tensorflow(
    precision: Optional[str] = None,
    ngpus: Optional[int] = None,
) -> int:
    """Initialize TensorFlow + Horovod for Distributed Training"""
    try:
        import tensorflow as tf  # type:ignore noqa

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import horovod.tensorflow as hvd  # type:ignore noqa
    except Exception:
        logger.warning(
            "Unable to import `tensorflow` or `horovod.tensorflow`. "
            "Install with `pip install tensorflow horovod`"
        )
        raise

    _ = None if hvd.is_initialized() else hvd.init()
    # hvd.init() if not hvd.is_initialized() else None
    if precision in [
        "fp16",
        "float16",
        "half",
        "16",
        "mixed_float16",
        # 'mixed_bfloat16'
    ]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")  # pyright:ignore
    TF_FLOAT = tf.keras.backend.floatx()  # pyright:ignore
    eager_mode = os.environ.get("TF_EAGER", None)
    if eager_mode is not None:
        logger.info("Detected `TF_EAGER` from env. Running eagerly.")
        tf.config.run_functions_eagerly(True)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    cpus = tf.config.experimental.list_physical_devices("CPU")
    if gpus:
        try:
            # Currently memory growth needs to be the same across GPUs
            if ngpus is not None:
                gpus = gpus[-ngpus:]

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()],
                "GPU",
            )
            _ = tf.config.experimental.list_logical_devices("GPU")  # pyright:ignore
        except RuntimeError as e:
            logger.info(e)
    elif cpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            logical_cpus = tf.config.experimental.list_logical_devices("CPU")
            logger.info(
                f"{len(cpus)}, Physical CPUs and "
                f"{len(logical_cpus)} Logical CPUs"
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info(e)
    RANK = hvd.rank()
    WORLD_SIZE = hvd.size()
    LOCAL_RANK = hvd.local_rank()
    # LOCAL_SIZE = hvd.local_size()
    os.environ["RANK"] = str(RANK)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["LOCAL_RANK"] = str(LOCAL_RANK)
    # logger.info(f'RANK: {RANK} / {WORLD_SIZE-1}')
    if RANK == 0:
        logger.info(f"Using {TF_FLOAT} precision")
    return RANK


def include_file(f: PathLike):
    """
    Check if a file should be included based on its extension.

    Args:
        f (PathLike): The file path to check.

    Returns:
        bool: True if the file should be included, False otherwise.
    """
    fpath = Path(f)
    return fpath.suffix in {
        ".py",
        ".yaml",
        ".sh",
        ".md",
        ".qmd",
        ".yml",
        ".toml",
    }


def get_machine(hostname: Optional[str] = None) -> str:
    """Get the machine name from the hostname.

    Args:
        hostname (str, optional): The hostname to check. Defaults to None.

    Returns:
        str: The machine name.

    Example:
        >>> get_machine("frontier")
        "Frontier"
    """

    if hostname is None:
        try:
            hostname = socket.gethostbyaddr(socket.gethostname())[0]
        except Exception:
            try:
                hostname = socket.gethostname()
            except Exception:
                logger.warning("Unable to determine hostname!")
                hostname = "unknown"
    if hostname.startswith("frontier"):
        return "Frontier"
    if hostname.startswith("sophia"):
        return "Sophia"
    if hostname.startswith("theta"):
        return "ThetaGPU"
    if hostname.startswith("x1"):
        return "SunSpot"
    if hostname.startswith("x3"):
        if (pbs_host := os.environ.get("PBS_O_HOST", None)) is not None:
            if str(pbs_host).startswith("sirius"):
                return "Sirius"
            return "Polaris"
        return "Polaris"
    if hostname.startswith("x4"):
        return "Aurora"
    if hostname.startswith("login"):
        return "Perlmutter"
    if hostname.startswith("nid"):
        return "Perlmutter"
    return f"{hostname}"


def setup_wandb(
    project_name: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[dict | DictConfig] = None,
    start_method: str = "thread",
    outdir: Optional[str | Path | os.PathLike] = None,
    init_timeout: int = 300,
):
    """Setup wandb for logging.

    Args:
        project_name (str, optional): The name of the project. Defaults to None.
        entity (str, optional): The entity name. Defaults to None.
        config (dict | DictConfig, optional): The configuration dictionary. Defaults to None.
        start_method (str, optional): The start method for wandb. Defaults to "thread".
        outdir (str | Path | os.PathLike, optional): The output directory. Defaults to None.
        init_timeout (int, optional): The timeout for wandb initialization. Defaults to 300.

    Example:
        >>> setup_wandb(project_name="my_project", entity="my_entity")
    """
    # try:
    #     import wandb
    # except Exception:
    #     wandb = None
    # try:
    #     import wandb
    #
    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
    WANDB_MODE = os.environ.get("WANDB_MODE", "").lower()

    # except Exception:
    #     wandb = None
    #     WANDB_DISABLED = True

    if WANDB_DISABLED or WANDB_MODE == "disabled":
        logger.warning(
            f"Logging with W&B is disabled!, caught: {WANDB_DISABLED=}"
        )
        return None

    try:
        import wandb
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(
            "Unable to import `wandb`. Install with `pip install wandb`"
        )
        raise e

    outdir = (
        Path(os.getcwd()).as_posix()
        if outdir is None
        else Path(outdir).as_posix()
    )
    rank = get_rank()
    project_name = (
        project_name
        if project_name is not None
        else os.environ.get(
            "WB_PROJECT",
            os.environ.get(
                "WANDB_PROJECT",
                os.environ.get("WB_PROJECT_NAME", None),
            ),
        )
    )
    if project_name is None:
        import sys

        frame = sys._getframe().f_back
        assert frame is not None
        calling_module = frame.f_code.co_filename
        fp = Path(calling_module)
        project_name = f"{fp.parent.stem}.{fp.stem}"

    logger.info(f"Setting up wandb from {rank=}")
    logger.info(f"Using WB_PROJECT={project_name}")
    tensorboard_dir = (
        os.environ.get("TENSORBOARD_DIR", None)
        if config is None
        else config.get("tensorboard_dir", None)
    )
    if tensorboard_dir is not None:
        logger.info(f"Patching tensorboard from {tensorboard_dir}")
        try:
            wandb.tensorboard.patch(root_logdir=tensorboard_dir)  # type:ignore
        except Exception as exc:
            logger.exception(exc)
    # wbrun_id = wandb.util.generate_id()
    now = datetime.datetime.now()
    dstr = now.strftime("%Y-%m-%d-%H%M%S")
    run = wandb.init(
        entity=entity,
        # resume='allow',
        dir=outdir,
        sync_tensorboard=(tensorboard_dir is not None),  # True,
        project=(project_name if project_name is not None else None),
        # dir=(tensorboard_dir if tensorboard_dir is not None else None),
        settings=wandb.Settings(
            start_method=start_method, init_timeout=init_timeout
        ),
    )
    assert run is not None and run is wandb.run
    # run.log_code(HERE.as_posix(), include_fn=include_file)
    logger.info(f"wandb.run=[{run.name}]({run.url})")
    if (
        wandb is not None
        and wandb.run is not None
        and "DIST_INFO" not in wandb.run.config
    ):
        wandb.run.config.update({"DIST_INFO": get_dist_info()})
    torch_version = torch.__version__
    torch_file = torch.__file__
    run.config.update(
        {
            "created_at": dstr,
            "day": ezpz.get_timestamp("%d"),
            "month": ezpz.get_timestamp("%m"),
            "outdir": os.getcwd(),
            "torch_version": torch_version,
            "torch_file": torch_file,
            "world_size": get_world_size(),
            "year": ezpz.get_timestamp("%Y"),
            "ezpz_version": ezpz.__version__,
            "ezpz_file": ezpz.__file__,
        }
    )
    if config is not None:
        if isinstance(config, DictConfig):
            cfg = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            run.config.update({"config": cfg})
        else:
            run.config.update({"config": config})
    env = {
        k: v
        for k, v in dict(os.environ).items()
        if not k.startswith("_ModuleTable")
    }
    _ = env.pop("LS_COLORS", None)
    _ = env.pop("PS1", None)
    run.config.update({"env": env})
    machine = get_machine()
    logger.info(f"Running on {machine=}")
    run.config.update({"machine": machine})
    model_size = os.environ.get("MODEL_SIZE", None)
    if model_size is not None:
        run.config.update({"MODEL_SIZE": model_size})
    return wandb.run


def run_bash_command(cmd: str) -> Any:
    """
    Run a bash command and return the output.
    Args:
        cmd (str): The command to run.

    Returns:
        Any: The output of the command.
    """
    import subprocess
    import shlex

    process = subprocess.Popen(
        shlex.split(cmd, posix=True),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = process.communicate()
    if process.returncode != 0:
        raise Exception(
            f"Command failed with return code {process.returncode}.\n"
            f"stdout: {output.decode().strip()}\n"
            f"stderr: {error.decode().strip()}"
        )
    if error:
        raise Exception(error.decode())
    else:
        return output


def get_nodes_from_hostfile(
    hostfile: PathLike,
) -> list[str]:
    """Get the nodes from the hostfile.

    Args:
        hostfile (PathLike): The path to the hostfile.

    Returns:
        list[str]: A list of nodes from the hostfile.
    """
    # cobalt_nodefile = get_cobalt_nodefile()
    fpath = Path(hostfile)
    assert fpath.is_file()
    with fpath.open("r") as f:
        nodes = [i.rstrip("\n") for i in f.readlines()]
    return nodes


def get_node_index() -> int:
    """Get the index of the current node in the hostfile"""
    return get_rank() % get_num_nodes()


def write_localhost_to_hostfile(hostfile: PathLike):
    """Write 'localhost' to the hostfile"""
    if get_rank() == 0:
        logger.debug(
            f"Writing {(hostname := get_hostname())} "
            f"to {Path(hostfile).as_posix()}"
        )
        hostname = get_hostname()
        with Path(hostfile).open("w") as f:
            f.write(f"{hostname}")


def write_hostfile_from_list_of_hosts(
    hosts: list[str],
    hostfile: Optional[PathLike] = None,
    rank_zero_only: bool = True,
):
    """Write a list of hosts to the hostfile.

    Args:
        hosts (list[str]): A list of hostnames to write to the hostfile.
        hostfile (PathLike, optional): The path to the hostfile. Defaults to None.
        rank_zero_only (bool, optional): If True, only rank 0 will write the hostfile.
            Defaults to True.
    """
    hostfile = (
        Path(hostfile).as_posix()
        if hostfile is not None
        else Path(os.getcwd()).joinpath("hostfile").as_posix()
    )
    if (rank_zero_only and get_rank() == 0) or not rank_zero_only:
        logger.info(f"Writing to {hostfile}")
        with Path(hostfile).open("w") as f:
            for host in hosts:
                f.write(f"{host}\n")


def make_hostfile_from_slurm_env(outfile: Optional[PathLike] = None) -> Path:
    """Make a hostfile from the SLURM_NODELIST environment variable"""
    nodes = os.environ.get("SLURM_NODELIST", None)
    # if nodes is not None:
    assert nodes is not None
    # machine = get_machine()
    prefix, idxs = nodes.split("[")
    idxs = idxs.rstrip("]")
    idxs = "-".join(idxs.split(",")).split("-")
    nodelist = [f"{prefix}{i}" for i in idxs]
    # idxs = (
    #     nodes.split
    # )
    # idxs = (
    #     nodes.lstrip('frontier').replace('[', '').replace(']', '').split('-')
    # )
    # nodelist = [f'frontier{i}' for i in idxs]
    if outfile is None:
        outfile = Path(os.getcwd()).joinpath("hostfile")
    else:
        outfile = Path(outfile)
    with outfile.open("w") as f:
        for node in nodelist:
            f.write(f"{node}\n")
    return outfile


def get_hostfile_with_fallback(hostfile: Optional[PathLike] = None) -> Path:
    """Get the hostfile from the environment or create one if it doesn't exist"""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler()
    if scheduler.lower() == "unknown":
        logger.debug("Unknown scheduler")
        hostfile = Path(os.getcwd()).joinpath("hostfile")
    if scheduler.lower() == "slurm":
        hostfile = make_hostfile_from_slurm_env()
        assert Path(hostfile).is_file()
    if hostfile is None:
        hfp = os.environ.get(
            "PBS_NODEFILE",
            os.environ.get(
                "HOSTFILE",
                None,  # fallback_hostfile.as_posix()
            ),
        )
        if (
            hfp is None or not Path(hfp).is_file()
            # and scheduler == 'PBS'
        ):
            if scheduler == "PBS":
                # hfp = Path(get_pbs_nodefile_from_qstat())
                nodefile = ezpz.pbs.get_pbs_nodefile()
                assert nodefile is not None, (
                    "Unable to get PBS_NODEFILE from `qstat` or `ezpz.pbs`!"
                )
                hfp = Path(nodefile)
            else:
                # create makeshift hostfile containing 'localhost'
                hfp = Path(os.getcwd()).joinpath("hostfile")
                hfp.touch(exist_ok=True)
                write_localhost_to_hostfile(hfp)
    else:
        hfp = Path(hostfile)
    assert hfp is not None and Path(hfp).is_file()
    assert Path(hfp).is_file()
    hostfile = Path(hfp).as_posix()
    # if hfp is not None:
    # hostfile, hosts = get_hosts_from_hostfile(hostfile)
    # hosts = [h.split('.')[0] for h in hosts]
    # if scheduler == 'PBS':
    #     os.environ['PBS_NODEFILE'] = hostfile  # hfp.as_posix()
    hfname = f"{scheduler.upper()}_NODEFILE"
    if hfname not in os.environ:
        os.environ |= {hfname: hostfile}
    # os.environ[f'{scheduler.upper()}_NODEFILE'] = hostfile
    return Path(hfp)


def get_num_nodes(hostfile: Optional[PathLike] = None) -> int:
    """Get the number of nodes from the hostfile"""
    num_nodes = os.environ.get("SLURM_NNODES", None)
    if num_nodes is not None:
        return int(num_nodes)
    hfp = get_hostfile_with_fallback(hostfile)
    hosts = [h.split(".")[0] for h in get_nodes_from_hostfile(hfp)]
    return len(hosts)


def get_cpus_per_node() -> int:
    """Get the number of CPUs per node"""
    from sh import getconf as sh_getconf  # type:ignore noqa

    return int(sh_getconf("_NPROCESSORS_ONLN").rstrip("\n"))


def get_gpus_per_node() -> int:
    """Get the number of GPUs per node"""
    # return torch.cuda.device_count() if torch.cuda.is_available() else (
    #     (
    #         ipex.xpu.device_count() if ipex is not None else (
    #             get_cpus_per_node()
    #         )
    #     )
    # )
    # if _assert:
    #     raise RuntimeError(
    #         'No {X, G}pus found; but _assert specified. Returning !!'
    #     )
    # logger.warning('No {x,g}-pus found, returning' + f'{cpus_per_node}')
    ngpu_per_host = os.environ.get("NGPU_PER_HOST", None)
    if ngpu_per_host is not None:
        return int(ngpu_per_host)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if torch.xpu.is_available():
        return torch.xpu.device_count()
    if ipex is not None and torch.xpu.is_available():
        return ipex.xpu.device_count()
    if torch.backends.mps.is_available():
        # XXX: Maybe we're running MPI with multiple MPS devices?
        return get_world_size_in_use()
    return 0


def check(
    framework: str = "pytorch",
    backend: str = "deepspeed",
    port: int | str = "5432",
):
    """Check if the framework is installed and working"""
    from ezpz.configs import FRAMEWORKS

    if framework in FRAMEWORKS["pytorch"]:
        _ = setup_torch(
            backend=backend,
            port=str(port),
        )
    elif framework in FRAMEWORKS["tensorflow"]:
        _ = setup_tensorflow()
    else:
        raise ValueError(f"Unable to parse framework: {framework}")
