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
from typing import Any, Callable, Optional, Union

import ezpz.tp

from mpi4py import MPI

import torch
import torch.distributed as tdist
from datetime import timedelta

from omegaconf import DictConfig, OmegaConf

try:
    import wandb

    WANDB_DISABLED = os.environ.get('WANDB_DISABLED', False)
except Exception:
    wandb = None
    WANDB_DISABLED = True


try:
    import intel_extension_for_pytorch as ipex  # type:ignore[missingTypeStubs]
except Exception:
    ipex = None

try:
    import oneccl_bindings_for_pytorch as oneccl_bpt  # type:ignore[missingTypeStubs]
except Exception:
    oneccl_bpt = None

# from dataclasses import dataclass

# @dataclass
# class TorchDistributedInfo:
#     backend: str  # [DDP, deepspeed, horovod]
#     rank: int    # [0, ..., world_size - 1]
#     local_rank: int  # [0, ..., ]
#     world_size: int

if not os.environ.get(
    'DUMB', os.environ.get('NOCOLOR', os.environ.get('NO_COLOR', False))
):
    os.environ['COLORTERM'] = 'truecolor'

PathLike = Union[str, os.PathLike, Path]

LOG_LEVEL = str(os.environ.get('LOG_LEVEL', 'INFO')).upper()
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
logging.getLogger('sh').setLevel('WARNING')


ACCELERATOR_TYPE = (
    'IntelGPU'
    if ipex is not None
    else (
        'NvidiaGPU'
        if (torch.cuda.is_available() and torch.cuda.device_count() > 0)
        else ('MPS' if torch.backends.mps.is_available() else 'CPU')
    )
)


def seed_everything(seed: int):
    import torch
    import numpy as np
    import random

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def log_dict_as_bulleted_list(d: dict, name: Optional[str] = None):
    """Print dictionary as list"""
    tag = name if name is not None else d.__qualname__
    logger.info(
        '\n'.join(
            ['\n', f'[{tag}]:']
            + [f'  • {k}={v}' for k, v in d.items()]
            + ['\n']
        )
    )


def timeitlogit(rank: Optional[int] = None, verbose: bool = True):
    rank = get_rank() if rank is None else rank

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            assert isinstance(rank, int)
            result = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            if verbose:
                if rank == 0:
                    tstr = [f'`{func.__name__}`']
                    if len(args) > 0:
                        tstr.append(f'({args}')
                    # _ = tstr.append(f"({args}") if len(args) > 0 else None
                    _ = (
                        tstr.append(f', {kwargs})')
                        if len(kwargs) > 0
                        else (tstr.append(')') if len(args) > 0 else '')
                    )
                    _ = tstr.append(f' took: {dt=:.4f}s')
                    logger.info(''.join(tstr))
                # try:
                #     import wandb
                # except:
                #     wandb = None
                if wandb is not None and wandb.run is not None:
                    # logger.info(
                    #     f'Logging timeit/{func.__name__}/{dt=:.4f} to W&B'
                    # )
                    wandb.run.log({f'timeit/{func.__name__}': dt}, commit=False)
            return result

        return wrapper

    return decorator


def timeit(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.info(f'{func.__name__}({args}, {kwargs}) took: {dt=:.4f}s')
        if wandb is not None and wandb.run is not None:
            wandb.run.log({f'timeit/{func.__name__}': dt})
        return result

    return wrapper


def get_hosts_from_hostfile(
    hostfile: Optional[str | Path] = None,  # type:ignore[reportDeprecated]
) -> tuple[str, list[str]]:
    hostname = get_hostname()
    hostfile = os.environ.get(
        'HOSTFILE',
        os.environ.get(
            'PBS_NODEFILE',
            os.environ.get(
                'COBALT_NODEFILE',
                None,
            ),
        ),
    )
    # hostfile = '' if hostfile is None else hostfile
    hosts: list[str] = []
    assert hostfile is not None
    if Path(hostfile).is_file():
        if get_rank() == 0:
            logger.debug(f'Reading hosts from {hostfile}')
        hpath = Path(hostfile).resolve().absolute()
        with hpath.open('r') as f:
            hosts.extend([h.rstrip('\n') for h in f.readlines()])
    else:
        hosts.append(hostname)
    return Path(hostfile).as_posix(), hosts


def get_hostname() -> str:
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
    return hostname.rstrip('\n')


def _get_dist_info(
    hostfile: Optional[PathLike] = None,
    framework: Optional[str] = None,
    # max_hosts_to_print: Optional[int] = None,  # truncate in logs
) -> dict:
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
        dist_info |= {'FRAMEWORK': framework}
    dist_info |= {
        'DEVICE': get_torch_device(),
        'DEVICE_ID': f'{get_torch_device()}:{get_local_rank()}',
        'DISTRIBUTED_BACKEND': get_torch_backend(),
        'GPUS_PER_NODE': num_gpus_per_node,
        'HOSTS': f'{hosts}',
        'HOSTFILE': hfp.absolute().resolve().as_posix(),
        'HOSTNAME': get_hostname(),
        'LOCAL_RANK': get_local_rank(),
        'MACHINE': get_machine(),
        'NUM_NODES': num_nodes,
        'NGPUS': num_gpus,
        'NGPUS_AVAILABLE': get_world_size_total(),
        # 'NGPUS': get_world_size_total(),
        'NODE_ID': get_node_index(),
        'RANK': get_rank(),
        'SCHEDULER': (scheduler := get_scheduler()),
        # 'WORLD_SIZE': get_world_size(),
        'WORLD_SIZE_TOTAL': get_world_size_total(),
        'WORLD_SIZE_IN_USE': get_world_size_in_use(),
        'LAUNCH_CMD': (
            get_pbs_launch_cmd(hostfile=hostfile)
            if scheduler.lower() == 'pbs'
            else None
        ),
    }
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
    dist_info = _get_dist_info(
        hostfile=hostfile,
        framework=framework,
    )
    if verbose:
        import json

        # logger.info(
        #     '\n'.join(
        #         ['\n', "[dist_info]:"]
        #         + [f"  • {k}={v}" for k, v in dist_info.items()]
        #         + ['\n']
        #     )
        # )
        # log_dict_as_bulleted_list(dist_info, name='dist_info')
        logger.info(
            f'DistInfo={json.dumps(dist_info, indent=4, sort_keys=True)}'
        )
    if (
        wandb is not None
        and wandb.run is not None
        and 'DIST_INFO' not in wandb.run.config
    ):
        logger.info(
            f'Updating wandb.run: {wandb.run.name} config with "DIST_INFO"'
        )
        wandb.run.config.update({'DIST_INFO': dist_info})
    return dist_info


def print_dist_setup(
    framework: Optional[str] = None,
    hostfile: Optional[PathLike] = None,
) -> str:
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
        f'[{device=}]',
        f'[{rank=:>{rank_len}}/{(wsa - 1):<{ws_len}}]',
        f'[{local_rank=:>{lr_len}}/{gpus_per_node - 1:<{gpn_len}}]',
        f'[{node=:>{node_len}}/{(num_nodes - 1):<{num_nodes_len}}]',
    ]
    if framework is not None:
        dist_list.append(f'[{framework=}]')
    dist_str = ''.join(dist_list)
    logger.info(f'{dist_str}')
    if rank == 0:
        if wsa > 1000:
            logger.warning(
                f'WORLD_SIZE={wsa} > 1000, only printing on RANK={rank}'
            )
        logger.warning(f'Using [{wsa} / {wst}] available "{device}" devices !!')
        if num_nodes_from_hostfile != num_nodes:
            logger.critical(
                f'num_nodes_from_hostfile = [{num_nodes_from_hostfile=}]'
                f'vs.'
                f'[{wsa=} // {gpus_per_node=}] = {num_nodes}'
                r'¯\_(ツ)_/¯ ??'
            )
    return dist_str


def synchronize(device: torch.device | int | str = 'cuda'):
    return (
        torch.cuda.synchronize(device)
        if torch.cuda.is_available()
        else (
            torch.xpu.synchronize(device)
            if torch.xpu.is_available()
            else torch.mps.synchronize()
            if torch.backends.mps.is_available()
            else torch.cpu.synchronize(device)
        )
    )


def setup(
    framework: str = 'pytorch',
    backend: str = 'DDP',
    port: str = '5432',
    seed: Optional[int] = None,
    precision: Optional[str] = None,
    ngpus: Optional[int] = None,
):
    return (
        setup_tensorflow(precision=precision, ngpus=ngpus)
        if framework in {'tensorflow', 'tf', 't'}
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
    rank = get_rank() if rank is None else rank
    world_size = get_world_size() if world_size is None else world_size
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
        logger.warning('Unable to `import deepspeed`. Exiting!')
        logger.exception(exc)
        raise exc


def get_torch_device_type(device_type: Optional[str] = None) -> str:
    if device_type is not None:
        assert device_type in (
            'cpu',
            'mps',
            'xpu',
            'cuda',
        )
        logger.warning(
            ' '.join(
                [
                    f'device_type: {device_type} passed to',
                    'ezpz.dist.get_torch_device_type',
                ]
            )
        )
        return device_type
    if (tdevice := os.environ.get('TORCH_DEVICE')) is not None:
        if get_rank() == 0:
            logger.warning(
                f"Caught 'TORCH_DEVICE'={tdevice}' from environment!"
            )
        tdevice = tdevice.lower()
        assert tdevice is not None and tdevice in (
            'cpu',
            'mps',
            'xpu',
            'cuda',
        )
        return tdevice.lower()
    return (
        'xpu'
        if torch.xpu.is_available()
        else (
            'cuda'
            if torch.cuda.is_available()
            else (
                'mps'
                if (
                    torch.backends.mps.is_available()
                    and torch.get_default_dtype() != torch.float64
                )
                else 'cpu'
            )
        )
    )


def get_torch_device(
    *,
    device_type: Optional[str] = None,
    as_torch_device: Optional[bool] = None,
) -> str | torch.device:
    device_type = get_torch_device_type(device_type)
    return torch.device(device_type) if as_torch_device else device_type


# def get_torch_backend() -> str:
#     tdevice = get_torch_device_type()
#     if tdevice == 'cuda' and torch.cuda.is_available():
#         return 'nccl'
#     if tdevice == 'xpu' and torch.xpu.is_available():
#         return 'ccl'
#         # return 'mpi'
#     return 'gloo'
#     #
#     # backend = (
#     #     'nccl'
#     #     if torch.cuda.is_available()
#     #     else (
#     #         'ccl' if torch.xpu.is_available() else 'gloo'
#     #         # 'ccl' if (ipex is not None and oneccl_bpt is not None) else 'gloo'
#     #     )
#     # )
#     # if backend is None:
#     #     logger.critical(f'Using "gloo" backend on {get_torch_device()}')
#     #     backend = 'gloo'
#     # return backend


def get_torch_version_as_float():
    return float('.'.join(torch.__version__.split('.')[:2]))


def get_torch_backend_on_xpu() -> str:
    torch_version = get_torch_version_as_float()
    assert torch.xpu.is_available()
    if torch_version >= 2.5:
        return 'xccl'
    return 'ccl'


def get_torch_backend() -> str:
    backend_from_env = os.environ.get('TORCH_BACKEND', None)
    if backend_from_env is not None:
        logger.warning(
            f'Caught `TORCH_BACKEND`={backend_from_env} from environment!'
        )
        return backend_from_env
    return (
        'nccl'
        if torch.cuda.is_available()
        else (
            get_torch_backend_on_xpu()
            if (ipex is not None and oneccl_bpt is not None)
            else 'gloo'
        )
    )


def init_process_group(
    rank: int | str,
    world_size: int | str,
    timeout: str | int | timedelta,
) -> None:
    backend = get_torch_backend()
    if get_rank() == 0:
        logger.info(f'Using {get_torch_device_type()=} with {backend=}')
    if not isinstance(timeout, timedelta):
        timeout = timedelta(
            seconds=int(timeout),
        )
    if not tdist.is_initialized():
        tdist.init_process_group(
            backend=backend,
            timeout=timeout,
            rank=int(rank),
            world_size=int(world_size),
            init_method='env://',
        )


def run_ddp(fn: Callable, world_size: int) -> None:
    import torch.multiprocessing as mp

    mp.spawn(  # type:ignore
        fn, args=(world_size,), nprocs=world_size, join=True
    )


def get_rank() -> int:
    """Get current MPI rank"""
    return int(MPI.COMM_WORLD.Get_rank())


def get_world_size_in_use() -> int:
    """Get number of currently in use MPI ranks"""
    return int(MPI.COMM_WORLD.Get_size())


def get_world_size_total() -> int:
    """Calculate total AVAILABLE *PUs as:

    total = [num_hosts] * [num_*pu_per_host]
    """
    # nhosts = get_num_nodes()
    # ngpu_per_host = get_gpus_per_node()
    # return ngpu_per_host * nhosts
    return get_gpus_per_node() * get_num_nodes()


def get_world_size(
    total: Optional[bool] = None,
    in_use: Optional[bool] = None,
) -> int:
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
            'MPI not initialized !!'
            'Calculating (and using!! ??) '
            '[world_size]=[(num_nodes) x (num_*pus_per_node)]=[num_*pus_total]'
            f'[{world_size}]=[({num_nodes}) x ({gpus_per_node})]'
        )
    # if world_size == 1:
    #     gpus_per_node = get_gpus_per_node()
    #     num_nodes = get_num_nodes()
    #     world_size = num_nodes * gpus_per_node
    return world_size


def get_local_rank() -> int:
    """Return `get_rank() % get_gpus_per_node()`"""
    return int(get_rank() % get_gpus_per_node())


def query_environment() -> dict[str, int]:
    """Query environment variables for info about distributed setup"""
    ws = os.environ.get('WORLD_SIZE', None)
    r = os.environ.get('RANK', None)
    lr = os.environ.get('LOCAL_RANK', None)
    if ws is not None and r is not None and lr is not None:
        return {
            'world_size': int(ws),
            'rank': int(r),
            'local_rank': int(lr),
            # 'machine': machine,
        }
    return {
        'world_size': int(get_world_size()),
        'rank': int(get_rank()),
        'local_rank': int(get_local_rank()),
    }


def setup_torch_DDP(
    port: str = '2345', timeout: int | str | timedelta = 3600
) -> dict[str, int]:
    if not isinstance(timeout, timedelta):
        timeout = timedelta(seconds=int(timeout))
    os_rank = os.environ.get('RANK', None)
    os_world_size = os.environ.get('WORLD_SIZE', None)
    os_local_rank = os.environ.get('LOCAL_RANK', None)
    world_size = int(get_world_size())
    rank = int(get_rank())
    local_rank = int(get_local_rank())
    # ensure there is no funny business going on
    if os_rank and int(os_rank) != int(rank):
        logger.warning(f'Mismatch between {os_rank=} and {rank=}')
    if os_world_size and int(os_world_size) != int(world_size):
        logger.warning(f'Mismatch between {os_world_size=} and {world_size=}')
    if os_local_rank and int(os_local_rank) != int(local_rank):
        logger.warning(f'Mismatch between {os_local_rank=} and {local_rank=}')
    # now, set these variables explicitly in the process' environment
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    # get `hostname` ONLY from rank 0
    master_addr = socket.gethostname() if rank == 0 else None
    # check if we have specified a 'MASTER_PORT' explicitly, if so, use this
    eport = os.environ.get('MASTER_PORT', None)
    if eport is not None:
        _ = (
            logger.info(f'Caught MASTER_PORT={eport} from environment!')
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
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    # now, torch is ready for us
    init_process_group(
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch_distributed(
    backend: str,
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
    """Returns {'world_size': int, 'rank': int, 'local_rank': int}"""
    assert backend.upper() in {'DDP', 'DEEPSPEED', 'DS', 'HOROVOD', 'HVD'}
    timeout = (
        3600
        if timeout is None
        else int(timeout)
        if isinstance(timeout, str)
        else timeout
    )
    port = (
        '1234' if port is None else str(port) if isinstance(port, int) else port
    )
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    be = backend.lower()
    # assert be in BACKENDS['pytorch']
    if be == 'ddp':
        dsetup = setup_torch_DDP(port, timeout)
        world_size = dsetup['world_size']
        rank = dsetup['rank']
        local_rank = dsetup['local_rank']
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    elif be in {'deepspeed', 'ds'}:
        init_deepspeed(timeout=timeout)
        world_size = get_world_size()
        rank = get_rank()
        local_rank = get_local_rank()
    elif be in {'horovod', 'hvd'}:
        import horovod.torch as hvd  # type:ignore noqa

        _ = None if hvd.is_initialized() else hvd.init()
        # hvd.init() if not hvd.is_initialized() else None
        rank = hvd.rank()
        world_size = hvd.size()
        local_rank = hvd.local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
    else:
        raise ValueError(f'Unable to parse backend: {be=}')

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

    os.environ['world_size'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch(
    backend: str = 'DDP',
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

    If launched with
    """

    device = get_torch_device()
    # if ACCELERATOR_TYPE == 'NvidiaGPU' and device == 'cuda':
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #     torch.backends.cudnn.deterministic = True     # type:ignore
    #     torch.backends.cudnn.benchmark = True         # type:ignore
    #     torch.backends.cudnn.allow_tf32 = True        # type:ignore
    #     torch.backends.cuda.matmul.allow_tf32 = True  # type:ignore
    # torch.use_deterministic_algorithms(True)
    ws_from_env = os.environ.get('WORLD_SIZE', None)
    if ws_from_env is not None and ws_from_env == '1':
        logger.info(
            f'Running on a single {device}, not initializing torch.distributed!'
        )
        rank = 0
        world_size = 1
        local_rank = 0
        local_size = 1
        num_nodes = 1
    else:
        dsetup = setup_torch_distributed(
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
        rank = dsetup['rank']
        world_size = dsetup['world_size']
        local_rank = dsetup['local_rank']
        local_size = get_gpus_per_node()
        num_nodes = get_num_nodes()
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['NUM_NODES'] = str(num_nodes)
    os.environ['LOCAL_SIZE'] = str(local_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    # nthreads = os.environ.get('OMP_NUM_THREADS', None)
    if ACCELERATOR_TYPE == 'IntelGPU' and device == 'xpu':
        # logger.warning(f'Using {get_torch_device()}:{get_local_rank()}')
        # os.environ['CCL_LOCAL_RANK'] = str(local_rank)
        # os.environ['CCL_LOCAL_SIZE'] = str(local_size)
        torch.xpu.set_device(local_rank)  # type:ignore
    if seed is not None:
        seed_everything(seed * (rank + 1) * (local_rank + 1))
    if rank == 0:
        if backend in {'ds', 'deepspeed', 'dspeed'}:
            from ezpz.configs import git_ds_info

            git_ds_info()
        _ = get_dist_info(verbose=verbose)
        if verbose:
            _ = print_dist_setup()
    if oneccl_bpt is not None:
        logger.debug(f'Using oneccl_bindings from: {oneccl_bpt.__file__}')
    if ipex is not None:
        logger.debug(f'Using ipex from: {ipex.__file__}')
    # if world_size > 1:
    #     tdist.barrier()

    if rank == 0:
        logger.info(
            f'Using {device=} with {backend=} '
            f"+ '{get_torch_backend()}' "
            'for distributed training.'
        )
    lrank = len(str(world_size - 1))
    # nz = lrank - len(str(rank))
    hn = socket.gethostname()
    psizes = [f"['{hn}']" + f'[{rank:>{lrank}}/{world_size - 1:<{lrank}}] ']
    if (
        tensor_parallel_size > 1
        or context_parallel_size > 1
        or pipeline_parallel_size > 1
    ):
        import ezpz.tp

        tprank = ezpz.tp.get_tensor_parallel_rank()
        # tpranks = ezpz.tp.get_tensor_parallel_ranks()
        tpsize = ezpz.tp.get_tensor_parallel_world_size()

        dprank = ezpz.tp.get_data_parallel_rank()
        # dpranks = ezpz.tp.get_data_parallel_ranks()
        dpsize = ezpz.tp.get_data_parallel_world_size()

        pprank = ezpz.tp.get_pipeline_parallel_rank()
        # ppranks = ezpz.tp.get_pipeline_parallel_ranks()
        ppsize = ezpz.tp.get_pipeline_parallel_world_size()

        # cpranks = ezpz.tp.get_context_parallel_ranks()
        cprank = ezpz.tp.get_context_parallel_rank()
        cpsize = ezpz.tp.get_context_parallel_world_size()

        # if cpsize > 1 or ppsize > 1 or tpsize > 1:
        #     if cpsize > 1:
        #         lcp = len(str(cpsize - 1))
        #         psizes.append(f'[cp:{cprank:>{lcp}}/{cpsize - 1:<{lcp}}]')
        #         tdist.barrier(group=ezpz.tp.get_context_parallel_group())
        #     if ppsize > 1:
        #         lpp = len(str(ppsize - 1))
        #         psizes.append(f'[pp:{pprank:>{lpp}}/{ppsize - 1:<{lpp}}]')
        #         tdist.barrier(group=ezpz.tp.get_pipeline_parallel_group())
        #     if tpsize > 1:
        #         ltp = len(str(tpsize - 1))
        #         psizes.append(f'[tp:{tprank:>{ltp}}/{tpsize - 1:<{ltp}}]')
        #         tdist.barrier(group=ezpz.tp.get_tensor_parallel_group())
        #     if dpsize > 1:
        #         ldp = len(str(dpsize - 1))
        #         psizes.append(f'[dp:{dprank:>{ldp}}/{dpsize - 1:<{ldp}}]')
        #         tdist.barrier(group=ezpz.tp.get_data_parallel_group())
    # tdist.all_gather(psizes)
    logger.info(''.join(psizes))
    MPI.COMM_WORLD.Barrier()
    return rank


def cleanup() -> None:
    tdist.destroy_process_group()


def setup_tensorflow(
    precision: Optional[str] = None,
    ngpus: Optional[int] = None,
) -> int:
    """Initialize TensorFlow + Horovod for Distributed Training"""
    import tensorflow as tf  # type:ignore noqa

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import horovod.tensorflow as hvd  # type:ignore noqa

    _ = None if hvd.is_initialized() else hvd.init()
    # hvd.init() if not hvd.is_initialized() else None
    if precision in [
        'fp16',
        'float16',
        'half',
        '16',
        'mixed_float16',
        # 'mixed_bfloat16'
    ]:
        tf.keras.mixed_precision.set_global_policy(  # pyright:ignore
            'mixed_float16'
        )
    TF_FLOAT = tf.keras.backend.floatx()  # pyright:ignore
    eager_mode = os.environ.get('TF_EAGER', None)
    if eager_mode is not None:
        logger.info('Detected `TF_EAGER` from env. Running eagerly.')
        tf.config.run_functions_eagerly(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    if gpus:
        try:
            # Currently memory growth needs to be the same across GPUs
            if ngpus is not None:
                gpus = gpus[-ngpus:]

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()],
                'GPU',
            )
            _ = (  # pyright:ignore
                tf.config.experimental.list_logical_devices('GPU')
            )
        except RuntimeError as e:
            logger.info(e)
    elif cpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            logical_cpus = tf.config.experimental.list_logical_devices('CPU')
            logger.info(
                f'{len(cpus)}, Physical CPUs and '
                f'{len(logical_cpus)} Logical CPUs'
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info(e)
    RANK = hvd.rank()
    WORLD_SIZE = hvd.size()
    LOCAL_RANK = hvd.local_rank()
    # LOCAL_SIZE = hvd.local_size()
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)
    # logger.info(f'RANK: {RANK} / {WORLD_SIZE-1}')
    if RANK == 0:
        logger.info(f'Using {TF_FLOAT} precision')
    return RANK


def include_file(f: PathLike):
    fpath = Path(f)
    return fpath.suffix in {
        '.py',
        '.yaml',
        '.sh',
        '.md',
        '.qmd',
        '.yml',
        '.toml',
    }


def get_machine(hostname: Optional[str] = None) -> str:
    if hostname is None:
        try:
            hostname = socket.gethostbyaddr(socket.gethostname())[0]
        except Exception:
            try:
                hostname = socket.gethostname()
            except Exception:
                logger.warning('Unable to determine hostname!')
                hostname = 'unknown'
    if hostname.startswith('frontier'):
        return 'Frontier'
    if hostname.startswith('sophia'):
        return 'Sophia'
    if hostname.startswith('theta'):
        return 'ThetaGPU'
    if hostname.startswith('x1'):
        return 'SunSpot'
    if hostname.startswith('x3'):
        if (pbs_host := os.environ.get('PBS_O_HOST', None)) is not None:
            if str(pbs_host).startswith('sirius'):
                return 'Sirius'
            return 'Polaris'
        return 'Polaris'
    if hostname.startswith('x4'):
        return 'Aurora'
    if hostname.startswith('login'):
        return 'Perlmutter'
    if hostname.startswith('nid'):
        return 'Perlmutter'
    return f'{hostname}'


def setup_wandb(
    project_name: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[dict | DictConfig] = None,
    start_method: str = 'thread',
    outdir: Optional[str | Path | os.PathLike] = None,
    init_timeout: int = 300,
):
    if WANDB_DISABLED:
        logger.warning(
            f'Logging with W&B is disabled!, caught: {WANDB_DISABLED=}'
        )
        return None

    try:
        import wandb
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(
            'Unable to import `wandb`. Install with `pip install wandb`'
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
            'WB_PROJECT',
            os.environ.get(
                'WANDB_PROJECT',
                os.environ.get('WB_PROJECT_NAME', None),
            ),
        )
    )
    if project_name is None:
        import sys

        frame = sys._getframe().f_back
        assert frame is not None
        calling_module = frame.f_code.co_filename
        fp = Path(calling_module)
        project_name = f'{fp.parent.stem}.{fp.stem}'

    logger.info(f'Setting up wandb from {rank=}')
    logger.info(f'Using=WB PROJECT={project_name}')
    tensorboard_dir = (
        os.environ.get('TENSORBOARD_DIR', None)
        if config is None
        else config.get('tensorboard_dir', None)
    )
    if tensorboard_dir is not None:
        logger.info(f'Patching tensorboard from {tensorboard_dir}')
        wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    # wbrun_id = wandb.util.generate_id()
    now = datetime.datetime.now()
    dstr = now.strftime('%Y-%m-%d-%H%M%S')
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
    logger.info(f'W&B RUN=[{run.name}]({run.url})')
    # run.config.update(
    #     {
    #         f'dist_info/{k}': v for k, v in get_dist_info().items()
    #     }
    # )
    if (
        wandb is not None
        and wandb.run is not None
        and 'DIST_INFO' not in wandb.run.config
    ):
        wandb.run.config.update({'dist_info': get_dist_info()})
    torch_version = torch.__version__
    torch_file = torch.__file__
    run.config.update({})
    run.config.update(
        {
            'created_at': dstr,
            'day': ezpz.get_timestamp('%d'),
            'month': ezpz.get_timestamp('%m'),
            'outdir': os.getcwd(),
            'torch_version': torch_version,
            'torch_file': torch_file,
            'world_size': get_world_size(),
            'year': ezpz.get_timestamp('%Y'),
        }
    )
    if config is not None:
        if isinstance(config, DictConfig):
            cfg = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            run.config.update({'config': cfg})
        else:
            run.config.update({'config': config})
    env = {
        k: v
        for k, v in dict(os.environ).items()
        if not k.startswith('_ModuleTable')
    }
    _ = env.pop('LS_COLORS', None)
    _ = env.pop('PS1', None)
    run.config.update({'env': env})
    machine = get_machine()
    logger.info(f'Running on {machine=}')
    run.config.update({'machine': machine})
    model_size = os.environ.get('MODEL_SIZE', None)
    if model_size is not None:
        run.config.update({'MODEL_SIZE': model_size})
    return wandb.run


def run_bash_command(cmd: str) -> Any:
    import subprocess

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        raise Exception(error)
    else:
        return output


def inspect_cobalt_running_job() -> dict[str, str | PathLike]:
    running_job_file = Path('/var/tmp/cobalt-running-job')
    with running_job_file.open('r') as f:
        tmp = f.readlines()
        jobid, uname = tmp[0].rstrip('\n').split(':')
        cobalt_nodefile = Path(f'/var/tmp/cobalt.{jobid}')
        os.environ['COBALT_NODEFILE'] = cobalt_nodefile.as_posix()
    return {
        'user': uname,
        'jobid': jobid,
        'COBALT_NODEFILE': cobalt_nodefile,
    }


def get_cobalt_nodefile() -> Path:
    cobalt_nodefile = os.environ.get('COBALT_NODEFILE', None)
    if cobalt_nodefile is None:
        logger.warning('COBALT_NODEFILE not in `env`!')
        logger.info(
            'Attempting to deduce from `/var/tmp/cobalt-running-job`...'
        )
        cobalt_info = inspect_cobalt_running_job()
        logger.info(f'Found COBALT info: {cobalt_info}')
        cobalt_nodefile = cobalt_info['COBALT_NODEFILE']
    return Path(cobalt_nodefile)


def get_nodes_from_hostfile(
    hostfile: PathLike,
) -> list[str]:
    # cobalt_nodefile = get_cobalt_nodefile()
    fpath = Path(hostfile)
    assert fpath.is_file()
    with fpath.open('r') as f:
        nodes = [i.rstrip('\n') for i in f.readlines()]
    return nodes


def get_node_index() -> int:
    # rank = get_rank()
    # logger.info(f'{rank=}')
    # logger.info(f'{get_num_nodes()=}')
    # logger.info(f'{get_rank() % get_num_nodes()=}')
    return get_rank() % get_num_nodes()


def write_localhost_to_hostfile(hostfile: PathLike):
    # hostfile = (
    #     Path(os.getcwd()).joinpath('hostfile') if hostfile is None
    #     else Path(hostfile)
    # )
    if get_rank() == 0:
        # logger.info(
        #     f'Writing {(hostname := get_hostname())} '
        #     f'to {Path(hostfile).as_posix()}'
        # )
        hostname = get_hostname()
        with Path(hostfile).open('w') as f:
            f.write(f'{hostname}')


def write_hostfile_from_list_of_hosts(
    hosts: list[str],
    hostfile: Optional[PathLike] = None,
    rank_zero_only: bool = True,
):
    hostfile = (
        Path(hostfile).as_posix()
        if hostfile is not None
        else Path(os.getcwd()).joinpath('hostfile').as_posix()
    )
    if (rank_zero_only and get_rank() == 0) or not rank_zero_only:
        logger.info(f'Writing to {hostfile}')
        with Path(hostfile).open('w') as f:
            for host in hosts:
                f.write(f'{host}\n')


def make_hostfile_from_slurm_env(outfile: Optional[PathLike] = None) -> Path:
    nodes = os.environ.get('SLURM_NODELIST', None)
    # if nodes is not None:
    assert nodes is not None
    # machine = get_machine()
    prefix, idxs = nodes.split('[')
    idxs = idxs.rstrip(']')
    idxs = '-'.join(idxs.split(',')).split('-')
    nodelist = [f'{prefix}{i}' for i in idxs]
    # idxs = (
    #     nodes.split
    # )
    # idxs = (
    #     nodes.lstrip('frontier').replace('[', '').replace(']', '').split('-')
    # )
    # nodelist = [f'frontier{i}' for i in idxs]
    if outfile is None:
        outfile = Path(os.getcwd()).joinpath('hostfile')
    else:
        outfile = Path(outfile)
    with outfile.open('w') as f:
        for node in nodelist:
            f.write(f'{node}\n')
    return outfile


def get_hostfile_with_fallback(hostfile: Optional[PathLike] = None) -> Path:
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler()
    if scheduler.lower() == 'unknown':
        logger.debug('Unknown scheduler')
        hostfile = Path(os.getcwd()).joinpath('hostfile"')
    if scheduler.lower() == 'slurm':
        hostfile = make_hostfile_from_slurm_env()
        assert Path(hostfile).is_file()
    if hostfile is None:
        hfp = os.environ.get(
            'PBS_NODEFILE',
            os.environ.get(
                'HOSTFILE',
                None,  # fallback_hostfile.as_posix()
            ),
        )
        if (
            hfp is None or not Path(hfp).is_file()
            # and scheduler == 'PBS'
        ):
            if scheduler == 'PBS':
                hfp = Path(get_pbs_nodefile_from_qstat())
            else:
                # create makeshift hostfile containing 'localhost'
                hfp = Path(os.getcwd()).joinpath('hostfile')
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
    hfname = f'{scheduler.upper()}_NODEFILE'
    if hfname not in os.environ:
        os.environ |= {hfname: hostfile}
    # os.environ[f'{scheduler.upper()}_NODEFILE'] = hostfile
    return Path(hfp)


def get_num_nodes(hostfile: Optional[PathLike] = None) -> int:
    num_nodes = os.environ.get('SLURM_NNODES', None)
    if num_nodes is not None:
        return int(num_nodes)
    hfp = get_hostfile_with_fallback(hostfile)
    hosts = [h.split('.')[0] for h in get_nodes_from_hostfile(hfp)]
    return len(hosts)


def get_cpus_per_node() -> int:
    from sh import getconf as sh_getconf  # type:ignore noqa

    return int(sh_getconf('_NPROCESSORS_ONLN').rstrip('\n'))


def get_gpus_per_node() -> int:
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
    ngpu_per_host = os.environ.get('NGPU_PER_HOST', None)
    if ngpu_per_host is not None:
        return int(ngpu_per_host)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if torch.xpu.is_available():
        return torch.xpu.device_count()
    if ipex is not None:
        return ipex.xpu.device_count()  # type:ignore
    return get_cpus_per_node()


def get_pbs_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[PathLike] = None,
) -> str:
    nhosts = get_num_nodes(hostfile=hostfile) if nhosts is None else nhosts
    ngpu_per_host = (
        get_gpus_per_node() if ngpu_per_host is None else ngpu_per_host
    )
    ngpus_available = get_world_size_total() if ngpus is None else ngpus
    ngpus_in_use = nhosts * ngpu_per_host
    hfp = Path(
        get_hostfile_with_fallback(hostfile) if hostfile is None else hostfile
    )
    if ngpus_available != (ngpus_in_use):
        logger.warning(
            'Mismatch in `ngpus_in_use` and `ngpus_available` '
            f'{ngpus_in_use=} vs. {ngpus_available=}'
        )
    return ' '.join(
        [
            'mpiexec',
            '--verbose',
            '--envall',
            # f'-n {ngpus}',
            f'-n {ngpus_in_use}',
            f'-ppn {ngpu_per_host}',
            f'--hostfile {hfp.as_posix()}',
            '--cpu-bind depth',
            '-d 16',
        ]
    )


def get_running_jobs_from_qstat() -> list[int]:
    try:
        from sh import qstat as shqstat  # type: ignore
    except Exception as e:
        raise e
    return [
        int(i.split('.')[0])
        for i in shqstat('-u', os.environ.get('USER')).split('\n')[2:-1]
        if ' R ' in i
    ]


def get_pbs_jobid_from_qstat() -> int:
    from ezpz.configs import get_scheduler

    assert get_scheduler() == 'PBS'
    try:
        from sh import qstat as sh_qstat  # pyright:ignore
    except Exception as exc:
        raise exc
    qstat_out = sh_qstat('-u', os.environ.get('USER')).split('\n')[2:-1]
    return int(qstat_out[-1].split('.')[0])
    # except Exception as exc:
    #     logger.error('Unable to determine PBS_JOBID from `qstat` command...')
    #     raise exc


def get_pbs_nodefile_from_qstat() -> Path:
    from ezpz.configs import get_scheduler

    assert get_scheduler() == 'PBS'
    nodefile = os.environ.get('PBS_NODEFILE', None)
    if nodefile is not None and (nf := Path(nodefile)).is_file():
        return nf
    pbs_jobid = get_pbs_jobid_from_qstat()
    matches = [
        i
        for i in Path('/var/spool/pbs/aux/').rglob(f'*{pbs_jobid}*')
        if i.is_file()
    ]
    assert len(matches) == 1
    return matches[0]


def get_pbs_launch_info(
    hostfile: Optional[str | Path] = None,  # type:ignore[reportDeprecated]
) -> dict[str, str]:
    from ezpz.configs import get_scheduler

    assert get_scheduler() == 'PBS'
    if hostfile is None:
        hostfile = os.environ.get('PBS_NODEFILE', get_pbs_nodefile_from_qstat())
    assert hostfile is not None
    hfp = Path(hostfile)
    # hostfile = os.environ.get("PBS_NODEFILE", None)
    # if hostfile is None:
    #     hostfile = (
    #             get_pbs_nodefile_from_qstat() if hostfile is None else
    #             Path(hostfile)
    #     )
    # assert hostfile is not None
    # hf = Path(hostfile)
    # assert hostfile is not None and hf.is_file()
    # hfp = Path(hostfile)
    hosts = get_nodes_from_hostfile(hfp)
    hosts = [h.split('.')[0] for h in hosts]
    nhosts = len(hosts)
    ngpu_per_host = get_gpus_per_node()
    # ngpus = nhosts * ngpu_per_host
    ngpus_available = get_world_size(total=True)
    ngpus = nhosts * ngpu_per_host
    world_size_total = get_world_size_total()
    # if ngpus != world_size_total:
    #     logger.warning('Disagreement in total world size!!')
    #     logger.warning(' '.join([
    #         f'{get_world_size(total=True)=}',
    #         f' vs. {get_world_size_total()=}'
    #     ]))
    #     logger.warning(' '.join([
    #         'Mismatch in: ',
    #         f'{ngpus=} vs. {ngpu_per_host=} * {nhosts=}'
    #     ]))
    launch_cmd = get_pbs_launch_cmd(hostfile=hostfile)
    return {
        'HOSTFILE': hfp.as_posix(),
        'HOSTS': (
            f'[{", ".join(hosts)}]'
            if nhosts < 1000
            else '[truncated (>1000 nodes)]'
        ),
        'NHOSTS': f'{nhosts}',
        'NGPU_PER_HOST': f'{ngpu_per_host}',
        'NGPUS': f'{ngpus}',
        'NGPUS_AVAILABLE': f'{ngpus_available}',
        'MACHINE': get_machine(),
        'DEVICE': get_torch_device_type(),
        'BACKEND': get_torch_backend(),
        'LAUNCH_CMD': launch_cmd,
        'world_size_total': f'{world_size_total}',
    }


def get_pbs_env(
    hostfile: Optional[Union[str, Path]] = None,
    verbose: Optional[bool] = None,
) -> dict[str, str]:
    from ezpz.configs import get_scheduler

    assert get_scheduler() == 'PBS'
    pbsenv = {k: v for k, v in dict(os.environ).items() if 'PBS' in k}
    if hostfile is None:
        hostfile = pbsenv.get('PBS_NODEFILE', get_pbs_nodefile_from_qstat())
    if (hfp := Path(hostfile)).is_file():
        pbsenv |= {
            f'{k.upper()}': f'{v}' for k, v in get_pbs_launch_info(hfp).items()
        }
        pbsenv |= {'LAUNCH_CMD': get_pbs_launch_cmd(hostfile=hostfile)}
    os.environ |= pbsenv
    if verbose and get_rank() == 0:
        # logger.debug(f'pbsenv={json.dumps(pbsenv, indent=4, sort_keys=True)}')
        log_dict_as_bulleted_list(pbsenv, name='pbsenv')
    return pbsenv


# def get_cobalt_resources() -> dict:
#     cobalt_info = inspect_cobalt_running_job()
#     # cobalt_nodefile = get_cobalt_nodefile()
#     nodes = get_nodes_from_hostfile(Path(cobalt_info["COBALT_NODEFILE"]))
#     gpus_per_node = get_gpus_per_node()
#     cobalt_info |= {
#         'nodes': nodes,
#         'num_nodes': len(nodes),
#         'gpus_per_node': gpus_per_node,
#         'num_gpus': len(nodes) * gpus_per_node,
#         'machine': 'ThetaGPU',
#     }
#     return cobalt_info


# def build_mpiexec_thetagpu():
#     jobenv = get_cobalt_resources()
#     return [
#         "mpirun",
#         f"-n {jobenv['num_nodes']}",
#         f"-npernode {jobenv['gpus_per_node']}",
#         f"--hostfile {jobenv['COBALT_NODEFILE']}",
#         "-x PATH",
#         "-x LD_LIBRARY_PATH",
#         "-x http_proxy",
#         "-x https_proxy",
#     ]


# def run_mpiexec(cmd: str):
#     import subprocess
#     mpiexec = ' '.join(build_mpiexec_thetagpu())
#     logger.info(f'Executing: {mpiexec} {cmd}')
#     return subprocess.Popen(f"{mpiexec} {cmd}", shell=True)


# def mpi_test_framework_backend(
#     framework: str = 'pytorch',
#     backend: str = 'DDP',
# ):
#     import sys
#     python3 = sys.executable
#     py_cmd = f'{python3} -m ezpz.check {framework} {backend}'
#     run_mpiexec(py_cmd)


def check(
    framework: str = 'pytorch',
    backend: str = 'deepspeed',
    port: int | str = '5432',
):
    from ezpz.configs import FRAMEWORKS

    if framework in FRAMEWORKS['pytorch']:
        _ = setup_torch(
            backend=backend,
            port=str(port),
        )
    elif framework in FRAMEWORKS['tensorflow']:
        _ = setup_tensorflow()
    else:
        raise ValueError(f'Unable to parse framework: {framework}')
