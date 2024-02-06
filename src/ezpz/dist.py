"""
dist.py

Contains methods for initializing distributed communication.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import logging.config
import os
from pathlib import Path
import time
from functools import wraps
from typing import Any, Callable, Optional
import socket
import json

import torch
import torch.distributed as dist
from datetime import timedelta

from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

from ezpz.configs import (
    FRAMEWORKS,
    HERE,
    git_ds_info,
    PathLike,
    get_scheduler,
    get_logging_config,
)

try:
    import wandb
except Exception:
    wandb = None


try:
    import intel_extension_for_pytorch as ipex        # type:ignore
except Exception:
    ipex = None

try:
    import oneccl_bindings_for_pytorch as oneccl_bpt  # type:ignore
except Exception:
    oneccl_bpt = None


os.environ['COLORTERM'] = 'truecolor'
log_config = logging.config.dictConfig(get_logging_config())
log = logging.getLogger(__name__)
log.setLevel('INFO')
# log = get_logger(__name__, level='INFO')
logging.getLogger('sh').setLevel('WARNING')


ACCELERATOR_TYPE = "IntelGPU" if ipex is not None else (
    "NvidiaGPU" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0
    ) else (
        "MPS" if torch.backends.mps.is_available()    # type:ignore
        else "CPU"
    )
)


def seed_everything(seed: int):
    import torch
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
                    tstr = [f"`{func.__name__}`"]
                    if len(args) > 0:
                        tstr.append(f"({args}")
                    # _ = tstr.append(f"({args}") if len(args) > 0 else None
                    _ = (
                        tstr.append(f", {kwargs})")
                        if len(kwargs) > 0 else (
                            tstr.append(")") if len(args) > 0
                            else ""
                        )
                    )
                    _ = tstr.append(f' took: {dt=:.4f}s')
                    log.info("".join(tstr))
                if wandb is not None and wandb.run is not None:
                    # log.info(
                    #     f'Logging timeit/{func.__name__}/{dt=:.4f} to W&B'
                    # )
                    wandb.run.log(
                        {f'timeit/{func.__name__}': dt},
                        commit=False
                    )
            return result
        return wrapper
    return decorator


def timeit(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        log.info(f'{func.__name__}({args}, {kwargs}) took: {dt=:.4f}s')
        if wandb is not None and wandb.run is not None:
            wandb.run.log({f'timeit/{func.__name__}': dt})
        return result
    return wrapper


def get_hosts_from_hostfile(
        hostfile: Optional[PathLike] = None
) -> tuple[str, list[str]]:
    hostname = get_hostname()
    hostfile = os.environ.get(
        'HOSTFILE',
        os.environ.get(
            'PBS_NODEFILE',
            os.environ.get(
                'COBALT_NODEFILE',
                None,
            )
        )
    )
    hostfile = '' if hostfile is None else hostfile
    hosts = []
    if hostfile is not None and Path(hostfile).is_file():
        if get_rank() == 0:
            log.debug(f'Reading hosts from {hostfile}')
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
        from sh import hostname as sh_hostname  # type:ignore noqa
        hostname = sh_hostname()
        # if get_rank() == 0:
        #     log.debug('Unable to determine hostname with `socket`.')
        #     log.debug(f'hostname from`sh`: {hostname}')
        #     # log.exception(exc)
    return hostname.rstrip('\n')


def _get_dist_info(
        hostfile: Optional[PathLike] = None,
        framework: Optional[str] = None,
        max_hosts: int = 1000,  # truncate in logs
) -> dict:
    hostfile = (
        Path(get_hostfile_with_fallback(hostfile)).as_posix()
    )
    # hostfile = (
    #     Path(get_hostfile_with_fallback(hostfile)).as_posix()
    #     if hostfile is None else hostfile
    # )
    assert hostfile is not None and Path(hostfile).is_file(), (
        f'{hostfile=} not None and {Path(hostfile).is_file()=}'
    )
    hosts = get_nodes_from_hostfile(Path(hostfile).as_posix())
    if len(hosts) > max_hosts:
        log.warning(f'{len(hosts)=} > {max_hosts=} in `dist.get_dist_info')
        log.warning(f'Truncating `hosts: [addr1, addr2, ...] at {max_hosts}')
    hosts = (
        [h.split('.')[0] for h in hosts] if len(hosts) < max_hosts
        else (
            [h.split('.')[0] for h in hosts[:max_hosts]].extend(
                [
                    f'[(...) truncated ({len(hosts)} > {max_hosts})]'
                ]
            )
        )
    )
    dist_info = {
        'DEVICE': get_torch_device(),
        'DEVICE_ID': f'{get_torch_device()}:{get_local_rank()}',
        'DISTRIBUTED_BACKEND': get_torch_backend(),
        'GPUS_PER_NODE': get_gpus_per_node(),
        'HOSTS': f'{hosts}',
        'HOSTFILE': hostfile,
        'HOSTNAME': get_hostname(),
        'LOCAL_RANK': get_local_rank(),
        'MACHINE': get_machine(),
        'NUM_NODES': get_num_nodes(),
        'NGPUS': get_world_size_total(),
        'NODE_ID': get_node_index(),
        'RANK': get_rank(),
        'SCHEDULER': get_scheduler(),
        # 'WORLD_SIZE': get_world_size(),
        'WORLD_SIZE_TOTAL': get_world_size_total(),
        'WORLD_SIZE_IN_USE': get_world_size_in_use(),
    }
    if framework is not None:
        dist_info |= {'FRAMEWORK': framework}
    return dist_info


def get_dist_info(
        framework: Optional[str] = None,
        verbose: Optional[bool] = None,
        max_hosts: int = 1000,
        hostfile: Optional[PathLike] = None,
) -> dict[str, str | int | list]:
    dist_info = _get_dist_info(
        hostfile=hostfile,
        framework=framework,
        max_hosts=max_hosts
    )
    if verbose:
        log.info(
            f'DistInfo={json.dumps(dist_info, indent=4, sort_keys=True)}'
        )
    if (
            wandb is not None
            and wandb.run is not None
            and 'DIST_INFO' not in wandb.run.config
    ):
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
    #     log.critical(f'{num_nodes=} vs. {num_nodes_from_hostfile=} ??')
    node = get_node_index()
    device = None
    # if framework.lower() in {'pt', 'torch', 'pytorch'}:
    device = get_torch_device()
    rank_len = len(str(rank))
    ws_len = len(str(wsa))
    lr_len = len(str(local_rank))
    gpn_len = len(str(gpus_per_node))
    node_len = len(str(node))
    num_nodes_len = len(str(num_nodes))
    dist_list = [
        f'[{device=}]',
        f'[{rank=:>{rank_len}}/{(wsa-1):<{ws_len}}]',
        f'[{local_rank=:>{lr_len}}/{gpus_per_node-1:<{gpn_len}}]',
        f'[{node=:>{node_len}}/{(num_nodes-1):<{num_nodes_len}}]',
    ]
    if framework is not None:
        dist_list.append(f'[{framework=}]')
    dist_str = ''.join(dist_list)
    log.info(f'{dist_str}')
    if rank == 0:
        if wsa > 1000:
            log.warning(
                f'WORLD_SIZE={wsa} > 1000, only printing on RANK={rank}'
            )
        log.warning(f'Using [{wsa} / {wst}] available "{device}" devices !!')
        if num_nodes_from_hostfile != num_nodes:
            log.critical(
                f'num_nodes_from_hostfile = [{num_nodes_from_hostfile=}]'
                f'vs.'
                f'[{wsa=} // {gpus_per_node=}] = {num_nodes}'
                r'¯\_(ツ)_/¯ ??'
            )
    return dist_str


def setup(
        framework: str = 'pytorch',
        backend: str = 'DDP',
        port: str = '5432',
        seed: Optional[int] = None,
        precision: Optional[str] = None,
        ngpus: Optional[int] = None
):
    return (
        setup_tensorflow(precision=precision, ngpus=ngpus)
        if framework in {'tensorflow', 'tf', 't'}
        else setup_torch(backend=backend, port=port, seed=seed)
    )


def init_deepspeed():
    try:
        import deepspeed  # type:ignore noqa
        deepspeed.init_distributed()
    except Exception as exc:
        log.warning('Unable to `import deepspeed`. Exiting!')
        log.exception(exc)
        raise exc


def get_torch_device() -> str:
    if (tdevice := os.environ.get('TORCH_DEVICE')) is not None:
        assert tdevice is not None
        return tdevice
    return 'xpu' if ipex is not None else (
        'cuda' if torch.cuda.is_available() else (
            'mps' if (
                (
                    torch.backends.mps.is_available()
                    and torch.get_default_dtype() != torch.float64
                )
            ) else 'cpu'
        )
    )


def get_torch_backend() -> str:
    backend = (
        'nccl' if torch.cuda.is_available() else (
            'ccl' if (ipex is not None and oneccl_bpt is not None)
            else 'gloo'
        )
    )
    if backend is None:
        log.critical(f'Using "gloo" backend on {get_torch_device()}')
        backend = 'gloo'
    return backend


def init_process_group(
        rank: int | str,
        world_size: int | str,
) -> None:
    backend = get_torch_backend()
    # log.warning(f'Using {backend=}')
    delta = timedelta(
        # days=50,
        seconds=300,
        # microseconds=10,
        # milliseconds=29000,
        # minutes=5,
        # hours=8,
        # weeks=2
    )
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=delta,
            rank=int(rank),
            world_size=int(world_size),
            init_method='env://',
        )


def run_ddp(fn: Callable, world_size: int) -> None:
    import torch.multiprocessing as mp
    mp.spawn(  # type:ignore
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
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
        log.warning(
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


def setup_torch_DDP(port: str = '2345') -> dict[str, int]:
    rank = os.environ.get('RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    local_rank = os.environ.get('LOCAL_RANK', None)
    world_size = int(get_world_size())
    rank = int(get_rank())
    local_rank = int(get_local_rank())
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    master_addr = (
        socket.gethostname() if rank == 0 else None
    )
    eport = os.environ.get("MASTER_PORT", None)
    if eport is not None:
        log.info(f'Caught MASTER_PORT: {eport=} from environment!')
    else:
        eport = port
    master_port = (
        eport if rank == 0 else None
    )
    master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    # if (eport := os.environ.get('MASTER_PORT', None)) is None:
    #     os.environ['MASTER_PORT'] = port
    # else:
    #     os.environ['MASTER_PORT'] = eport
    #     if rank == 0:
    #         log.info(f'Caught MASTER_PORT:{eport} from environment!')
    init_process_group(
        rank=rank,
        world_size=world_size,
        # backend=get_torch_backend(),
    )
    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch_distributed(
        backend: str,
        port: str = '2345',
) -> dict[str, int]:
    """Returns {'world_size': int, 'rank': int, 'local_rank': int}"""
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    be = backend.lower()
    # assert be in BACKENDS['pytorch']
    if be == 'ddp':
        dsetup = setup_torch_DDP(port)
        world_size = dsetup['world_size']
        rank = dsetup['rank']
        local_rank = dsetup['local_rank']
        # if rank == 0:
        #     import pudb; pudb.set_trace()
        # if torch.cuda.is_available():
        #     torch.cuda.set_device(local_rank)
    elif be in {'deepspeed', 'ds'}:
        init_deepspeed()
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
    os.environ['world_size'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch(
        backend: str = 'DDP',
        port: str = '2345',
        seed: Optional[int] = None,
) -> int:
    """Returns RANK"""
    import torch
    # import torch.distributed as tdist
    device = get_torch_device()
    # if torch.cuda.is_available() and device == 'cuda':
    if ACCELERATOR_TYPE == 'NvidiaGPU' and device == 'cuda':
        # from rich import log.info
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True     # type:ignore
        torch.backends.cudnn.benchmark = True         # type:ignore
        torch.backends.cudnn.allow_tf32 = True        # type:ignore
        torch.backends.cuda.matmul.allow_tf32 = True  # type:ignore
    # try:
    #     import intel_extension_for_pytorch as ipex
    # except (ImportError, ModuleNotFoundError):
    torch.use_deterministic_algorithms(True)
    dsetup = setup_torch_distributed(backend=backend, port=port)
    rank = dsetup['rank']
    world_size = dsetup['world_size']
    local_rank = dsetup['local_rank']
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    nthreads = os.environ.get('OMP_NUM_THREADS', None)
    if nthreads is not None:
        torch.set_num_threads(int(nthreads))
    # if torch.cuda.is_available() and device == 'cuda':
    # if ACCELERATOR_TYPE == 'NvidiaGPU' and device == 'cuda':
    #     torch.cuda.set_device(local_rank)
    # #     # torch.cuda.set_device('cuda')
    # # elif device == 'xpu':
    # else:
    #     log.warning(
    #         f'No Intel or NVIDIA GPUs found, using: {ACCELERATOR_TYPE=}'
    #     )
    if ACCELERATOR_TYPE == 'IntelGPU' and device == 'xpu':
        # log.warning(f'Using {get_torch_device()}:{get_local_rank()}')
        torch.xpu.set_device(local_rank)  # type:ignore
    if seed is not None:
        seed_everything(seed * (rank + 1) * (local_rank + 1))
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        _ = get_dist_info(verbose=True)
        if backend in {'ds', 'deepspeed', 'dspeed'}:
            git_ds_info()
        if oneccl_bpt is not None:
            log.info(f'Using oneccl_bindings from: {oneccl_bpt.__file__}')
        if ipex is not None:
            log.info(f'Using ipex from: {ipex.__file__}')
        log.info(
            f"[{rank}/{world_size}] Using {device=} with {backend=} "
            f"+ '{get_torch_backend()}' "
            "for distributed training."
        )
    _ = print_dist_setup()
    return rank


def cleanup() -> None:
    import torch.distributed as tdist
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
        tf.keras.mixed_precision.set_global_policy(
            'mixed_float16'
        )
        # tf.keras.backend.set_floatx('float16')
        # mixed_precision.set_global_policy('mixed_float16')
    # else:
    #     tf.keras.backend.set_floatx(precision)
    TF_FLOAT = tf.keras.backend.floatx()
    eager_mode = os.environ.get('TF_EAGER', None)
    if eager_mode is not None:
        log.info('Detected `TF_EAGER` from env. Running eagerly.')
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
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            log.info(e)
    elif cpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            logical_cpus = tf.config.experimental.list_logical_devices('CPU')
            log.info(
                f'{len(cpus)}, Physical CPUs and '
                f'{len(logical_cpus)} Logical CPUs'
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            log.info(e)
    RANK = hvd.rank()
    WORLD_SIZE = hvd.size()
    LOCAL_RANK = hvd.local_rank()
    # LOCAL_SIZE = hvd.local_size()
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)
    # log.info(f'RANK: {RANK} / {WORLD_SIZE-1}')
    if RANK == 0:
        log.info(f"Using {TF_FLOAT} precision")
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
                log.warning('Unable to determine hostname!')
                hostname = 'unknown'
    if hostname.startswith('theta'):
        return 'ThetaGPU'
    if hostname.startswith('x1'):
        return 'SunSpot'
    if hostname.startswith('x3'):
        return 'Polaris'
    if hostname.startswith('x4'):
        return 'Aurora'
    if hostname.startswith('login'):
        return 'NERSC'
    if hostname.startswith('nid'):
        return 'Perlmutter'
    return f'{hostname}'


def setup_wandb(
        project_name: Optional[str] = None,
        config: Optional[dict | DictConfig] = None,
        start_method: str = 'thread',
        init_timeout: int = 300,
):
    import wandb
    rank = get_rank()
    project_name = (
        project_name if project_name is not None
        else os.environ.get(
            'WB_PROJECT',
            os.environ.get(
                'WANDB_PROJECT',
                os.environ.get(
                    "WB_PROJECT_NAME",
                    'ezpz'
                )
            )
        )
    )
    log.info(f"Setting up wandb from rank: {rank}")
    log.info(f"Using: WB PROJECT: {project_name}")
    tensorboard_dir = None
    if config is None:
        tensorboard_dir = os.environ.get('TENSORBOARD_DIR', None)
    else:
        tensorboard_dir = (
            config.get(
                'tensorboard_dir',
                None,  # os.getcwd()
            )
        )
    if tensorboard_dir is not None:
        log.info(f'Patching tensorboard from {tensorboard_dir}')
        wandb.tensorboard.patch(root_logdir=tensorboard_dir)
    # wbrun_id = wandb.util.generate_id()
    now = datetime.datetime.now()
    dstr = now.strftime('%Y-%m-%d-%H%M%S')
    run = wandb.init(
        # resume='allow',
        dir=os.getcwd(),
        sync_tensorboard=(tensorboard_dir is not None),  # True,
        project=(project_name if project_name is not None else None),
        # dir=(tensorboard_dir if tensorboard_dir is not None else None),
        settings=wandb.Settings(
            start_method=start_method,
            init_timeout=init_timeout
        )
    )
    assert run is not None and run is wandb.run
    run.log_code(HERE.as_posix(), include_fn=include_file)
    log.info(f"W&B RUN: [{run.name}]({run.url})")
    run.config.update(
        {
            f'dist_info/{k}': v for k, v in get_dist_info().items()
        }
    )
    run.config.update({'created_at': dstr})
    run.config.update({'world_size': get_world_size()})
    run.config.update({'outdir': os.getcwd()})
    # wandb.run.config.update({'hostname': rank})
    if config is not None:
        if isinstance(config, DictConfig):
            cfg = OmegaConf.to_container(
                config,
                resolve=True,
                throw_on_missing=True
            )
            run.config.update({'config': cfg})
        else:
            run.config.update({'config': config})
    env = {
        k: v for k, v in dict(os.environ).items()
        if not k.startswith('_ModuleTable')
    }
    _ = env.pop('LS_COLORS', None)
    _ = env.pop('PS1', None)
    run.config.update({'env': env})
    machine = get_machine()
    log.info(f'Running on {machine=}')
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
        log.warning('COBALT_NODEFILE not in `env`!')
        log.info('Attempting to deduce from `/var/tmp/cobalt-running-job`...')
        cobalt_info = inspect_cobalt_running_job()
        log.info(f"Found COBALT info: {cobalt_info}")
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
    return get_rank() % get_num_nodes()


def write_localhost_to_hostfile(hostfile: PathLike):
    # hostfile = (
    #     Path(os.getcwd()).joinpath('hostfile') if hostfile is None
    #     else Path(hostfile)
    # )
    if get_rank() == 0:
        # log.info(
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
    if (
            (rank_zero_only and get_rank() == 0)
            or not rank_zero_only
    ):
        log.info(f'Writing to {hostfile}')
        with Path(hostfile).open('w') as f:
            for host in hosts:
                f.write(f'{host}\n')


def get_hostfile_with_fallback(
        hostfile: Optional[PathLike] = None
) -> Path:
    scheduler = get_scheduler()
    if scheduler.lower() == 'unknown':
        log.debug('Unknown scheduler')
        hostfile = Path(os.getcwd()).joinpath('hostfile"')
    if hostfile is None:
        hfp = (
            os.environ.get(
                'PBS_NODEFILE',
                os.environ.get(
                    'HOSTFILE',
                    None,  # fallback_hostfile.as_posix()
                )
            )
        )
        if (
                (hfp is None or not Path(hfp).is_file())
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
    hfp = get_hostfile_with_fallback(hostfile)
    hosts = [h.split('.')[0] for h in get_nodes_from_hostfile(hfp)]
    return len(hosts)


def get_cpus_per_node() -> int:
    from sh import getconf as sh_getconf  # type:ignore noqa
    return int(sh_getconf("_NPROCESSORS_ONLN").rstrip('\n'))


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
    # log.warning('No {x,g}-pus found, returning' + f'{cpus_per_node}')
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if ipex is not None:
        return ipex.xpu.device_count()
    return get_cpus_per_node()


def get_pbs_launch_cmd(
        ngpus: Optional[int] = None,
        nhosts: Optional[int] = None,
        ngpu_per_host: Optional[int] = None,
        hostfile: Optional[PathLike] = None,
) -> str:
    if hostfile is None:
        hostfile = get_hostfile_with_fallback(hostfile)
    ngpus = get_world_size_total() if ngpus is None else ngpus
    nhosts = get_num_nodes() if nhosts is None else nhosts
    ngpu_per_host = (
        get_gpus_per_node() if ngpu_per_host is None else ngpu_per_host
    )
    hfp = Path(
        get_hostfile_with_fallback(hostfile) if hostfile is None else hostfile
    )
    if ngpus != (ngpu_per_host * nhosts):
        log.critical(
            'Mismatch in `ngpu_per_host * nhosts` and `ngpus` '
            f'{ngpus=} vs. {ngpu_per_host=} * {nhosts=}'
        )
    return ' '.join([
        'mpiexec',
        '--verbose',
        '--envall',
        f'-n {ngpus}',
        f'-ppn {ngpu_per_host}',
        f'--hostfile {hfp.as_posix()}'
    ])


def get_pbs_jobid_from_qstat() -> int:
    from ezpz.configs import get_scheduler
    assert get_scheduler() == 'PBS'
    try:
        from sh import qstat as sh_qstat
    except Exception as exc:
        raise exc
    qstat_out = sh_qstat("-u", os.environ.get("USER")).split('\n')[2:-1]
    return int(qstat_out[-1].split('.')[0])
    # except Exception as exc:
    #     log.error('Unable to determine PBS_JOBID from `qstat` command...')
    #     raise exc


def get_pbs_nodefile_from_qstat() -> Path:
    assert get_scheduler() == 'PBS'
    pbs_jobid = get_pbs_jobid_from_qstat()
    matches = [
        i for i in Path('/var/spool/pbs/aux/').rglob(f'*{pbs_jobid}*')
        if i.is_file()
    ]
    assert len(matches) == 1
    # if len(matches) > 1:
    #     raise RuntimeError(
    #         'More than one candidate PBS_NODEFILE found? '
    #         f'{matches=}'
    #     )
    return matches[0]


def get_pbs_launch_info(
        hostfile: Optional[PathLike] = None
) -> dict:
    assert get_scheduler() == 'PBS'
    if hostfile is None:
        hostfile = get_pbs_nodefile_from_qstat()
    assert hostfile is not None and Path(hostfile).is_file()
    hfp = Path(hostfile)
    hosts = get_nodes_from_hostfile(hfp)
    hosts = [h.split('.')[0] for h in hosts]
    nhosts = len(hosts)
    ngpu_per_host = get_gpus_per_node()
    # ngpus = nhosts * ngpu_per_host
    ngpus = get_world_size(total=True)
    world_size_total = get_world_size_total()
    if ngpus != world_size_total:
        log.critical('Disagreement in total world size!!')
        log.critical(
            f'{get_world_size(total=True)=}'
            f' vs. {get_world_size_total()=}'
        )
        # if ngpus != (ngpu_per_host * nhosts):
        log.critical(
            'Mismatch in `ngpu_per_host * nhosts` and `ngpus` '
            f'{ngpus=} vs. {ngpu_per_host=} * {nhosts=}'
        )
    # launch_cmd = ' '.join([
    #     'mpiexec',
    #     '--verbose',
    #     '--envall',
    #     f'-n {ngpus}',
    #     f'-ppn {ngpu_per_host}',
    #     f'--hostfile {hfp.as_posix()}'
    # ])
    launch_cmd = get_pbs_launch_cmd(hostfile=hostfile)
    return {
        'HOSTFILE': hfp.as_posix(),
        'HOSTS': (
            f'[{", ".join(hosts)}]' if nhosts < 1000
            else '[truncated (>1000 nodes)]'
        ),
        'NHOSTS': f'{nhosts}',
        'NGPU_PER_HOST': f'{ngpu_per_host}',
        'NGPUS': f'{ngpus}',
        'MACHINE': get_machine(),
        'DEVICE': get_torch_device(),
        'BACKEND': get_torch_backend(),
        'LAUNCH_CMD': launch_cmd,
    }


def get_pbs_env(verbose: bool = False) -> dict[str, str]:
    assert get_scheduler() == 'PBS'
    pbsenv = {
        k: v for k, v in dict(os.environ).items() if 'PBS' in k
    }
    hostfile = pbsenv.get('PBS_NODEFILE')
    if hostfile is None:
        hostfile = get_pbs_nodefile_from_qstat()
    if hostfile is not None and (hfp := Path(hostfile)).is_file():
        launch_info = {
            f'{k.upper()}': f'{v}' for k, v in get_pbs_launch_info(hfp).items()
        }
        pbsenv |= launch_info
        # os.environ |= launch_info
    # dist_info = get_dist_info(framework='pytorch', verbose=verbose)
    # dist_info.pop('')
    # pbsenv |= {k: f'{v}' for k, v in dist_info.items()}
    os.environ |= pbsenv
    if verbose and get_rank() == 0:
        log.debug(f'pbsenv={json.dumps(pbsenv, indent=4, sort_keys=True)}')
    return pbsenv


def get_cobalt_resources() -> dict:
    cobalt_info = inspect_cobalt_running_job()
    # cobalt_nodefile = get_cobalt_nodefile()
    nodes = get_nodes_from_hostfile(Path(cobalt_info["COBALT_NODEFILE"]))
    gpus_per_node = get_gpus_per_node()
    cobalt_info |= {
        'nodes': nodes,
        'num_nodes': len(nodes),
        'gpus_per_node': gpus_per_node,
        'num_gpus': len(nodes) * gpus_per_node,
        'machine': 'ThetaGPU',
    }
    return cobalt_info


def build_mpiexec_thetagpu():
    jobenv = get_cobalt_resources()
    return [
        "mpirun",
        f"-n {jobenv['num_nodes']}",
        f"-npernode {jobenv['gpus_per_node']}",
        f"--hostfile {jobenv['COBALT_NODEFILE']}",
        "-x PATH",
        "-x LD_LIBRARY_PATH",
        "-x http_proxy",
        "-x https_proxy",
    ]


def run_mpiexec(cmd: str):
    import subprocess
    mpiexec = ' '.join(build_mpiexec_thetagpu())
    log.info(f'Executing: {mpiexec} {cmd}')
    return subprocess.Popen(f"{mpiexec} {cmd}", shell=True)


def mpi_test_framework_backend(
    framework: str = 'pytorch',
    backend: str = 'DDP',
):
    import sys
    python3 = sys.executable
    py_cmd = f'{python3} -m ezpz.check {framework} {backend}'
    run_mpiexec(py_cmd)


def check(
        framework: str = 'pytorch',
        backend: str = 'deepspeed',
        port: int | str = '5432',
):
    if framework in FRAMEWORKS['pytorch']:
        _ = setup_torch(
            backend=backend,
            port=str(port),
        )
    elif framework in FRAMEWORKS['tensorflow']:
        _ = setup_tensorflow()
    else:
        raise ValueError(f"Unable to parse framework: {framework}")
