"""
dist.py

Contains methods for initializing distributed communication.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path
import time
from functools import wraps
from typing import Any, Callable, Optional
import socket
import json
from enrich.console import get_console
from rich import print_json
from rich.logging import RichHandler
from rich.style import Style
from rich.text import Text

import torch
import torch.distributed as dist
from datetime import timedelta

from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

from ezpz.configs import BACKENDS, FRAMEWORKS, HERE, git_ds_info

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None

ipex = None
try:
    import oneccl_bindings_for_pytorch  # type:ignore  # noqa
    import intel_extension_for_pytorch as ipex  # type:ignore  # noqa
    ACCELERATOR_TYPE = "IntelGPU"
except (ImportError, ModuleNotFoundError):
    if torch.cuda.is_available():
        ACCELERATOR_TYPE = "NvidiaGPU"
    else:
        ACCELERATOR_TYPE = "CPU"

log = logging.getLogger(__name__)
logging.getLogger('sh').setLevel('WARNING')


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
                    _ = tstr.append(f"({args}") if len(args) > 0 else None
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
        hostfile: Optional[str | os.PathLike | Path] = None
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


def get_dist_info(
        framework: str = 'pytorch',
        verbose: Optional[bool] = None
) -> dict[str, str | int | list]:
    # master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    hostname = get_hostname()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    num_nodes = get_num_nodes()
    gpus_per_node = get_gpus_per_node()
    node_id = get_node_index()
    device = local_rank
    distributed_backend = None
    if framework in {'pt', 'torch', 'pytorch'}:
        device = get_torch_device()
        distributed_backend = get_torch_backend()
    machine = get_machine()
    hostfile, hosts = get_hosts_from_hostfile()
    dist_info = {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'num_nodes': num_nodes,
        'gpus_per_node': gpus_per_node,
        'node_id': node_id,
        'machine': machine,
        'hostfile': hostfile,
        'hostname': hostname,
        'hosts': hosts,
        'device': device,
        'distributed_backend': distributed_backend,
    }
    if verbose:
        # cjson = config.to_json()
        # text = Text("DistInfo:")
        # text += json.dumps(dist_info, indent=4)
        # log.info(f'{text}')
        # console = get_console()
        # from enrich.handler import RichHandler as EnrichHandler
        # from rich.logging import RichHandler
        # console = None
        # for handler in log.handlers:
        #     if isinstance(handler, (RichHandler, EnrichHandler)):
        #         console = handler.console
        # if console is None:
        #     console = get_console()
        # console.print_json(data=dist_info, indent=4, highlight=True)
        # # get_console().print_json(data=dist_info, indent=4, highlight=True)
        # from ezpz.configs import print_config
        # print_config(dist_info)
        # from rich.json import JSON
        # log.info(
        #     f'DistInfo: {JSON(json.dumps(dist_info, indent=4))}'
        # )
        # log.info(
        #     ' '.join([
        #         'DistInfo:',
        #         "\n".join([f"{k}: {v}" for k, v in dist_info.items()])
        #     ])
        # )
        from ezpz import get_console_from_logger
        console = get_console_from_logger(log)
        console.print(
            Text(
                'DistInfo: ',
                style=Style(color='bright_green', bold=True, underline=True)
            )
        )
        console.print_json(data=dist_info, indent=4)
    if wandb is not None and wandb.run is not None:
        wandb.run.config.update({'DIST_INFO': dist_info})
    return dist_info


def print_dist_setup(framework: str = 'torch') -> str:
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    gpus_per_node = get_gpus_per_node()
    # num_nodes = get_num_nodes()
    node = get_node_index()
    device = None
    if framework.lower() in ['pt', 'torch', 'pytorch']:
        device = get_torch_device()
    rank_len = len(str(rank))
    ws_len = len(str(world_size))
    lr_len = len(str(local_rank))
    gpn_len = len(str(gpus_per_node))
    dist_str = ''.join([
        f'[{node=}]',
        f'[{rank=:>{rank_len}}/{(world_size-1):<{ws_len}}]',
        f'[{local_rank=:>{lr_len}}/{gpus_per_node-1:<{gpn_len}}]',
        f'[{device=}]'
    ])
    log.info(f'{dist_str}')
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
    except (ImportError, ModuleNotFoundError) as exc:
        log.warning('Unable to `import deepspeed`. Exiting!')
        log.exception(exc)
        raise exc


def get_torch_device() -> str:
    try:
        import intel_extension_for_pytorch as ipex  # type:ignore noqa
        device = "xpu"
    except (ImportError, ModuleNotFoundError):
        # if torch.cuda.is_available():
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if (
                torch.backends.mps.is_available()
                and torch.get_default_dtype() != torch.float64
            )
            else 'cpu'
        )
    return device


def get_torch_backend() -> str:
    import torch
    backend = 'nccl' if torch.cuda.is_available() else None
    if backend is None:
        try:
            import oneccl_bindings_for_pytorch  # type:ignore noqa
            import intel_extension_for_pytorch as ipex  # type:ignore noqa
            backend = "ccl"  # if backend is None else str(backend)
        except (ImportError, ModuleNotFoundError):
            try:
                import torch_ccl  # type: ignore  noqa
                backend = 'ccl'
            except (ImportError, ModuleNotFoundError):
                backend = 'gloo'
            # if torch.cuda.is_available():
            #     backend = 'nccl' if backend is None else str(backend)
            # else:
            # backend = 'gloo'  # if backend is None else str(backend)
    if backend is None:
        log.critical(f'Using "gloo" backend on {get_torch_device()}')
        backend = 'gloo'
    return backend


def init_process_group(
        rank: int | str,
        world_size: int | str,
        # backend: Optional[str] = None,
) -> None:
    # import torch
    # import torch.distributed as dist
    # try:
    #     import oneccl_bindings_for_pytorch
    #     import intel_extension_for_pytorch as ipex
    #     backend = "ccl" if backend is None else str(backend)
    # except (ImportError, ModuleNotFoundError):
    #     if torch.cuda.is_available():
    #         backend = 'nccl' if backend is None else str(backend)
    #     else:
    #         backend = 'gloo' if backend is None else str(backend)
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
    return int(MPI.COMM_WORLD.Get_rank())


def get_world_size() -> int:
    return int(MPI.COMM_WORLD.Get_size())


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
    assert be in BACKENDS['pytorch']
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
        raise ValueError(f'Unable to parse backend: {be}')
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
        log.info(
            f"Using {device=} with {backend=} + '{get_torch_backend()}' "
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
    hvd.init() if not hvd.is_initialized() else None
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
            # if hvd.rank() == 0:
            #     log.info(
            #         f'{len(gpus)}, Physical GPUs and '
            #         f'{len(logical_gpus)} Logical GPUs'
            #     )
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


def include_file(f: os.PathLike | str | Path):
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
        machine = 'ThetaGPU'
    elif hostname.startswith('x1'):
        machine = 'SunSpot'
    elif hostname.startswith('x3'):
        machine = 'Polaris'
    elif hostname.startswith('x4'):
        machine = 'Aurora'
    elif hostname.startswith('login'):
        machine = 'NERSC'
    elif hostname.startswith('nid'):
        machine = 'Perlmutter'
    else:
        machine = hostname
        log.warning(f'Unknown machine, setting {machine=}')
    return machine


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
                os.environ.get("WB_PROJECT_NAME", None)
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


def inspect_cobalt_running_job() -> dict[str, str | os.PathLike]:
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


def get_nodes_from_hostfile(hostfile: os.PathLike) -> list[str]:
    # cobalt_nodefile = get_cobalt_nodefile()
    fpath = Path(hostfile)
    assert fpath.is_file()
    with fpath.open('r') as f:
        nodes = [i.rstrip('\n') for i in f.readlines()]
    return nodes


def get_node_index() -> int:
    return get_rank() % get_num_nodes()


def get_num_nodes() -> int:
    return (
        1 if (ws := get_world_size()) < (gpn := get_gpus_per_node())
        else ws // gpn
    )


def get_cpus_per_node() -> int:
    from sh import getconf as sh_getconf  # type:ignore noqa
    return int(sh_getconf("_NPROCESSORS_ONLN").rstrip('\n'))


def get_gpus_per_node(_assert: Optional[bool] = None) -> int:
    import torch
    gpus_per_node = None
    try:
        import intel_extension_for_pytorch as ipex  # pyright:ignore  # noqa
        try:
            import oneccl_bindings_for_pytorch  # type:ignore noqa
        except (ImportError, ModuleNotFoundError):
            import torch_ccl  # type: ignore  # noqa
        gpus_per_node = ipex.xpu.device_count()
    except (ImportError, ModuleNotFoundError):
        if torch.cuda.is_available():
            gpus_per_node = torch.cuda.device_count()
    if _assert:
        try:
            # import sh  # pyright: ignore
            from sh import wc as sh_wc  # pyright: ignore
            from sh import nvidia_smi as sh_nvidia_smi  # pyright: ignore
            gpus_per_node = int(
                sh_wc(
                    "-l",
                    _in=sh_nvidia_smi("-L")
                ).rstrip("\n")
            )
        except (ImportError, ModuleNotFoundError):
            if torch.cuda.is_available():
                gpus_per_node = torch.cuda.device_count()
            else:
                gpus_per_node = int(run_bash_command('nvidia-smi -L | wc -l'))
            raise ValueError('No GPUs found. Exiting!')
    if gpus_per_node is None:
        ncpus = get_cpus_per_node()
        return ncpus
    return gpus_per_node


def get_cobalt_resources(_assert: Optional[bool] = None) -> dict:
    cobalt_info = inspect_cobalt_running_job()
    # cobalt_nodefile = get_cobalt_nodefile()
    nodes = get_nodes_from_hostfile(Path(cobalt_info["COBALT_NODEFILE"]))
    gpus_per_node = get_gpus_per_node(_assert=_assert)
    cobalt_info |= {
        'nodes': nodes,
        'num_nodes': len(nodes),
        'gpus_per_node': gpus_per_node,
        'num_gpus': len(nodes) * gpus_per_node,
        'machine': 'ThetaGPU',
    }
    return cobalt_info


def build_mpiexec_thetagpu(
        # ngpus: Optional[int] = None,
        # hostfile: Optional[os.PathLike] = None
):
    # import subprocess
    # import subprocess
    jobenv = get_cobalt_resources()
    # which_mpi = subprocess.Popen('which mpirun', shell=True)
    # try:
    #     import sh
    #     which_mpi = sh.which('mpirun').rstrip('\n')
    # except (ImportError, ModuleNotFoundError):
    mpiexec = [
        "mpirun",
        f"-n {jobenv['num_nodes']}",
        f"-npernode {jobenv['gpus_per_node']}",
        f"--hostfile {jobenv['COBALT_NODEFILE']}",
        "-x PATH",
        "-x LD_LIBRARY_PATH",
        "-x http_proxy",
        "-x https_proxy",
    ]
    return mpiexec


def run_mpiexec(cmd: str):
    import subprocess
    mpiexec = ' '.join(build_mpiexec_thetagpu())
    # python3 = sys.executable
    # ezpz_test = '-m ezpz.check'
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
