"""
dist.py

Contains methods for initializing distributed communication.
"""
from __future__ import absolute_import, annotations, division, print_function

import os

from typing import Optional, Callable, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from mpi4py import MPI
import logging
from ezpz.configs import FRAMEWORKS, BACKENDS, HERE, PROJECT_ROOT

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    import torch
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_dist_setup():
    # print(' '.join(setup_strings))
    # log.info(f"{get_log_prefix()}: {get_rank()} / {get_world_size() - 1}")
    log.info(f"RANK: {get_rank()} / {get_world_size() - 1}")


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
        if framework in ['tensorflow', 'tf', 't']
        else setup_torch(backend=backend, port=port, seed=seed)
    )


def init_deepspeed():
    import deepspeed
    deepspeed.init_distributed()


def init_process_group(
        rank: int | str,
        world_size: int | str,
        backend: Optional[str] = None,
) -> None:
    import torch
    import torch.distributed as dist
    if torch.cuda.is_available():
        backend = 'nccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
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
    return int(
        os.environ.get(
            'PMI_LOCAL_RANK',  # PMI_* for Polaris (/ Aurora) @ ALCF
            # OMPI_* for ThetaGPU @ ALCF
            os.environ.get(
                'LOCAL_RANK',
                # '0'
                os.environ.get(
                    'OMPI_COMM_WORLD_LOCAL_RANK',
                    os.environ.get(
                        'RANK',
                        os.environ.get(
                            'SLURM_PROCID',
                            '0',
                        )
                    )
                )
            )
        )
    )


def query_environment() -> dict[str, int]:
    """Query environment variables for info about distributed setup"""
    ws = os.environ.get('WORLD_SIZE', None)
    r = os.environ.get('RANK', None)
    lr = os.environ.get('LOCAL_RANK', None)
    if ws is not None and r is not None and lr is not None:
        return {
            'world_size': int(ws),
            'rank': int(r),
            'local_rank': int(lr)
        }

    return {
        'world_size': int(get_world_size()),
        'rank': int(get_rank()),
        'local_rank': int(get_local_rank()),
    }


def setup_torch_DDP(port: str = '2345') -> dict[str, int]:
    import torch
    rank = os.environ.get('RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    local_rank = os.environ.get('LOCAL_RANK', None)

    import socket
    world_size = int(get_world_size())
    rank = int(get_rank())
    local_rank = int(get_local_rank())
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    master_addr = (
        socket.gethostname() if rank == 0 else None
    )
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    if (eport := os.environ.get('MASTER_PORT', None)) is None:
        os.environ['MASTER_PORT'] = port
    else:
        os.environ['MASTER_PORT'] = eport
        if rank == 0:
            log.info(f'Caught MASTER_PORT:{eport} from environment!')
    init_process_group(
        rank=rank,
        world_size=world_size,
        backend='nccl' if torch.cuda.is_available() else 'gloo'
    )
    return {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}


def setup_torch_distributed(
        backend: str,
        port: str = '2345',
) -> dict:
    import torch
    rank = os.environ.get('RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    be = backend.lower()
    assert be in BACKENDS['pytorch']
    if rank == 0 and local_rank == 0:
        log.info(f'Using {backend} for distributed training')
    if be == 'ddp':
        dsetup = setup_torch_DDP(port)
        world_size = dsetup['world_size']
        rank = dsetup['rank']
        local_rank = dsetup['local_rank']
    elif be in ['deepspeed', 'ds']:
        init_deepspeed()
        world_size = get_world_size()
        rank = get_rank()
        local_rank = get_local_rank()
    elif be in ['horovod', 'hvd']:
        import horovod.torch as hvd
        hvd.init() if not hvd.is_initialized() else None
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
        backend: str = 'deepspeed',
        port: str = '2345',
        seed: Optional[int] = None,
) -> int:
    import torch
    # from rich import log.info
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True     # type:ignore
    torch.backends.cudnn.benchmark = True         # type:ignore
    torch.backends.cudnn.allow_tf32 = True        # type:ignore
    torch.backends.cuda.matmul.allow_tf32 = True  # type:ignore
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
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    log.info(f'RANK: {rank} / {world_size-1}')
    if seed is not None:
        seed_everything(seed * (rank + 1) * (local_rank + 1))
    return rank


def cleanup() -> None:
    import torch.distributed as tdist
    tdist.destroy_process_group()


def setup_tensorflow(
        precision: Optional[str] = None,
        ngpus: Optional[int] = None,
) -> int:
    """Initialize TensorFlow + Horovod for Distributed Training"""
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import horovod.tensorflow as hvd
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
    log.info(f'RANK: {RANK} / {WORLD_SIZE-1}')
    if RANK == 0:
        log.info(f"Using {TF_FLOAT} precision")
    return RANK


def setup_wandb(
        project_name: Optional[str] = None,
        config: Optional[dict | DictConfig] = None,
        start_method: Optional[str] = None,
):
    import wandb
    import socket
    import time
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
    # if get_rank() == 0:
    # tensorboard_dir = args.tensorboard_dir
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
    current_time = time.time()
    # local_time = time.localtime(current_time)
    # if wandb.run is None:
    wbsettings = None
    if start_method is not None:
        wbsettings = wandb.Settings(start_method=start_method)
    wandb.init(
        # resume='allow',
        dir=os.getcwd(),
        sync_tensorboard=(tensorboard_dir is not None),  # True,
        project=(project_name if project_name is not None else None),
        settings=wbsettings,
        # dir=(tensorboard_dir if tensorboard_dir is not None else None),
    )
    assert wandb.run is not None
    wandb.run.log_code(HERE.as_posix())
    log.info(f"W&B RUN: [{wandb.run.name}]({wandb.run.url})")
    wandb.run.config.update({'current_time': current_time})
    wandb.run.config.update({'world_size': get_world_size()})
    wandb.run.config.update({'outdir': os.getcwd()})
    wandb.run.config.update({'hostname': rank})
    if config is not None:
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(
                config,
                resolve=True,
                throw_on_missing=True
            )
        wandb.run.config.update({'config': config})
    env = {
        k: v for k, v in dict(os.environ).items()
        if not k.startswith('_ModuleTable')
    }
    _ = env.pop('LS_COLORS', None)
    _ = env.pop('PS1', None)
    wandb.run.config.update({'env': env})
    hostname = socket.gethostbyaddr(socket.gethostname())[0]
    hostfile = os.environ.get(
        'HOSTFILE',
        os.environ.get(
            'PBS_NODEFILE',
            os.environ.get(
                'COBALT_NODEFILE',
                os.environ.get(
                    'SLURM_JOB_NODELIST',
                    None
                )
            )
        )
    )
    # if (hpath := Path(hostfile).resolve().is_file()):
    if hostfile is not None:
        hpath = Path(hostfile).resolve().absolute()
        if hpath.is_file():
            with hpath.open('r') as f:
                hosts = f.readlines()
            wandb.run.config['hosts'] = hosts
    if hostname.startswith('theta'):
        wandb.run.config.update({'machine': 'ThetaGPU'})
    elif hostname.startswith('x3'):
        wandb.run.config.update({'machine': 'Polaris'})
    elif hostname.startswith('x1'):
        wandb.run.config.update({'machine': 'Sunspot'})
    elif hostname.startswith('nid'):
        wandb.run.config.update({'machine': 'Perlmutter'})
    elif hostname.startswith('login'):
        wandb.run.config.update({'machine': 'NERSC'})
    else:
        wandb.run.config.update({'machine': hostname})
    model_size = os.environ.get('MODEL_SIZE', None)
    if model_size is not None:
        wandb.run.config.update({'MODEL_SIZE': model_size})


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


def get_gpus_per_node(_assert: Optional[bool] = None) -> int:
    import sh
    gpus_per_node = int(sh.wc("-l", _in=sh.nvidia_smi("-L")).rstrip("\n"))
    if _assert:
        import torch
        assert gpus_per_node == torch.cuda.device_count()
    return gpus_per_node


def get_cobalt_resources(_assert: Optional[bool] = None) -> dict:
    cobalt_info = inspect_cobalt_running_job()
    # cobalt_nodefile = get_cobalt_nodefile()
    nodes = get_nodes_from_hostfile(cobalt_info["COBALT_NODEFILE"])
    gpus_per_node = get_gpus_per_node()
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
    import subprocess
    jobenv = get_cobalt_resources()
    # which_mpi = subprocess.Popen('which mpirun', shell=True)
    import sh
    which_mpi = sh.which('mpirun').rstrip('\n')
    mpiexec = [
        f"{which_mpi}",
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
            port=port,
        )
    elif framework in FRAMEWORKS['tensorflow']:
        _ = setup_tensorflow()
    else:
        raise ValueError(f"Unable to parse framework: {framework}")
