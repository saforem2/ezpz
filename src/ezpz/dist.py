"""Compatibility shim -- delegates to :mod:`ezpz.distributed`.

``ezpz.dist`` is **deprecated**.  All functionality has moved to
:mod:`ezpz.distributed`.  This module re-exports every public symbol so
that existing callers (``import ezpz.dist as dist``) continue to work,
but a :class:`DeprecationWarning` is emitted on first import.
"""

from __future__ import annotations

import logging
import os
import warnings as _warnings
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

_warnings.warn(
    "ezpz.dist is deprecated and will be removed in a future release. "
    "Use ezpz.distributed instead.",
    DeprecationWarning,
    stacklevel=2,
)

# ---------------------------------------------------------------------------
# Re-export the full public API from ezpz.distributed
# ---------------------------------------------------------------------------
from ezpz.distributed import (  # noqa: F401 -- re-exports
    TORCH_DTYPES_MAP,
    all_reduce,
    barrier,
    broadcast,
    cleanup,
    get_cpus_per_node,
    get_device_properties,
    get_dist_info,
    get_gpus_per_node,
    get_hostname,
    get_hostfile_with_fallback,
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
    get_world_size_in_use,
    get_world_size_total,
    log_dict_as_bulleted_list,
    print_dist_setup,
    query_environment,
    seed_everything,
    setup_torch,
    setup_wandb,
    synchronize,
    timeitlogit,
    verify_wandb,
    wrap_model,
    wrap_model_for_ddp,
    wrap_model_for_fsdp,
    wrap_model_for_fsdp2,
    write_hostfile_from_list_of_hosts,
    write_localhost_to_hostfile,
    # private helpers referenced by tests
    _expand_slurm_nodelist,
)

PathLike = Union[str, os.PathLike, Path]

logger = logging.getLogger(__name__)

# Flags previously exposed as module-level state (referenced by tests/test.py)
_ENV_TORCH_DEVICE_LOGGED = False
_ENV_TORCH_DEVICE_APPLIED = False


# ---------------------------------------------------------------------------
# Legacy functions that only existed in dist.py
#
# These are preserved here so that ``from ezpz.dist import <name>`` keeps
# working, but they are thin wrappers or trivial implementations.  None of
# them are imported by the main library code.
# ---------------------------------------------------------------------------


def setup(
    framework: str = "pytorch",
    backend: str = "DDP",
    port: Optional[str] = None,
    seed: Optional[int] = None,
    precision: Optional[str] = None,
    ngpus: Optional[int] = None,
) -> int:
    """Legacy entry point -- delegates to :func:`ezpz.distributed.setup_torch`."""
    return setup_torch(port=port, seed=seed)


def timeit(func: Callable) -> Callable:
    """Simple timing decorator (legacy)."""
    import time

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if get_rank() == 0:
            logger.info(f"[timeit] {func.__name__}: {dt:.4f}s")
        return result

    return wrapper


def get_hosts_from_hostfile(
    hostfile: Union[str, os.PathLike, Path],
) -> tuple[Path, list[str]]:
    """Return ``(hostfile_path, hosts)`` from a hostfile."""
    hfp = Path(hostfile)
    hosts = get_nodes_from_hostfile(hfp)
    return hfp, hosts


def wrap_with_ddp(model: Any) -> Any:
    """Alias for :func:`wrap_model_for_ddp`."""
    return wrap_model_for_ddp(model)


def wrap_with_fsdp(
    model: Any,
    dtype: str = "bfloat16",
    device_id: Optional[Any] = None,
) -> Any:
    """Alias for :func:`wrap_model_for_fsdp`."""
    return wrap_model_for_fsdp(model, dtype=dtype, device_id=device_id)


def wrap_with_fsdp2(
    model: Any,
    dtype: str = "bfloat16",
    device_id: Optional[Any] = None,
    device_mesh: Optional[Any] = None,
) -> Any:
    """Alias for :func:`wrap_model_for_fsdp2`."""
    return wrap_model_for_fsdp2(
        model, dtype=dtype, device_id=device_id, device_mesh=device_mesh
    )


def init_deepspeed(timeout: int = 3600) -> None:
    """Legacy DeepSpeed init -- use ``setup_torch(backend='deepspeed')``."""
    import torch.distributed

    if not torch.distributed.is_initialized():
        import deepspeed  # type: ignore[import-not-found]

        deepspeed.init_distributed(
            dist_backend=get_torch_backend(),
            timeout=timedelta(seconds=timeout),
        )


def get_device(
    as_torch_device: bool = False,
) -> str:
    """Legacy device helper -- use :func:`get_torch_device`."""
    dev = get_torch_device()
    return str(dev)


def get_torch_version_as_float() -> float:
    """Return the PyTorch version as a float (e.g. 2.5)."""
    import torch

    parts = torch.__version__.split(".")[:2]
    return float(f"{parts[0]}.{parts[1]}")


def get_torch_backend_on_xpu() -> str:
    """Return the distributed backend name for Intel XPU."""
    return "xccl"


def init_process_group(
    backend: Optional[str] = None,
    timeout: int = 3600,
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> None:
    """Thin wrapper around ``torch.distributed.init_process_group``."""
    import torch.distributed

    if torch.distributed.is_initialized():
        return
    backend = backend or get_torch_backend()
    kwargs: dict[str, Any] = {
        "backend": backend,
        "timeout": timedelta(seconds=timeout),
    }
    if init_method is not None:
        kwargs["init_method"] = init_method
    if rank is not None:
        kwargs["rank"] = rank
    if world_size is not None:
        kwargs["world_size"] = world_size
    torch.distributed.init_process_group(**kwargs)


def run_ddp(fn: Callable, world_size: int) -> None:
    """Spawn *fn* across *world_size* processes using ``mp.spawn``."""
    import torch.multiprocessing as mp

    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)  # type: ignore[arg-type]


def get_free_port() -> int:
    """Return a free TCP port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_torch_env(
    master_port: Optional[str] = None,
    backend: Optional[str] = None,
    device_id: Any = None,
) -> dict[str, int]:
    """Legacy helper -- use :func:`setup_torch` instead."""
    return {"rank": setup_torch(port=master_port)}


def setup_torch_DDP(
    port: str = "2345",
    timeout: int | str | timedelta = 3600,
    backend: Optional[str] = None,
    device_id: Any = None,
) -> dict[str, Any]:
    """Legacy DDP setup -- use :func:`setup_torch` instead."""
    rank = setup_torch(port=port, timeout=timeout)
    return {
        "rank": rank,
        "local_rank": get_local_rank(),
        "world_size": get_world_size(),
        "master_port": port,
    }


def setup_torch_distributed(
    framework: Optional[str] = None,
    backend: Optional[str] = None,
    tensor_parallel_size: int = 1,
    port: Optional[str | int] = None,
    timeout: int | str | timedelta = 3600,
    device_id: Any = None,
) -> dict[str, Any]:
    """Legacy distributed setup -- use :func:`setup_torch` instead."""
    rank = setup_torch(port=port, seed=None, timeout=timeout)
    return {
        "rank": rank,
        "local_rank": get_local_rank(),
        "world_size": get_world_size(),
    }


def setup_tensorflow(
    precision: Optional[str] = None,
    ngpus: Optional[int] = None,
) -> int:
    """Legacy TF setup stub (no longer supported)."""
    _warnings.warn(
        "setup_tensorflow() is no longer supported. Use setup_torch() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return 0


def include_file(f: Union[PathLike, str]) -> bool:
    """Return True if file *f* should be included in wandb code logging."""
    fpath = Path(f)
    return fpath.suffix in {".py", ".yaml", ".sh", ".md"}


def get_wandb_mode(
    mode: Optional[str] = None,
) -> str:
    """Resolve the effective wandb mode."""
    if mode is not None:
        return mode
    env_mode = os.environ.get("WANDB_MODE", "online")
    return env_mode if env_mode else "online"


def get_git_branch_name() -> str | None:
    """Return the current git branch name, or None."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_bash_command(cmd: str) -> Any:
    """Run a bash command and return stdout."""
    import subprocess

    proc = subprocess.run(
        cmd,
        shell=True,
        check=False,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def make_hostfile_from_slurm_env(
    outfile: Optional[Union[PathLike, str]] = None,
) -> Path:
    """Create a hostfile from SLURM environment variables."""
    nodelist_str = os.environ.get("SLURM_NODELIST", "")
    hosts = _expand_slurm_nodelist(nodelist_str)
    if not hosts:
        import socket

        hosts = [socket.gethostname()]
    if outfile is None:
        outfile = Path(os.getcwd()) / "hostfile"
    return write_hostfile_from_list_of_hosts(hosts, outfile)


def check(
    framework: str = "pytorch",
    backend: str = "deepspeed",
    port: int | str = "5432",
) -> None:
    """Legacy check -- use ``ezpz doctor`` instead."""
    setup_torch(port=str(port))


class TorchDistributedEnvironment:
    """Legacy class -- use :func:`setup_torch` directly."""

    def __init__(
        self,
        master_port: str | int | None = None,
        backend: str | None = None,
        device_id: Any = None,
    ):
        rank = setup_torch(port=str(master_port) if master_port else None)
        self.env = {
            "rank": rank,
            "local_rank": get_local_rank(),
            "world_size": get_world_size(),
            "master_port": master_port,
        }
        self.master_port = master_port
        self.master_addr = os.environ.get("MASTER_ADDR")
        self.world_size = get_world_size()
        self.rank = rank
        self.local_rank = get_local_rank()

    def as_dict(self) -> dict[str, Any]:
        return self.env

    def ensure_env_vars(self) -> None:
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(get_gpus_per_node())
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
