"""Simplified distributed training primitives for ``ezpz``.

This module is a clean rewrite of :pymod:`ezpz.dist`.  It preserves the
same public API surface that the rest of the codebase relies on while
eliminating module-level side effects, dead code, redundant aliases, and
tangled responsibilities.

Design principles:

* **No side effects on import** -- no env vars mutated, no devices set,
  no wandb probed until the caller explicitly asks for it.
* **Lazy heavy imports** -- ``mpi4py``, ``torch``, optional deps are
  imported inside the functions that need them so that
  ``import ezpz.distributed`` stays fast.
* **Single responsibility per function** -- no 200-line monoliths.
* **Flat public API** -- every symbol listed in ``__all__`` is a
  first-class citizen; everything else is prefixed with ``_``.
"""

from __future__ import annotations

import datetime
import logging
import os
import socket
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
)

if TYPE_CHECKING:
    import torch
    import torch.distributed

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("EZPZ_LOG_LEVEL", "INFO").upper())

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # -- rank / topology --
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "get_world_size_total",
    "get_world_size_in_use",
    "get_num_nodes",
    "get_gpus_per_node",
    "get_cpus_per_node",
    "get_node_index",
    "get_device_properties",
    # -- device / backend --
    "get_torch_device",
    "get_torch_device_type",
    "get_torch_backend",
    "use_torchcomms",
    "TORCH_DTYPES_MAP",
    # -- lifecycle --
    "setup_torch",
    "cleanup",
    # -- collectives / sync --
    "synchronize",
    "barrier",
    "broadcast",
    "all_reduce",
    # -- model wrapping --
    "wrap_model",
    "wrap_model_for_ddp",
    "wrap_model_for_fsdp",
    "wrap_model_for_fsdp2",
    "init_device_mesh_safe",
    # -- diagnostics --
    "get_dist_info",
    "get_hostname",
    "get_machine",
    "query_environment",
    "print_dist_setup",
    "seed_everything",
    "log_dict_as_bulleted_list",
    # -- timing --
    "timeitlogit",
    # -- wandb / mlflow --
    "setup_wandb",
    "setup_mlflow",
    "verify_wandb",
    # -- hostfile helpers --
    "get_nodes_from_hostfile",
    "get_hostfile_with_fallback",
    "write_localhost_to_hostfile",
    "write_hostfile_from_list_of_hosts",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TORCH_DTYPES_MAP: dict[str, Any] = {
    "bf16": None,
    "bfloat16": None,
    "fp16": None,
    "float16": None,
    "half": None,
    "fp32": None,
    "float32": None,
}
"""Mapping of short dtype names to :class:`torch.dtype` objects.

Populated lazily on first access via :func:`_ensure_dtype_map`.
"""

_DTYPE_MAP_READY = False

_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "xpu", "mps"})

# Lazily-initialised singletons ------------------------------------------------
_MPI_COMM = None  # cached MPI.COMM_WORLD


def _ensure_dtype_map() -> dict[str, Any]:
    """Fill *TORCH_DTYPES_MAP* with real :class:`torch.dtype` values."""
    global _DTYPE_MAP_READY
    if _DTYPE_MAP_READY:
        return TORCH_DTYPES_MAP
    import torch

    TORCH_DTYPES_MAP.update(
        {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
    )
    _DTYPE_MAP_READY = True
    return TORCH_DTYPES_MAP


def _get_mpi_comm() -> Any:
    """Return ``MPI.COMM_WORLD``, importing ``mpi4py`` on first call."""
    global _MPI_COMM
    if _MPI_COMM is None:
        from mpi4py import MPI

        _MPI_COMM = MPI.COMM_WORLD
    return _MPI_COMM


# ===================================================================
# Rank / topology
# ===================================================================


def get_rank() -> int:
    """Return the global MPI rank of the current process.

    The value is resolved from well-known environment variables set by
    common MPI implementations and job schedulers.  If none are set the
    function falls back to querying the MPI communicator.

    Returns:
        Global rank (0-indexed).
    """
    _ENV_VARS = (
        "RANK",
        "PMI_RANK",
        "OMPI_COMM_WORLD_RANK",
        "SLURM_PROCID",
    )
    for var in _ENV_VARS:
        val = os.environ.get(var)
        if val is not None and val != "":
            return int(val)
    try:
        return int(_get_mpi_comm().Get_rank())
    except Exception:
        return 0


def get_local_rank() -> int:
    """Return the local rank (GPU index) of the current process on its node.

    The value is resolved from well-known environment variables set by
    common MPI implementations and job schedulers.  If none are set the
    function falls back to ``get_rank() % ranks_per_node``.

    Returns:
        Local rank (0-indexed within the node).
    """
    _ENV_VARS = (
        "LOCAL_RANK",
        "PALS_LOCAL_RANKID",  # cray-pals mpiexec (Aurora / Sunspot)
        "PMIX_LOCAL_RANK",  # PMIx layer (Aurora / Sunspot)
        "PMI_LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MPICH_LOCALRANKID",
        "SLURM_LOCAL_ID",
    )
    for var in _ENV_VARS:
        val = os.environ.get(var)
        if val is not None and val != "":
            return int(val)
    ws = get_world_size()
    if ws <= 1:
        return 0
    # Last-resort fallback (no launcher local-rank var found): derive the
    # local rank from the global rank and the number of ranks placed on
    # each node. Use ranks-per-node (world_size // num_nodes), NOT the
    # device count — they diverge whenever a job under-subscribes a node
    # (e.g. `-ppn 6` on a 12-device node, or COMPOSITE collapsing 12
    # tiles into 6 GPUs), and a device-count modulo would then hand
    # `set_device` an out-of-range index (the original COMPOSITE crash).
    nnodes = get_num_nodes()
    if nnodes > 0 and ws % nnodes == 0:
        ranks_per_node = ws // nnodes
    else:
        # Can't cleanly determine ranks-per-node; best-effort with the
        # device count (historical behavior).
        ranks_per_node = get_gpus_per_node()
    return get_rank() % ranks_per_node if ranks_per_node > 0 else 0


def get_world_size(
    *,
    total: bool = False,
    in_use: bool = False,
) -> int:
    """Return the distributed world size.

    Args:
        total: If ``True``, return the *total available* accelerator count
            (``num_nodes * gpus_per_node``).
        in_use: If ``True``, return ``MPI.COMM_WORLD.Get_size()``.

    Returns:
        World size as an integer.
    """
    if total:
        return get_world_size_total()
    if in_use:
        return get_world_size_in_use()
    _ENV_VARS = (
        "WORLD_SIZE",
        "PMI_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
    )
    for var in _ENV_VARS:
        val = os.environ.get(var)
        if val is not None and val != "":
            return int(val)
    try:
        return int(_get_mpi_comm().Get_size())
    except Exception:
        return 1


def get_world_size_in_use() -> int:
    """Return the number of MPI ranks currently participating."""
    try:
        return int(_get_mpi_comm().Get_size())
    except Exception:
        return 1


def get_world_size_total() -> int:
    """Return ``max(get_gpus_per_node(), 1) * get_num_nodes()``."""
    return max(get_gpus_per_node(), 1) * get_num_nodes()


def get_num_nodes(hostfile: str | os.PathLike | None = None) -> int:
    """Return the number of nodes in the current allocation.

    Checks ``SLURM_NNODES`` first, then counts lines in *hostfile*.

    Args:
        hostfile: Explicit path; resolved automatically when ``None``.
    """
    slurm_nnodes = os.environ.get("SLURM_NNODES")
    if slurm_nnodes is not None:
        return int(slurm_nnodes)
    hfp = get_hostfile_with_fallback(hostfile)
    hosts = [h.split(".")[0] for h in get_nodes_from_hostfile(hfp)]
    return len(hosts)


def get_gpus_per_node() -> int:
    """Return the number of accelerators on the local node.

    Prefers environment variables (``NGPU_PER_HOST``, ``LOCAL_WORLD_SIZE``,
    ``PMI_LOCAL_SIZE``, ``SLURM_NTASKS_PER_NODE``) then falls back to
    ``torch.{cuda,xpu}.device_count()``.

    On Intel XPU the env hint is *clamped down* to the number of devices
    the runtime actually exposes: ezpz's shell setup hardcodes
    ``NGPU_PER_HOST=12`` (the FLAT tile count), but under
    ``ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`` only 6 composite GPUs exist.
    Reporting 12 there over-counts devices and skews ``get_node_index``,
    the per-host summary, and the local-rank fallback. Clamping is
    one-directional (never bumps a smaller, deliberate under-subscription
    up) and XPU-only (COMPOSITE has no CUDA analogue), so CUDA behavior
    is unchanged.
    """
    env_hint: int | None = None
    for var in (
        "NGPU_PER_HOST",
        "LOCAL_WORLD_SIZE",
        "PMI_LOCAL_SIZE",
        "SLURM_NTASKS_PER_NODE",
    ):
        val = os.environ.get(var)
        if val is not None and val != "":
            env_hint = int(val)
            break

    xpu_count = _xpu_device_count_safe()
    if env_hint is not None:
        # Only reconcile against XPU: clamp a stale FLAT-mode hint down to
        # the visible composite-GPU count. Never bump up (under-
        # subscription is intentional), never touch the CUDA path.
        if xpu_count is not None and 0 < xpu_count < env_hint:
            return xpu_count
        return env_hint

    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if xpu_count is not None:
        return xpu_count
    if torch.backends.mps.is_available():
        return get_world_size_in_use()
    return 0


def _xpu_device_count_safe() -> int | None:
    """Return ``torch.xpu.device_count()`` or ``None`` if unavailable.

    Guards against builds without ``torch.xpu`` (CPU-/CUDA-only) and
    against login nodes where probing raises (no Level-Zero loader) —
    ``get_gpus_per_node`` runs during launch-time topology inference
    there, so a raised exception must degrade to "unknown", not crash.
    """
    import torch

    if not hasattr(torch, "xpu"):
        return None
    try:
        if torch.xpu.is_available():
            return torch.xpu.device_count()
    except Exception:
        return None
    return None


def get_cpus_per_node() -> int:
    """Return the number of CPUs available on the local node."""
    return os.cpu_count() or 1


def get_node_index() -> int:
    """Return the index of the current node (``rank // gpus_per_node``)."""
    gpn = get_gpus_per_node()
    return get_rank() // gpn if gpn > 0 else 0


# ===================================================================
# Device / backend
# ===================================================================


def get_torch_device_type(device_type: str | None = None) -> str:
    """Return the accelerator type as a string (``"cuda"``, ``"xpu"``, …).

    Respects the ``TORCH_DEVICE`` environment variable when set.

    Args:
        device_type: Override value; returned as-is after validation.
    """
    if device_type is not None:
        if device_type not in _SUPPORTED_DEVICE_TYPES:
            raise ValueError(
                f"Unsupported device_type={device_type!r}; "
                f"expected one of {sorted(_SUPPORTED_DEVICE_TYPES)}"
            )
        return device_type
    env = os.environ.get("TORCH_DEVICE")
    if env:
        base = env.strip().lower().split(":", 1)[0]
        if base in _SUPPORTED_DEVICE_TYPES:
            return base
        logger.warning(
            "Ignoring unsupported TORCH_DEVICE=%s; expected one of %s",
            env,
            sorted(_SUPPORTED_DEVICE_TYPES),
        )
    import torch

    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_device(
    *,
    device_type: str | None = None,
    as_torch_device: bool | None = None,
) -> str | torch.device:
    """Return the current accelerator device.

    Args:
        device_type: Force a specific device type.
        as_torch_device: If ``True``, return a :class:`torch.device` object
            instead of a plain string.
    """
    import torch

    env = os.environ.get("TORCH_DEVICE")
    if env:
        normalized = env.strip().lower()
        base = normalized.split(":", 1)[0]
        if base in _SUPPORTED_DEVICE_TYPES:
            return torch.device(normalized) if as_torch_device else normalized
    dt = device_type if device_type is not None else get_torch_device_type()
    return torch.device(dt) if as_torch_device else dt


def get_device_properties(device: int | None = None) -> dict[str, Any]:
    """Return device properties as a dictionary.

    Args:
        device: Device index.  Defaults to ``get_local_rank()``.
    """
    import torch

    device_type = get_torch_device_type()
    idx = device if device is not None else get_local_rank()
    if device_type == "cuda":
        props = torch.cuda.get_device_properties(idx)
        # torch.cuda.DeviceProperties uses `total_memory` (in bytes),
        # not `total_mem`. The old name silently AttributeError'd to a
        # plain `props.total_mem` call that doesn't exist.
        return {"name": props.name, "total_memory": props.total_memory}
    if device_type == "xpu" and hasattr(torch, "xpu"):
        props = torch.xpu.get_device_properties(idx)
        return {
            "name": props.name,
            "total_memory": getattr(props, "total_memory", -1),
        }
    return {"name": device_type, "total_memory": -1}


def get_torch_backend() -> str:
    """Return the appropriate ``torch.distributed`` backend string.

    Checks ``TORCH_BACKEND`` env, then probes hardware availability to
    select ``nccl`` / ``xccl`` / ``gloo``.
    """
    env = os.environ.get("TORCH_BACKEND")
    if env is not None:
        return env
    import torch

    if torch.cuda.is_available() and torch.distributed.is_backend_available(
        "nccl"
    ):
        return "nccl"
    if torch.xpu.is_available():
        if torch.distributed.is_backend_available("xccl"):
            return "xccl"
        return "ccl"
    return "gloo"


# Module-scope cache for the torchcomms availability probe so the import
# check runs once and any warning fires exactly once per process.
_TORCHCOMMS_CACHE: "tuple[bool, str] | None" = None


def _reset_torchcomms_cache() -> None:
    """Clear the cached torchcomms probe result (test hook)."""
    global _TORCHCOMMS_CACHE
    _TORCHCOMMS_CACHE = None


def _probe_torchcomms() -> "tuple[bool, str]":
    """Return ``(usable, reason)`` for torchcomms.

    ``usable`` is True only when torchcomms is importable AND torch exposes the
    ``use_torchcomms`` switch. ``reason`` is a human-readable explanation when
    not usable, else ``""``.

    Result is cached at module scope; call :func:`_reset_torchcomms_cache` to
    reprobe.
    """
    global _TORCHCOMMS_CACHE
    if _TORCHCOMMS_CACHE is not None:
        return _TORCHCOMMS_CACHE
    try:
        import torchcomms  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure → unavailable
        _TORCHCOMMS_CACHE = (False, f"torchcomms import failed: {exc}")
        return _TORCHCOMMS_CACHE
    import torch.distributed as _td

    if not hasattr(getattr(_td, "config", None), "use_torchcomms"):
        _TORCHCOMMS_CACHE = (
            False,
            "installed torch lacks torch.distributed.config.use_torchcomms",
        )
        return _TORCHCOMMS_CACHE
    _TORCHCOMMS_CACHE = (True, "")
    return _TORCHCOMMS_CACHE


def _torchcomms_requested() -> bool:
    """Whether ``EZPZ_USE_TORCHCOMMS`` is set to a truthy value."""
    return os.environ.get("EZPZ_USE_TORCHCOMMS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def use_torchcomms() -> bool:
    """Whether to route ``torch.distributed`` through torchcomms.

    True only when ``EZPZ_USE_TORCHCOMMS`` is truthy (``1``/``true``/``yes``/
    ``on``) AND torchcomms is usable (package importable + torch exposes
    ``torch.distributed.config.use_torchcomms``). Best-effort: returns False
    when requested-but-unavailable; use :func:`_torchcomms_unavailable_reason`
    to surface why.
    """
    if not _torchcomms_requested():
        return False
    usable, _ = _probe_torchcomms()
    return usable


def _torchcomms_unavailable_reason() -> str:
    """Reason torchcomms is unusable when requested; ``""`` otherwise."""
    if not _torchcomms_requested():
        return ""
    _, reason = _probe_torchcomms()
    return reason


def _maybe_enable_torchcomms(*, rank: int, backend: str) -> bool:
    """Enable torchcomms for the standard PG if requested + available.

    Returns True if torchcomms was activated. When requested but unavailable,
    logs a single rank-0 warning naming the reason and returns False. No-op
    (returns False) when ``EZPZ_USE_TORCHCOMMS`` is unset.
    """
    if use_torchcomms():
        import torch.distributed as _td

        _td.config.use_torchcomms = True
        if rank == 0:
            logger.info("Using torchcomms over backend=%s", backend)
        return True
    if _torchcomms_requested() and rank == 0:
        logger.warning(
            "EZPZ_USE_TORCHCOMMS set but torchcomms unavailable (%s); "
            "using standard %s backend.",
            _torchcomms_unavailable_reason(),
            backend,
        )
    return False


# ===================================================================
# Lifecycle
# ===================================================================


def _configure_rank_warnings(rank: int) -> None:
    """Suppress warnings on non-rank-0 processes to prevent duplicates.

    In distributed training, ``warnings.warn()`` calls from libraries like
    ``torch._dynamo`` fire independently on every rank, producing N identical
    copies.  This helper silences them on all ranks except 0.

    Set ``EZPZ_WARN_FROM_ALL_RANKS=1`` to keep warnings on every rank.
    """
    import logging
    import warnings

    logging.captureWarnings(True)

    warn_from_all = os.environ.get(
        "WARN_FROM_ALL_RANKS",
        os.environ.get("EZPZ_WARN_FROM_ALL_RANKS", ""),
    )
    if warn_from_all.lower() in ("1", "true", "yes", "on"):
        return

    if rank != 0:
        warnings.filterwarnings("ignore")


def _set_local_device(device_type: str, device_index: int) -> None:
    """Set the per-process current accelerator device, guarded.

    Single source of truth for "make ``torch.{cuda,xpu}.current_device()``
    return ``device_index``". Used in two places by ``setup_torch``:

      1. Pre-``init_process_group``, so the PG binds to the right device
         on XPU (without this, xccl/foreach_all_gather routes collectives
         to the wrong device and FSDP2 silently hangs).
      2. Post-``init_process_group``, only if the resolved device
         differs from what we pre-bound.

    Both call sites must use the same resolution rule (caller-provided
    ``device_id`` if any, else ``local_rank``) so the current device and
    the PG's ``device_id=`` always agree. Centralising the dispatch here
    keeps that contract intact across future refactors.

    Guards:

    * ``torch.xpu`` doesn't exist on every torch build (CPU-only, CUDA-
      only nightlies, ROCm). ``hasattr(torch, "xpu")`` short-circuits
      before ``torch.xpu.is_available()`` would AttributeError.
    * ``device_type`` not in ``{"cuda", "xpu"}`` is a no-op — there's
      nothing to bind for ``cpu``/``mps``/``hip``.
    """
    import torch

    if device_type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.set_device(device_index)
    elif device_type == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_device(device_index)


def setup_torch(
    port: str | int | None = None,
    seed: int | None = None,
    timeout: str | int | None = None,
    verbose: bool = False,
    *,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
    tensor_parallel_backend: str | None = None,
    pipeline_parallel_backend: str | None = None,
    context_parallel_backend: str | None = None,
    data_parallel_backend: str | None = None,
    device_id: int | None = None,
) -> int:
    """Initialise ``torch.distributed`` and return the global rank.

    This is the main entry point.  It:

    1. Uses MPI to discover rank / world_size / master_addr / master_port.
    2. Calls ``torch.distributed.init_process_group``.
    3. Sets the local CUDA/XPU device.
    4. Optionally seeds RNGs and initialises tensor parallelism.
    5. Prints a one-line-per-rank summary.

    Args:
        port: Fallback master port (rank 0 picks a free port otherwise).
        seed: If given, call :func:`seed_everything` with a rank-aware seed.
        timeout: ``init_process_group`` timeout in seconds (default 3600).
        verbose: Print verbose dist info on rank 0.
        tensor_parallel_size: TP degree (default 1 = disabled).
        pipeline_parallel_size: PP degree (default 1 = disabled).
        context_parallel_size: CP degree (default 1 = disabled).
        tensor_parallel_backend: Override backend for TP group.
        pipeline_parallel_backend: Override backend for PP group.
        context_parallel_backend: Override backend for CP group.
        data_parallel_backend: Override backend for DP group.
        device_id: Explicit device ordinal for ``init_process_group``.

    Returns:
        The global rank of this process.
    """
    import torch

    device_type = get_torch_device_type()
    backend = get_torch_backend()
    timeout_s = (
        int(timeout)
        if timeout is not None
        else int(os.environ.get("TORCH_DDP_TIMEOUT", 3600))
    )

    # -- Single-device fast path --
    ws_env = os.environ.get("WORLD_SIZE")
    if ws_env is not None and ws_env == "1":
        if get_rank() == 0:
            logger.info(
                "Running on a single %s, not initialising torch.distributed!",
                device_type,
            )
        _set_env_vars(rank=0, local_rank=0, world_size=1)
        return 0

    # -- Multi-device init --
    #
    # Set the per-process local device BEFORE init_process_group so
    # that whatever "current device" the process group binds to is
    # actually the same device. Without this, every rank's PG would
    # bind to the default device (e.g. xpu:0 on Aurora) at
    # construction time; later `set_device(...)` switches the current
    # device but the PG remains stuck on xpu:0. xccl/foreach_all_gather
    # then routes collectives to two different XPU queues and they
    # never meet up — FSDP2 deadlocks on the very first
    # all_gather_into_tensor.
    #
    # CUDA tends to mask this by being more forgiving about
    # current-device-at-init-time, but on XPU it's load-bearing.
    #
    # IMPORTANT: the device we set here must match whatever `_setup_ddp`
    # will bind via `device_id=` in init_process_group. That's the
    # caller-provided `device_id` if given, otherwise `local_rank`.
    # Using `LOCAL_RANK` unconditionally would reintroduce the wrong-
    # device hang on XPU whenever a caller passes an explicit
    # `device_id` (e.g. `setup_torch(device_id=2)` on local_rank 5).
    pre_local_rank = get_local_rank()
    pre_device_index = device_id if device_id is not None else pre_local_rank
    _set_local_device(get_torch_device_type(), pre_device_index)

    dsetup = _setup_ddp(
        port=str(port) if port is not None else "1234",
        timeout=timedelta(seconds=timeout_s),
        backend=backend,
        device_id=device_id,
    )

    rank = dsetup["rank"]
    world_size = dsetup["world_size"]
    local_rank = dsetup["local_rank"]

    # Re-set in case `_setup_ddp` resolved a different device than we
    # pre-bound (rare — happens if `get_local_rank()`'s pre-init guess
    # differs from what `_setup_ddp` saw in env vars). Compare against
    # the device we ACTUALLY set above, not raw `LOCAL_RANK` — those
    # diverge whenever `device_id` was passed.
    post_device_index = device_id if device_id is not None else local_rank
    if post_device_index != pre_device_index:
        _set_local_device(get_torch_device_type(), post_device_index)

    _set_env_vars(rank=rank, local_rank=local_rank, world_size=world_size)

    # -- Tensor / pipeline / context parallelism --
    if (
        tensor_parallel_size > 1
        or pipeline_parallel_size > 1
        or context_parallel_size > 1
    ):
        import ezpz.tp

        ezpz.tp.initialize_tensor_parallel(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            tensor_parallel_backend=tensor_parallel_backend,
            pipeline_parallel_backend=pipeline_parallel_backend,
            context_parallel_backend=context_parallel_backend,
            data_parallel_backend=data_parallel_backend,
            timeout=timedelta(seconds=float(timeout_s)),
        )

    # -- Seed --
    if seed is not None:
        if rank == 0:
            logger.warning("Manually specifying seed=%d", seed)
        seed_everything(seed * (rank + 1) * (local_rank + 1))

    # -- Diagnostics --
    if rank == 0:
        _ = get_dist_info(verbose=verbose)
        logger.info(
            "Using device=%s with backend=%s for distributed training.",
            device_type,
            backend,
        )

    if world_size > 1:
        barrier()

    # Only log the per-host summary from local_rank=0. At large scale
    # (96+ ranks across 8 hosts) logging from every rank produces
    # ~world_size lines of nearly-identical noise that drowns out
    # real training output. local_rank=0 gives one line per host —
    # enough to confirm topology + spot hostname/device mismatches —
    # without the per-tile spam. Override with EZPZ_LOG_ALL_RANKS=1
    # if you really do need every rank's line (debugging weird
    # local-rank assignments).
    if local_rank == 0 or os.environ.get("EZPZ_LOG_ALL_RANKS") == "1":
        logger.info(print_dist_setup(display=False))
    barrier()
    _configure_rank_warnings(rank)
    return rank


def cleanup() -> None:
    """Destroy the ``torch.distributed`` process group if active."""
    import torch.distributed

    if get_rank() == 0 and verify_wandb():
        try:
            import wandb  # noqa: F811

            if wandb.run is not None and not getattr(wandb.run, "disabled", False):
                logger.info("wandb.run=[%s](%s)", wandb.run.name, wandb.run.url)
        except Exception:
            pass
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ===================================================================
# Collectives / synchronization
# ===================================================================


def synchronize(device: str | int | None = None) -> None:
    """Block until all work on the given accelerator has finished.

    Args:
        device: Device specifier; auto-detected when ``None``.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif torch.xpu.is_available():
        torch.xpu.synchronize(device)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def barrier(
    group: Any = None,
    implementation: str | None = None,
) -> None:
    """Barrier across all ranks.

    Tries MPI first (fast), falls back to ``torch.distributed.barrier``.

    Args:
        group: Optional ``torch.distributed`` process group.
        implementation: Force ``"mpi"`` or ``"torch"``.
    """
    if implementation is not None and implementation.lower() not in {
        "mpi",
        "torch",
    }:
        raise ValueError(
            f"Unsupported barrier implementation: {implementation}"
        )
    if implementation is None or implementation.lower() == "mpi":
        try:
            _get_mpi_comm().barrier()
            return
        except Exception:
            if get_rank() == 0:
                logger.warning(
                    "MPI barrier failed; falling back to torch.distributed.barrier"
                )
    # torch fallback
    import torch.distributed

    if torch.distributed.is_initialized():
        kwargs: dict[str, Any] = {}
        if group is not None:
            kwargs["group"] = group
        torch.distributed.barrier(**kwargs)


def broadcast(obj: Any, root: int = 0) -> Any:
    """Broadcast a picklable *obj* from *root* to all ranks via MPI.

    Args:
        obj: Payload to broadcast.
        root: Originating rank.

    Returns:
        The broadcast payload on every rank.
    """
    return _get_mpi_comm().bcast(obj, root=root)


def all_reduce(
    obj: Any,
    op: Any = None,
    implementation: str | None = None,
) -> Any:
    """All-reduce *obj* across all ranks.

    Args:
        obj: Numeric value to reduce.
        op: Reduction operation (defaults to ``MPI.SUM``).
        implementation: ``"mpi"`` (default) or ``"torch"``.

    Returns:
        The reduced value.
    """
    impl = (implementation or "mpi").lower()
    if impl == "mpi":
        from mpi4py import MPI

        op = MPI.SUM if op is None else op
        return _get_mpi_comm().allreduce(obj, op=op)
    if impl in {"torch", "pytorch", "pt"}:
        import torch
        import torch.distributed as tdist

        op = tdist.ReduceOp.SUM if op is None else op
        tensor = torch.tensor(obj)
        tdist.all_reduce(tensor, op=op)
        return tensor.item()
    raise ValueError(
        f"Unsupported all_reduce implementation: {implementation}"
    )


# ===================================================================
# Model wrapping
# ===================================================================


FSDP_SHARDING_STRATEGIES = {
    "full-shard": True,
    "shard-grad-op": False,
    "no-shard": None,
    "hybrid-shard": "hybrid",
}
"""Map CLI sharding strategy names to ``reshard_after_forward`` values.

Used by :func:`wrap_model` and the example CLI parsers::

    --fsdp-sharding-strategy full-shard   # reshard_after_forward=True  (ZeRO-3)
    --fsdp-sharding-strategy shard-grad-op # reshard_after_forward=False (ZeRO-2)
    --fsdp-sharding-strategy hybrid-shard  # reshard to intra-node size
    --fsdp-sharding-strategy no-shard      # fall back to DDP
"""


def resolve_fsdp_strategy(
    strategy: str,
) -> bool | int | None:
    """Convert a CLI sharding strategy name to a ``reshard_after_forward`` value.

    Returns ``None`` when the strategy is ``"no-shard"`` (caller should
    use DDP instead).
    """
    if strategy not in FSDP_SHARDING_STRATEGIES:
        raise ValueError(
            f"Unknown FSDP sharding strategy {strategy!r}. "
            f"Choose from: {', '.join(FSDP_SHARDING_STRATEGIES)}"
        )
    val = FSDP_SHARDING_STRATEGIES[strategy]
    if val == "hybrid":
        return get_gpus_per_node()
    return val


def wrap_model(
    model: torch.nn.Module,
    use_fsdp: bool = True,
    dtype: str = "bfloat16",
    device_mesh: Any = None,
    reshard_after_forward: bool | int = True,
    device_id: torch.device | int | None = None,
) -> torch.nn.Module:
    """Wrap *model* with DDP or FSDP for distributed training.

    Args:
        model: Model to wrap.
        use_fsdp: Use FSDP when ``True``, DDP when ``False``.
        dtype: Mixed-precision parameter dtype for FSDP (e.g. ``"bf16"``).
        device_id: Explicit device ordinal for FSDP (legacy FSDP1 only).
        device_mesh: Optional :class:`torch.distributed.device_mesh.DeviceMesh`.
        reshard_after_forward: Controls parameter lifetime after forward:

            - ``True`` (default): reshard after forward (FULL_SHARD / ZeRO-3).
            - ``False``: keep unsharded (SHARD_GRAD_OP / ZeRO-2).
            - ``int``: reshard to this world-size (HYBRID_SHARD).

    Returns:
        The wrapped model.  If ``world_size <= 1`` the original model is
        returned unchanged.
    """
    import torch

    ws = get_world_size()
    if ws <= 1:
        logger.warning(
            "%s requested but world_size=%d; returning unwrapped model.",
            "FSDP" if use_fsdp else "DDP",
            ws,
        )
        return model
    if get_rank() == 0:
        logger.info("Wrapping model with %s", "fsdp" if use_fsdp else "ddp")
    if not use_fsdp:
        return wrap_model_for_ddp(model)
    device_type = get_torch_device_type()
    if device_type in ("cpu", "mps"):
        logger.warning(
            "FSDP is not supported on %s devices; falling back to DDP.",
            device_type,
        )
        return wrap_model_for_ddp(model)
    # Auto-create a 1D DeviceMesh when none is provided so FSDP2
    # (fully_shard) is the default sharding strategy.
    if device_mesh is None:
        device_mesh = init_device_mesh_safe(device_type, (ws,))
    return _wrap_fsdp2(
        model,
        dtype=dtype,
        device_mesh=device_mesh,
        reshard_after_forward=reshard_after_forward,
    )


def wrap_model_for_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap *model* with :class:`~torch.nn.parallel.DistributedDataParallel`.

    Args:
        model: Model to wrap (should already be on the correct device).

    Returns:
        A DDP-wrapped model.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_type = get_torch_device_type()
    local_rank = get_local_rank()
    if device_type in {"cuda", "xpu"}:
        return DDP(model, device_ids=[local_rank])
    return DDP(model)


def wrap_model_for_fsdp(
    model: torch.nn.Module,
    dtype: str = "bfloat16",
    device_id: int | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Wrap *model* with FSDP 1 and mixed precision.

    Args:
        model: Model to wrap (should already be on the correct device).
        dtype: Mixed-precision parameter dtype (e.g. ``"bf16"``).
        device_id: Explicit device ordinal for FSDP.
        **kwargs: Extra keyword arguments forwarded to the FSDP constructor.

    Returns:
        An FSDP-wrapped model.
    """
    return _wrap_fsdp(model, dtype=dtype, device_id=device_id, **kwargs)


def wrap_model_for_fsdp2(
    model: torch.nn.Module,
    dtype: str = "bfloat16",
    device_mesh: Any = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Wrap *model* with FSDP2 (per-module ``fully_shard``).

    .. note:: **Experimental** -- the FSDP2 API is subject to change in
       future PyTorch releases.

    Args:
        model: Model to wrap.
        dtype: Mixed-precision parameter dtype (e.g. ``"bf16"``).
        device_mesh: Optional :class:`torch.distributed.device_mesh.DeviceMesh`.
        **kwargs: Extra keyword arguments forwarded to ``fully_shard``.

    Returns:
        The model after applying ``fully_shard`` to every sub-module.
    """
    return _wrap_fsdp2(model, dtype=dtype, device_mesh=device_mesh, **kwargs)


# ===================================================================
# Diagnostics
# ===================================================================


def get_hostname() -> str:
    """Return the hostname of the current machine (lowercased)."""
    try:
        name = socket.gethostname()
        if name:
            try:
                return socket.gethostbyaddr(name)[0].strip().lower()
            except OSError:
                return name.strip().lower()
    except Exception:
        pass
    for var in ("HOSTNAME", "HOST"):
        val = os.environ.get(var)
        if val is not None and val != "":
            return val.strip().lower()
    try:
        import platform

        node = platform.node()
        if node:
            return node.strip().lower()
    except Exception:
        pass
    return "localhost"


def get_machine(hostname: str | None = None) -> str:
    """Identify the ALCF / HPC machine from its hostname prefix.

    Args:
        hostname: Override; auto-detected when ``None``.

    Returns:
        A human-readable machine name (e.g. ``"Polaris"``, ``"Aurora"``).
    """
    if hostname is None:
        hostname = get_hostname()
    _PREFIX_MAP = (
        ("frontier", "Frontier"),
        ("sophia", "Sophia"),
        ("theta", "ThetaGPU"),
        ("sunspot", "SunSpot"),
        ("x1", "SunSpot"),
        ("aurora", "Aurora"),
        ("x4", "Aurora"),
        ("login", "Perlmutter"),
        ("nid", "Perlmutter"),
    )
    for prefix, name in _PREFIX_MAP:
        if hostname.startswith(prefix):
            return name
    if hostname.startswith("x3"):
        return "Sirius" if "sirius" in hostname else "Polaris"
    return hostname


def get_dist_info(
    verbose: bool | None = None,
    hostfile: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Return a dictionary summarising the distributed environment.

    Args:
        verbose: If ``True``, log the info dict as formatted JSON.
        hostfile: Explicit hostfile path.
    """
    import sys

    from ezpz.configs import get_scheduler

    hfp = (
        get_hostfile_with_fallback(hostfile)
        if hostfile is None
        else Path(hostfile)
    )
    if hfp is not None and Path(hfp).is_file():
        hosts = get_nodes_from_hostfile(hfp)
        hostfile_path = Path(hfp).resolve().as_posix()
    else:
        hosts = [get_hostname()]
        hostfile_path = str(hfp) if hfp is not None else ""
    num_nodes = len(hosts)
    gpus = get_gpus_per_node()
    info: dict[str, Any] = {}
    info.update(
        {
            "DEVICE": get_torch_device(),
            "DEVICE_ID": f"{get_torch_device()}:{get_local_rank()}",
            "DISTRIBUTED_BACKEND": get_torch_backend(),
            "TORCHCOMMS": use_torchcomms(),
            "GPUS_PER_NODE": gpus,
            "HOSTS": str(hosts),
            "HOSTFILE": hostfile_path,
            "HOSTNAME": get_hostname(),
            "LOCAL_RANK": get_local_rank(),
            "MACHINE": get_machine(),
            "NUM_NODES": num_nodes,
            "NGPUS": num_nodes * gpus,
            "NGPUS_AVAILABLE": get_world_size_total(),
            "NODE_ID": get_node_index(),
            "RANK": get_rank(),
            "SCHEDULER": get_scheduler(),
            "WORLD_SIZE_TOTAL": get_world_size_total(),
            "WORLD_SIZE_IN_USE": get_world_size_in_use(),
            "world_size": get_world_size(),
            "EZPZ_RUN_COMMAND": os.environ.get(
                "EZPZ_RUN_COMMAND", sys.argv[0]
            ),
        }
    )
    if verbose:
        import json

        logger.info("DistInfo=%s", json.dumps(info, indent=4, sort_keys=True))
    return info


def query_environment() -> dict[str, int]:
    """Return ``{world_size, rank, local_rank}`` from env vars or MPI."""
    ws = os.environ.get("WORLD_SIZE")
    r = os.environ.get("RANK")
    lr = os.environ.get("LOCAL_RANK")
    if ws is not None and r is not None and lr is not None:
        return {"world_size": int(ws), "rank": int(r), "local_rank": int(lr)}
    return {
        "world_size": get_world_size(),
        "rank": get_rank(),
        "local_rank": get_local_rank(),
    }


def print_dist_setup(
    hostfile: str | os.PathLike | None = None,
    display: bool = True,
) -> str:
    """Build (and optionally log) a one-line-per-rank summary string.

    Args:
        hostfile: Explicit hostfile path.
        display: If ``True``, emit the string via :func:`logger.info`.

    Returns:
        The formatted summary string.
    """
    rank = get_rank()
    world_size = get_world_size(in_use=True)
    local_rank = get_local_rank()
    gpn = max(get_gpus_per_node(), 1)
    num_nodes = max(world_size // gpn, 1)
    node = get_node_index()
    device = get_torch_device_type()
    hn = socket.gethostname()

    rw = len(str(max(0, world_size - 1)))
    lw = len(str(max(0, gpn - 1)))
    nnw = len(str(max(0, num_nodes - 1)))
    nw = nnw

    parts = [
        f"['{hn}']",
        f"[{device=}]",
        f"[node={node:>0{nw}d}/{num_nodes - 1:<0{nnw}d}]",
        f"[local_rank={local_rank:>0{lw}d}/{gpn - 1:<0{lw}d}]",
        f"[rank={rank:>0{rw}d}/{world_size - 1:<0{rw}d}]",
    ]
    dist_str = "".join(parts)
    if display:
        logger.info(dist_str)
    if rank == 0:
        wst = get_world_size(total=True)
        logger.warning(
            'Using [%d / %d] available "%s" devices !!',
            world_size,
            wst,
            device,
        )
    return dist_str


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: The random seed.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed)


def log_dict_as_bulleted_list(d: dict, name: str | None = None) -> None:
    """Log a dictionary as a bulleted list.

    Args:
        d: Dictionary to format.
        name: Optional label for the header.
    """
    tag = name or getattr(d, "__qualname__", "dict")
    lines = [f"[{tag}]:"] + [f"  - {k}={v}" for k, v in d.items()]
    logger.info("\n\n%s\n", "\n".join(lines))


# ===================================================================
# Timing
# ===================================================================


def timeitlogit(
    rank: int | None = None,
    record: bool = True,
    verbose: bool = False,
    prefix: str | None = None,
) -> Callable:
    """Decorator factory to time a function, optionally logging to wandb.

    Args:
        rank: Rank whose logger emits messages (defaults to :func:`get_rank`).
        record: Whether to ``wandb.log`` the timing.
        verbose: Whether to log to stdout on the selected rank.
        prefix: Metric prefix for wandb (default ``"timeit"``).

    Returns:
        A decorator that wraps the target function.

    Example::

        @timeitlogit(rank=0, verbose=True)
        def train_step(batch): ...
    """
    _rank = rank if rank is not None else get_rank()
    _prefix = prefix or "timeit"

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            fname = getattr(
                func, "__qualname__", getattr(func, "__name__", "unknown")
            )
            if record:
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log({f"{_prefix}/{fname}": dt}, commit=False)
                except Exception:
                    pass
            if verbose and _rank == 0:
                arg_str = ", ".join(map(str, args))
                kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                inner = ", ".join(filter(None, [arg_str, kw_str]))
                logger.info("%s(%s) took %.4f s", fname, inner, dt)
            return result

        return wrapper

    return decorator


# ===================================================================
# Wandb
# ===================================================================


def verify_wandb() -> bool:
    """Return ``True`` if wandb is importable, enabled, and usable.

    "Usable" means one of:

    1. ``WANDB_MODE=offline`` is set — offline runs don't need
       network or credentials; they write to a local
       ``wandb/offline-run-*`` directory and can be synced later.
       This is the documented compute-node workflow (see
       ``docs/troubleshooting.md`` + ``doctor.py``'s permissive
       handling of offline mode).
    2. An API key is reachable (``$WANDB_API_KEY``, ``wandb.api.api_key``,
       or a ``~/.netrc`` entry for ``api.wandb.ai``).

    Returns ``False`` when wandb is uninstalled, ``WANDB_DISABLED`` is
    truthy, ``WANDB_MODE=disabled``, or neither offline mode nor any
    credential source is configured.
    """
    rank = get_rank()
    try:
        import wandb
    except Exception:
        if rank == 0:
            logger.warning(
                "Unable to import wandb; install with `pip install wandb`"
            )
        return False
    if os.environ.get("WANDB_DISABLED"):
        return False
    wm = os.environ.get("WANDB_MODE", "").lower()
    if wm == "disabled":
        return False
    # Offline mode is fine without credentials — wandb.init(mode="offline")
    # writes locally and never touches the network. Return True so callers
    # (esp. setup_wandb) don't silently no-op on compute nodes that use
    # the offline-then-sync workflow.
    if wm == "offline":
        return True
    if (
        wandb.api.api_key is not None
        or os.environ.get("WANDB_API_KEY") is not None
    ):
        return True
    # Last resort: check ~/.netrc
    try:
        import netrc as _netrc

        netrc_path = Path(os.path.expanduser("~/.netrc"))
        if netrc_path.is_file():
            auth = _netrc.netrc(netrc_path).authenticators("api.wandb.ai")
            return bool(auth)
    except Exception:
        pass
    return False


def _get_ezpz_git_sha() -> str | None:
    """Return the short SHA of the ezpz checkout, or ``None`` if not a git repo.

    Useful for distinguishing wandb runs from feature branches vs main
    when ``ezpz_version`` alone is ambiguous (e.g. dev installs from a
    branch that has not been version-bumped yet).

    Returns ``None`` on any failure mode — pip-installed packages aren't
    git repos, ``git`` may not be on PATH, the call could time out, etc.
    Logging is silenced; this helper is best-effort by design.
    """
    try:
        import subprocess  # noqa: PLC0415 — only needed here

        ezpz_dir = Path(__file__).resolve().parent
        proc = subprocess.run(
            ["git", "-C", str(ezpz_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        sha = proc.stdout.strip()
        return sha or None
    except Exception:
        return None


def _build_wandb_settings(
    *,
    wandb: Any,
    init_timeout: int | float | None,
    start_method: str | None,
) -> Any:
    """Construct a ``wandb.Settings`` object without spurious deprecation noise.

    Two cleanups vs the previous inline form:

    1. ``start_method`` is only forwarded when the caller explicitly
       requested one. Default-passing ``start_method="fork"`` triggers
       wandb's "start_method is deprecated and will be removed in a
       future version of wandb. This setting is currently
       non-functional and safely ignored" warning on every run; the
       intent (defend against the wrong default) is now wandb's
       responsibility.
    2. ``init_timeout`` defaults to 60 s only when the caller didn't
       provide one — same as before, just hoisted out.
    """
    kwargs: dict[str, Any] = {
        "init_timeout": init_timeout if init_timeout is not None else 60,
    }
    if start_method is not None:
        kwargs["start_method"] = start_method
    return wandb.Settings(**kwargs)


def setup_wandb(
    project_name: str | None = None,
    entity: str | None = None,
    config: dict[str, Any] | None = None,
    outdir: str | os.PathLike | None = None,
    project: str | None = None,
    dir: str | os.PathLike | None = None,
    id: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    tags: Sequence[str] | None = None,
    config_exclude_keys: list[str] | None = None,
    config_include_keys: list[str] | None = None,
    allow_val_change: bool | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: Literal["online", "offline", "disabled", "shared"] | None = None,
    force: bool = False,
    reinit: bool | str | None = None,
    resume: bool | str | None = None,
    resume_from: str | None = None,
    fork_from: str | None = None,
    save_code: bool | None = None,
    init_timeout: int | float | None = None,
    start_method: Literal["fork", "spawn", "thread", "process"] | None = None,
    tensorboard: bool | None = None,
    sync_tensorboard: bool | None = None,
    monitor_gym: bool | None = None,
    settings: dict[str, Any] | None = None,
    **kwargs,
) -> Any:
    """Initialise a wandb run (rank 0 only — non-zero ranks return ``None``).

    Most parameters are forwarded directly to :func:`wandb.init`.  See
    the `wandb docs <https://docs.wandb.ai/ref/python/init/>`_ for
    details.

    Returns:
        The :obj:`wandb.run` object, ``None`` if called from a non-zero
        rank, or ``None`` if wandb is unavailable.

    .. note::
       Returns ``None`` on every rank other than 0. Previously
       non-zero ranks got a ``mode="disabled"`` wandb.run back — that
       still meant verify_wandb(), wandb.init(), and full
       run.config.update() ran on every rank, which on a 96-rank job
       produced 96 dummy runs and a wall of "Setting up wandb from
       rank=N" log spam. Callers that need a no-op tracker on
       non-zero ranks should test for ``None`` and use
       :class:`ezpz.tracker.NullTracker` (or just ignore the return —
       ``log()`` calls against ``None`` should be guarded by the
       caller anyway).
    """
    # Hard rank gate: non-zero ranks skip all wandb work entirely.
    # verify_wandb() and wandb.init() each take real time and produce
    # log spam; on a 96-rank job that's 95x wasted work + a 96x
    # multiplier on every log line in this function.
    rank = get_rank()
    if rank != 0:
        return None

    import wandb

    if not verify_wandb():
        logger.warning("verify_wandb() failed; not initialising run")
        return None

    outdir_str = Path(outdir).as_posix() if outdir else os.getcwd()

    # Resolve project name
    _project = project or project_name
    if _project is None:
        _project = os.environ.get(
            "WB_PROJECT",
            os.environ.get("WANDB_PROJECT", os.environ.get("WB_PROJECT_NAME")),
        )
    if _project is None:
        import sys

        frame = sys._getframe().f_back
        if frame is not None:
            fp = Path(frame.f_code.co_filename)
            _project = f"{fp.parent.stem}.{fp.stem}"

    # Resolve mode
    _mode = _resolve_wandb_mode(mode)

    logger.info("Setting up wandb from rank=%d", rank)
    logger.info("Using WB_PROJECT=%s", _project)

    try:
        run = wandb.init(
            entity=entity,
            project=_project,
            dir=str(dir) if dir is not None else outdir_str,
            id=id,
            name=name,
            notes=notes,
            tags=tags,
            config_exclude_keys=config_exclude_keys,
            config_include_keys=config_include_keys,
            allow_val_change=allow_val_change,
            group=group,
            job_type=job_type,
            mode=_mode,
            force=force,
            reinit=reinit,
            resume=resume,
            resume_from=resume_from,
            fork_from=fork_from,
            save_code=save_code,
            tensorboard=tensorboard if tensorboard is not None else False,
            sync_tensorboard=sync_tensorboard
            if sync_tensorboard is not None
            else False,
            monitor_gym=monitor_gym,
            settings=(
                settings
                if settings is not None
                else _build_wandb_settings(
                    wandb=wandb,
                    init_timeout=init_timeout,
                    start_method=start_method,
                )
            ),
            **kwargs,
        )
        if run is not None:
            logger.info("wandb.run=[%s](%s)", run.name, run.url)
            import sys  # noqa: PLC0415
            import torch  # noqa: PLC0415

            import ezpz  # noqa: PLC0415
            from ezpz.configs import get_scheduler  # noqa: PLC0415

            now = datetime.datetime.now()

            # Best-effort resolution of the active scheduler jobid.
            # ezpz.launch.get_active_jobid imports ezpz.pbs / ezpz.slurm
            # lazily and returns None when no job is detected. Wrapped
            # in try/except because the launch module pulls a non-
            # trivial chain on first import — any failure here should
            # NOT block the wandb run from being created.
            jobid: str | None = None
            try:
                from ezpz.launch import (
                    get_active_jobid,
                )  # noqa: PLC0415

                jobid = get_active_jobid()
            except Exception:
                pass

            # num_nodes / gpus_per_node have their own getters that
            # already swallow failures internally (return 1 / fall
            # back). Safe to call directly.
            run.config.update(
                {
                    # "DIST_INFO": get_dist_info(),
                    # --- existing fields (unchanged) ---
                    "hostname": get_hostname(),
                    "pytorch_backend": get_torch_backend(),
                    "torch_version": torch.__version__,
                    "world_size": get_world_size(),
                    "ezpz_version": ezpz.__version__,
                    "machine": get_machine(),
                    "working_directory": os.getcwd(),
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "tstamp": now.isoformat(),
                    # --- new dimensions for filtering / grouping ---
                    # Pivot from a wandb run → the cluster job that
                    # ran it. None when not inside a PBS/SLURM job.
                    "jobid": jobid,
                    # "pbs" / "slurm" / "" — useful when you have
                    # runs from both systems in the same project.
                    "scheduler": get_scheduler(),
                    # Distinct from world_size: same world_size can be
                    # 8x8 or 4x16, lets you separate the two.
                    "num_nodes": get_num_nodes(),
                    "ranks_per_node": get_gpus_per_node(),
                    # "cuda" / "xpu" / "mps" / "cpu" — at-a-glance
                    # distinction between Aurora vs NVIDIA vs CPU runs.
                    "device_type": get_torch_device_type(),
                    # --- debugging / postmortems ---
                    "python_version": sys.version.split()[0],
                    # None when ezpz is a pip-install, not a git
                    # checkout. Disambiguates dev branches sharing the
                    # same ezpz_version.
                    "ezpz_git_sha": _get_ezpz_git_sha(),
                }
            )
            if config is not None:
                run.config.update({"config": config})
        return wandb.run
    except Exception as exc:
        logger.exception("wandb.init() failed from rank=%d: %s", rank, exc)
        logger.warning("Continuing without wandb logging.")
        return None


def setup_mlflow(
    project_name: str | None = None,
    config: dict[str, Any] | None = None,
    outdir: str | os.PathLike | None = None,
    **kwargs: Any,
) -> Any:
    """Initialise an MLflow run (rank 0 only logs, others return ``None``).

    Convenience wrapper around :class:`~ezpz.tracker.MLflowBackend` that
    mirrors :func:`setup_wandb`.  Handles dotenv loading, auth, experiment
    name resolution, and system-param logging automatically.

    Args:
        project_name: MLflow experiment name.  Falls back to
            ``MLFLOW_EXPERIMENT_NAME``, then wandb project env vars,
            then a script-derived default.
        config: Run-level config dict logged as MLflow params.
        outdir: Artifact output directory.
        **kwargs: Forwarded to ``mlflow.start_run``.

    Returns:
        The ``mlflow.ActiveRun`` object, or ``None`` if MLflow is
        unavailable or the current rank is not 0.
    """
    try:
        from ezpz.tracker import MLflowBackend

        backend = MLflowBackend(
            project_name=project_name,
            config=config,
            outdir=outdir,
            **kwargs,
        )
        return backend.run
    except ImportError:
        logger.warning("mlflow is not installed; skipping MLflow setup.")
        return None
    except Exception as exc:
        logger.warning("setup_mlflow() failed: %s — continuing without MLflow", exc)
        return None


# ===================================================================
# Hostfile helpers
# ===================================================================


def get_nodes_from_hostfile(hostfile: str | os.PathLike) -> list[str]:
    """Read hostnames from *hostfile*, one per line.

    Args:
        hostfile: Path to the hostfile.

    Returns:
        List of hostnames.
    """
    fpath = Path(hostfile)
    if not fpath.is_file():
        return [get_hostname()]
    with fpath.open("r") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def get_hostfile_with_fallback(
    hostfile: str | os.PathLike | None = None,
) -> Path:
    """Locate (or create) a usable hostfile.

    Checks PBS, SLURM, and environment variables.  As a last resort,
    writes ``localhost`` to a file in the current directory.

    Args:
        hostfile: Explicit path; auto-detected when ``None``.

    Returns:
        :class:`Path` to the hostfile.
    """
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler()

    if scheduler.lower() == "slurm":
        return _make_hostfile_from_slurm()

    if hostfile is not None:
        hfp = Path(hostfile)
        if hfp.is_file():
            return hfp

    # Try standard env vars
    for var in ("PBS_NODEFILE", "HOSTFILE"):
        val = os.environ.get(var)
        if val and Path(val).is_file():
            return Path(val)

    # PBS without env var
    if scheduler == "PBS":
        try:
            import ezpz.pbs

            nodefile = ezpz.pbs.get_pbs_nodefile()
            if nodefile is not None:
                return Path(nodefile)
        except Exception:
            pass

    # Fallback: write localhost
    hfp = Path(os.getcwd()) / "hostfile"
    hfp.touch(exist_ok=True)
    write_localhost_to_hostfile(hfp)
    return hfp


def write_localhost_to_hostfile(hostfile: str | os.PathLike) -> None:
    """Write the current hostname to *hostfile* (rank 0 only).

    Args:
        hostfile: Path to write to.
    """
    if get_rank() == 0:
        hn = get_hostname()
        Path(hostfile).write_text(hn)


def write_hostfile_from_list_of_hosts(
    hosts: Sequence[str],
    hostfile: str | os.PathLike,
) -> Path:
    """Write a hostfile from a list of hostnames.

    Args:
        hosts: Sequence of hostnames to write.
        hostfile: Path to write to.
    """
    hfp = Path(hostfile)
    hfp.parent.mkdir(parents=True, exist_ok=True)
    hfp.write_text("\n".join(hosts) + "\n")
    return hfp


# ===================================================================
# Private helpers
# ===================================================================


def _set_env_vars(rank: int, local_rank: int, world_size: int) -> None:
    """Set standard distributed env vars."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def _setup_ddp(
    port: str = "1234",
    timeout: timedelta | None = None,
    backend: str | None = None,
    device_id: int | None = None,
) -> dict[str, int]:
    """Bootstrap ``torch.distributed`` using MPI for address/port discovery.

    Returns:
        Dict with ``rank``, ``world_size``, ``local_rank`` keys.
    """
    import torch
    import torch.distributed

    timeout = timeout or timedelta(seconds=3600)
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    backend = backend or get_torch_backend()

    # Env vars needed by torch.distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(get_gpus_per_node())

    if torch.distributed.is_initialized():
        if rank == 0:
            logger.info("torch.distributed already initialised, skipping.")
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
        }

    # If MASTER_ADDR and MASTER_PORT are already set (e.g. by torchrun),
    # use them directly instead of re-deriving via MPI broadcast.
    env_master_addr = os.environ.get("MASTER_ADDR")
    env_master_port = os.environ.get("MASTER_PORT")
    if env_master_addr and env_master_port:
        master_addr = env_master_addr
        master_port = env_master_port
    else:
        # Rank 0 determines master_addr and port, then broadcasts
        master_addr = socket.gethostname() if rank == 0 else None
        if master_addr is not None:
            machine = get_machine().lower()
            if machine in {"aurora", "polaris", "sirius"}:
                master_addr = f"{master_addr}.hsn.cm.{machine}.alcf.anl.gov"
            elif machine == "sophia":
                master_addr = f"{master_addr}.lab.alcf.anl.gov"

        free_port = str(_get_free_port()) if rank == 0 else None
        master_port = (
            os.environ.get("MASTER_PORT", free_port) if rank == 0 else None
        )

        master_addr = broadcast(master_addr, root=0)
        master_port = broadcast(master_port, root=0)

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    if rank == 0:
        logger.info(
            "init_process_group: master_addr=%s, master_port=%s, "
            "world_size=%d, rank=%d, backend=%s, timeout=%s",
            master_addr,
            master_port,
            world_size,
            rank,
            backend,
            timeout,
        )

    _maybe_enable_torchcomms(rank=rank, backend=backend)

    if not torch.distributed.is_initialized():
        init_kwargs: dict[str, Any] = {
            "backend": backend,
            "timeout": timeout,
            "rank": rank,
            "world_size": world_size,
            "init_method": "env://",
        }
        # Resolve the device the PG should bind to. Two principles:
        #
        #   1. An explicit `device_id` arg (int OR torch.device) wins,
        #      but we must honor get_torch_device_type() to pick the
        #      right device family. Hardcoding "cuda:N" for an int
        #      breaks XPU/MPS/HIP callers — same bug class as the
        #      barrier() warning this code originally fixed.
        #
        #   2. Auto-detect binds for CUDA AND XPU. Caller must have
        #      already called `set_device(local_rank)` BEFORE
        #      _setup_ddp (setup_torch does this) so that "current
        #      device" agrees with the device_id we're about to pass.
        #
        # XPU note: an earlier comment here said we MUST NOT pass
        # device_id on xpu because xccl didn't support split_group
        # (which DeviceMesh._unflatten needs when PGs are
        # device-bound). That was incorrect for FSDP2: without
        # device-bound PGs, foreach_all_gather routes some ranks'
        # collectives to xpu:0 (the current device at PG-construction
        # time, when set_device(local_rank) hadn't yet run) and
        # others to xpu:LOCAL_RANK (post-set_device). They never meet
        # up → silent deadlock on the first all_gather_into_tensor.
        # Caught on Aurora torchtitan job 8518207 — see PR for the
        # py-spy stack trace. If DeviceMesh._unflatten regresses on a
        # newer xccl, we'll see a loud failure (not a silent hang) and
        # can revisit.
        device_type = get_torch_device_type()
        resolved_device: Any = None
        if device_id is not None:
            if isinstance(device_id, int):
                resolved_device = torch.device(f"{device_type}:{device_id}")
            else:
                # Caller passed a torch.device — honor verbatim.
                resolved_device = device_id
        elif device_type in ("cuda", "xpu"):
            resolved_device = torch.device(f"{device_type}:{local_rank}")
        if resolved_device is not None:
            init_kwargs["device_id"] = resolved_device
        torch.distributed.init_process_group(**init_kwargs)

    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def init_device_mesh_safe(
    device_type: str,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim_names: tuple[str, ...] | None = None,
) -> Any:
    """Drop-in replacement for ``torch.distributed.init_device_mesh``.

    Works around xccl's missing ``ProcessGroup.split_group`` support.

    For FSDP2 to route ``foreach_all_gather`` correctly on XPU,
    ``_setup_ddp`` binds the default PG to a device by passing
    ``device_id=`` to ``init_process_group``. Torch then sees
    ``default_group.bound_device_id is not None`` and, inside
    ``DeviceMesh._init_one_process_group``, prefers
    ``split_group(parent_pg, ...)`` over ``new_group(ranks, ...)``.
    On the current xccl backend ``parent_backend.supports_splitting``
    is ``False``, so ``split_group`` raises:

        RuntimeError: No backend for the parent process group or its
                      backend does not support splitting

    We temporarily clear ``default_group.bound_device_id`` for the
    duration of the ``init_device_mesh`` call so torch takes the
    ``new_group`` fallback (which xccl supports), then restore it so
    FSDP2's per-device PG resolution still works.

    No-op on non-xpu devices and when no default PG exists yet.
    """
    import torch
    from torch.distributed.device_mesh import init_device_mesh as _imd

    default_pg = None
    saved: Any = None
    needs_workaround = False
    if device_type == "xpu" and torch.distributed.is_initialized():
        try:
            default_pg = torch.distributed.distributed_c10d._get_default_group()
            saved = getattr(default_pg, "bound_device_id", None)
            if saved is not None:
                default_pg.bound_device_id = None  # type: ignore[attr-defined]
                needs_workaround = True
        except (AttributeError, RuntimeError):
            needs_workaround = False
    try:
        return _imd(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)
    finally:
        if needs_workaround and default_pg is not None:
            try:
                default_pg.bound_device_id = saved  # type: ignore[attr-defined]
            except AttributeError:
                pass


def _init_deepspeed(timeout: int = 3600) -> None:
    """Initialise DeepSpeed distributed backend."""
    try:
        import deepspeed
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed is not installed. Install with `pip install deepspeed`"
        ) from exc
    rank = get_rank()
    world_size = get_world_size()
    os.environ["WORLD_SIZE"] = str(world_size)
    deepspeed.init_distributed(
        dist_backend=None,
        auto_mpi_discovery=True,
        verbose=True,
        timeout=timedelta(seconds=timeout),
        rank=rank,
        world_size=world_size,
    )


def _init_horovod() -> dict[str, int]:
    """Initialise Horovod and return rank/world_size/local_rank."""
    import horovod.torch as hvd

    if not hvd.is_initialized():
        hvd.init()
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
    return {
        "rank": hvd.rank(),
        "world_size": hvd.size(),
        "local_rank": hvd.local_rank(),
    }


def _wrap_fsdp(
    model: torch.nn.Module,
    dtype: str = "bfloat16",
    device_id: torch.device | int | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Wrap *model* with FSDP 1 and mixed precision."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    import torch

    dtypes = _ensure_dtype_map()
    if get_rank() == 0:
        logger.info("Wrapping model with FSDP + %s", dtype)
    return FSDP(
        model,
        device_id=device_id,
        mixed_precision=MixedPrecision(
            param_dtype=dtypes[dtype],
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        ),
        **kwargs,
    )


def _wrap_fsdp2(
    model: torch.nn.Module,
    dtype: str = "bfloat16",
    device_mesh: Any = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Wrap *model* with FSDP2 (per-module ``fully_shard``)."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    import torch

    dtypes = _ensure_dtype_map()
    if get_rank() == 0:
        logger.info("Wrapping model with FSDP2 + %s", dtype)
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=dtypes[dtype],
            reduce_dtype=torch.float32,
        ),
        **kwargs,
    }
    # Skip container modules (ModuleList, ModuleDict) that don't implement
    # forward — fully_shard raises ValueError on these.
    _CONTAINERS = (torch.nn.ModuleList, torch.nn.ModuleDict)
    for module in model.children():
        if not isinstance(module, _CONTAINERS):
            fully_shard(module, mesh=device_mesh, **fsdp_kwargs)
    return fully_shard(model, mesh=device_mesh, **fsdp_kwargs)


def _get_free_port() -> int:
    """Find and return an available TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_hostfile_from_slurm() -> Path:
    """Parse ``SLURM_NODELIST`` and write a hostfile.

    Handles the common ``SLURM_NODELIST`` formats:

    * Single node: ``node001``
    * Bracket list: ``node[001,002,003]``
    * Bracket range: ``node[001-004]``
    * Mixed: ``node[001-003,007,010-012]``
    """
    nodes_str = os.environ.get("SLURM_NODELIST")
    if nodes_str is None:
        raise RuntimeError(
            "SLURM_NODELIST not set but scheduler=slurm detected"
        )
    nodelist = _expand_slurm_nodelist(nodes_str)
    outfile = Path(os.getcwd()) / "hostfile"
    with outfile.open("w") as f:
        for node in nodelist:
            f.write(f"{node}\n")
    return outfile


def _expand_slurm_nodelist(nodelist_str: str) -> list[str]:
    """Expand a ``SLURM_NODELIST`` string into individual hostnames.

    Examples::

        >>> _expand_slurm_nodelist("node001")
        ['node001']
        >>> _expand_slurm_nodelist("node[001,003]")
        ['node001', 'node003']
        >>> _expand_slurm_nodelist("node[001-003]")
        ['node001', 'node002', 'node003']
        >>> _expand_slurm_nodelist("node[001-003,007,010-012]")
        ['node001', 'node002', 'node003', 'node007', 'node010', 'node011', 'node012']
    """
    s = nodelist_str.strip()
    if "[" not in s:
        # Single node, no bracket notation
        return [s]
    bracket_start = s.index("[")
    prefix = s[:bracket_start]
    rest = s[bracket_start + 1 :].rstrip("]")
    nodes: list[str] = []
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            width = len(lo)  # preserve zero-padding width
            for i in range(int(lo), int(hi) + 1):
                nodes.append(f"{prefix}{i:0{width}d}")
        else:
            nodes.append(f"{prefix}{part}")
    return nodes


def _resolve_wandb_mode(
    mode: str | None = None,
) -> str:
    """Determine the effective wandb mode from env vars and explicit arg."""
    disabled = os.environ.get("WANDB_DISABLED")
    env_mode = os.environ.get("WANDB_MODE")
    explicit = mode or env_mode
    if disabled:
        return "disabled"
    if explicit:
        m = explicit.lower()
        if m not in {"online", "offline", "disabled", "shared"}:
            raise ValueError(
                f"Invalid wandb mode={m!r}; "
                "expected 'online', 'offline', 'disabled', or 'shared'"
            )
        return m
    return "online"
