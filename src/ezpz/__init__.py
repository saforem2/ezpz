"""Lightweight package initializer for ezpz.

The goal is to keep ``import ezpz`` fast and side-effect free while still
exposing the familiar helpers at the package level.  Submodules are imported
on demand the first time one of their attributes is requested.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict

from .__about__ import __version__  # re-exported symbol

import socket

if socket.getfqdn().startswith("x3"):
    from mpi4py import MPI  # noqa
    import torch  # noqa

# ---------------------------------------------------------------------------
# Static re-exports for type checkers and editor tooling
# ---------------------------------------------------------------------------
#
# At runtime, attribute access on `ezpz.foo` flows through `__getattr__`
# below — it walks `_MODULE_SEARCH_ORDER` to find which submodule
# defines `foo` and imports lazily.  That preserves the "import ezpz
# stays fast" property (no eager torch/mpi4py import).
#
# But Pyright / Pylance / Jedi / LSP servers can't follow `__getattr__`
# resolution.  Users reported (issue #133) that `ezpz.setup_torch` and
# friends don't show up in completion, auto-import, or symbol search —
# even though they work at runtime.
#
# Fix: a `TYPE_CHECKING` block with explicit re-exports.  Type checkers
# evaluate this branch (TYPE_CHECKING is True at analysis time), so
# they see real `from .module import name` statements and can resolve
# everything statically.  At runtime, TYPE_CHECKING is False and the
# imports are skipped — `__getattr__` still does the lazy work.
#
# Standard pattern; same shape pandas/numpy/sqlalchemy use.
# noqa: F401  — these imports are pure re-exports for static analysis.
if TYPE_CHECKING:
    # -- distributed: rank / topology / device / lifecycle / collectives --
    from .distributed import (  # noqa: F401
        all_reduce,
        barrier,
        broadcast,
        cleanup,
        get_cpus_per_node,
        get_device_properties,
        get_dist_info,
        get_gpus_per_node,
        get_hostname,
        get_local_rank,
        get_machine,
        get_node_index,
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
        setup_mlflow,
        setup_torch,
        setup_wandb,
        synchronize,
        timeitlogit,
        verify_wandb,
        wrap_model,
        wrap_model_for_ddp,
        wrap_model_for_fsdp,
        wrap_model_for_fsdp2,
        init_device_mesh_safe,
    )
    # -- utils: helpers + memory + tarball plumbing --
    from .utils import (  # noqa: F401
        Color,
        DistributedPdb,
        ForkedPdb,
        NoColor,
        check_for_tarball,
        format_compact_summary,
        format_memory_summary,
        format_pair,
        get_current_memory_allocated,
        get_current_memory_reserved,
        get_max_memory_allocated,
        get_max_memory_reserved,
        get_memory_metrics,
        get_timestamp,
        grab_tensor,
        is_memory_metric_key,
        make_tarfile,
        model_summary,
        reset_peak_memory_stats,
        summarize_dict,
    )
    # -- log: structured logging --
    from .log import get_logger, silence_noisy_loggers  # noqa: F401
    # -- history: distributed metric tracking --
    from .history import History  # noqa: F401
    # -- flops: MFU + peak FLOPS database --
    from .flops import compute_mfu, try_estimate  # noqa: F401
    # -- configs: scheduler / machine detection --
    from .configs import get_scheduler  # noqa: F401
    # -- launch: scheduler-aware launcher --
    from .launch import (  # noqa: F401
        get_active_jobid,
        get_nodelist_of_active_job,
    )

# ---------------------------------------------------------------------------
# Lazy module loading (runtime path)
# ---------------------------------------------------------------------------

_LAZY_MODULES: Dict[str, str] = {
    "configs": "ezpz.configs",
    # "dist": "ezpz.dist",
    "distributed": "ezpz.distributed",
    "doctor": "ezpz.doctor",
    "examples": "ezpz.examples",
    "flops": "ezpz.flops",
    "history": "ezpz.history",
    "jobs": "ezpz.jobs",
    "launch": "ezpz.launch",
    "log": "ezpz.log",
    "models": "ezpz.models",
    "pbs": "ezpz.pbs",
    "profile": "ezpz.profile",
    "tp": "ezpz.tp",
    "tplot": "ezpz.tplot",
    "tracker": "ezpz.tracker",
    "utils": "ezpz.utils",
}

_MODULE_SEARCH_ORDER: tuple[str, ...] = (
    "ezpz.log",
    "ezpz.configs",
    "ezpz.distributed",
    "ezpz.doctor",
    "ezpz.examples",
    "ezpz.utils",
    "ezpz.flops",
    "ezpz.history",
    "ezpz.profile",
    "ezpz.tp",
    "ezpz.models",
    "ezpz.pbs",
    "ezpz.jobs",
    "ezpz.launch",
    "ezpz.tplot",
    "ezpz.tracker",
    # "ezpz.test",
)

# Symbols re-exported via the TYPE_CHECKING block above.  Listing them
# in __all__ makes them discoverable to `dir(ezpz)`, `from ezpz import *`,
# and stub-aware tools.  The runtime resolution is unchanged — these all
# come from the lazy __getattr__ walker below.
_STATIC_REEXPORTS: tuple[str, ...] = (
    # distributed
    "all_reduce", "barrier", "broadcast", "cleanup",
    "get_cpus_per_node", "get_device_properties", "get_dist_info",
    "get_gpus_per_node", "get_hostname", "get_local_rank",
    "get_machine", "get_node_index", "get_num_nodes", "get_rank",
    "get_torch_backend", "get_torch_device", "get_torch_device_type",
    "get_world_size", "get_world_size_in_use", "get_world_size_total",
    "log_dict_as_bulleted_list", "print_dist_setup", "query_environment",
    "seed_everything", "setup_mlflow", "setup_torch", "setup_wandb",
    "synchronize", "timeitlogit", "verify_wandb",
    "wrap_model", "wrap_model_for_ddp",
    "wrap_model_for_fsdp", "wrap_model_for_fsdp2",
    "init_device_mesh_safe",
    # utils
    "Color", "DistributedPdb", "ForkedPdb", "NoColor",
    "check_for_tarball",
    "format_compact_summary", "format_memory_summary", "format_pair",
    "get_current_memory_allocated", "get_current_memory_reserved",
    "get_max_memory_allocated", "get_max_memory_reserved",
    "get_memory_metrics", "get_timestamp", "grab_tensor",
    "is_memory_metric_key", "make_tarfile",
    "model_summary", "reset_peak_memory_stats", "summarize_dict",
    # log
    "get_logger", "silence_noisy_loggers",
    # history
    "History",
    # flops
    "compute_mfu", "try_estimate",
    # configs
    "get_scheduler",
    # launch
    "get_active_jobid", "get_nodelist_of_active_job",
)

__all__ = [
    "__version__",
    *sorted(_LAZY_MODULES.keys()),
    *_STATIC_REEXPORTS,
]  # type: ignore

_IMPORT_CACHE: Dict[str, ModuleType] = {}
# Track the underlying ImportError for any submodule whose import
# failed, so __getattr__ can surface it in the AttributeError instead
# of giving a misleading "no attribute" message that doesn't say WHY.
_IMPORT_ERRORS: Dict[str, ImportError] = {}


def _load_module(module_name: str) -> ModuleType | None:
    """Import *module_name* once and cache the result.

    Returns ``None`` on import failure; the underlying ImportError is
    stashed in :data:`_IMPORT_ERRORS` so the lazy attribute lookup
    below can include it in the final error message.
    """
    if module_name in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_name]
    if module_name in _IMPORT_ERRORS:
        return None
    try:
        module = import_module(module_name)
    except ImportError as exc:
        # Catch ImportError (not just ModuleNotFoundError) so failures
        # inside the imported module itself — e.g. `import torch`
        # failing inside `ezpz.utils.__init__` — are also stashed and
        # reported, not silently swallowed.
        _IMPORT_ERRORS[module_name] = exc
        return None
    _IMPORT_CACHE[module_name] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised via tests
    """Dynamically resolve attributes from submodules on first access."""

    if name in _LAZY_MODULES:
        module_name = _LAZY_MODULES[name]
        module = _load_module(module_name)
        if module is None:
            err = _IMPORT_ERRORS.get(module_name)
            detail = f": {err}" if err else ""
            raise AttributeError(
                f"Module {module_name!r} cannot be imported{detail}"
            )
        globals()[name] = module
        return module

    for module_name in _MODULE_SEARCH_ORDER:
        module = _load_module(module_name)
        if module is None:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    # Attribute not found in any successfully-loaded submodule. Include
    # any import failures we saw along the way — chances are that's
    # the real reason the attribute is "missing". Without this, a
    # missing `torch` would silently mask every util as "no attribute",
    # which is brutal to debug (the user sees `AttributeError: module
    # 'ezpz' has no attribute 'get_timestamp'` with no hint that the
    # underlying cause is `ModuleNotFoundError: No module named
    # 'torch'` inside `ezpz.utils.__init__`).
    if _IMPORT_ERRORS:
        failures = "; ".join(
            f"{m}: {e}" for m, e in _IMPORT_ERRORS.items()
        )
        raise AttributeError(
            f"module 'ezpz' has no attribute {name!r} "
            f"(submodule import failures: {failures})"
        )
    raise AttributeError(f"module 'ezpz' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    visible = set(__all__)
    visible.update(globals().keys())
    return sorted(visible)


def try_import_wandb() -> object | None:
    import os

    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
    WANDB_MODE = os.environ.get("WANDB_MODE", "").lower()
    if not WANDB_DISABLED and WANDB_MODE != "disabled":
        try:
            import wandb
        except Exception as e:
            wandb = None
    else:
        wandb = None
        WANDB_DISABLED = True
    return wandb


def get_torch_version_as_float() -> float:
    """Return the PyTorch version as a float (e.g. 2.5).

    .. deprecated::
        Use :func:`get_torch_version_tuple` for semver-safe comparisons.
    """
    import torch
    return float(".".join(torch.__version__.split(".")[:2]))


def get_torch_version_tuple() -> tuple[int, int]:
    """Return the PyTorch (major, minor) version as an integer tuple.

    Semver-safe: ``(2, 12)`` compares correctly unlike the float ``2.12``.
    """
    import torch
    parts = torch.__version__.split(".")[:2]
    return (int(parts[0]), int(parts[1]))


# Record the package version in the environment for compatibility with callers
# that previously relied on the eager side effects from the old initializer.
try:  # pragma: no cover - best-effort, avoid failures in restricted envs
    import os

    os.environ.setdefault("EZPZ_VERSION", __version__)
except Exception:
    pass
