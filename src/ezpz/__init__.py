"""Lightweight package initializer for ezpz.

The goal is to keep ``import ezpz`` fast and side-effect free while still
exposing the familiar helpers at the package level.  Submodules are imported
on demand the first time one of their attributes is requested.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Dict

from .__about__ import __version__  # re-exported symbol

import socket
if socket.getfqdn().startswith("x3"):
    from mpi4py import MPI  # noqa
    import torch  # noqa

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

_LAZY_MODULES: Dict[str, str] = {
    "configs": "ezpz.configs",
    "dist": "ezpz.dist",
    "doctor": "ezpz.doctor",
    "examples": "ezpz.examples",
    "history": "ezpz.history",
    "jobs": "ezpz.jobs",
    "launch": "ezpz.launch",
    "log": "ezpz.log",
    "models": "ezpz.models",
    "pbs": "ezpz.pbs",
    "profile": "ezpz.profile",
    "tp": "ezpz.tp",
    "tplot": "ezpz.tplot",
    "utils": "ezpz.utils",
}

_MODULE_SEARCH_ORDER: tuple[str, ...] = (
    "ezpz.log",
    "ezpz.configs",
    "ezpz.doctor",
    "ezpz.examples",
    "ezpz.utils",
    "ezpz.history",
    "ezpz.profile",
    "ezpz.tp",
    "ezpz.dist",
    "ezpz.models",
    "ezpz.pbs",
    "ezpz.jobs",
    "ezpz.launch",
    "ezpz.tplot",
)

__all__ = ["__version__", *sorted(_LAZY_MODULES.keys())]  # type:ignore

_IMPORT_CACHE: Dict[str, ModuleType] = {}


def _load_module(module_name: str) -> ModuleType | None:
    """Import *module_name* once and cache the result."""

    if module_name in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        return None
    _IMPORT_CACHE[module_name] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised via tests
    """Dynamically resolve attributes from submodules on first access."""

    if name in _LAZY_MODULES:
        module = _load_module(_LAZY_MODULES[name])
        if module is None:
            raise AttributeError(f"Module {_LAZY_MODULES[name]!r} cannot be imported")
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
    raise AttributeError(f"module 'ezpz' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    visible = set(__all__)
    visible.update(globals().keys())
    return sorted(visible)


# Record the package version in the environment for compatibility with callers
# that previously relied on the eager side effects from the old initializer.
try:  # pragma: no cover - best-effort, avoid failures in restricted envs
    import os

    os.environ.setdefault("EZPZ_VERSION", __version__)
except Exception:
    pass
