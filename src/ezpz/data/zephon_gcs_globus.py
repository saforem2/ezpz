"""Globus adapter for Zephon: lets zephon globus:// URIs ride globusfs fsspec.

Zephon's RouterStorageBackend only knows local/s3/hf/gs/az. This module
adds a ``globus://`` branch by wrapping ``globusfs.GlobusFileSystem`` as
a ``StorageBackend`` and monkey-patching the router. Import side-effect
only; call ``register_globusfs_with_zephon()`` once before any
``Dataset.from_path("globus://...")``.

Requires ``ezpz`` extras ``zephon`` + ``globusfs`` (``saforem2/globusfs``).
"""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

def register_globusfs_with_zephon() -> bool:
    """Register ``globus://`` with Zephon's storage router. No-op if unavailable."""
    try:
        import globusfs  # noqa: F401
    except ImportError:
        logger.debug("globusfs not installed — globus:// routes disabled")
        return False
    try:
        import fsspec
        fsspec.get_filesystem_class("globus")
    except Exception:
        logger.debug("globusfs fsspec protocol not registered")
        return False
    try:
        from zephon._internal.io.storage.router import RouterStorageBackend
        if getattr(RouterStorageBackend, "_ezpz_globus_patched", False):
            return True
        orig = RouterStorageBackend._backend_for
        _globus = None
        class GlobusBackend:
            valid_schemes = frozenset({"globus"})
            def __init__(self):
                self._fs = fsspec.filesystem("globus")
            def open(self, path, mode="rb", **kw):
                return self._fs.open(path, mode, **kw)
            def exists(self, path):
                return self._fs.exists(path)
            def canonical_root(self, path, fmt=None):
                return path
            def download(self, src, dst, timeout=None):
                self._fs.get(src, dst)
            def glob(self, pattern, **kw):
                return self._fs.glob(pattern, **kw)
            def put(self, path, data, **kw):
                d = data if isinstance(data, bytes) else data.encode()
                with self._fs.open(path, "wb") as f:
                    f.write(d)
            def delete(self, path, **kw):
                try:
                    self._fs.rm(path)
                except FileNotFoundError:
                    pass
            def mkdir(self, path, **kw):
                return None
            def listdir(self, path):
                return self._fs.ls(path, detail=False)
            def stat(self, path):
                return self._fs.info(path)
            def read_range(self, path, start, end=None, length=None):
                with self._fs.open(path, "rb") as f:
                    f.seek(start)
                    if end is not None and length is None:
                        length = end - start
                    return f.read(length) if length is not None else f.read()
        def _patched(self2, path: str):
            if "://" in path:
                import urllib.parse as _url
                if _url.urlparse(path).scheme == "globus":
                    nonlocal _globus
                    if _globus is None:
                        _globus = GlobusBackend()
                    return _globus
            return orig(self2, path)
        RouterStorageBackend._backend_for = _patched
        RouterStorageBackend._ezpz_globus_patched = True
        logger.info("globus:// registered with Zephon")
        return True
    except Exception as exc:
        logger.warning("globusfs Zephon registration failed: %s", exc)
        return False
