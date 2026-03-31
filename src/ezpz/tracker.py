"""Multi-backend experiment tracking for distributed training.

Provides a :class:`Tracker` that fans out metric logging to one or more
:class:`TrackerBackend` instances (e.g. wandb, CSV).  Use the
:func:`setup_tracker` factory to create a tracker from a list of backend
names::

    tracker = setup_tracker(
        project_name="my-project",
        backends="wandb,csv",
        outdir="outputs/run-001",
    )
    tracker.log({"loss": 0.42, "lr": 1e-4})
    tracker.finish()

The tracker is rank-aware: on non-rank-0 processes, backends that perform
external I/O (like wandb) are silently disabled.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "CSVBackend",
    "NullTracker",
    "Tracker",
    "TrackerBackend",
    "WandbBackend",
    "setup_tracker",
]


# ── Backend ABC ──────────────────────────────────────────────────────────────


class TrackerBackend(ABC):
    """Base class for experiment tracking backends.

    Subclasses must implement :meth:`log`, :meth:`log_config`, and
    :meth:`finish`.  The remaining methods have no-op defaults and can be
    overridden when the backend supports richer features.
    """

    name: str = "base"

    @abstractmethod
    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log a dictionary of scalar metrics.

        Args:
            metrics: Key-value pairs to record.
            step: Optional global step number.
            commit: If ``True``, flush this entry immediately (relevant for
                backends that batch writes).
        """

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Record run-level configuration (hyperparameters, env info, etc.)."""

    @abstractmethod
    def finish(self) -> None:
        """Finalise the run and release resources."""

    # ── Optional capabilities (no-op defaults) ───────────────────────────

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a tabular dataset.  No-op unless overridden."""

    def log_image(
        self,
        key: str,
        image_path: str | Path,
        caption: str | None = None,
    ) -> None:
        """Log an image asset.  No-op unless overridden."""

    def watch(self, model: Any, **kwargs: Any) -> None:
        """Attach gradient/parameter tracking to a model.  No-op unless overridden."""


# ── Wandb backend ────────────────────────────────────────────────────────────


class WandbBackend(TrackerBackend):
    """Backend that delegates to `Weights & Biases <https://wandb.ai>`_.

    Wraps ``wandb.init`` with the same rank-aware, env-var-respecting logic
    used by :func:`ezpz.distributed.setup_wandb`.

    Args:
        project_name: W&B project name.
        config: Run-level config dict passed to ``wandb.init``.
        outdir: Directory for local wandb files.
        rank: Distributed rank (non-0 gets ``mode="disabled"``).
        **kwargs: Forwarded to ``wandb.init``.
    """

    name: str = "wandb"

    def __init__(
        self,
        project_name: str | None = None,
        config: dict[str, Any] | None = None,
        outdir: str | os.PathLike | None = None,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        import wandb

        self._wandb = wandb

        if rank is None:
            try:
                from ezpz.distributed import get_rank
                rank = get_rank()
            except Exception:
                rank = 0

        # Resolve project name from args / env
        _project = project_name
        if _project is None:
            _project = os.environ.get(
                "WB_PROJECT",
                os.environ.get(
                    "WANDB_PROJECT",
                    os.environ.get("WB_PROJECT_NAME"),
                ),
            )

        # Resolve mode
        from ezpz.distributed import _resolve_wandb_mode
        _mode = _resolve_wandb_mode(kwargs.pop("mode", None))

        outdir_str = Path(outdir).as_posix() if outdir else os.getcwd()

        init_kwargs: dict[str, Any] = {
            "project": _project,
            "dir": outdir_str,
            "mode": _mode,
            **kwargs,
        }
        # Remove None values so wandb.init uses its own defaults
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            logger.warning("wandb.init() failed: %s — continuing without wandb", exc)
            self._run = None
            return

        if self._run is not None and config is not None:
            self._run.config.update(config)

        # Auto-populate system info
        if self._run is not None:
            try:
                import torch
                import ezpz

                self._run.config.update(
                    {
                        "hostname": os.environ.get("HOSTNAME", ""),
                        "torch_version": torch.__version__,
                        "ezpz_version": ezpz.__version__,
                        "working_directory": os.getcwd(),
                    }
                )
            except Exception:
                pass

    @property
    def run(self) -> Any:
        """The underlying ``wandb.Run``, or ``None``."""
        return self._run

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        if self._run is None:
            return
        kwargs: dict[str, Any] = {"commit": commit}
        if step is not None:
            kwargs["step"] = step
        self._wandb.log(metrics, **kwargs)

    def log_config(self, config: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(config)

    def finish(self) -> None:
        if self._run is not None:
            self._wandb.finish()

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        if self._run is None:
            return
        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({key: table})

    def log_image(
        self,
        key: str,
        image_path: str | Path,
        caption: str | None = None,
    ) -> None:
        if self._run is None:
            return
        self._wandb.log(
            {key: self._wandb.Image(str(image_path), caption=caption)}
        )

    def watch(self, model: Any, **kwargs: Any) -> None:
        if self._run is not None:
            self._run.watch(model, **kwargs)


# ── CSV backend ──────────────────────────────────────────────────────────────


class CSVBackend(TrackerBackend):
    """Backend that writes metrics to a CSV file and config to JSON.

    Args:
        outdir: Directory where ``metrics.csv`` and ``config.json`` are written.
    """

    name: str = "csv"

    def __init__(self, outdir: str | os.PathLike) -> None:
        self._outdir = Path(outdir)
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._outdir / "metrics.csv"
        self._fieldnames: list[str] = []
        self._rows: list[dict[str, Any]] = []

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        row = dict(metrics)
        if step is not None:
            row.setdefault("step", step)
        # Track all seen keys to build the header
        for key in row:
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(row)
        if commit:
            self._flush()

    def log_config(self, config: dict[str, Any]) -> None:
        config_path = self._outdir / "config.json"
        existing: dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(config)
        config_path.write_text(
            json.dumps(existing, indent=2, default=str) + "\n"
        )

    def finish(self) -> None:
        self._flush()

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        table_path = self._outdir / f"{key}.csv"
        with table_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)

    def _flush(self) -> None:
        """Write all buffered rows to the CSV file."""
        if not self._rows:
            return
        with self._csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self._rows)


# ── Tracker multiplexer ──────────────────────────────────────────────────────


class Tracker:
    """Multiplexer that fans out tracking calls to multiple backends.

    Args:
        backends: List of :class:`TrackerBackend` instances.

    Example::

        tracker = Tracker([WandbBackend(...), CSVBackend("./logs")])
        tracker.log({"loss": 0.5})
        tracker.finish()
    """

    def __init__(self, backends: list[TrackerBackend]) -> None:
        self._backends = list(backends)
        self._backend_map: dict[str, TrackerBackend] = {
            b.name: b for b in self._backends
        }

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to all backends."""
        for backend in self._backends:
            try:
                backend.log(metrics, step=step, commit=commit)
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.log() failed: %s", backend.name, exc
                )

    def log_config(self, config: dict[str, Any]) -> None:
        """Record config on all backends."""
        for backend in self._backends:
            try:
                backend.log_config(config)
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.log_config() failed: %s",
                    backend.name,
                    exc,
                )

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table to all backends that support it."""
        for backend in self._backends:
            try:
                backend.log_table(key, columns, data)
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.log_table() failed: %s",
                    backend.name,
                    exc,
                )

    def log_image(
        self,
        key: str,
        image_path: str | Path,
        caption: str | None = None,
    ) -> None:
        """Log an image to all backends that support it."""
        for backend in self._backends:
            try:
                backend.log_image(key, image_path, caption=caption)
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.log_image() failed: %s",
                    backend.name,
                    exc,
                )

    def watch(self, model: Any, **kwargs: Any) -> None:
        """Attach model watching on all backends that support it."""
        for backend in self._backends:
            try:
                backend.watch(model, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.watch() failed: %s",
                    backend.name,
                    exc,
                )

    def finish(self) -> None:
        """Finalise all backends."""
        for backend in self._backends:
            try:
                backend.finish()
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.finish() failed: %s",
                    backend.name,
                    exc,
                )

    def get_backend(self, name: str) -> TrackerBackend | None:
        """Return a specific backend by name, or ``None`` if not active.

        Use this for backend-specific features::

            wb = tracker.get_backend("wandb")
            if wb is not None:
                wb.watch(model, log="all")
        """
        return self._backend_map.get(name)

    @property
    def wandb_run(self) -> Any:
        """Convenience accessor for the underlying ``wandb.Run``.

        Returns ``None`` if no wandb backend is active.
        """
        wb = self.get_backend("wandb")
        if wb is not None and isinstance(wb, WandbBackend):
            return wb.run
        return None


class NullTracker(Tracker):
    """A no-op tracker used when no backends are configured.

    All methods are inherited from :class:`Tracker` with an empty backend
    list, so every call is a silent no-op.
    """

    def __init__(self) -> None:
        super().__init__(backends=[])


# ── Backend registry & factory ───────────────────────────────────────────────

_BACKEND_REGISTRY: dict[str, type[TrackerBackend]] = {}


def register_backend(name: str, cls: type[TrackerBackend]) -> None:
    """Register a :class:`TrackerBackend` class under the given name.

    Args:
        name: Short name used in ``--tracker`` flags (e.g. ``"csv"``).
        cls: The backend class to instantiate.
    """
    _BACKEND_REGISTRY[name] = cls


# Built-in registrations
register_backend("wandb", WandbBackend)
register_backend("csv", CSVBackend)


def setup_tracker(
    project_name: str | None = None,
    backends: str | Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    outdir: str | os.PathLike | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> Tracker:
    """Create a :class:`Tracker` with the requested backends.

    Args:
        project_name: Project name passed to backends that support it.
        backends: Comma-separated string or list of backend names.
            Defaults to ``["wandb"]``.  Use ``"none"`` to disable tracking.
            Also reads ``EZPZ_TRACKER_BACKENDS`` env var as fallback.
        config: Run-level configuration dict.
        outdir: Output directory for file-based backends (csv, etc.).
        rank: Distributed rank.  Auto-detected if not provided.
        **kwargs: Extra keyword arguments forwarded to backend constructors
            (e.g. wandb-specific init params).

    Returns:
        A :class:`Tracker` (or :class:`NullTracker` if no backends activate).
    """
    # Resolve rank
    if rank is None:
        try:
            from ezpz.distributed import get_rank
            rank = get_rank()
        except Exception:
            rank = 0

    # Parse backends
    if backends is None:
        backends = os.environ.get("EZPZ_TRACKER_BACKENDS", "wandb")
    if isinstance(backends, str):
        backends = [b.strip() for b in backends.split(",") if b.strip()]

    if backends == ["none"] or not backends:
        return NullTracker()

    active: list[TrackerBackend] = []
    for name in backends:
        cls = _BACKEND_REGISTRY.get(name)
        if cls is None:
            logger.warning(
                "Unknown tracker backend %r (registered: %s)",
                name,
                ", ".join(_BACKEND_REGISTRY),
            )
            continue
        try:
            if name == "wandb":
                backend = cls(
                    project_name=project_name,
                    config=config,
                    outdir=outdir,
                    rank=rank,
                    **kwargs,
                )
            elif name == "csv":
                csv_outdir = outdir or os.getcwd()
                backend = cls(outdir=csv_outdir)
                if config is not None:
                    backend.log_config(config)
            else:
                # Generic instantiation for custom backends
                backend = cls(
                    project_name=project_name,
                    config=config,
                    outdir=outdir,
                    rank=rank,
                    **kwargs,
                )
            active.append(backend)
        except ImportError as exc:
            logger.warning(
                "Tracker backend %r not available (missing dependency: %s)",
                name,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialise tracker backend %r: %s", name, exc
            )

    if not active:
        return NullTracker()
    return Tracker(active)
