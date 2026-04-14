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
    "MLflowBackend",
    "NullTracker",
    "Tracker",
    "TrackerBackend",
    "WandbBackend",
    "setup_tracker",
]


def _default_project_name() -> str:
    """Derive a project name from the running script, e.g. ``ezpz.examples.fsdp``.

    Falls back to ``"ezpz"`` when ``__main__`` has no file path.
    """
    import sys

    main = sys.modules.get("__main__")
    if main is None:
        return "ezpz"
    fpath = getattr(main, "__file__", None)
    if fpath is None:
        return "ezpz"
    fp = Path(fpath)
    return f"ezpz.{fp.parent.stem}.{fp.stem}"


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

    def log_artifacts(self, paths: dict[str, str]) -> None:
        """Upload local files as run artifacts.  No-op unless overridden."""

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

        # Resolve project name from args / env / script name
        _project = project_name
        if _project is None:
            _project = os.environ.get(
                "WB_PROJECT",
                os.environ.get(
                    "WANDB_PROJECT",
                    os.environ.get("WB_PROJECT_NAME"),
                ),
            )
        if _project is None:
            _project = _default_project_name()

        # Resolve mode — disable on non-rank-0 processes
        from ezpz.distributed import _resolve_wandb_mode

        _mode_arg = kwargs.pop("mode", None)
        if rank != 0:
            _mode = "disabled"
        else:
            _mode = _resolve_wandb_mode(_mode_arg)

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
            logger.warning(
                "wandb.init() failed: %s — continuing without wandb", exc
            )
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

    Only rank 0 writes files; other ranks buffer in memory (no-op I/O).

    Args:
        outdir: Directory where ``metrics.csv`` and ``config.json`` are written.
        rank: Distributed rank. Non-0 ranks skip all file I/O.
    """

    name: str = "csv"

    def __init__(self, outdir: str | os.PathLike, rank: int = 0) -> None:
        self._rank = rank
        self._outdir = Path(outdir)
        if self._rank == 0:
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
        if self._rank != 0:
            return
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
        if self._rank != 0:
            return
        table_path = self._outdir / f"{key}.csv"
        with table_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)

    def _flush(self) -> None:
        """Write all buffered rows to the CSV file."""
        if self._rank != 0 or not self._rows:
            return
        with self._csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self._rows)


# ── MLflow backend ──────────────────────────────────────────────────────────


class MLflowBackend(TrackerBackend):
    """Backend that delegates to `MLflow Tracking <https://mlflow.org>`_.

    Logs metrics via ``mlflow.log_metrics``, config via ``mlflow.log_params``,
    and tables/images as artifacts.

    Args:
        project_name: MLflow experiment name.
        config: Run-level config dict logged as MLflow params.
        outdir: Artifact location (forwarded to ``mlflow.start_run``).
        rank: Distributed rank (non-0 is silently disabled).
        **kwargs: Forwarded to ``mlflow.start_run``.
    """

    name: str = "mlflow"

    def __init__(
        self,
        project_name: str | None = None,
        config: dict[str, Any] | None = None,
        outdir: str | os.PathLike | None = None,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        import mlflow

        self._mlflow = mlflow

        if rank is None:
            try:
                from ezpz.distributed import get_rank

                rank = get_rank()
            except Exception:
                rank = 0

        if rank != 0:
            self._run = None
            self._active = False
            return

        # ── Tracking URI & auth ─────────────────────────────────────────
        # Load .env if python-dotenv is available (picks up
        # MLFLOW_TRACKING_URI, AMSC_API_KEY, etc.)
        try:
            from dotenv import find_dotenv, load_dotenv
        except Exception:
            logger.warning(f"python-dotenv not available, skipping .env loading for mlflow credentials")

        try:
            # Load ~/.amsc.env first (user-level credentials), then
            # project-level .env (which can override/extend).
            amsc_env = Path.home() / ".amsc.env"
            if amsc_env.is_file():
                logger.info(f"Loading AMSC credentials from {amsc_env}")
                load_dotenv(amsc_env)
            env_file = find_dotenv(usecwd=True) or find_dotenv()
            if env_file:
                logger.info(f"Loading AMSC credentials from {env_file}")
                load_dotenv(env_file, override=True)
        except Exception as exc:
            logger.exception(exc)
            logger.exception("Failed to load .env files for mlflow credentials")
            logger.info("Proceeding without .env credentials for mlflow (if required, set env vars manually)")

        # Suppress urllib3 TLS warnings when insecure mode is enabled
        if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in (
            "true",
            "1",
            "yes",
        ):
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Inject API key auth. Try MLFLOW_TRACKING_TOKEN (Bearer) first,
        # then AMSC_API_KEY (X-API-Key header via request hook).
        api_key = os.environ.get("MLFLOW_TRACKING_TOKEN")
        if api_key:
            pass  # MLflow handles Bearer auth natively
        else:
            api_key = os.environ.get(
                "AMSC_API_KEY", os.environ.get("AM_SC_API_KEY")
            )
            if api_key:
                # Patch mlflow's request layer to inject X-Api-Key header.
                # Signature must match http_request(host_creds, endpoint,
                # method, ...) — see zhenghh04/alcf_mlflow_deployment.
                try:
                    import mlflow.utils.rest_utils as _rest

                    _orig_call = _rest.http_request

                    def _patched_call(  # type: ignore[no-untyped-def]
                        host_creds: Any,
                        endpoint: str,
                        method: str,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Any:
                        headers = dict(kwargs.get("extra_headers") or {})
                        if kwargs.get("headers") is not None:
                            headers.update(dict(kwargs["headers"]))
                        headers["X-Api-Key"] = api_key
                        kwargs["extra_headers"] = headers
                        kwargs.pop("headers", None)
                        return _orig_call(host_creds, endpoint, method, *args, **kwargs)

                    _rest.http_request = _patched_call  # type: ignore[assignment]
                except Exception as exc:
                    logger.warning(
                        "Failed to patch mlflow auth headers: %s", exc
                    )

        # Resolve experiment name: MLFLOW_EXPERIMENT_NAME (explicit override)
        # → project_name arg → wandb project env vars → script-derived default.
        # MLFLOW_EXPERIMENT_NAME wins over project_name so users can redirect
        # MLflow to a different experiment without changing code.
        _experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if _experiment is None:
            _experiment = project_name
        if _experiment is None:
            _experiment = os.environ.get(
                "WB_PROJECT",
                os.environ.get(
                    "WANDB_PROJECT",
                    os.environ.get("WB_PROJECT_NAME"),
                ),
            )
        if _experiment is None:
            _experiment = _default_project_name()
        # Replace dots with hyphens for cleaner experiment names
        # (e.g. "ezpz.examples.vit" → "ezpz-examples-vit")
        _experiment = _experiment.replace(".", "-")

        try:
            mlflow.set_experiment(_experiment)
            self._run = mlflow.start_run(**kwargs)
            self._active = True
        except Exception as exc:
            _msg = str(exc)
            if "403" in _msg or "Permission" in _msg:
                logger.warning(
                    "mlflow.start_run() got 403 Permission Denied for "
                    "experiment %r — this is a server-side auth issue, "
                    "not an ezpz bug. Check your API key or try a "
                    "different experiment name. Continuing without mlflow.",
                    _experiment,
                )
            else:
                logger.warning(
                    "mlflow.start_run() failed: %s — continuing without mlflow",
                    exc,
                )
            self._run = None
            self._active = False
            return

        self._step = 0  # auto-increment when caller doesn't provide step
        self._log_errors_warned = False  # suppress repeated 403 warnings

        # Log base system/environment params under the "ezpz." prefix
        self._log_system_params()

        if config is not None:
            self.log_config({"config": config})

        # Cache tracking info for later retrieval
        self._tracking_uri = self._mlflow.get_tracking_uri()
        run_info = self._run.info
        self._experiment_id = run_info.experiment_id
        self._run_id = run_info.run_id
        self._run_name = run_info.run_name or self._run_id
        if self._tracking_uri.startswith("http"):
            self._run_url: str | None = (
                f"{self._tracking_uri.rstrip('/')}/"
                f"#/experiments/{self._experiment_id}/"
                f"runs/{self._run_id}"
            )
        else:
            self._run_url = None

        # Print setup info directly (like wandb does) so it's always visible
        # regardless of log level.
        import sys

        _w = sys.stderr.write
        _mlflow = "\033[91mmlflow\033[0m"
        _w(f"🔄 {_mlflow}: Tracking run with mlflow\n")
        _w(f"🔄 {_mlflow}: Experiment: {_experiment}\n")
        _w(f"🔄 {_mlflow}: Run name: {self._run_name}\n")
        _w(f"🔄 {_mlflow}: Tracking URI: {self._tracking_uri}\n")
        if self._run_url:
            _w(f"🔄 {_mlflow}: 🔗 View run at {self._run_url}\n")
        else:
            _w(
                f"🔄 {_mlflow}: Run ID: {self._run_id} "
                f"(view with: mlflow ui --port 5000)\n"
            )
        sys.stderr.flush()

    @property
    def run(self) -> Any:
        """The underlying ``mlflow.ActiveRun``, or ``None``."""
        return self._run

    @property
    def run_url(self) -> str | None:
        """Dashboard URL for this run, or ``None`` for local file stores."""
        return getattr(self, "_run_url", None)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        if not self._active:
            return
        # Auto-increment step so MLflow records a time series (line chart)
        # rather than overwriting step 0 each call (bar chart).
        if step is None:
            step = self._step
        self._step = step + 1
        # MLflow only accepts float-coercible values
        safe: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                safe[k] = float(v)
            except (TypeError, ValueError):
                continue
        if safe:
            # Rename raw keys to {key}/local when distributed aggregates
            # ({key}/mean, /min, /max, /std) are present, so MLflow groups
            # them under a shared prefix.
            agg_suffixes = ("/mean", "/max", "/min", "/std")
            agg_prefixes = {
                k.rsplit("/", 1)[0]
                for k in safe
                if k.endswith(agg_suffixes)
            }
            if agg_prefixes:
                safe = {
                    (f"{k}/local" if k in agg_prefixes else k): v
                    for k, v in safe.items()
                }
            try:
                self._mlflow.log_metrics(safe, step=step)
            except Exception as exc:
                if not self._log_errors_warned:
                    _msg = str(exc)
                    if "403" in _msg or "Permission" in _msg:
                        logger.warning(
                            "mlflow.log_metrics() got 403 — server is "
                            "rejecting writes. Suppressing further warnings."
                        )
                    else:
                        logger.warning("mlflow.log_metrics() failed: %s", exc)
                    self._log_errors_warned = True

    @staticmethod
    def _flatten(
        d: dict[str, Any], prefix: str = "", sep: str = "."
    ) -> dict[str, str]:
        """Flatten a nested dict into dot-separated keys with string values."""
        out: dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(MLflowBackend._flatten(v, prefix=key, sep=sep))
            else:
                out[key] = str(v)
        return out

    def _log_system_params(self) -> None:
        """Log base ezpz/system info as MLflow params under ``ezpz.*``."""
        try:
            import sys

            import torch

            import ezpz
            from ezpz.configs import get_scheduler
            from ezpz.distributed import (
                get_hostname,
                get_local_rank,
                get_machine,
                get_rank,
                get_torch_backend,
                get_torch_device,
                get_world_size,
            )

            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)

            # Git branch (best-effort)
            git_branch = ""
            try:
                import subprocess

                git_branch = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).strip()
                )
            except Exception:
                pass

            params: dict[str, Any] = {
                "ezpz": {
                    "version": ezpz.__version__,
                    "hostname": get_hostname(),
                    "machine": get_machine(),
                    "scheduler": get_scheduler(),
                    "device": str(get_torch_device()),
                    "backend": get_torch_backend(),
                    "world_size": get_world_size(),
                    "rank": get_rank(),
                    "local_rank": get_local_rank(),
                    "working_directory": os.getcwd(),
                    "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "date": now.strftime("%Y-%m-%d"),
                    "year": now.strftime("%Y"),
                    "month": now.strftime("%m"),
                    "day": now.strftime("%d"),
                    "python_version": sys.version.split()[0],
                    "torch_version": torch.__version__,
                    "command": " ".join(sys.argv),
                    "git_branch": git_branch,
                },
            }
            self._mlflow.log_params(self._flatten(params))
        except Exception as exc:
            logger.warning("mlflow system params failed: %s", exc)

    def log_config(self, config: dict[str, Any]) -> None:
        if not self._active:
            return
        flat = self._flatten(config)
        if flat:
            try:
                self._mlflow.log_params(flat)
            except Exception as exc:
                logger.warning("mlflow.log_params failed: %s", exc)

    def log_artifacts(self, paths: dict[str, str]) -> None:
        """Upload local files as MLflow artifacts.

        Args:
            paths: Mapping of ``label → file_path``.  Each existing file
                   is uploaded under an artifact subdirectory matching the
                   label (e.g. ``"Report" → "Report/report.md"``).
        """
        if not self._active:
            return
        for label, fpath in paths.items():
            p = Path(fpath)
            if not p.exists():
                continue
            try:
                if p.is_dir():
                    self._mlflow.log_artifacts(str(p), artifact_path=label)
                else:
                    self._mlflow.log_artifact(str(p), artifact_path=label)
            except Exception as exc:
                logger.warning(
                    "mlflow log_artifact(%s) failed: %s", label, exc
                )

    def finish(self) -> None:
        if not self._active:
            return
        try:
            self._mlflow.end_run()
        except Exception as exc:
            logger.warning("mlflow.end_run failed: %s", exc)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        if not self._active:
            return
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix=f"{key}_"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(data)
                temp_path = f.name
            self._mlflow.log_artifact(temp_path, artifact_path="tables")
            Path(temp_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("mlflow log_table failed: %s", exc)

    def log_image(
        self,
        key: str,
        image_path: str | Path,
        caption: str | None = None,
    ) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(str(image_path), artifact_path="images")
        except Exception as exc:
            logger.warning("mlflow log_image failed: %s", exc)


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

    def log_artifacts(self, paths: dict[str, str]) -> None:
        """Upload local files as artifacts on all backends that support it."""
        for backend in self._backends:
            try:
                backend.log_artifacts(paths)
            except Exception as exc:
                logger.warning(
                    "Tracker backend %s.log_artifacts() failed: %s",
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

    @property
    def mlflow_run(self) -> Any:
        """Convenience accessor for the underlying ``mlflow.ActiveRun``.

        Returns ``None`` if no MLflow backend is active.
        """
        ml = self.get_backend("mlflow")
        if ml is not None and isinstance(ml, MLflowBackend):
            return ml.run
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
register_backend("mlflow", MLflowBackend)


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
        backends = os.environ.get(
            "EZPZ_TRACKER_BACKENDS",
            os.environ.get("EZPZ_TRACKERS", "wandb"),
        )
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
                backend = cls(outdir=csv_outdir, rank=rank)
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
