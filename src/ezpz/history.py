"""
history.py

Contains implementation of History object for tracking / aggregating metrics.
"""

from __future__ import absolute_import, annotations, division, print_function

import json
import os
import platform
import shutil
import sys
import time
from contextlib import ContextDecorator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import ezpz
import ezpz.dist

from ezpz.lazy import lazy_import
#
# ezpz = lazy_import("ezpz")
# assert ezpz is not None

# import matplotlib.pyplot as plt
import numpy as np
import torch

# import torch.distributed
import xarray as xr

# from ezpz.dist import get_rank

# import ezpz

# import ezpz.plot as ezplot

# import ezpz.dist

from ezpz import get_rank
from ezpz import plot as ezplot
from ezpz import timeitlogit
from ezpz.configs import OUTPUTS_DIR, PathLike
from ezpz.log import get_logger
from ezpz.tplot import tplot as eztplot

# from ezpz import tplot as eztplot
from ezpz.utils import get_timestamp, grab_tensor, save_dataset, summarize_dict

# xr = lazy_import("xarray")

# from jaxtyping import ScalarLike
#
ScalarLike = Union[float, int, bool]


# RANK = get_rank()

logger = get_logger(__name__)

# try:
#     import wandb
#
#     WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
# except Exception:
#     wandb = None
#     WANDB_DISABLED = True
ENABLE_WANDB = False
try:
    wandb = lazy_import("wandb")
    if wandb.api.api_key is not None and not os.environ.get(
        "WANDB_DISABLED", False
    ):
        ENABLE_WANDB = True
except Exception:
    wandb = None

# try:
#     import torch.distributed as dist
# except Exception:  # pragma: no cover - optional dependency
#     dist = None

TensorLike = Union[torch.Tensor, np.ndarray, list]

PT_FLOAT = torch.get_default_dtype()

xplt = xr.plot  # type:ignore

AUTO_USE_DISTRIBUTED_HISTORY = True if ezpz.get_world_size() <= 384 else False


class StopWatch(ContextDecorator):
    """
    A simple stopwatch context manager for measuring time taken by a block of code.
    """

    def __init__(
        self,
        msg: str,
        wbtag: Optional[str] = None,
        iter: Optional[int] = None,
        commit: Optional[bool] = False,
        prefix: str = "StopWatch/",
        log_output: bool = True,
    ) -> None:
        """
        Initialize the StopWatch.

        Args:
            msg (str): Message to log when the stopwatch is started.
            wbtag (Optional[str]): Optional tag for logging to Weights & Biases.
            iter (Optional[int]): Optional iteration number to log.
            commit (Optional[bool]): Whether to commit the log to Weights & Biases.
            prefix (str): Prefix for the log data.
            log_output (bool): Whether to log the output message.
        """
        self.msg = msg
        self.data = {}
        self.iter = iter if iter is not None else None
        self.prefix = prefix
        self.wbtag = wbtag if wbtag is not None else None
        self.log_output = log_output
        self.commit = commit
        if wbtag is not None:
            self.data = {
                f"{self.wbtag}/dt": None,
            }
            if iter is not None:
                self.data |= {
                    f"{self.wbtag}/iter": self.iter,
                }

    def __enter__(self):
        """Start the stopwatch."""
        self.time = time.perf_counter()
        return self

    def __exit__(self, t, v, traceback):
        """Stop the stopwatch and log the time taken."""
        dt = time.perf_counter() - self.time
        # if self.wbtag is not None and wandb.run is not None:
        # if len(self.data) > 0 and wandb.run is not None:
        try:
            if (
                len(self.data) > 0
                and wandb is not None
                and (wbrun := getattr(wandb, "run", None)) is not None
            ):
                self.data |= {f"{self.wbtag}/dt": dt}
                wbrun.log({self.prefix: self.data}, commit=self.commit)
        except Exception as e:
            logger.error(f"Unable to log to wandb: {e}")
        if self.log_output:
            logger.info(f"{self.msg} took {dt:.3f} seconds")


class History:
    """
    A class to track and log metrics during training or evaluation.
    """

    def __init__(
        self,
        keys: Optional[list[str]] = None,
        *,
        report_dir: Optional[PathLike] = None,
        report_enabled: bool = True,
        jsonl_path: Optional[PathLike] = None,
        jsonl_overwrite: bool = False,
        distributed_history: bool = AUTO_USE_DISTRIBUTED_HISTORY,
    ) -> None:
        """
        Initialize the History object.

        Args:
            keys (Optional[list[str]]): List of keys to initialize the history with.
                If None, initializes with an empty list.
            report_dir (Optional[PathLike]): Directory for markdown reports. Defaults
                to ``OUTPUTS_DIR/history``.
            report_enabled (bool): Toggle automatic markdown generation.
            jsonl_path (Optional[PathLike]): Destination for JSONL metric logging.
            jsonl_overwrite (bool): Whether to truncate an existing JSONL log.
            distributed_history (bool): Enable distributed history tracking.
        """
        self.keys = [] if keys is None else keys
        self.history: dict[str, list[Any]] = {}
        self.data = self.history
        if (
            os.environ.get("EZPZ_NO_DISTRIBUTED_HISTORY", None)
            or os.environ.get("EZPZ_LOCAL_HISTORY", False)
            or ezpz.dist.get_world_size() <= 1
        ):
            logger.info(
                "Not using distributed metrics! Will only be tracked from a single rank..."
            )
            distributed_history = False
            # aggregate_metrics = False
        self.distributed_history = distributed_history
        logger.info(
            f"Using {self.__class__.__name__} with distributed_history={self.distributed_history}"
        )
        # self._aggregate_metrics = aggregate_metrics
        self._rank = get_rank()
        now = datetime.now(timezone.utc)
        self._run_id = now.strftime("%Y%m%d-%H%M%S")
        self.report_enabled = report_enabled
        base_report_root = (
            Path(report_dir)
            if report_dir is not None
            else Path(OUTPUTS_DIR).joinpath("history")
        )
        self._report_root = Path(base_report_root).expanduser().resolve()
        self._report_dir = self._report_root.joinpath(self._run_id)
        self._report_path: Optional[Path] = None
        self._asset_dir: Optional[Path] = None
        self._report_filename = "report.md"
        self._report_initialized = False
        if jsonl_path is None:
            default_jsonl_dir = (
                self._report_dir if report_enabled else Path(OUTPUTS_DIR)
            )
            self._jsonl_path = (
                Path(default_jsonl_dir)
                .expanduser()
                .resolve()
                .joinpath(f"{self._run_id}.jsonl")
            )
        else:
            self._jsonl_path = Path(jsonl_path).expanduser().resolve()
        if jsonl_overwrite and self._jsonl_path.exists():
            try:
                self._jsonl_path.unlink()
            except OSError:
                logger.warning(
                    "Unable to remove existing JSONL log at %s",
                    self._jsonl_path,
                )
        self._jsonl_enabled = True
        self._dist = torch.distributed
        self._environment_written = False
        self._metric_summary_written = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _utc_iso() -> str:
        """Return the current UTC timestamp in ISO-8601 format with trailing Z."""

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _configure_report_destination(self, base_dir: Path) -> None:
        """Configure the report directory to live under *base_dir*."""

        base_dir = base_dir.expanduser().resolve()
        self._report_root = base_dir
        self._report_dir = base_dir
        self._report_path = base_dir.joinpath(self._report_filename)
        self._asset_dir = base_dir.joinpath("assets")
        self._report_initialized = False
        if self._jsonl_enabled:
            self._jsonl_path = base_dir.joinpath(f"{self._run_id}.jsonl")
        self._environment_written = False
        self._metric_summary_written = False

    def _ensure_report_file(self) -> Optional[Path]:
        """Ensure the markdown report and asset directories exist."""

        if not self.report_enabled:
            return None
        if not self._report_initialized:
            self._report_dir.mkdir(parents=True, exist_ok=True)
            self._asset_dir = self._report_dir.joinpath("assets")
            self._asset_dir.mkdir(parents=True, exist_ok=True)
            self._report_path = self._report_dir.joinpath(
                self._report_filename
            )
            header = (
                f"# History Report ({self._run_id})\n\n"
                f"_Generated at {self._utc_iso()}_\n\n"
            )
            self._report_path.write_text(header, encoding="utf-8")
            self._report_initialized = True
        return self._report_path

    def _prepare_report_asset(self, source: Path) -> Optional[Path]:
        """Copy plot artifacts into the report asset directory."""

        report_file = self._ensure_report_file()
        if report_file is None:
            return None
        assert self._asset_dir is not None
        source = source.resolve()
        try:
            if source.is_relative_to(self._asset_dir):
                return source
        except AttributeError:  # Python < 3.9 fallback (not expected)
            pass
        destination = self._asset_dir.joinpath(source.name)
        if destination != source:
            try:
                shutil.copy2(source, destination)
            except OSError:
                logger.warning(
                    "Unable to copy asset %s into report directory.", source
                )
                return source
        return destination

    def _write_plot_report(
        self,
        key: Optional[str],
        asset_path: Path,
        *,
        kind: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a markdown section describing the generated plot."""

        report_file = self._ensure_report_file()
        if report_file is None:
            return
        asset_path = asset_path.resolve()
        if not asset_path.exists():
            return
        asset_path = self._prepare_report_asset(asset_path) or asset_path
        try:
            rel_path = asset_path.relative_to(report_file.parent)
        except ValueError:
            rel_path = asset_path
        title = key or asset_path.stem
        timestamp = self._utc_iso()
        lines = [
            f"## {title}",
            "",
            f"_Kind_: `{kind}`  ",
            f"_Generated_: {timestamp}",
            "",
        ]
        if asset_path.suffix.lower() in {".txt", ".log"}:
            try:
                text = asset_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            snippet = "\n".join(text.splitlines()[:40]).strip()
            lines.extend(["```", snippet, "```", ""])
        else:
            lines.append(f"![{title}]({rel_path.as_posix()})")
            lines.append("")
        if metadata:
            for meta_key, meta_val in metadata.items():
                lines.append(f"- **{meta_key}**: {meta_val}")
            lines.append("")
        with report_file.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if not lines[-1].endswith("\n"):
                handle.write("\n")

    def _write_environment_section(
        self, env_info: Optional[dict[str, Any]]
    ) -> None:
        """Write environment details into the report."""

        if (
            not self.report_enabled
            or env_info is None
            or self._environment_written
        ):
            return
        report_file = self._ensure_report_file()
        if report_file is None:
            return
        lines: list[str] = ["## Environment", ""]
        for section, details in env_info.items():
            if isinstance(details, dict):
                lines.append(f"### {section}")
                lines.append("")
                lines.extend((f"### {section}", ""))
                lines.extend(
                    f"- **{key}**: {value}" for key, value in details.items()
                )
                lines.append("")
            else:
                lines.append(f"- **{section}**: {details}")
        with report_file.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if not lines[-1].endswith("\n"):
                handle.write("\n")
        self._environment_written = True

    def _default_environment_info(self) -> dict[str, dict[str, str]]:
        """Return a minimal environment summary."""

        python_info = {
            "Version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "Implementation": sys.implementation.name,
            "Executable": sys.executable,
            "Platform": platform.platform(),
        }

        try:
            torch_version = torch.__version__
        except Exception:  # pragma: no cover - torch should be importable
            torch_version = "unknown"

        torch_info = {
            "Version": torch_version,
        }

        path_info = {
            "Working directory": str(Path.cwd()),
        }

        env_vars: dict[str, str] = {}
        for key in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
        ):
            value = os.environ.get(key)
            if value is not None:
                env_vars[key] = value

        summary: dict[str, dict[str, str]] = {
            "Paths": path_info,
            "Python": python_info,
            "Torch": torch_info,
        }
        if env_vars:
            summary["Environment Variables"] = env_vars
        return summary

    def _collect_metric_groups(
        self, dataset: xr.Dataset
    ) -> dict[str, dict[str, float]]:
        """Return metric statistics grouped by base metric name."""

        groups: dict[str, dict[str, float]] = {}
        for name in sorted(dataset.data_vars):
            arr = dataset[name]
            if arr.size == 0:
                continue
            try:
                latest = arr.isel({arr.dims[0]: -1})
            except Exception:
                latest = arr
            data = np.asarray(latest)
            if data.size == 0:
                continue
            value = float(data.mean()) if data.ndim > 0 else float(data.item())
            base, _, suffix = name.partition("_")
            if suffix in {"mean", "max", "min", "std"}:
                groups.setdefault(base, {})[suffix] = value
            else:
                groups.setdefault(name, {})["latest"] = value
        return groups

    def _write_metric_summary(self, dataset: xr.Dataset) -> None:
        """Append a metric overview table grouped by metric."""

        if not self.report_enabled or self._metric_summary_written:
            return
        groups = self._collect_metric_groups(dataset)
        if not groups:
            return
        report_file = self._ensure_report_file()
        if report_file is None:
            return
        with report_file.open("a", encoding="utf-8") as handle:
            handle.write("## Metric Overview\n\n")
            for metric_name, stats in groups.items():
                handle.write(f"### {metric_name}\n\n")
                handle.write("| Statistic | Value |\n")
                handle.write("| --- | --- |\n")
                for label in ("latest", "mean", "max", "min", "std"):
                    if label in stats:
                        value = stats[label]
                        handle.write(
                            f"| {label.capitalize()} | {value:.6f} |\n"
                        )
                handle.write("\n")
        self._metric_summary_written = True

    def _series_from_dataarray(self, data: xr.DataArray) -> np.ndarray:
        """Convert an xarray DataArray into a 1-D numerical series."""

        arr = np.asarray(data.values)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 0:
            return np.array([float(arr)])
        axes = tuple(range(arr.ndim - 1))
        return arr.mean(axis=axes)

    def _group_metric_variables(
        self, dataset: xr.Dataset
    ) -> dict[str, dict[str, xr.DataArray]]:
        """Group metric variables by base name and associated aggregates."""

        groups: dict[str, dict[str, xr.DataArray]] = {}
        for name, data_array in dataset.data_vars.items():
            base, sep, suffix = name.rpartition("_")
            if sep and base and suffix in {"mean", "max", "min", "std"}:
                groups.setdefault(base, {})[suffix] = data_array
            else:
                groups.setdefault(name, {})["raw"] = data_array
        return groups

    def _plot_metric_group(
        self,
        name: str,
        metric_vars: dict[str, xr.DataArray],
        *,
        warmup: Optional[float | int] = 0.0,
        title: Optional[str] = None,
        outdir: Optional[Path] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        verbose: bool = False,
    ) -> Optional[Path]:
        """Render a single matplotlib figure combining metric aggregates."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        subplots_kwargs = (
            {} if subplots_kwargs is None else dict(subplots_kwargs)
        )
        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)

        series_candidates = [
            metric_vars.get("raw"),
            metric_vars.get("mean"),
            metric_vars.get("min"),
            metric_vars.get("max"),
        ]
        # base_series = None
        # for candidate in series_candidates:
        #     if candidate is not None:
        #         base_series = self._series_from_dataarray(candidate)
        #         break
        base_series = next(
            (
                self._series_from_dataarray(candidate)
                for candidate in series_candidates
                if candidate is not None
            ),
            None,
        )
        if base_series is None or len(base_series) == 0:
            return None

        x = np.arange(base_series.shape[-1])
        fig, ax = plt.subplots(**subplots_kwargs)
        color = plot_kwargs.get("color")

        raw_da = metric_vars.get("raw")
        if raw_da is not None:
            raw_series = self._series_from_dataarray(raw_da)
            ax.plot(
                x,
                raw_series,
                label=name,
                color=color,
                alpha=0.35,
                linewidth=1.25,
            )
        mean_da = metric_vars.get("mean")
        std_da = metric_vars.get("std")
        min_da = metric_vars.get("min")
        max_da = metric_vars.get("max")

        mean_series = None
        if mean_da is not None:
            mean_series = self._series_from_dataarray(mean_da)
            ax.plot(
                x,
                mean_series,
                label=f"{name} mean",
                color=color,
                linewidth=2.0,
            )

        if mean_series is not None and std_da is not None:
            std_series = self._series_from_dataarray(std_da)
            upper = mean_series + std_series
            lower = mean_series - std_series
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.2,
                label=f"{name} ± std",
            )
        elif min_da is not None and max_da is not None:
            min_series = self._series_from_dataarray(min_da)
            max_series = self._series_from_dataarray(max_da)
            ax.fill_between(
                x,
                min_series,
                max_series,
                color=color,
                alpha=0.15,
                label=f"{name} range",
            )

        if (
            mean_da is None
            and raw_da is None
            and min_da is None
            and max_da is None
        ):
            # fall back to plotting whichever aggregate is available
            for label, array in metric_vars.items():
                series = self._series_from_dataarray(array)
                ax.plot(x, series, label=f"{name} {label}", linewidth=1.75)

        ax.set_xlabel("draw")
        ax.set_ylabel(name)
        if title is not None:
            ax.set_title(title)
        sns.despine(ax=ax, top=True, right=True)
        ax.legend(loc="best", frameon=False)

        if outdir is None and self.report_enabled:
            save_dir = self._report_dir.joinpath("mplot")
        elif outdir is not None:
            save_dir = Path(outdir)
        else:
            save_dir = None

        primary_asset: Optional[Path] = None
        if save_dir is not None:
            save_dir = save_dir.expanduser().resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            asset_name = name.replace("/", "_")
            dirs = {
                "png": save_dir.joinpath("pngs"),
                "svg": save_dir.joinpath("svgs"),
            }
            for directory in dirs.values():
                directory.mkdir(parents=True, exist_ok=True)
            if verbose:
                logger.info("Saving %s plot to: %s", name, save_dir)
            for ext, directory in dirs.items():
                outfile = directory.joinpath(f"{asset_name}.{ext}")
                if outfile.exists():
                    outfile = directory.joinpath(
                        f"{asset_name}-{get_timestamp()}.{ext}"
                    )
                fig.savefig(outfile, dpi=400, bbox_inches="tight")
                if primary_asset is None and ext == "png":
                    primary_asset = outfile
        plt.close(fig)
        return primary_asset

    def _tplot_metric_group(
        self,
        name: str,
        metric_vars: dict[str, xr.DataArray],
        *,
        warmup: Optional[float | int] = 0.0,
        outdir: Optional[Path] = None,
        plot_type: Optional[str] = None,
        verbose: bool = False,
        logfreq: Optional[int] = None,
    ) -> Optional[Path]:
        """Render grouped metrics into a single text-based plot asset."""

        outdir = Path(outdir) if outdir is not None else None
        if outdir is None and self.report_enabled:
            outdir = self._report_dir.joinpath("tplot")
        if outdir is None:
            return None
        outdir = outdir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        asset_path = outdir.joinpath(f"{name.replace('/', '_')}.txt")

        order = [
            ("raw", name),
            ("mean", f"{name} mean"),
            ("max", f"{name} max"),
            ("min", f"{name} min"),
            ("std", f"{name} std"),
        ]
        wrote_any = False
        points = 0
        append_flag = False
        for key, label in order:
            data_array = metric_vars.get(key)
            if data_array is None:
                continue
            series = self._series_from_dataarray(data_array)
            points = max(points, len(series))
            self._tplot(
                y=series,
                xlabel="iter",
                ylabel=label,
                append=append_flag,
                outfile=asset_path.as_posix(),
                verbose=verbose,
                plot_type=plot_type,
                logfreq=(1 if logfreq is None else logfreq),
                record_report=False,
            )
            append_flag = True
            wrote_any = True
        if wrote_any and self.report_enabled:
            self._write_plot_report(
                name,
                asset_path,
                kind="tplot",
                metadata={
                    "components": ", ".join(
                        key for key, _ in order if key in metric_vars
                    ),
                    "points": points,
                },
            )
        return asset_path

    def _write_jsonl_entry(
        self,
        metrics: dict[str, Any],
        aggregated: Optional[dict[str, float]] = None,
    ) -> None:
        """Append metrics to the configured JSONL log."""

        if not self._jsonl_enabled:
            return
        if self._jsonl_path is None:
            return
        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "rank": self._rank,
            "metrics": metrics,
        }
        if aggregated and self._rank == 0:
            payload["aggregated"] = aggregated
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, default=self._to_serializable)
                )
                handle.write("\n")
        except OSError:
            logger.warning(
                "Unable to write JSONL metrics to %s", self._jsonl_path
            )

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        """Convert values to JSON-serializable structures."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, Path):
            return value.as_posix()
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.numel() == 1:
                return tensor.item()
            return tensor.cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {
                key: cls._to_serializable(sub_value)
                for key, sub_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._to_serializable(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    @classmethod
    def _sanitize_metrics(cls, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of metrics with values converted to JSON-safe types."""

        return {
            key: cls._to_serializable(value) for key, value in metrics.items()
        }

    def _iter_scalar_metrics(
        self, metrics: dict[str, Any]
    ) -> Iterable[tuple[str, float]]:
        """Yield scalar metrics suitable for distributed reductions."""

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                yield key, float(value)
            elif isinstance(value, np.ndarray) and value.shape == ():
                yield key, float(value.item())
            elif torch.is_tensor(value) and value.numel() == 1:
                yield key, float(value.item())

    def _select_metric_device(self) -> torch.device:
        """Return the device to use for distributed metric aggregation."""

        candidate = ezpz.get_torch_device(as_torch_device=True)
        device = (
            candidate
            if isinstance(candidate, torch.device)
            else torch.device(str(candidate))
        )
        device_type = device.type
        if device_type == "mps":
            return torch.device("cpu")
        if device_type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        xpu_backend = getattr(torch, "xpu", None)
        if device_type == "xpu" and not (
            xpu_backend and xpu_backend.is_available()
        ):
            return torch.device("cpu")
        return device

    def _compute_distributed_metrics(
        self, metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Compute distributed reductions for scalar metrics."""

        if not self.distributed_history or self._dist is None:
            return {}
        try:
            if (
                not self._dist.is_available()
                or not torch.distributed.is_initialized()  # type: ignore[attr-defined]
            ):
                return {}
        except AttributeError:
            return {}
        scalars = dict(self._iter_scalar_metrics(metrics))
        if not scalars:
            return {}
        metric_device = self._select_metric_device()
        dtype = torch.get_default_dtype()
        values = torch.tensor(
            list(scalars.values()),
            dtype=dtype,
            device=metric_device,
        )
        sum_vals = values.clone()
        sq_vals = values.square()
        max_vals = values.clone()
        min_vals = values.clone()
        # world_size = ezpz.dist.get_world_size()
        # world_size = self._dist.get_world_size()
        if (world_size := ezpz.dist.get_world_size()) <= 1:
            return {
                f"{key}/{suffix}": (value if suffix != "std" else 0.0)
                for key, value in scalars.items()
                for suffix in ("mean", "max", "min", "std")
            }

        # ezpz.dist.all_reduce(sum_vals, op=ops.SUM, implementation="torch")
        # ezpz.dist.all_reduce(sq_vals, op=ops.SUM, implementation="torch")
        # ezpz.dist.all_reduce(max_vals, op=ops.MAX, implementation="torch")
        # ezpz.dist.all_reduce(min_vals, op=ops.MIN, implementation="torch")
        # ops = self._dist.ReduceOp  # type: ignore[attr-defined]
        ops = torch.distributed.ReduceOp  # type: ignore[attr-defined]
        torch.distributed.all_reduce(sum_vals, op=ops.SUM)
        torch.distributed.all_reduce(sq_vals, op=ops.SUM)
        torch.distributed.all_reduce(max_vals, op=ops.MAX)
        torch.distributed.all_reduce(min_vals, op=ops.MIN)
        mean_vals = sum_vals.div(world_size)
        var_vals = sq_vals.div(world_size).sub(mean_vals.square())
        std_vals = var_vals.clamp_min_(0.0).sqrt_()
        stats: dict[str, float] = {}
        for idx, key in enumerate(scalars):
            # if any([s in key] for s in ["iter", "epoch", "step", "batch"]):
            #     continue
            stats[f"{key}/mean"] = float(mean_vals[idx].item())
            stats[f"{key}/max"] = float(max_vals[idx].item())
            stats[f"{key}/min"] = float(min_vals[idx].item())
            stats[f"{key}/std"] = float(std_vals[idx].item())
        return stats

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def _update(
        self,
        key: str,
        val: Union[Any, ScalarLike, list, torch.Tensor, np.ndarray],
    ):
        """
        Update the history with a new key-value pair.

        Args:
            key (str): The key to update in the history.
            val (Union[Any, ScalarLike, list, torch.Tensor, np.ndarray]): The value
                to associate with the key.
        """
        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]
        return val

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def update(
        self,
        metrics: dict,
        precision: int = 6,
        use_wandb: Optional[bool] = True,
        commit: Optional[bool] = True,
        summarize: Optional[bool] = True,
    ) -> str:
        """
        Update the history with a dictionary of metrics.

        Args:
            metrics (dict): Dictionary of metrics to update the history with.
            precision (int): Precision for summarizing the metrics.
            use_wandb (Optional[bool]): Whether to log the metrics to Weights & Biases.
            commit (Optional[bool]): Whether to commit the log to Weights & Biases.
            summarize (Optional[bool]): Whether to summarize the metrics.
        """
        for key, val in metrics.items():
            # if isinstance(val, (list, np.ndarray, torch.Tensor)):
            #     val = grab_tensor(val)
            try:
                self.history[key].append(val)
            except KeyError:
                self.history[key] = [val]
        aggregated_metrics = self._compute_distributed_metrics(metrics)
        if aggregated_metrics and self._rank == 0:
            for agg_key, agg_val in aggregated_metrics.items():
                self._update(agg_key, agg_val)
        metrics_for_logging = dict(metrics)
        if aggregated_metrics and self._rank == 0:
            metrics_for_logging.update(aggregated_metrics)
        sanitized_metrics = self._sanitize_metrics(metrics_for_logging)
        summary_source = (
            sanitized_metrics
            if aggregated_metrics and self._rank == 0
            else self._sanitize_metrics(metrics)
        )
        if (
            wandb is not None
            and use_wandb
            # and not WANDB_DISABLED
            and getattr(wandb, "run", None) is not None
        ):
            wandb.log(sanitized_metrics, commit=commit)
        self._write_jsonl_entry(sanitized_metrics, aggregated_metrics)
        if summarize:
            scalar_summary = {
                key: value
                for key, value in summary_source.items()
                # skip keys like "train/iter/min", "eval/step/std", etc.,
                if not any(
                    count_str in key
                    for count_str in [
                        "iter/min",
                        "iter/max",
                        "iter/std",
                        "iter/avg",
                        "iter/mean",
                        "step/min",
                        "step/max",
                        "step/std",
                        "step/avg",
                        "step/mean",
                        "epoch/min",
                        "epoch/max",
                        "epoch/std",
                        "epoch/avg",
                        "epoch/mean",
                        "batch/min",
                        "batch/max",
                        "batch/std",
                        "batch/avg",
                        "batch/mean",
                        "idx/min",
                        "idx/max",
                        "idx/std",
                        "idx/avg",
                        "idx/mean",
                    ]
                )
            }
            # _ss = {"max", "min", "std"}
            # _sk = {"iter", "step", "epoch", "batch", "idx"}
            # keys_to_skip = [
            #     f"{i}/{s}" for s in _ss for i in _sk
            # ]
            if scalar_summary:
                return summarize_dict(
                    scalar_summary, precision=precision
                ).replace("/", "/")
            return ""
        return ""

    @staticmethod
    def split_metrics_for_logging(
        metrics: dict[str, Any],
        debug_prefixes: tuple[str, ...] = ("hist/",),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        info_metrics: dict[str, Any] = {}
        debug_metrics: dict[str, Any] = {}
        for key, value in metrics.items():
            if key.startswith(debug_prefixes):
                debug_metrics[key] = value
            else:
                info_metrics[key] = value
        return info_metrics, debug_metrics

    @staticmethod
    def summarize_min_max_std(
        metrics: dict[str, Any],
    ) -> dict[str, float]:
        numeric: dict[str, list[float]] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                numeric[key] = [float(value)]
            elif torch.is_tensor(value) and value.numel() == 1:
                numeric[key] = [float(value.item())]
        summary: dict[str, float] = {}
        for key, values in numeric.items():
            if not values:
                continue
            t = torch.tensor(values)
            summary[f"{key}/mean"] = float(t.mean().item())
            summary[f"{key}/min"] = float(t.min().item())
            summary[f"{key}/max"] = float(t.max().item())
            summary[f"{key}/std"] = float(t.std(unbiased=False).item())
        return summary

    def summarize_distributed_min_max_std(
        self, metrics: dict[str, Any]
    ) -> dict[str, float]:
        summary_stats = self._compute_distributed_metrics(metrics)
        if not summary_stats:
            summary_stats = self.summarize_min_max_std(metrics)
        filtered: dict[str, float] = {
            k: v
            for k, v in summary_stats.items()
            if k.endswith(("/mean", "/min", "/max", "/std"))
        }
        keys = {k.rsplit("/", 1)[0] for k in filtered}
        pruned: dict[str, float] = {}
        for base in keys:
            mean_v = filtered.get(f"{base}/mean")
            min_v = filtered.get(f"{base}/min")
            max_v = filtered.get(f"{base}/max")
            std_v = filtered.get(f"{base}/std")
            if (
                mean_v == 0.0
                and min_v == 0.0
                and max_v == 0.0
                and std_v == 0.0
            ):
                continue
            if mean_v is not None:
                pruned[f"{base}/mean"] = mean_v
            if min_v is not None:
                pruned[f"{base}/min"] = min_v
            if max_v is not None:
                pruned[f"{base}/max"] = max_v
            if std_v is not None:
                pruned[f"{base}/std"] = std_v
        return pruned

    def log_metrics(
        self,
        metrics: dict[str, Any],
        *,
        logger: Optional[Any] = None,
        debug_prefixes: tuple[str, ...] = ("hist/",),
        include_summary: bool = True,
        rank0_only_summary: bool = True,
        precision: int = 6,
    ) -> None:
        log = logger if logger is not None else get_logger(__name__)
        info_metrics, debug_metrics = self.split_metrics_for_logging(
            metrics, debug_prefixes=debug_prefixes
        )
        info_msg = summarize_dict(
            info_metrics, precision=precision
        ).replace("train/", "")
        if info_msg:
            log.info(info_msg)
        if include_summary:
            summary_stats = self.summarize_distributed_min_max_std(
                info_metrics
            )
            if summary_stats and (
                not rank0_only_summary or self._rank == 0
            ):
                summary_msg = summarize_dict(
                    summary_stats, precision=precision
                ).replace("train/", "")
                if summary_msg:
                    log.info(summary_msg)
        debug_msg = summarize_dict(
            debug_metrics, precision=precision
        ).replace("train/", "")
        if debug_msg:
            log.debug(debug_msg)

    def _tplot(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        append: bool = True,
        title: Optional[str] = None,
        verbose: bool = False,
        outfile: Optional[str] = None,
        logfreq: Optional[int] = None,
        plot_type: Optional[str] = None,
        record_report: bool = True,
    ):
        """
        Create a text plot of the given data.

        Args:
            y (np.ndarray): The data to plot.
            x (Optional[np.ndarray]): The x-axis data.
            xlabel (Optional[str]): The x-axis label.
            ylabel (Optional[str]): The y-axis label.
            append (bool): Whether to append to an existing plot.
            title (Optional[str]): The title of the plot.
            verbose (bool): Whether to print the plot.
            outfile (Optional[str]): The path to save the plot to.
            logfreq (Optional[int]): The log frequency of the plot.
            plot_type (Optional[str]): The type of plot to create.
        """
        outfile_path: Optional[Path] = None
        if outfile is None and self.report_enabled:
            label = (ylabel or xlabel or "metric").replace("/", "_")
            default_dir = self._report_dir.joinpath("tplot")
            default_dir.mkdir(parents=True, exist_ok=True)
            outfile_path = default_dir.joinpath(
                f"{label}-{get_timestamp()}.txt"
            )
            outfile = outfile_path.as_posix()
        elif outfile is not None:
            outfile_path = Path(outfile)
        if xlabel is not None and ylabel == xlabel:
            return
        if len(y) > 1:
            x = x if x is not None else np.arange(len(y))
            assert x is not None
            eztplot(
                y=y,
                x=x,
                xlabel=xlabel,
                ylabel=ylabel,
                logfreq=(1 if logfreq is None else logfreq),
                append=append,
                verbose=verbose,
                outfile=outfile,
                plot_type=plot_type,
                title=title,
                # plot_type=('scatter' if 'dt' in ylabel else None),
            )
            if (
                record_report
                and self.report_enabled
                and outfile_path is not None
            ):
                self._write_plot_report(
                    ylabel,
                    outfile_path,
                    kind="tplot",
                    metadata={"points": len(y)},
                )
        if ylabel is not None and "dt" in ylabel:
            of = Path(outfile) if outfile is not None else None
            if of is not None:
                of = Path(of.parent).joinpath(f"{of.stem}-hist{of.suffix}")
            eztplot(
                y=y,
                xlabel=ylabel,
                title=title,
                ylabel="freq",
                append=append,
                verbose=verbose,
                outfile=(of if of is not None else None),
                plot_type="hist",
            )
            if record_report and self.report_enabled and of is not None:
                self._write_plot_report(
                    f"{ylabel}-hist",
                    of,
                    kind="tplot-hist",
                    metadata={"points": len(y)},
                )

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def plot(
        self,
        val: np.ndarray,
        key: Optional[str] = None,
        warmup: Optional[float] = 0.0,
        num_chains: Optional[int] = 128,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """
        Plot a single variable from the history.

        NOTE: The `warmup` argument can be used to drop the first `warmup`
        iterations (as a percent of the total number of iterations) from the
        plot.

        Args:
            val (np.ndarray): The data to plot.
            key (Optional[str]): The key for the data.
            warmup (Optional[float]): The percentage of iterations to drop from the
                beginning of the plot.
            num_chains (Optional[int]): The number of chains to plot.
            title (Optional[str]): The title of the plot.
            outdir (Optional[os.PathLike]): The directory to save the plot to.
            subplots_kwargs (Optional[dict[str, Any]]): Additional arguments for
                subplots.
            plot_kwargs (Optional[dict[str, Any]]): Additional arguments for plotting.
            verbose (bool): Emit additional logging when saving plots.
        """
        import matplotlib.pyplot as plt

        LW = plt.rcParams.get("axes.linewidth", 1.75)
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get("figsize", ezplot.set_size())
        subplots_kwargs.update({"figsize": figsize})
        num_chains = 16 if num_chains is None else num_chains

        # tmp = val[0]
        arr = np.array(val)

        subfigs = None
        steps = np.arange(arr.shape[0])
        if warmup is not None and warmup > 0 and arr.size > 0:
            if isinstance(warmup, int) or warmup >= 1:
                warmup_frac = float(warmup) / float(arr.shape[0])
            else:
                warmup_frac = float(warmup)
            warmup_frac = min(max(warmup_frac, 0.0), 1.0)
            drop = min(int(round(warmup_frac * arr.shape[0])), arr.shape[0])
            if drop > 0:
                arr = arr[drop:]
                steps = steps[drop:]

        if len(arr.shape) == 2:
            import seaborn as sns

            _ = subplots_kwargs.pop("constrained_layout", True)
            figsize = (3 * figsize[0], 1.5 * figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)

            gs_kw = {"width_ratios": [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(
                1, 2, sharey=True, gridspec_kw=gs_kw
            )
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get("color", None)
            label = r"$\langle$" + f" {key} " + r"$\rangle$"
            ax.plot(
                steps, arr.mean(-1), lw=1.5 * LW, label=label, **plot_kwargs
            )
            sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, shade=True)
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            # ax1.set_yticks([])
            # ax1.set_yticklabels([])
            sns.despine(ax=ax, top=True, right=True)
            sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
            # ax.legend(loc='best', frameon=False)
            ax1.set_xlabel("")
            # ax1.set_ylabel('')
            # ax.set_yticks(ax.get_yticks())
            # ax.set_yticklabels(ax.get_yticklabels())
            # ax.set_ylabel(key)
            # _ = subfigs[1].subplots_adjust(wspace=-0.75)
            axes = (ax, ax1)
        else:
            if len(arr.shape) == 1:
                fig, ax = plt.subplots(**subplots_kwargs)
                # assert isinstance(ax, plt.Axes)
                ax.plot(steps, arr, **plot_kwargs)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = plt.subplots(**subplots_kwargs)
                # assert isinstance(ax, plt.Axes)
                cmap = plt.get_cmap("viridis")
                nlf = arr.shape[1]
                for idx in range(nlf):
                    # y = arr[:, idx, :].mean(-1)
                    # pkwargs = {
                    #     'color': cmap(idx / nlf),
                    #     'label': f'{idx}',
                    # }
                    # ax.plot(steps, y, **pkwargs)
                    label = plot_kwargs.pop("label", None)
                    if label is not None:
                        label = f"{label}-{idx}"
                    y = arr[:, idx, :]
                    color = cmap(idx / y.shape[1])
                    plot_kwargs["color"] = cmap(idx / y.shape[1])
                    if len(y.shape) == 2:
                        # TOO: Plot chains
                        if num_chains > 0:
                            for idx in range(min((num_chains, y.shape[1]))):
                                _ = ax.plot(
                                    steps,
                                    y[:, idx],  # color,
                                    lw=LW / 2.0,
                                    alpha=0.8,
                                    **plot_kwargs,
                                )

                        _ = ax.plot(
                            steps,
                            y.mean(-1),  # color=color,
                            label=label,
                            **plot_kwargs,
                        )
                    else:
                        _ = ax.plot(
                            steps,
                            y,  # color=color,
                            label=label,
                            **plot_kwargs,
                        )
                axes = ax
            else:
                raise ValueError("Unexpected shape encountered")

            ax.set_ylabel(key)

        if num_chains > 0 and len(arr.shape) > 1:
            # lw = LW / 2.
            for idx in range(min(num_chains, arr.shape[1])):
                # ax = subfigs[0].subplots(1, 1)
                # plot values of invidual chains, arr[:, idx]
                # where arr[:, idx].shape = [ndraws, 1]
                ax.plot(
                    steps, arr[:, idx], alpha=0.5, lw=LW / 2.0, **plot_kwargs
                )

        ax.set_xlabel("draw")
        if title is not None:
            fig.suptitle(title)

        save_dir: Optional[Path]
        if outdir is not None:
            save_dir = Path(outdir).expanduser().resolve()
        elif self.report_enabled:
            save_dir = self._report_dir.joinpath("mplot")
        else:
            save_dir = None

        if save_dir is not None:
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            save_dir.mkdir(parents=True, exist_ok=True)
            outfile = save_dir.joinpath(f"{key}.svg")
            if outfile.is_file():
                tstamp = ezpz.get_timestamp()
                pngdir = save_dir.joinpath("pngs")
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f"{key}-{tstamp}.png")
                svgfile = save_dir.joinpath(f"{key}-{tstamp}.svg")
                plt.savefig(pngfile, dpi=400, bbox_inches="tight")
                plt.savefig(svgfile, dpi=400, bbox_inches="tight")
        primary_asset: Optional[Path] = None
        if save_dir is not None:
            dirs = {
                "png": Path(save_dir).joinpath("pngs/"),
                "svg": Path(save_dir).joinpath("svgs/"),
            }
            _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
            for ext, d in dirs.items():
                outfile = d.joinpath(f"{key}.{ext}")
                if outfile.is_file():
                    outfile = d.joinpath(f"{key}-subfig.{ext}")
                if verbose:
                    logger.info(f"Saving {key} plot to: {outfile.resolve()}")
                plt.savefig(outfile, dpi=400, bbox_inches="tight")
                if primary_asset is None and ext == "png":
                    primary_asset = outfile
        if (
            self.report_enabled
            and primary_asset is not None
            and Path(primary_asset).exists()
        ):
            self._write_plot_report(
                key,
                primary_asset,
                kind="matplotlib",
                metadata={"shape": list(arr.shape)},
            )

        return fig, subfigs, axes

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def plot_dataArray(
        self,
        val: xr.DataArray,
        key: Optional[str] = None,
        warmup: Optional[float] = 0.0,
        num_chains: Optional[int] = 0,
        title: Optional[str] = None,
        outdir: Optional[str] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        verbose: bool = False,
        line_labels: bool = False,
        logfreq: Optional[int] = None,
    ):
        """
        Plot a single variable from the history as an xarray DataArray.

        Args:
            val (xr.DataArray): The data to plot.
            key (Optional[str]): The key for the data.
            warmup (Optional[float]): The percentage of iterations to drop from the
                beginning of the plot.
            num_chains (Optional[int]): The number of chains to plot.
            title (Optional[str]): The title of the plot.
            outdir (Optional[str]): The directory to save the plot to.
            subplots_kwargs (Optional[dict[str, Any]]): Additional arguments for
                subplots.
            plot_kwargs (Optional[dict[str, Any]]): Additional arguments for plotting.
            verbose (bool): Whether to print the plot.
            line_labels (bool): Whether to label lines in the plot.
            logfreq (Optional[int]): The log frequency of the plot.
        """
        import matplotlib.pyplot as plt

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        ezplot.set_plot_style()
        plt.rcParams["axes.labelcolor"] = "#bdbdbd"
        figsize = subplots_kwargs.get("figsize", ezplot.set_size())
        subplots_kwargs.update({"figsize": figsize})
        subfigs = None
        # if key == 'dt':
        #     warmup = 0.2
        arr = val.values  # shape: [nchains, ndraws]
        # steps = np.arange(len(val.coords['draw']))
        steps = val.coords["draw"]
        if warmup is not None and warmup > 0.0 and arr.size > 0:
            if isinstance(warmup, int) or warmup >= 1:
                warmup_frac = float(warmup) / float(arr.shape[0])
            else:
                warmup_frac = float(warmup)
            warmup_frac = min(max(warmup_frac, 0.0), 1.0)
            drop = min(int(round(warmup_frac * arr.shape[0])), arr.shape[0])
            if drop > 0:
                arr = arr[drop:]
            steps = steps[drop:]
        if len(arr.shape) == 2:
            fig, axes = ezplot.plot_combined(
                val,
                key=key,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
        else:
            if len(arr.shape) == 1:
                fig, ax = ezplot.subplots(**subplots_kwargs)
                try:
                    ax.plot(steps, arr, **plot_kwargs)
                except ValueError:
                    try:
                        ax.plot(steps, arr[~np.isnan(arr)], **plot_kwargs)
                    except Exception:
                        logger.error(f"Unable to plot {key}! Continuing")
                _ = ax.grid(True, alpha=0.2)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = ezplot.subplots(**subplots_kwargs)
                cmap = plt.get_cmap("viridis")
                y = val.mean("chain")
                for idx in range(len(val.coords["leapfrog"])):
                    pkwargs = {
                        "color": cmap(idx / len(val.coords["leapfrog"])),
                        "label": f"{idx}",
                    }
                    ax.plot(steps, y[idx], **pkwargs)
                axes = ax
            else:
                raise ValueError("Unexpected shape encountered")
            ax = plt.gca()
            # assert isinstance(ax, plt.Axes)
            assert key is not None
            _ = ax.set_ylabel(key)
            _ = ax.set_xlabel("draw")
            # if num_chains > 0 and len(arr.shape) > 1:
            #     lw = LW / 2.
            #     #for idx in range(min(num_chains, arr.shape[1])):
            #     nchains = len(val.coords['chains'])
            #     for idx in range(min(nchains, num_chains)):
            #         # ax = subfigs[0].subplots(1, 1)
            #         # plot values of invidual chains, arr[:, idx]
            #         # where arr[:, idx].shape = [ndraws, 1]
            #         ax.plot(steps, val
            #                 alpha=0.5, lw=lw/2., **plot_kwargs)
        if title is not None:
            fig = plt.gcf()
            _ = fig.suptitle(title)
        if logfreq is not None:
            ax = plt.gca()
            xticks = ax.get_xticks()  # type: ignore
            _ = ax.set_xticklabels(  # type: ignore
                [f"{logfreq * int(i)}" for i in xticks]  # type: ignore
            )
        save_dir: Optional[Path]
        if outdir is not None:
            save_dir = Path(outdir).expanduser().resolve()
        elif self.report_enabled:
            save_dir = self._report_dir.joinpath("dataarray")
        else:
            save_dir = None
        primary_asset: Optional[Path] = None
        if save_dir is not None:
            dirs = {
                "png": Path(save_dir).joinpath("pngs/"),
                "svg": Path(save_dir).joinpath("svgs/"),
            }
            _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
            if verbose:
                logger.info(
                    f"Saving {key} plot to: {Path(save_dir).resolve()}"
                )
            for ext, d in dirs.items():
                outfile = d.joinpath(f"{key}.{ext}")
                plt.savefig(outfile, dpi=400, bbox_inches="tight")
                if primary_asset is None and ext == "png":
                    primary_asset = outfile
        if (
            self.report_enabled
            and primary_asset is not None
            and Path(primary_asset).exists()
        ):
            metadata = {"dims": list(val.dims)}
            self._write_plot_report(
                key,
                primary_asset,
                kind="dataarray",
                metadata=metadata,
            )
        return (fig, subfigs, axes)

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def plot_dataset(
        self,
        title: Optional[str] = None,
        nchains: Optional[int] = None,
        outdir: Optional[os.PathLike] = None,
        dataset: Optional[xr.Dataset] = None,
        data: Optional[dict] = None,
        warmup: Optional[int | float] = None,
        # subplots_kwargs: Optional[dict[str, Any]] = None,
        # plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        dataset = (
            dataset
            if dataset is not None
            else (
                self.get_dataset(
                    data=(data if data is not None else self.history),
                    warmup=warmup,
                )
            )
        )
        return ezplot.plot_dataset(
            dataset=dataset,
            nchains=nchains,
            title=title,
            outdir=outdir,
        )

    def plot_2d_xarr(
        self,
        xarr: xr.DataArray,
        label: Optional[str] = None,
        num_chains: Optional[int] = None,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        LW = plt.rcParams.get("axes.linewidth", 1.75)
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        assert len(xarr.shape) == 2
        assert "draw" in xarr.coords and "chain" in xarr.coords
        num_chains = len(xarr.chain) if num_chains is None else num_chains
        # _ = subplots_kwargs.pop('constrained_layout', True)
        figsize = plt.rcParams.get("figure.figsize", (8, 6))
        figsize = (3 * figsize[0], 1.5 * figsize[1])
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        subfigs = fig.subfigures(1, 2)
        gs_kw = {"width_ratios": [1.33, 0.33]}
        (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
        ax.grid(alpha=0.2)
        ax1.grid(False)
        color = plot_kwargs.get("color", f"C{np.random.randint(6)}")
        label = r"$\langle$" + f" {label} " + r"$\rangle$"
        ax.plot(
            xarr.draw.values,
            xarr.mean("chain"),
            color=color,
            lw=1.5 * LW,
            label=label,
            **plot_kwargs,
        )
        for idx in range(num_chains):
            # ax = subfigs[0].subplots(1, 1)
            # plot values of invidual chains, arr[:, idx]
            # where arr[:, idx].shape = [ndraws, 1]
            # ax0.plot(
            #     xarr.draw.values,
            #     xarr[xarr.chain == idx][0],
            #     lw=1.,
            #     alpha=0.7,
            #     color=color
            # )
            ax.plot(
                xarr.draw.values,
                xarr[xarr.chain == idx][0],
                color=color,
                alpha=0.5,
                lw=LW / 2.0,
                **plot_kwargs,
            )

        axes = (ax, ax1)
        sns.kdeplot(y=xarr.values.flatten(), ax=ax1, color=color, shade=True)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        # ax1.set_yticks([])
        # ax1.set_yticklabels([])
        sns.despine(ax=ax, top=True, right=True)
        sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
        # ax.legend(loc='best', frameon=False)
        ax1.set_xlabel("")
        # ax1.set_ylabel('')
        # ax.set_yticks(ax.get_yticks())
        # ax.set_yticklabels(ax.get_yticklabels())
        # ax.set_ylabel(key)
        # _ = subfigs[1].subplots_adjust(wspace=-0.75)
        # if num_chains > 0 and len(arr.shape) > 1:
        # lw = LW / 2.
        # num_chains = np.min([
        #     16,
        #     len(xarr.coords['chain']),
        # ])
        sns.despine(subfigs[0])
        ax0 = subfigs[0].subplots(1, 1)
        im = xarr.plot(ax=ax0)  # type:ignore
        im.colorbar.set_label(label)  # type:ignore
        # ax0.plot(
        #     xarr.draw.values,
        #     xarr.mean('chain'),
        #     lw=2.,
        #     color=color
        # )
        # for idx in range(min(num_chains, i.shape[1])):
        ax.set_xlabel("draw")
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            assert label is not None
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f"{label}.svg")
            if outfile.is_file():
                tstamp = get_timestamp("%Y-%m-%d-%H%M%S")
                pngdir = Path(outdir).joinpath("pngs")
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f"{label}-{tstamp}.png")
                svgfile = Path(outdir).joinpath(f"{label}-{tstamp}.svg")
                plt.savefig(pngfile, dpi=400, bbox_inches="tight")
                plt.savefig(svgfile, dpi=400, bbox_inches="tight")

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def tplot_all(
        self,
        outdir: Optional[os.PathLike] = None,
        warmup: Optional[float] = 0.0,
        append: bool = True,
        xkey: Optional[str] = None,
        dataset: Optional[xr.Dataset] = None,
        data: Optional[dict] = None,
        logfreq: Optional[int] = None,
        plot_type: Optional[str] = None,
        verbose: bool = False,
    ):
        dataset = (
            dataset
            if dataset is not None
            else (
                self.get_dataset(
                    data=(data if data is not None else self.history),
                    warmup=warmup,
                )
            )
        )

        outdir_path = Path(os.getcwd()) if outdir is None else Path(outdir)
        groups = self._group_metric_variables(dataset)
        for metric_name, metric_vars in sorted(groups.items()):
            if (xkey is not None and metric_name == xkey) or xkey in [
                "iter",
                "draw",
            ]:
                continue
            self._tplot_metric_group(
                metric_name,
                metric_vars,
                warmup=warmup,
                outdir=outdir_path,
                plot_type=plot_type,
                verbose=verbose,
                logfreq=logfreq,
            )

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def plot_all(
        self,
        num_chains: int = 128,
        warmup: Optional[float | int] = 0.0,
        title: Optional[str] = None,
        verbose: bool = False,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        dataset: Optional[xr.Dataset] = None,
        data: Optional[dict] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        subplots_kwargs = (
            {} if subplots_kwargs is None else dict(subplots_kwargs)
        )

        dataset = (
            dataset
            if dataset is not None
            else (
                self.get_dataset(
                    data=(data if data is not None else self.history),
                    warmup=warmup,
                )
            )
        )

        _ = ezplot.make_ridgeplots(
            dataset,
            outdir=outdir,
            drop_nans=True,
            drop_zeros=False,
            num_chains=num_chains,
            cmap="viridis",
            save_plot=(outdir is not None),
        )

        groups = self._group_metric_variables(dataset)
        for idx, (metric_name, metric_vars) in enumerate(
            sorted(groups.items())
        ):
            plot_kwargs["color"] = f"C{idx % 9}"
            asset = self._plot_metric_group(
                metric_name,
                metric_vars,
                warmup=warmup,
                title=title,
                outdir=Path(outdir) if outdir is not None else None,
                subplots_kwargs=subplots_kwargs,
                plot_kwargs=plot_kwargs,
                verbose=verbose,
            )
            if asset is not None and self.report_enabled and asset.exists():
                components = sorted(metric_vars.keys())
                sample_series = self._series_from_dataarray(
                    metric_vars[components[0]]
                )
                self._write_plot_report(
                    metric_name,
                    asset,
                    kind="matplotlib",
                    metadata={
                        "components": ", ".join(components),
                        "points": len(sample_series),
                    },
                )
        return dataset

    def history_to_dict(self) -> dict:
        # return {k: np.stack(v).squeeze() for k, v in self.history.items()}
        return {
            k: torch.Tensor(v).numpy(force=True)
            for k, v in self.history.items()
        }

    def to_DataArray(
        self,
        x: Union[list, np.ndarray, torch.Tensor],
        warmup: Optional[float] = 0.0,
    ) -> xr.DataArray:
        if isinstance(x, list) and isinstance(x[0], torch.Tensor):
            x = torch.Tensor(x).numpy(force=True)
        try:
            arr = grab_tensor(x)
        except ValueError:
            arr = np.array(x).real
            # arr = np.array(x)
            logger.info(f"len(x): {len(x)}")
            logger.info(f"x[0].shape: {x[0].shape}")
            logger.info(f"arr.shape: {arr.shape}")
        assert isinstance(arr, np.ndarray)
        if warmup is not None and warmup > 0 and len(arr) > 0:
            if isinstance(warmup, int):
                warmup = warmup / len(arr)
            # drop = int(warmup * arr.shape[0])
            drop = int(warmup * len(arr))
            arr = arr[drop:]
        # steps = np.arange(len(arr))
        if len(arr.shape) == 1:  # [ndraws]
            ndraws = arr.shape[0]
            dims = ["draw"]
            coords = [np.arange(len(arr))]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        if len(arr.shape) == 2:  # [nchains, ndraws]
            arr = arr.T
            nchains, ndraws = arr.shape
            dims = ("chain", "draw")
            coords = [np.arange(nchains), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        if len(arr.shape) == 3:  # [nchains, nlf, ndraws]
            arr = arr.T
            nchains, nlf, ndraws = arr.shape
            dims = ("chain", "leapfrog", "draw")
            coords = [np.arange(nchains), np.arange(nlf), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        else:
            print(f"arr.shape: {arr.shape}")
            raise ValueError("Invalid shape encountered")

    def get_dataset(
        self,
        data: Optional[
            dict[str, Union[list, np.ndarray, torch.Tensor]]
        ] = None,
        warmup: Optional[float] = 0.0,
    ):
        data = self.history_to_dict() if data is None else data
        data_vars = {}
        for key, val in data.items():
            name = key.replace("/", "_")
            try:
                data_vars[name] = self.to_DataArray(val, warmup)
            except ValueError:
                logger.error(
                    f"Unable to create DataArray for {key}! Skipping!"
                )
                logger.error(f"{key}.shape= {np.stack(val).shape}")  # type:ignore
        return xr.Dataset(data_vars)

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def save_dataset(
        self,
        outdir: PathLike,
        fname: str = "dataset",
        use_hdf5: bool = True,
        data: Optional[
            dict[str, Union[list, np.ndarray, torch.Tensor]]
        ] = None,
        dataset: Optional[xr.Dataset] = None,
        warmup: Optional[int | float] = None,
        **kwargs,
    ) -> Path:
        dataset = (
            dataset
            if dataset is not None
            else (
                self.get_dataset(
                    data=(data if data is not None else self.history),
                    warmup=warmup,
                )
            )
        )
        return save_dataset(
            dataset,
            outdir=outdir,
            fname=fname,
            use_hdf5=use_hdf5,
            **kwargs,
        )

    @timeitlogit(rank=get_rank(), record=True, verbose=False, prefix="history")
    def finalize(
        self,
        outdir: Optional[PathLike] = None,
        run_name: Optional[str] = None,
        dataset_fname: Optional[str] = None,
        num_chains: int = 128,
        warmup: Optional[int | float] = 0.0,
        verbose: bool = False,
        save: bool = True,
        plot: bool = True,
        append_tplot: bool = True,
        title: Optional[str] = None,
        data: Optional[
            dict[str, Union[list, np.ndarray, torch.Tensor]]
        ] = None,
        dataset: Optional[xr.Dataset] = None,
        xkey: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        tplot_type: Optional[str] = None,
        env_info: Optional[dict[str, Any]] = None,
    ) -> xr.Dataset:
        dataset = (
            dataset
            if dataset is not None
            else (
                self.get_dataset(
                    data=(data if data is not None else self.history),
                    warmup=warmup,
                )
            )
        )
        run_name = (
            f"History-{get_timestamp()}" if run_name is None else run_name
        )
        if outdir is None:
            base_dir = (
                Path(os.getcwd())
                .joinpath("outputs", run_name, get_timestamp())
                .expanduser()
                .resolve()
            )
        else:
            base_dir = Path(outdir).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        self._configure_report_destination(base_dir)
        env_details = (
            env_info
            if env_info is not None
            else self._default_environment_info()
        )
        self._write_environment_section(env_details)
        self._write_metric_summary(dataset)
        if plot:
            logger.info(
                "Saving plots to %s (matplotlib) and %s (tplot)",
                base_dir.joinpath("plots", "mplot"),
                base_dir.joinpath("plots", "tplot"),
            )
            plotdir = base_dir.joinpath("plots")
            tplotdir = plotdir.joinpath("tplot")
            mplotdir = plotdir.joinpath("mplot")
            tplotdir.mkdir(exist_ok=True, parents=True)
            mplotdir.mkdir(exist_ok=True, parents=True)
            _ = self.plot_all(
                dataset=dataset,
                outdir=mplotdir,
                verbose=verbose,
                num_chains=num_chains,
                warmup=warmup,
                title=title,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            _ = self.tplot_all(
                dataset=dataset,
                outdir=tplotdir,
                warmup=warmup,
                append=append_tplot,
                plot_type=tplot_type,
                xkey=xkey,
                verbose=verbose,
            )
        if save:
            try:
                import h5py

                use_hdf5 = True
            except ImportError:
                logger.warning(
                    "h5py not found! Saving dataset as netCDF instead."
                )
                use_hdf5 = False

            fname = "dataset" if dataset_fname is None else dataset_fname
            _ = self.save_dataset(
                dataset=dataset,
                outdir=base_dir,
                fname=fname,
                use_hdf5=use_hdf5,
            )
        if self.report_enabled:
            logger.info(
                "Saving history report to %s",
                base_dir.joinpath(self._report_filename),
            )
        return dataset
