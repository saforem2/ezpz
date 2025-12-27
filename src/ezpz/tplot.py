"""
tplot.py
"""

import os
from typing import Optional, Union

import numpy as np
import torch
import ezpz

try:
    import plotext as pltx
except ImportError:  # pragma: no cover - optional dependency
    pltx = None
from pathlib import Path


logger = ezpz.get_logger(__name__)

DEFAULT_MARKER = "hd"  # fixed-width block characters for consistent text plots
MAX_PLOT_WIDTH = 120
MAX_PLOT_HEIGHT = 40


def _clamp_width(width: Optional[int]) -> int | None:
    env_max_width = int(os.environ.get("EZPZ_TPLOT_MAX_WIDTH", MAX_PLOT_WIDTH))
    cw = None if width is None else min(width, env_max_width)
    return cw

def _clamp_height(height: Optional[int]) -> int | None:
    env_max_height = int(os.environ.get("EZPZ_TPLOT_MAX_HEIGHT", MAX_PLOT_HEIGHT))
    ch = None if height is None else min(height, env_max_height)
    return ch

def _clamp_size(
    width: Optional[int], height: Optional[int]
) -> tuple[int | None, int | None]:
    cw = _clamp_width(width)
    ch = _clamp_height(height)
    return cw, ch


def _resolve_plot_type(
    plot_type: Optional[str], *, default: Optional[str] = None
) -> Optional[str]:
    env_type = os.environ.get("EZPZ_TPLOT_TYPE")
    if plot_type is not None:
        return plot_type
    if env_type is not None:
        return env_type
    return default


def _resolve_marker(
    marker: Optional[str] = None, *, plot_type: Optional[str] = None
) -> Optional[str]:
    env_marker = os.environ.get("EZPZ_TPLOT_MARKER")
    resolved = marker if marker is not None else env_marker
    if resolved is None and plot_type != "hist":
        resolved = DEFAULT_MARKER
    return resolved


def _require_plotext():
    if pltx is None:
        raise ModuleNotFoundError(
            "plotext is required for terminal plotting. Install it via "
            "`pip install plotext` to use ezpz.tplot functions."
        )
    return pltx


def plotext_prepare_figure(theme: str = "clear"):
    plotext = _require_plotext()
    if hasattr(plotext, "clear_figure"):
        plotext.clear_figure()
    elif hasattr(plotext, "clf"):
        plotext.clf()
    if theme and hasattr(plotext, "theme"):
        plotext.theme(theme)
    return plotext


def plotext_set_size(
    plotext,
    *,
    width: Optional[int] = None,
    height_scale: float = 2.0,
    min_height: int = 20,
) -> None:
    if not hasattr(plotext, "plot_size"):
        return
    if width is None and hasattr(plotext, "tw"):
        width = plotext.tw()
    height = None
    if hasattr(plotext, "th"):
        height = max(min_height, int(plotext.th() * height_scale))
    if width is None:
        width = 80
    width, height = _clamp_size(width, height)
    plotext.plot_size(width, height)


def plotext_plot_series(
    plotext,
    series: np.ndarray,
    *,
    yerr: Optional[np.ndarray] = None,
    xerr: Optional[np.ndarray] = None,
    label: Optional[str],
    color: Optional[str] = None,
    marker: Optional[str] = None,
    plot_type: Optional[str] = None,
) -> None:
    # marker: Optional[str] = "braille",
    plot_type = _resolve_plot_type(plot_type)
    marker = _resolve_marker(marker, plot_type=plot_type)
    if plot_type is not None:
        logger.debug(f"Using plot type: {plot_type} for {label}")
    if marker is not None:
        logger.debug(f"Using plot marker: {marker} for {label}")
    if plot_type is None:
        # plotext.plot(y, label=label, marker=marker)
        try:
            if label:
                if color is not None:
                    plotext.plot(
                        series, label=label, color=color, marker=marker
                    )
                else:
                    plotext.plot(series, label=label, marker=marker)
            else:
                if color is not None:
                    plotext.plot(series, color=color, marker=marker)
                else:
                    plotext.plot(series, marker=marker)
        except TypeError:
            if label:
                plotext.plot(series, label=label, marker=marker)
            else:
                plotext.plot(series, marker=marker)
    else:
        if plot_type == "scatter":
            # marker = env_marker if env_marker is not None else "braille"
            plotext.scatter(series, label=label, marker=marker, color=color)
        elif plot_type == "line":
            # marker = env_marker if env_marker is not None else "braille"
            plotext.plot(series, marker=marker, label=label, color=color)
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            plotext.plot(series, label=label, color=color)
    if yerr is not None or xerr is not None:
        plotext.error(
            series,
            yerr=yerr,
            xerr=xerr,
            label=f"{label} error" if label else None,
            color=color,
            marker=marker,
        )


    # if len(y.shape) == 2:
    # else:


def plotext_hist_series(
    plotext,
    series: np.ndarray,
    *,
    label: Optional[str],
    bins: Optional[int] = None,
) -> None:
    if label:
        if bins is not None:
            plotext.hist(series, bins=bins, label=label)
        else:
            plotext.hist(series, label=label)
    else:
        if bins is not None:
            plotext.hist(series, bins=bins)
        else:
            plotext.hist(series)


def plotext_subplots(
    *,
    layout: tuple[int, int] = (1, 2),
    left_layout: tuple[int, int] = (3, 1),
    right_layout: tuple[int, int] = (2, 1),
    theme: str = "clear",
    tick_style: Optional[str] = None,
    height_scale: float = 1.0,
):
    plotext = plotext_prepare_figure(theme=theme)
    assert plotext is not None
    width = None
    height = None
    if hasattr(plotext, "tw"):
        width = _clamp_width(plotext.tw())
    if hasattr(plotext, "th") and height_scale is not None:
        height = _clamp_height(int(plotext.th() * height_scale))
    if width is None:
        width = _clamp_width(80)
    plotext.plot_size(width, height)
    plotext.subplots(*layout)
    left = plotext.subplot(1, 1)
    right = plotext.subplot(1, 2)

    if (
        hasattr(left, "plotsize")
        and hasattr(right, "plotsize")
    ):
        total_width = width if width is not None else _clamp_width(80)
        half_width = max(20, (total_width or 80) // 2)
        half_width, height = _clamp_size(half_width, height)
        left.plotsize(half_width, height)
        right.plotsize(half_width, height)

    if hasattr(left, "subplots"):
        left.subplots(*left_layout)
    if hasattr(right, "subplots"):
        right.subplots(*right_layout)
    if tick_style and hasattr(left, "ticks_style"):
        left.ticks_style(tick_style)
    if tick_style and hasattr(right, "ticks_style"):
        right.ticks_style(tick_style)

    return plotext, left, right


def get_plot_title(
    ylabel: Optional[str], xlabel: Optional[str], label: Optional[str]
) -> str:
    if ylabel is not None and xlabel is not None:
        return f"{ylabel} vs {xlabel}"
    if ylabel is not None:
        return ylabel
    if label is not None:
        return label
    return ""


def tplot_dict(
    data: dict,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    append: bool = True,
    figsize: Optional[tuple[int, int]] = None,
) -> None:
    figsize = (75, 25) if figsize is None else figsize

    plotext = _require_plotext()

    plotext.clear_figure()
    plotext.theme("clear")  # pyright[ReportUnknownMemberType]
    w, h = _clamp_size(*figsize)
    plotext.plot_size(w, h)
    plotext.plot(list(data.values()))
    if ylabel is not None:
        plotext.ylabel(ylabel)
    if xlabel is not None:
        plotext.xlabel(xlabel)
    if title is not None:
        plotext.title(title)
    plotext.show()
    if outfile is not None:
        logger.info(f"Appending plot to: {outfile}")
        if not Path(outfile).parent.exists():
            _ = Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plotext.save_fig(outfile, append=append)


def tplot(
    y: Union[list, np.ndarray, torch.Tensor],
    x: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    marker: Optional[str] = None,
    bins: Optional[int] = None,
    # bins: int = 60,
    logfreq: int = 1,
    outfile: Optional[os.PathLike | str | Path] = None,
    plot_type: Optional[str] = None,
    append: bool = True,
    verbose: bool = False,
    figsize: Optional[tuple[int, int]] = None,
):
    # if isinstance(y, list):
    #     if len(y) > 0 and isinstance(y[0], torch.Tensor):
    #         y = torch.stack(y)
    #     if isinstance(y[0], )
    # tstamp = get_timestamp()
    plot_type = _resolve_plot_type(plot_type, default="line")
    title = (
        get_plot_title(ylabel=ylabel, xlabel=xlabel, label=label)
        if title is None
        else title
    )
    if isinstance(y, list):
        y = torch.stack(y).numpy()
    if x is not None and isinstance(x, list):
        x = torch.stack(x).numpy()
    assert isinstance(y, (np.ndarray, torch.Tensor))
    y = np.nan_to_num(y, nan=0.0)

    figsize = (60, 20) if figsize is None else figsize
    plotext = _require_plotext()

    plotext.clear_figure()
    plotext.theme("clear")
    plotext.plot_size(*_clamp_size(*figsize))
    # marker = "braille" if (marker is None and type == 'scatter') else marker
    marker = _resolve_marker(marker, plot_type=plot_type)
    if plot_type is not None:
        logger.info(f"Using plot type: {plot_type}")
    if marker is not None:
        logger.info(f"Using plot marker: {marker}")
    # if len(y.shape) == 2:
    if (yshape := getattr(y, "shape")) and yshape and len(yshape) == 2:
        plotext.hist(y.flatten(), bins=bins, label=label)
    else:
        if plot_type is None:
            plotext.plot(y, label=label, marker=marker)
        else:
            if plot_type == "scatter":
                plotext.scatter(y, label=label, marker=marker)
            elif plot_type == "line":
                plotext.plot(y, marker=marker, label=label)
            elif plot_type == "hist":
                marker = None
                plotext.hist(y, bins=bins, label=label)
            else:
                logger.warning(f"Unknown plot type: {plot_type}")
                plotext.plot(y, label=label)

    # elif len(y.shape) == 1:
    #     # if type is not None:
    #     #     assert type in ['scatter', 'line']
    #     if plot_type is not None and plot_type == "scatter":
    #         marker = "braille" if marker is None else marker
    #         plotext.scatter(y, label=label, marker=marker)
    #     elif plot_type is None or plot_type == "line":
    #         marker = "braille" if marker is None else marker
    #         plotext.plot(y, marker=marker, label=label)
    #     elif plot_type is not None and plot_type == "hist":
    #         plotext.hist(y, bins=bins, label=label)
    #     else:
    #         logger.warning(f"Unknown plot type: {plot_type}")
    #         plotext.plot(y, label=label)
    #     # else:
    #     #     pltx.plot(y, label=label)
    if title is not None:
        plotext.title(title)
    if ylabel is not None:
        plotext.ylabel(ylabel)
    if xlabel is not None:
        plotext.xlabel(xlabel)
    if plot_type != "hist":
        if x is None:
            x = np.arange(len(y))
            x = x * logfreq
        if x is not None:
            assert len(x.shape) == 1
            plotext.xticks(x.tolist())
    plotext.show()
    if outfile is not None:
        if verbose:
            if append:
                logger.info(f"Appending plot to: {outfile}")
            else:
                logger.info(f"Saving plot to: {outfile}")
        if not Path(outfile).parent.exists():
            _ = Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plotext.savefig(
            Path(outfile).resolve().as_posix(), append=append, keep_colors=True
        )
