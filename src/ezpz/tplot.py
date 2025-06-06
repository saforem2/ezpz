"""
tplot.py
"""
import os

import torch

import numpy as np
from typing import Optional, Union
import plotext as pltx
from pathlib import Path

from ezpz.log import get_logger

logger = get_logger(__name__)

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

    pltx.clear_figure()
    pltx.theme("clear")  # pyright[ReportUnknownMemberType]
    pltx.plot(list(data.values()))
    if ylabel is not None:
        pltx.ylabel(ylabel)
    if xlabel is not None:
        pltx.xlabel(xlabel)
    if title is not None:
        pltx.title(title)
    pltx.show()
    if outfile is not None:
        logger.info(f"Appending plot to: {outfile}")
        if not Path(outfile).parent.exists():
            _ = Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        pltx.save_fig(outfile, append=append)

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
    plot_type = "line" if plot_type is None else plot_type
    title = (
        get_plot_title(ylabel=ylabel, xlabel=xlabel, label=label)
        if title is None
        else title
    )
    if isinstance(y, list):
        y = torch.stack(y)
    if isinstance(x, list):
        x = torch.stack(x)
    figsize = (60, 20) if figsize is None else figsize
    import plotext as pltx

    pltx.clear_figure()
    pltx.theme("clear")
    pltx.plot_size(*figsize)
    # marker = "braille" if (marker is None and type == 'scatter') else marker
    y = np.nan_to_num(y, nan=0.0)
    if len(y.shape) == 2:
        pltx.hist(y.flatten(), bins=bins, label=label)
    elif len(y.shape) == 1:
        # if type is not None:
        #     assert type in ['scatter', 'line']
        if plot_type is not None and plot_type == "scatter":
            marker = "braille" if marker is None else marker
            pltx.scatter(y, label=label, marker=marker)
        elif plot_type is None or plot_type == "line":
            # marker = "braille" if marker is None else marker
            pltx.plot(y, marker=marker, label=label)
        elif plot_type is not None and plot_type == "hist":
            pltx.hist(y, bins=bins, label=label)
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            pltx.plot(y, label=label)
        # else:
        #     pltx.plot(y, label=label)
    if title is not None:
        pltx.title(title)
    if ylabel is not None:
        pltx.ylabel(ylabel)
    if xlabel is not None:
        pltx.xlabel(xlabel)
    if plot_type != "hist":
        if x is None:
            x = np.arange(len(y))
            x = x * logfreq
        if x is not None:
            assert len(x.shape) == 1
            pltx.xticks(x.tolist())
    pltx.show()
    if outfile is not None:
        if verbose:
            if append:
                logger.info(f"Appending plot to: {outfile}")
            else:
                logger.info(f"Saving plot to: {outfile}")
        if not Path(outfile).parent.exists():
            _ = Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        pltx.savefig(
            Path(outfile).resolve().as_posix(), append=append, keep_colors=True
        )
