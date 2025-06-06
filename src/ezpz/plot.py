"""
plot_helpers.py

Contains helpers for plotting.
"""

from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
import time
from typing import Any, Optional, Union

import torch

# import ezpz
from ezpz.dist import get_rank
from ezpz.log import get_logger
from ezpz.utils import get_timestamp

# from ezpz.dist import get_rank
import numpy as np
import xarray as xr

RANK = get_rank()

logger = get_logger(__name__)

xplt = xr.plot  # type: ignore

PLOTS_LOG = Path(os.getcwd()).joinpath("plots.txt")

COLORS = {
    "blue": "#2196F3",
    "red": "#EF5350",
    "green": "#4CAF50",
    "orange": "#FFA726",
    "purple": "#AE81FF",
    "yellow": "#ffeb3b",
    "pink": "#EC407A",
    "teal": "#009688",
    "white": "#CFCFCF",
}

# ["#2196F3", "#EF5350", "#4CAF50", "#FFA726", "#AE81FF", "#ffeb3b", "#EC407A", "#009688", "#CFCFCF"]

# sns.set_context('talk')
# set_plot_style()
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# set_plot_style()


# def _get_timestamp(fstr: Optional[str] = None) -> str:
#     """Get formatted timestamp."""
#     import datetime
#
#     now = datetime.datetime.now()
#     if fstr is None:
#         return now.strftime('%Y-%m-%d-%H%M%S')
#     return now.strftime(fstr)


def set_plot_style(**kwargs):
    import matplotlib.pyplot as plt

    # LW = plt.rcParams.get('axes.linewidth', 1.75)
    # FigAxes = Tuple[plt.Figure, plt.Axes]
    plt.style.use("default")
    # plt.style.use('default')
    try:
        # import ambivalent

        # STYLES = ambivalent.STYLES
        # from toolbox import set_plot_style
        from ambivalent import STYLES

        plt.style.use(STYLES["opinionated_min"])
    except (ImportError, ModuleNotFoundError):
        STYLES = {}

    plt.rcParams.update(
        {
            "image.cmap": "viridis",
            "savefig.transparent": True,
            # 'text.color': '#666666',
            # 'xtick.color': '#66666604',
            # 'ytick.color': '#66666604',
            # 'ytick.labelcolor': '#666666',
            # 'xtick.labelcolor': '#666666',
            # 'axes.edgecolor': '#66666600',
            # 'axes.labelcolor': '#666666',
            # 'grid.linestyle': ':',
            # 'grid.alpha': 0.4,
            # 'grid.color': '#353535',
            "path.simplify": True,
            "savefig.bbox": "tight",
            "legend.labelcolor": "#838383",
            # 'axes.labelcolor': (189, 189, 189, 1.0),
            # 'grid.color': (0.434, 0.434, 0.434, 0.2),  # #66666602
            # 'axes.facecolor': (1.0, 1.0, 1.0, 0.0),
            # 'figure.facecolor': (1.0, 1.0, 1.0, 0.0),
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.format": "svg",
            "axes.edgecolor": "none",
            "axes.grid": True,
            "axes.labelcolor": "#838383",
            "axes.titlecolor": "#838383",
            "grid.color": "#838383",
            "text.color": "#838383",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.33,
            "xtick.color": "none",
            "ytick.color": "none",
            "xtick.labelcolor": "#838383",
            "legend.edgecolor": "none",
            "ytick.labelcolor": "#838383",
            # 'savefig.transparent': True,
        }
    )
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", list(COLORS.values()))
    plt.rcParams["axes.labelcolor"] = "#838383"
    plt.rcParams.update(**kwargs)
    # plt.rcParams |= {'figure.figsize': [12.4, 4.8]}
    # plt.style.use(STYLES['opinionated_min'])
    # plt.rcParams['ytick.labelsize'] = 14.0
    # plt.rcParams['xtick.labelsize'] = 14.0
    # plt.rcParams['grid.alpha'] = 0.4
    # grid_color = plt.rcParams['grid.color']
    # if not is_interactive():
    #     figsize = plt.rcParamsDefault.get('figure.figsize', (4.5, 3))
    #     x = figsize[0]
    #     y = figsize[1]
    #     plt.rcParams['figure.figsize'] = [2.5 * x, 2. * y]
    #     plt.rcParams['figure.dpi'] = 400


def save_figure(fig: Any, fname: str, outdir: os.PathLike):
    pngdir = Path(outdir).joinpath("pngs")
    pngdir.mkdir(exist_ok=True, parents=True)
    pngfile = pngdir.joinpath(f"{fname}.png")
    svgfile = Path(outdir).joinpath(f"{fname}.svg")

    _ = fig.savefig(pngfile, dpi=400, bbox_inches="tight")
    _ = fig.savefig(svgfile, dpi=400, bbox_inches="tight")
    with PLOTS_LOG.open("a") as f:
        f.write(f"{fname}: {svgfile.as_posix()}\n")


def savefig(fig: Any, outfile: os.PathLike):
    fout = Path(outfile)
    parent = fout.parent
    parent.mkdir(exist_ok=True, parents=True)
    relpath = fout.relative_to(os.getcwd())
    logger.info(
        "Saving figure to: " rf"[link={fout.as_posix()}]{relpath}[\link]"
    )
    with PLOTS_LOG.open("a") as f:
        f.write(f"{fout.as_posix()}\n")
    _ = fig.savefig(fout.as_posix(), dpi=400, bbox_inches="tight")


def subplots(**kwargs) -> tuple:
    """Returns (fig, axis) tuple"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(**kwargs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    return fig, ax


def plot_arr(
    metric: list,
    name: Optional[str] = None,
) -> tuple:
    """Returns (fig, axis) tuple"""
    assert len(metric) > 0
    y = np.stack(metric)
    if isinstance(metric[0], (int, float, bool, np.floating)):
        return plot_scalar(y, ylabel=name)
    element_shape = metric[0].shape
    if len(element_shape) == 2:
        # y = grab_tensor(torch.stack(metric))
        return plot_leapfrogs(y, ylabel=name)
    if len(element_shape) == 1:
        # y = grab_tensor(torch.stack(metric))
        return plot_chains(y, ylabel=name)
    raise ValueError


def plot_scalar(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fig_axes: Optional[tuple] = None,
    outfile: Optional[os.PathLike] = None,
    **kwargs,
) -> tuple:
    assert len(y.shape) == 1
    if x is None:
        x = np.arange(len(y))

    if fig_axes is None:
        fig, ax = subplots()
    else:
        fig, ax = fig_axes

    _ = ax.plot(x, y, label=label, **kwargs)
    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)
    # if label is not None and MATPLOTX:
    #     _ = matplotx.line_labels()

    if outfile is not None:
        savefig(fig, outfile)

    return fig, ax


def plot_chains(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    num_chains: Optional[int] = 8,
    fig_axes: Optional[tuple] = None,
    label: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    outfile: Optional[os.PathLike] = None,
    **kwargs,
) -> tuple:
    assert len(y.shape) == 2
    # y.shape = [ndraws, nchains]
    num_chains = 8 if num_chains is None else num_chains
    nchains = min(num_chains, y.shape[1])
    if x is None:
        x = np.arange(y.shape[0])

    if fig_axes is None:
        fig, ax = subplots()
    else:
        fig, ax = fig_axes

    if label is not None:
        label = f"{label}, avg: {y.mean():4.3g}"

    _ = kwargs.pop("color", None)
    color = f"C{np.random.randint(8)}"
    _ = ax.plot(x, y.mean(-1), label=label, color=color, lw=1.5, **kwargs)
    for idx in range(nchains):
        _ = ax.plot(
            x,
            y[:, idx],
            # ls='--',
            lw=0.6,
            color=color,
            alpha=0.6,
            **kwargs,
        )

    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)

    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)

    # if label is not None:
    #     _ = matplotx.line_labels()

    if outfile is not None:
        savefig(fig, outfile)
    xlim = ax.get_xlim()
    _ = ax.set_xlim(xlim[0], xlim[1] + 1)
    return fig, ax


def plot_leapfrogs(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    fig_axes: Optional[tuple] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    outfile: Optional[os.PathLike] = None,
    # line_labels: Optional[bool] = False,
) -> tuple:
    import matplotlib.pyplot as plt

    assert len(y.shape) == 3

    if fig_axes is None:
        fig, ax = subplots()
    else:
        fig, ax = fig_axes

    if x is None:
        x = np.arange(y.shape[0])

    # y.shape = [ndraws, nleapfrog, nchains]
    nlf = y.shape[1]
    yavg = y.mean(-1)
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(n / nlf) for n in range(nlf)}
    for lf in range(nlf):
        _ = ax.plot(x, yavg[:, lf], color=colors[lf], label=f"{lf}")

    # if line_labels:
    #     _ = matplotx.line_labels(font_kwargs={'size': 'small'})

    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)

    if outfile is not None:
        savefig(fig, outfile)

    return fig, ax


def plot_dataset(
    dataset: xr.Dataset,
    nchains: Optional[int] = 10,
    logfreq: Optional[int] = None,
    outdir: Optional[os.PathLike] = None,
    title: Optional[str] = None,
    job_type: Optional[str] = None,
    save_plots: bool = True,
) -> None:
    tstamp = get_timestamp()
    outdir = (
        Path(outdir)
        if outdir is not None
        else (Path(os.getcwd()).joinpath(f"{tstamp}"))
    )
    outdir.mkdir(exist_ok=True, parents=True)
    job_type = job_type if job_type is not None else f"job-{tstamp}"
    # set_plot_style()

    _ = make_ridgeplots(
        dataset,
        outdir=outdir,
        drop_nans=True,
        drop_zeros=False,
        num_chains=nchains,
        cmap="viridis",
    )
    for key, val in dataset.data_vars.items():
        if key == "x":
            continue

        fig, _, _ = plot_dataArray(
            val,
            key=key,
            logfreq=logfreq,
            outdir=outdir,
            title=title,
            # line_labels=False,
            num_chains=nchains,
            save_plot=save_plots,
        )
        if save_plots:
            _ = save_figure(fig=fig, fname=key, outdir=outdir)


def plot_combined(
    val: xr.DataArray,
    key: Optional[str] = None,
    num_chains: Optional[int] = 10,
    subplots_kwargs: Optional[dict[str, Any]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
) -> tuple:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    figsize = subplots_kwargs.get("figsize", set_size())
    subplots_kwargs.update({"figsize": figsize})
    subfigs = None
    num_chains = 10 if num_chains is None else num_chains

    _ = subplots_kwargs.pop("constrained_layout", True)
    figsize = (3 * figsize[0], 1.5 * figsize[1])
    line_width = plt.rcParams.get("axes.linewidth", 1.75)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    gs_kw = {"width_ratios": [1.33, 0.33]}
    vmin = np.min(val)
    vmax = np.max(val)
    if vmin < 0 < vmax:
        color = "#FF5252" if val.mean() > 0 else "#2979FF"
    elif 0 < vmin < vmax:
        color = "#3FB5AD"
    else:
        color = plot_kwargs.get("color", f"C{np.random.randint(5)}")

    (ax1, ax2) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
    ax1.grid(alpha=0.2)
    ax2.grid(False)
    sns.kdeplot(y=val.values.flatten(), ax=ax2, color=color, fill=True)
    axes = (ax1, ax2)
    ax0 = subfigs[0].subplots(1, 1)
    if "chain" in val.dims:
        val = val.dropna("chain")
        _ = xplt.imshow(
            val[:num_chains, :],
            "draw",
            "chain",
            ax=ax0,
            robust=True,
            add_colorbar=True,
        )
    # _ = xplt.pcolormesh(val, 'draw', 'chain', ax=ax0,
    #                     robust=True, add_colorbar=True)
    # sns.despine(ax0)
    nchains = min(num_chains, len(val.coords["chain"]))
    label = f"{key}_avg"
    # label = r'$\langle$' + f'{key} ' + r'$\rangle$'
    # steps = np.arange(len(val.coords['draw']))
    # steps = val.coords['draw']
    chain_axis = val.get_axis_num("chain")
    if chain_axis == 0:
        for idx in range(nchains):
            _ = ax1.plot(
                val.coords["draw"].values,
                val.values[idx, :],
                color=color,
                lw=line_width / 2.0,
                alpha=0.6,
            )

    _ = ax1.plot(
        val.coords["draw"].values,
        val.mean("chain"),
        color=color,
        label=label,
        lw=1.5 * line_width,
    )
    if key is not None and "eps" in key:
        _ = ax0.set_ylabel("leapfrog")
    _ = ax2.set_xticks([])
    _ = ax2.set_xticklabels([])
    # sns.despine(ax=ax0, top=True, right=True, left=True, bottom=True)
    _ = sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
    _ = sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
    _ = ax2.set_xlabel("")
    _ = ax1.set_xlabel("draw")
    _ = sns.despine(subfigs[0])
    _ = plt.autoscale(enable=True, axis=ax0)
    return (fig, axes)


def plot_dataArray(
    val: xr.DataArray,
    key: Optional[str] = None,
    therm_frac: Optional[float] = 0.0,
    logfreq: Optional[int] = None,
    num_chains: Optional[int] = 10,
    title: Optional[str] = None,
    outdir: Optional[str | Path] = None,
    subplots_kwargs: Optional[dict[str, Any]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    # line_labels: Optional[bool] = False,
    save_plot: bool = True,
) -> tuple:
    import matplotlib.pyplot as plt

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    set_plot_style()
    plt.rcParams["axes.labelcolor"] = "#bdbdbd"
    figsize = subplots_kwargs.get("figsize", set_size())
    subplots_kwargs.update({"figsize": figsize})
    subfigs = None
    # if key == 'dt':
    #     therm_frac = 0.2
    arr = val.values  # shape: [nchains, ndraws]
    # steps = np.arange(len(val.coords['draw']))
    steps = val.coords["draw"]
    if therm_frac is not None and therm_frac > 0.0:
        drop = int(therm_frac * arr.shape[0])
        arr = arr[drop:]
        steps = steps[drop:]
    if len(arr.shape) == 2:
        fig, axes = plot_combined(
            val,
            key=key,
            num_chains=num_chains,
            plot_kwargs=plot_kwargs,
            subplots_kwargs=subplots_kwargs,
        )
    else:
        if len(arr.shape) == 1:
            fig, ax = subplots(**subplots_kwargs)
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
            fig, ax = subplots(**subplots_kwargs)
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
        assert isinstance(ax, plt.Axes)
        _ = ax.set_ylabel(key)
        _ = ax.set_xlabel("draw")
        # matplotx.line_labels()
        # if line_labels:
        #     matplotx.line_labels()
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
        xticks = ax.get_xticks()  # type:ignore
        _ = ax.set_xticklabels(  # type:ignore
            [  # type:ignore
                f"{logfreq * int(i)}" for i in xticks
            ]
        )
    if outdir is not None and save_plot:
        outfile = Path(outdir).joinpath(f"{key}.svg")
        if outfile.is_file():
            tstamp = get_timestamp("%Y-%m-%d-%H%M%S")
            pngdir = Path(outdir).joinpath("pngs")
            pngdir.mkdir(exist_ok=True, parents=True)
            pngfile = pngdir.joinpath(f"{key}-{tstamp}.png")
            svgfile = Path(outdir).joinpath(f"{key}-{tstamp}.svg")
            _ = plt.savefig(pngfile, dpi=400, bbox_inches="tight")
            _ = plt.savefig(svgfile, dpi=400, bbox_inches="tight")
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
    return (fig, subfigs, axes)


def plot_array(
    val: list | np.ndarray,
    key: Optional[str] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    num_chains: Optional[int] = 10,
    outdir: Optional[str | Path] = None,
    **kwargs,
) -> tuple:
    import matplotlib.pyplot as plt

    fig, ax = subplots(constrained_layout=True)
    arr = np.array(val)
    if num_chains is None:
        num_chains = 10
    # arr.shape = [ndraws, nleapfrog, nchains]
    if len(arr.shape) == 3:
        ndraws, nlf, _ = arr.shape
        lfarr = np.arange(nlf)
        cmap = plt.get_cmap("viridis")
        colors = {lf: cmap(lf / nlf) for lf in lfarr}
        yarr = arr.transpose((1, 0, 2))  # shape: [nleapfrog, ndraws, nchains]
        for idx, ylf in enumerate(yarr):
            y = ylf.mean(-1)  # average over chains, shape = [ndraws]
            x = np.arange(len(y))
            _ = ax.plot(x, y, label=f"{idx}", color=colors[idx], **kwargs)
        x = np.arange(ndraws)
        _ = ax.plot(x, yarr.mean((0, 1)), **kwargs)
        # arr = arr.mean()
    # arr.shape = [ndraws, nchains]
    elif len(arr.shape) == 2:
        # ndraws, nchains = arr.shape
        for idx in range(min((arr.shape[1], num_chains))):
            y = arr[:, idx]
            x = np.arange(len(y))
            _ = ax.plot(x, y, lw=1.0, alpha=0.7, **kwargs)
        y = arr.mean(-1)
        x = np.arange(len(y))
        _ = ax.plot(x, y, label=key, **kwargs)
    elif len(arr.shape) == 1:
        y = arr
        x = np.arange(y.shape[0])
        _ = ax.plot(x, y, label=key, **kwargs)
    else:
        raise ValueError(f"Unexpected shape encountered: {arr.shape}")
    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if title is not None:
        _ = ax.set_title(title)
    _ = ax.legend(loc="best")
    if outdir is not None:
        outfile = Path(outdir).joinpath(f"{key}.svg")
        if outfile.is_file():
            tstamp = get_timestamp("%Y-%m-%d-%H%M%S")
            pngdir = Path(outdir).joinpath("pngs")
            pngdir.mkdir(exist_ok=True, parents=True)
            pngfile = pngdir.joinpath(f"{key}-{tstamp}.png")
            svgfile = Path(outdir).joinpath(f"{key}-{tstamp}.svg")
            _ = plt.savefig(pngfile, dpi=400, bbox_inches="tight")
            _ = plt.savefig(svgfile, dpi=400, bbox_inches="tight")
    return fig, ax


def set_size(
    width: Optional[str] = None,
    fraction: Optional[float] = None,
    subplots: Optional[tuple] = None,
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX."""
    width_pt = 345.0
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    fraction = 1.0 if fraction is None else fraction
    subplots = (1, 1) if subplots is None else subplots
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set asethetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_metric(
    val: np.ndarray | xr.DataArray,
    key: Optional[str] = None,
    therm_frac: Optional[float] = 0.0,
    num_chains: Optional[int] = 0,
    logfreq: Optional[int] = None,
    title: Optional[str] = None,
    outdir: Optional[os.PathLike] = None,
    subplots_kwargs: Optional[dict[str, Any]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
    ext: Optional[str] = "png",
    # line_labels: Optional[bool] = False,
) -> tuple:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    figsize = subplots_kwargs.get("figsize", set_size())
    subplots_kwargs.update({"figsize": figsize})
    therm_frac = 0.0 if therm_frac is None else therm_frac
    num_chains = 16 if num_chains is None else num_chains
    line_width = plt.rcParams.get("axes.linewidth", 1.75)

    # tmp = val[0]
    arr = np.array(val)

    subfigs = None
    steps = np.arange(arr.shape[0])
    if therm_frac > 0:
        drop = int(therm_frac * arr.shape[0])
        arr = arr[drop:]
        steps = steps[drop:]

    # arr.shape = [draws, chains]
    if len(arr.shape) == 2:
        _ = subplots_kwargs.pop("constrained_layout", True)
        figsize = (3 * figsize[0], 1.5 * figsize[1])

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        subfigs = fig.subfigures(1, 2)

        gs_kw = {"width_ratios": [1.33, 0.33]}
        (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
        _ = ax.grid(alpha=0.4)
        _ = ax1.grid(False)
        color = plot_kwargs.get("color", "C0")
        label = plot_kwargs.pop("label", None)
        # label = r'$\langle$' + f' {key} ' + r'$\rangle$'
        label = f"{key}_avg"
        _ = ax.plot(
            steps,
            arr.mean(-1),
            lw=1.5 * line_width,
            label=label,
            **plot_kwargs,
        )
        if num_chains > 0:
            for chain in range(min((num_chains, arr.shape[1]))):
                plot_kwargs.update({"label": None})
                ax.plot(
                    steps, arr[:, chain], lw=line_width / 2.0, **plot_kwargs
                )
        sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, fill=True)
        _ = ax1.set_xticks([])
        _ = ax1.set_xticklabels([])
        _ = sns.despine(ax=ax, top=True, right=True)
        _ = sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
        _ = ax1.set_xlabel("")
        if logfreq is not None:
            xticks = ax.get_xticks()
            _ = ax.set_xticklabels([f"{logfreq * int(i)}" for i in xticks])
        axes = (ax, ax1)
    else:
        # arr.shape = [draws]
        if len(arr.shape) == 1:
            fig, ax = subplots(**subplots_kwargs)
            _ = ax.plot(steps, arr, **plot_kwargs)
            _ = ax.grid(True, alpha=0.2)
            axes = ax
        # arr.shape = [draws, nleapfrog, chains]
        elif len(arr.shape) == 3:
            fig, ax = subplots(**subplots_kwargs)
            cmap = plt.get_cmap("viridis", lut=arr.shape[1])
            _ = plot_kwargs.pop("color", None)
            for idx in range(arr.shape[1]):
                label = plot_kwargs.pop("label", None)
                if label is not None:
                    label = f"{label}-{idx}"
                y = arr[:, idx]
                color = cmap(idx / y.shape[1])
                _ = plot_kwargs.pop("color", None)
                if len(y.shape) == 2:
                    # TOO: Plot chains
                    if num_chains > 0:
                        for idx in range(min((num_chains, y.shape[1]))):
                            _ = ax.plot(
                                steps,
                                y[:, idx],
                                color=color,
                                lw=line_width / 4.0,
                                alpha=0.7,
                                **plot_kwargs,
                            )
                    _ = ax.plot(
                        steps,
                        y.mean(-1),
                        color=color,
                        label=label,
                        **plot_kwargs,
                    )
                else:
                    _ = ax.plot(
                        steps, y, color=color, label=label, **plot_kwargs
                    )
            axes = ax
        else:
            raise ValueError("Unexpected shape encountered")

        _ = ax.set_ylabel(key)
    if num_chains > 0 and len(arr.shape) > 1:
        lw = line_width / 2.0
        for idx in range(min(num_chains, arr.shape[1])):
            # plot values of invidual chains, arr[:, idx]
            # where arr[:, idx].shape = [ndraws, 1]
            _ = ax.plot(
                steps, arr[:, idx], alpha=0.5, lw=lw / 2.0, **plot_kwargs
            )

    _ = ax.set_xlabel("draw")
    if title is not None:
        _ = fig.suptitle(title)

    if logfreq is not None:
        assert isinstance(ax, plt.Axes)
        xticks = ax.get_xticks()  # type:ignore
        _ = ax.set_xticklabels(
            [  # type:ignore
                f"{logfreq * int(i)}" for i in xticks
            ]
        )

    if outdir is not None:
        _ = Path(outdir).mkdir(exist_ok=True, parents=True)
        outfile = Path(outdir).joinpath(f"{key}.{ext}")
        if not outfile.is_file():
            fig = plt.gcf() if fig is None else fig
            save_figure(fig, fname=f"{key}", outdir=outdir)
            # savefig(fig)
            # _ = Path(outdir).parent().mkdir(exist_ok=True, parents=True)
            # _ = plt.savefig(Path(outdir).joinpath(f'{key}.{ext}'),
            #                 dpi=400, bbox_inches='tight')

    return fig, subfigs, axes


def plot_history(
    data: dict[str, np.ndarray],
    num_chains: Optional[int] = 0,
    therm_frac: Optional[float] = 0.0,
    title: Optional[str] = None,
    outdir: Optional[os.PathLike] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
):
    for key, val in data.items():
        _ = plot_metric(
            val=val,
            key=str(key),
            title=title,
            outdir=outdir,
            therm_frac=therm_frac,
            num_chains=num_chains,
            plot_kwargs=plot_kwargs,
        )


def make_ridgeplots(
    dataset: xr.Dataset,
    num_chains: Optional[int] = None,
    outdir: Optional[os.PathLike] = None,
    drop_zeros: Optional[bool] = False,
    drop_nans: Optional[bool] = True,
    cmap: Optional[str] = "viridis_r",
    save_plot: bool = True,
):
    """Make ridgeplots."""
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = {}
    # with sns.axes_style('white', rc={'axes.facecolor': (0, 0, 0, 0)}):
    # sns.set(style='white', palette='bright', context='paper')
    # with sns.set_style(style='white'):
    outdir = Path(os.getcwd()) if outdir is None else Path(outdir)
    outdir = outdir.joinpath("ridgeplots")
    with sns.plotting_context(
        context="paper",
    ):
        sns.set_theme(
            style="white",
            palette="bright",
        )
        plt.rcParams["axes.facecolor"] = (0, 0, 0, 0.0)
        plt.rcParams["figure.facecolor"] = (0, 0, 0, 0.0)
        for key, val in dataset.data_vars.items():
            tstart = time.time()
            if "leapfrog" in val.coords.dims:
                lf_data = {
                    key: [],
                    "lf": [],
                    "avg": [],
                }
                for lf in val.leapfrog.values:
                    # val.shape = (chain, leapfrog, draw)
                    # x.shape = (chain, draw);  selects data for a single lf
                    x = val[{"leapfrog": lf}].values
                    # if num_chains is not None, keep `num_chains` for plotting
                    if num_chains is not None:
                        x = x[:num_chains, :]

                    x = x.flatten()
                    if drop_zeros:
                        x = x[x != 0]
                    #  x = val[{'leapfrog': lf}].values.flatten()
                    if drop_nans:
                        x = x[np.isfinite(x)]

                    lf_arr = np.array(len(x) * [f"{lf}"])
                    avg_arr = np.array(len(x) * [x.mean()])
                    lf_data[key].extend(x)
                    lf_data["lf"].extend(lf_arr)
                    lf_data["avg"].extend(avg_arr)

                lfdf = pd.DataFrame(lf_data)
                lfdf_avg = lfdf.groupby("lf")["avg"].mean()
                lfdf["lf_avg"] = lfdf["lf"].map(lfdf_avg)  # type:ignore

                # Initialize the FacetGrid object
                ncolors = len(val.leapfrog.values)
                pal = sns.color_palette(cmap, n_colors=ncolors)
                g = sns.FacetGrid(
                    lfdf,
                    row="lf",
                    hue="lf_avg",
                    aspect=15,
                    height=0.25,  # type:ignore
                    palette=pal,  # type:ignore
                )
                # avgs = lfdf.groupby('leapfrog')[f'Mean {key}']

                # Draw the densities in a few steps
                _ = g.map(
                    sns.kdeplot,
                    key,
                    cut=1,
                    bw_adjust=1.0,
                    clip_on=False,
                    fill=True,
                    alpha=0.7,
                    linewidth=1.25,
                )
                # _ = sns.histplot()
                # _ = g.map(sns.histplot, key)
                #           # rug=False, kde=False, norm_hist=False,
                #           # shade=True, alpha=0.7, linewidth=1.25)
                _ = g.map(plt.axhline, y=0, lw=1.0, alpha=0.9, clip_on=False)

                # Define and use a simple function to
                # label the plot in axes coords:

                def label(_, color, label):  # type: ignore # noqa
                    ax = plt.gca()
                    # assert isinstance(ax, plt.Axes)
                    _ = ax.set_ylabel("")  # type:ignore
                    _ = ax.set_yticks([])  # type:ignore
                    _ = ax.set_yticklabels([])  # type:ignore
                    color = ax.lines[-1].get_color()  # type:ignore
                    _ = ax.text(  # type:ignore
                        0,
                        0.10,
                        label,
                        fontweight="bold",
                        color=color,
                        ha="left",
                        va="center",
                        transform=ax.transAxes,  # type:ignore
                    )

                # _ = g.map(label, key)
                for i, ax in enumerate(g.axes.flat):
                    _ = ax.set_ylabel("")
                    _ = ax.set_yticks([])
                    _ = ax.set_yticklabels([])
                    ax.text(
                        0,
                        0.10,
                        f"{i}",
                        fontweight="bold",
                        ha="left",
                        va="center",
                        transform=ax.transAxes,
                        color=ax.lines[-1].get_color(),
                    )
                # Set the subplots to overlap
                _ = g.figure.subplots_adjust(hspace=-0.75)
                # Remove the axes details that don't play well with overlap
                _ = g.set_titles("")
                _ = g.set(yticks=[])
                _ = g.set(yticklabels=[])
                plt.rcParams["axes.labelcolor"] = "#bdbdbd"
                _ = g.set(xlabel=f"{key}")
                _ = g.despine(bottom=True, left=True)
                if outdir is not None and save_plot:
                    outdir = Path(outdir)
                    pngdir = outdir.joinpath("pngs")
                    svgdir = outdir.joinpath("svgs")
                    fsvg = Path(svgdir).joinpath(f"{key}_ridgeplot.svg")
                    fpng = Path(pngdir).joinpath(f"{key}_ridgeplot.png")

                    svgdir.mkdir(exist_ok=True, parents=True)
                    pngdir.mkdir(exist_ok=True, parents=True)

                    logger.info(f"Saving figure to: {fsvg.as_posix()}")
                    _ = plt.savefig(
                        fsvg.as_posix(), dpi=400, bbox_inches="tight"
                    )
                    _ = plt.savefig(
                        fpng.as_posix(), dpi=400, bbox_inches="tight"
                    )

                logger.debug(
                    f"Ridgeplot for {key} took {time.time() - tstart:.3f}s"
                )

    #  sns.set(style='whitegrid', palette='bright', context='paper')
    fig = plt.gcf()
    ax = plt.gca()

    return fig, ax, data
