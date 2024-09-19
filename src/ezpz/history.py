"""
history.py

Contains implementation of History object for tracking / aggregating metrics.
"""

from __future__ import absolute_import, annotations, division, print_function
import ezpz as ez
import logging
import logging.config
from contextlib import ContextDecorator
import os
from pathlib import Path
import time
from typing import Any, Optional, Union
from collections.abc import Iterable

# from ezpz import get_logging_config
from ezpz.configs import PathLike
from ezpz.plot import plot_dataset, tplot_dict
from ezpz.utils import save_dataset, grab_tensor
from ezpz.log.console import is_interactive
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
import xarray as xr
from jaxtyping import Array, Float, Scalar, PyTree, ScalarLike

from ezpz.utils import grab_tensor
import ezpz.plot as ezplot

RANK = ez.get_rank()

try:
    import wandb

    WANDB_DISABLED = os.environ.get("WANDB_DISABLED", False)
except Exception:
    wandb = None
    WANDB_DISABLED = True

# TensorLike = Union[tf.Tensor, torch.Tensor, np.ndarray]
TensorLike = Union[torch.Tensor, np.ndarray, list]
# ScalarLike = Union[float, int, bool, np.floating, np.integer]

PT_FLOAT = torch.get_default_dtype()
# TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())
# Scalar = Union[float, int, np.floating, bool]
# Scalar = Shaped[Array, ""]
# ScalarLike = Shaped[ArrayLike, ""]
# Scalar = TF_FLOAT | PT_FLOAT | np.floating | int | bool

# log = logging.getLogger(__name__)

# log_config = logging.config.dictConfig(get_logging_config())
log = logging.getLogger(__name__)

log.setLevel("INFO") if RANK == 0 else log.setLevel("CRITICAL")

xplt = xr.plot  # type:ignore
LW = plt.rcParams.get("axes.linewidth", 1.75)


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f"{k}={v}"
    # return f'{k}={v:<3.4f}'
    return f"{k}={v:<.3f}"


def summarize_dict(d: dict) -> str:
    return " ".join([format_pair(k, v) for k, v in d.items()])


# def subsample_dict(d: dict) -> dict:
#     for key, val in d.items():
#         pass

# def timeit(func):
#     @functools.wraps(func)
#     def time_closure(*args, **kwargs):
#         start = time.perf_counter()
#         result = func(*args, **kwargs)
#         end = time.perf_counter() - start
#         log.info(f')


class StopWatch(ContextDecorator):
    def __init__(
        self,
        msg: str,
        wbtag: Optional[str] = None,
        iter: Optional[int] = None,
        commit: Optional[bool] = False,
        prefix: str = "StopWatch/",
        log_output: bool = True,
    ) -> None:
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
        self.time = time.perf_counter()
        return self

    def __exit__(self, t, v, traceback):
        dt = time.perf_counter() - self.time
        # if self.wbtag is not None and wandb.run is not None:
        if len(self.data) > 0 and wandb.run is not None:
            self.data |= {f"{self.wbtag}/dt": dt}
            wandb.run.log({self.prefix: self.data}, commit=self.commit)
        if self.log_output:
            log.info(f"{self.msg} took " f"{dt:.3f} seconds")


class BaseHistory:
    def __init__(self):
        self.history = {}
        # nera = 1 if steps is None else steps.nera
        # self.era_metrics = {str(era): {} for era in range(nera)}

    def metric_to_numpy(
        self,
        metric: PyTree,
    ) -> np.ndarray | Scalar | None | ScalarLike:
        if isinstance(metric, (Scalar, np.ndarray)):
            return metric
        if isinstance(metric, list):
            if isinstance(metric[0], np.ndarray):
                return np.stack(metric)
            if isinstance(metric[0], torch.Tensor):
                return grab_tensor(torch.stack(metric))
            if callable(getattr(metric, "numpy", None)):
                return grab_tensor(np.stack(metric))
            # if isinstance(metric[0], tf.Tensor):
            #     return grab_tensor(tf.stack(metric))
        try:
            marr = grab_tensor(metric)
        except Exception:
            marr = np.array(metric)
        # if (
        #     isinstance(metric, tf.Tensor)
        #     or isinstance(metric, torch.Tensor)
        # ):
        #     return grab_tensor(metric)
        return marr

    def _update(
        self,
        key: str,
        val: Union[Iterable, Float[Array, "..."]],
    ) -> float | int | bool | np.floating | np.integer | None:
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return None
            v = grab_tensor(torch.stack(val))
        elif isinstance(val, (int, bool, np.floating)):
            return val
        elif isinstance(val, torch.Tensor):
            v = grab_tensor(val)
        else:
            v = val
        #
        #
        # # if (_len := getattr(val, "__len__", None)) is not None and _len > 0:
        # #     try:
        # #         v = grab_tensor(val)
        # #     except Exception:
        # #         # try:
        # #         v = grab_tensor(torch.stack(val))
        # #     finally:
        # #         import pudb
        # #
        # #         pudb.set_trace()
        # #         # except Exception:
        # #         #     v = val
        # #         #     # import pudb; pudb.set_trace()
        # # else:
        # #     v = val
        # # if isinstance(val, ScalarLike):
        # #     v = val
        # try:
        #     v = grab_tensor(val)
        # except Exception:
        #     # if (m := getattr(val, "__len__", 0)) > 0:
        #     v = grab_tensor(torch.stack(list(val)))
        #     # else:
        #     # if isinstance(val, (list, tuple)) or hasattr(val, "__len__"):
        #     #     if getattr(val, '__len__', 0) > 0:
        #     #         v = grab_tensor(torch.stack(val))  # type:ignore
        #     # else:
        # else:
        #     v = val
        #     #     v = val
        # finally:
        #     log.error(f"Unable to _update({key})")
        #     log.error(f"{val=}")
        #     log.error("Continuing!")
        #     return None
        #
        # v = None
        # if isinstance(val, Iterable) and len(val) > 0:
        #     try:
        #         v = grab_tensor(torch.stack(val))
        #     except Exception:
        #         v = val
        #         # import pudb; pudb.set_trace()
        #
        #     # if isinstance(val[0], torch.Tensor):
        #     #     v = grab_tensor(torch.stack(val))
        #     # # elif isinstance(val, np.ndarray):
        #     # #     v = np.stack(val)
        #     # else:
        #     #     v = val
        # else:
        #     # if isinstance(v, (tf.Tensor, torch.Tensor)):
        #     try:
        #         v = grab_tensor(val)
        #     except Exception:
        #         import pudb
        #         pudb.set_trace()
        assert v is not None
        try:
            self.history[key].append(v)
        except KeyError:
            self.history[key] = [v]

        # ScalarLike = Union[float, int, bool, np.floating]
        if isinstance(v, (float, int, bool, np.floating, np.integer)):
            return v
        #     return v
        # # avg = np.mean(v)
        # assert isinstance(avg, np.floating)
        # return avg
        assert v is not None
        # return v.mean()
        # return np.mean(v)

    def update(self, metrics: dict) -> dict[str, Any]:
        avgs = {}
        avg = 0.0
        for key, val in metrics.items():
            if isinstance(val, dict):
                # if isinstance(val, dict):
                avgs |= self.update({f"{key}/{k}": v for k, v in val.items()})
                # for k, v in val.items():
                #     kk = f"{key}/{k}"
                #     avg = self._update(kk, v)
                #     avgs[kk] = avg
            else:
                # try:
                avg = self._update(key, val)
                if avg is not None:
                    avgs[key] = avg
                # except Exception:
                #     import pudb
                #
                #     pudb.set_trace()

        return avgs

    def tplot(
        self,
        outdir: Optional[PathLike] = None,
        logfreq: int = 1,
    ):
        dset = self.get_dataset()
        for key, val in dset.items():
            outdir = Path(os.getcwd()) if outdir is None else outdir
            outfile = Path(outdir).joinpath(f"{key}.txt").as_posix()
            # x = dset.get('iter')
            # if x is not None:
            #     x = x.values
            ezplot.tplot(
                y=val.values,
                # x=x,
                label=str(key),
                # xlabel='iter' if x is not None else None,
                outfile=outfile,
            )

    def plot(
        self,
        val: np.ndarray,
        key: Optional[str] = None,
        therm_frac: Optional[float] = 0.0,
        num_chains: Optional[int] = 128,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get("figsize", ezplot.set_size())
        subplots_kwargs.update({"figsize": figsize})
        num_chains = 16 if num_chains is None else num_chains

        # tmp = val[0]
        arr = np.array(val)

        subfigs = None
        steps = np.arange(arr.shape[0])
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        if len(arr.shape) == 2:
            _ = subplots_kwargs.pop("constrained_layout", True)
            figsize = (3 * figsize[0], 1.5 * figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)

            gs_kw = {"width_ratios": [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get("color", None)
            label = r"$\langle$" + f" {key} " + r"$\rangle$"
            ax.plot(steps, arr.mean(-1), lw=1.5 * LW, label=label, **plot_kwargs)
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
                assert isinstance(ax, plt.Axes)
                ax.plot(steps, arr, **plot_kwargs)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = plt.subplots(**subplots_kwargs)
                assert isinstance(ax, plt.Axes)
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
                ax.plot(steps, arr[:, idx], alpha=0.5, lw=LW / 2.0, **plot_kwargs)

        ax.set_xlabel("draw")
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f"{key}.svg")
            if outfile.is_file():
                tstamp = ezplot.get_timestamp("%Y-%m-%d-%H%M%S")
                pngdir = Path(outdir).joinpath("pngs")
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f"{key}-{tstamp}.png")
                svgfile = Path(outdir).joinpath(f"{key}-{tstamp}.svg")
                plt.savefig(pngfile, dpi=400, bbox_inches="tight")
                plt.savefig(svgfile, dpi=400, bbox_inches="tight")

        return fig, subfigs, axes

    def plot_dataArray(
        self,
        val: xr.DataArray,
        key: Optional[str] = None,
        therm_frac: Optional[float] = 0.0,
        num_chains: Optional[int] = 0,
        title: Optional[str] = None,
        outdir: Optional[str] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        line_labels: bool = False,
        logfreq: Optional[int] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        ezplot.set_plot_style()
        plt.rcParams["axes.labelcolor"] = "#bdbdbd"
        figsize = subplots_kwargs.get("figsize", ezplot.set_size())
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
                        log.error(f"Unable to plot {key}! Continuing")
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
            assert isinstance(ax, plt.Axes)
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
            _ = ax.set_xticklabels(
                [f"{logfreq * int(i)}" for i in xticks]  # type: ignore
            )
        if outdir is not None:
            dirs = {
                "png": Path(outdir).joinpath("pngs/"),
                "svg": Path(outdir).joinpath("svgs/"),
            }
            _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
            # from l2hmc.configs import PROJECT_DIR
            # from ezpz
            log.info(f"Saving {key} plot to: " f"{Path(outdir).resolve()}")
            for ext, d in dirs.items():
                outfile = d.joinpath(f"{key}.{ext}")
                plt.savefig(outfile, dpi=400, bbox_inches="tight")
        return (fig, subfigs, axes)

    def plot_dataset(
        self,
        # therm_frac: float = 0.,
        title: Optional[str] = None,
        nchains: Optional[int] = None,
        outdir: Optional[os.PathLike] = None,
        # subplots_kwargs: Optional[dict[str, Any]] = None,
        # plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        dataset = self.get_dataset()
        return plot_dataset(
            dataset=dataset,
            nchains=nchains,
            # therm_frac=therm_frac,
            title=title,
            outdir=outdir,
            # subplots_kwargs=subplots_kwargs,
            # plot_kwargs=plot_kwargs
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
                tstamp = ezplot.get_timestamp("%Y-%m-%d-%H%M%S")
                pngdir = Path(outdir).joinpath("pngs")
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f"{label}-{tstamp}.png")
                svgfile = Path(outdir).joinpath(f"{label}-{tstamp}.svg")
                plt.savefig(pngfile, dpi=400, bbox_inches="tight")
                plt.savefig(svgfile, dpi=400, bbox_inches="tight")

    def plot_all(
        self,
        num_chains: int = 128,
        therm_frac: float = 0.0,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        dataset = self.get_dataset()

        _ = ezplot.make_ridgeplots(
            dataset,
            outdir=outdir,
            drop_nans=True,
            drop_zeros=False,
            num_chains=num_chains,
            cmap="viridis",
            save_plot=(outdir is not None),
        )

        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            color = f"C{idx%9}"
            plot_kwargs["color"] = color

            fig, subfigs, ax = self.plot(
                val=val.values.T.real,
                key=str(key),
                title=title,
                outdir=outdir,
                therm_frac=therm_frac,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            if fig is not None:
                _ = sns.despine(fig, top=True, right=True, bottom=True, left=True)

            # _ = plt.grid(True, alpha=0.4)
            if subfigs is not None:
                # edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams["axes.edgecolor"] = plt.rcParams["axes.facecolor"]
                ax = subfigs[0].subplots(1, 1)
                # ax = fig[1].subplots(constrained_layout=True)
                _ = xplt.pcolormesh(
                    val, "draw", "chain", ax=ax, robust=True, add_colorbar=True
                )
                # im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)
                # im.colorbar.set_label(f'{key}')  # , labelpad=1.25)
                sns.despine(subfigs[0], top=True, right=True, left=True, bottom=True)
            if outdir is not None:
                dirs = {
                    "png": Path(outdir).joinpath("pngs/"),
                    "svg": Path(outdir).joinpath("svgs/"),
                }
                _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
                log.info(f"Saving {key} plot to: " f"{Path(outdir).resolve()}")
                for ext, d in dirs.items():
                    outfile = d.joinpath(f"{key}.{ext}")
                    if outfile.is_file():
                        log.info(f"Saving {key} plot to: " f"{outfile.resolve()}")
                        outfile = d.joinpath(f"{key}-subfig.{ext}")
                    # log.info(f"Saving {key}.ext to: {outfile}")
                    plt.savefig(outfile, dpi=400, bbox_inches="tight")
            if is_interactive():
                plt.show()

        return dataset

    def history_to_dict(self) -> dict:
        return {k: np.stack(v).squeeze() for k, v in self.history.items()}

    def to_DataArray(
        self,
        x: Union[list, np.ndarray],
        therm_frac: Optional[float] = 0.0,
    ) -> xr.DataArray:
        try:
            arr = np.array(x).real
        except ValueError:
            arr = np.array(x)
            log.info(f"len(x): {len(x)}")
            log.info(f"x[0].shape: {x[0].shape}")
            log.info(f"arr.shape: {arr.shape}")
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
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
        data: Optional[dict[str, Union[list, np.ndarray]]] = None,
        therm_frac: Optional[float] = 0.0,
    ):
        data = self.history_to_dict() if data is None else data
        data_vars = {}
        for key, val in data.items():
            name = key.replace("/", "_")
            try:
                data_vars[name] = self.to_DataArray(val, therm_frac)
            except ValueError:
                log.error(f"Unable to create DataArray for {key}! Skipping!")
                log.error(f"{key}.shape= {np.stack(val).shape}")  # type:ignore
        return xr.Dataset(data_vars)

    def save_dataset(
        self,
        outdir: PathLike,
        fname: str = "dataset",
        use_hdf5: bool = True,
        data: Optional[dict[str, Union[list, np.ndarray]]] = None,
        **kwargs,
    ) -> Path:
        data = self.history if data is None else data
        dataset = self.get_dataset(data)
        return save_dataset(
            dataset,
            outdir=outdir,
            fname=fname,
            use_hdf5=use_hdf5,
            **kwargs,
        )


class History:
    def __init__(self, keys: Optional[list[str]] = None) -> None:
        self.keys = [] if keys is None else keys
        self.history = {}

    def _update_alt(
        self,
        key: str,
        val: Any,
        # val: Float[Array, "..."],
    ) -> float | int | bool | np.floating | np.integer:
        if isinstance(val, (list, tuple)):
            if isinstance(val[0], torch.Tensor):
                val = grab_tensor(torch.stack(val))
            elif isinstance(val, np.ndarray):
                val = np.stack(val)
            else:
                val = val
        val = grab_tensor(val)
        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]

        # ScalarLike = Union[float, int, bool, np.floating]
        if isinstance(val, (float, int, bool, np.floating, np.integer)):
            return val
        #     return val
        avg = np.mean(val).real
        assert isinstance(avg, np.floating)
        return avg

    def update_alt(self, metrics: dict) -> dict[str, Any]:
        avgs = {}
        avg = 0.0
        for key, val in metrics.items():
            if val is None:
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    kk = f"{key}/{k}"
                    avg = self._update(kk, v)
                    avgs[kk] = avg
            else:
                avg = self._update(key, val)
                avgs[key] = avg

        return avgs

    def _update(
        self, key: str, val: Union[Any, ScalarLike, list, torch.Tensor, np.ndarray]
    ):
        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]
        return val

    def update(self, metrics: dict):
        for key, val in metrics.items():
            if isinstance(val, (list, np.ndarray, torch.Tensor)):
                val = grab_tensor(val)
            try:
                self.history[key].append(val)
            except KeyError:
                self.history[key] = [val]
        if (
            wandb is not None
            and not WANDB_DISABLED
            and getattr(wandb, "run", None) is not None
        ):
            metrics |= {
                f"{k}/hist": (
                    wandb.Histogram(v.tolist()) if isinstance(v, np.ndarray) else v
                )
                for k, v in metrics.items()
            }
            wandb.log(metrics)

    def tplot(
        self,
        outdir: Optional[PathLike] = None,
        logfreq: int = 1,
    ):
        dset = self.get_dataset()
        for key, val in dset.items():
            outdir = Path(os.getcwd()) if outdir is None else outdir
            outfile = Path(outdir).joinpath(f"{key}.txt").as_posix()
            # x = dset.get('iter')
            # if x is not None:
            #     x = x.values
            ezplot.tplot(
                y=val.values,
                # x=x,
                label=str(key),
                # xlabel='iter' if x is not None else None,
                outfile=outfile,
            )

    def plot(
        self,
        val: np.ndarray,
        key: Optional[str] = None,
        therm_frac: Optional[float] = 0.0,
        num_chains: Optional[int] = 128,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get("figsize", ezplot.set_size())
        subplots_kwargs.update({"figsize": figsize})
        num_chains = 16 if num_chains is None else num_chains

        # tmp = val[0]
        arr = np.array(val)

        subfigs = None
        steps = np.arange(arr.shape[0])
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        if len(arr.shape) == 2:
            _ = subplots_kwargs.pop("constrained_layout", True)
            figsize = (3 * figsize[0], 1.5 * figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)

            gs_kw = {"width_ratios": [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get("color", None)
            label = r"$\langle$" + f" {key} " + r"$\rangle$"
            ax.plot(steps, arr.mean(-1), lw=1.5 * LW, label=label, **plot_kwargs)
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
                assert isinstance(ax, plt.Axes)
                ax.plot(steps, arr, **plot_kwargs)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = plt.subplots(**subplots_kwargs)
                assert isinstance(ax, plt.Axes)
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
                ax.plot(steps, arr[:, idx], alpha=0.5, lw=LW / 2.0, **plot_kwargs)

        ax.set_xlabel("draw")
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f"{key}.svg")
            if outfile.is_file():
                tstamp = ezplot.get_timestamp("%Y-%m-%d-%H%M%S")
                pngdir = Path(outdir).joinpath("pngs")
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f"{key}-{tstamp}.png")
                svgfile = Path(outdir).joinpath(f"{key}-{tstamp}.svg")
                plt.savefig(pngfile, dpi=400, bbox_inches="tight")
                plt.savefig(svgfile, dpi=400, bbox_inches="tight")

        return fig, subfigs, axes

    def plot_dataArray(
        self,
        val: xr.DataArray,
        key: Optional[str] = None,
        therm_frac: Optional[float] = 0.0,
        num_chains: Optional[int] = 0,
        title: Optional[str] = None,
        outdir: Optional[str] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        line_labels: bool = False,
        logfreq: Optional[int] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        ezplot.set_plot_style()
        plt.rcParams["axes.labelcolor"] = "#bdbdbd"
        figsize = subplots_kwargs.get("figsize", ezplot.set_size())
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
                        log.error(f"Unable to plot {key}! Continuing")
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
            assert isinstance(ax, plt.Axes)
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
            _ = ax.set_xticklabels(
                [f"{logfreq * int(i)}" for i in xticks]  # type: ignore
            )
        if outdir is not None:
            dirs = {
                "png": Path(outdir).joinpath("pngs/"),
                "svg": Path(outdir).joinpath("svgs/"),
            }
            _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
            # from l2hmc.configs import PROJECT_DIR
            # from ezpz
            log.info(f"Saving {key} plot to: " f"{Path(outdir).resolve()}")
            for ext, d in dirs.items():
                outfile = d.joinpath(f"{key}.{ext}")
                plt.savefig(outfile, dpi=400, bbox_inches="tight")
        return (fig, subfigs, axes)

    def plot_dataset(
        self,
        # therm_frac: float = 0.,
        title: Optional[str] = None,
        nchains: Optional[int] = None,
        outdir: Optional[os.PathLike] = None,
        # subplots_kwargs: Optional[dict[str, Any]] = None,
        # plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        dataset = self.get_dataset()
        return plot_dataset(
            dataset=dataset,
            nchains=nchains,
            # therm_frac=therm_frac,
            title=title,
            outdir=outdir,
            # subplots_kwargs=subplots_kwargs,
            # plot_kwargs=plot_kwargs
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
                tstamp = ez.get_timestamp("%Y-%m-%d-%H%M%S")
                pngdir = Path(outdir).joinpath("pngs")
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f"{label}-{tstamp}.png")
                svgfile = Path(outdir).joinpath(f"{label}-{tstamp}.svg")
                plt.savefig(pngfile, dpi=400, bbox_inches="tight")
                plt.savefig(svgfile, dpi=400, bbox_inches="tight")

    def tplot_all(
        self,
        outdir: Optional[os.PathLike] = None,
        append: bool = True,
        xkey: Optional[str] = None,
        dataset: Optional[xr.Dataset] = None,
    ):
        dataset = self.get_dataset() if dataset is None else dataset
        outdir = Path(os.getcwd()) if outdir is None else Path(outdir)
        for idx, (key, val) in enumerate(dataset.items()):
            if xkey is not None and key == xkey:
                continue
            if callable(getattr(val, "to_numpy", None)):
                arr = val.to_numpy()
            else:
                try:
                    # arr = grab_tensor(val)
                    arr = np.array(val)
                except Exception:
                    arr = torch.Tensor(val).cpu().numpy()
                finally:
                    arr = grab_tensor(val)
            # assert arr is not None and len(arr) > 1
            if len(arr) > 1:
                if xkey is None:
                    xarr = np.arange(len(arr))
                else:
                    xval = dataset.get(xkey.replace("/", "_"))
                    assert xval is not None
                    xarr = xval.to_numpy()
                assert xarr is not None
                ez.tplot(
                    y=arr,
                    x=xarr,
                    xlabel=xkey,
                    ylabel=str(key),
                    append=append,
                    title=f"{key} [{ez.get_timestamp()}]",
                    outfile=outdir.joinpath(f"{key}.txt").as_posix(),
                )
                # tplot_dict(
                #     data=dict(zip(xarr, arr)),
                #     xlabel=xkey,
                #     ylabel=str(key),
                #     append=append,
                #     title=f"{key} [{ez.get_timestamp()}]",
                #     outfile=Path(outdir).joinpath(f"{key}.txt").as_posix(),
                # )
            else:
                log.warning(f"{key}: {len(arr)=}")

    def plot_all(
        self,
        num_chains: int = 128,
        therm_frac: float = 0.0,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        dataset = self.get_dataset()

        _ = ezplot.make_ridgeplots(
            dataset,
            outdir=outdir,
            drop_nans=True,
            drop_zeros=False,
            num_chains=num_chains,
            cmap="viridis",
            save_plot=(outdir is not None),
        )

        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            color = f"C{idx%9}"
            plot_kwargs["color"] = color

            fig, subfigs, ax = self.plot(
                val=val.values.T.real,
                key=str(key),
                title=title,
                outdir=outdir,
                therm_frac=therm_frac,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            if fig is not None:
                _ = sns.despine(fig, top=True, right=True, bottom=True, left=True)

            # _ = plt.grid(True, alpha=0.4)
            if subfigs is not None:
                # edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams["axes.edgecolor"] = plt.rcParams["axes.facecolor"]
                ax = subfigs[0].subplots(1, 1)
                # ax = fig[1].subplots(constrained_layout=True)
                _ = xplt.pcolormesh(
                    val, "draw", "chain", ax=ax, robust=True, add_colorbar=True
                )
                # im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)
                # im.colorbar.set_label(f'{key}')  # , labelpad=1.25)
                sns.despine(subfigs[0], top=True, right=True, left=True, bottom=True)
            if outdir is not None:
                dirs = {
                    "png": Path(outdir).joinpath("pngs/"),
                    "svg": Path(outdir).joinpath("svgs/"),
                }
                _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
                log.info(f"Saving {key} plot to: " f"{Path(outdir).resolve()}")
                for ext, d in dirs.items():
                    outfile = d.joinpath(f"{key}.{ext}")
                    if outfile.is_file():
                        log.info(f"Saving {key} plot to: " f"{outfile.resolve()}")
                        outfile = d.joinpath(f"{key}-subfig.{ext}")
                    # log.info(f"Saving {key}.ext to: {outfile}")
                    plt.savefig(outfile, dpi=400, bbox_inches="tight")
            if is_interactive():
                plt.show()

        return dataset

    def history_to_dict(self) -> dict:
        return {k: np.stack(v).squeeze() for k, v in self.history.items()}

    def to_DataArray(
        self,
        x: Union[list, np.ndarray],
        therm_frac: Optional[float] = 0.0,
    ) -> xr.DataArray:
        try:
            arr = np.array(x).real
        except ValueError:
            arr = np.array(x)
            log.info(f"len(x): {len(x)}")
            log.info(f"x[0].shape: {x[0].shape}")
            log.info(f"arr.shape: {arr.shape}")
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
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
        data: Optional[dict[str, Union[list, np.ndarray]]] = None,
        therm_frac: Optional[float] = 0.0,
    ):
        data = self.history_to_dict() if data is None else data
        data_vars = {}
        for key, val in data.items():
            name = key.replace("/", "_")
            try:
                data_vars[name] = self.to_DataArray(val, therm_frac)
            except ValueError:
                log.error(f"Unable to create DataArray for {key}! Skipping!")
                log.error(f"{key}.shape= {np.stack(val).shape}")  # type:ignore
        return xr.Dataset(data_vars)

    def save_dataset(
        self,
        outdir: PathLike,
        fname: str = "dataset",
        use_hdf5: bool = True,
        data: Optional[dict[str, Union[list, np.ndarray]]] = None,
        **kwargs,
    ) -> Path:
        data = self.history if data is None else data
        dataset = self.get_dataset(data)
        return save_dataset(
            dataset,
            outdir=outdir,
            fname=fname,
            use_hdf5=use_hdf5,
            **kwargs,
        )
