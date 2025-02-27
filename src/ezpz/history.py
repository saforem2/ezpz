"""
history.py

Contains implementation of History object for tracking / aggregating metrics.
"""

from __future__ import absolute_import, annotations, division, print_function
from contextlib import ContextDecorator
import os
from pathlib import Path
import time
from typing import Any, Optional, Union

import ezpz

import ezpz.plot as ezplot

from ezpz.configs import PathLike, get_timestamp
from ezpz.utils import save_dataset, grab_tensor
from ezpz.log.console import is_interactive

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import xarray as xr
from jaxtyping import ScalarLike


RANK = ezpz.get_rank()

logger = ezpz.get_logger(__name__)

try:
    import wandb

    WANDB_DISABLED = os.environ.get('WANDB_DISABLED', False)
except Exception:
    wandb = None
    WANDB_DISABLED = True

TensorLike = Union[torch.Tensor, np.ndarray, list]

PT_FLOAT = torch.get_default_dtype()

xplt = xr.plot  # type:ignore
LW = plt.rcParams.get('axes.linewidth', 1.75)


class StopWatch(ContextDecorator):
    def __init__(
        self,
        msg: str,
        wbtag: Optional[str] = None,
        iter: Optional[int] = None,
        commit: Optional[bool] = False,
        prefix: str = 'StopWatch/',
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
                f'{self.wbtag}/dt': None,
            }
            if iter is not None:
                self.data |= {
                    f'{self.wbtag}/iter': self.iter,
                }

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, t, v, traceback):
        dt = time.perf_counter() - self.time
        # if self.wbtag is not None and wandb.run is not None:
        # if len(self.data) > 0 and wandb.run is not None:
        if (
            len(self.data) > 0
            and (wbrun := getattr(wandb, 'run', None)) is not None
        ):
            self.data |= {f'{self.wbtag}/dt': dt}
            wbrun.log({self.prefix: self.data}, commit=self.commit)
        if self.log_output:
            logger.info(f'{self.msg} took {dt:.3f} seconds')


class History:
    def __init__(self, keys: Optional[list[str]] = None) -> None:
        self.keys = [] if keys is None else keys
        self.history = {}

    def _update(
        self,
        key: str,
        val: Union[Any, ScalarLike, list, torch.Tensor, np.ndarray],
    ):
        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]
        return val

    def update(
        self,
        metrics: dict,
        precision: int = 6,
        use_wandb: Optional[bool] = True,
        commit: Optional[bool] = True,
        summarize: Optional[bool] = True,
    ) -> str:
        for key, val in metrics.items():
            # if isinstance(val, (list, np.ndarray, torch.Tensor)):
            #     val = grab_tensor(val)
            try:
                self.history[key].append(val)
            except KeyError:
                self.history[key] = [val]
        if (
            wandb is not None
            and use_wandb
            and not WANDB_DISABLED
            and getattr(wandb, 'run', None) is not None
        ):
            wandb.log(metrics, commit=commit)
        if summarize:
            return ezpz.summarize_dict(metrics, precision=precision)
        return ''

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
    ):
        if xlabel is not None and ylabel == xlabel:
            return
        if len(y) > 1:
            x = x if x is not None else np.arange(len(y))
            assert x is not None
            ezplot.tplot(
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
        if ylabel is not None and 'dt' in ylabel:
            of = Path(outfile) if outfile is not None else None
            if of is not None:
                of = Path(of.parent).joinpath(f'{of.stem}-hist{of.suffix}')
            ezplot.tplot(
                y=y,
                xlabel=ylabel,
                title=title,
                ylabel='freq',
                append=append,
                verbose=verbose,
                outfile=(of if of is not None else None),
                plot_type='hist',
            )

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
    ):
        """
        Plot a single variable from the history.

        NOTE: The `warmup` argument can be used to drop the first `warmup`
        iterations (as a percent of the total number of iterations) from the
        plot.
        """
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', ezplot.set_size())
        subplots_kwargs.update({'figsize': figsize})
        num_chains = 16 if num_chains is None else num_chains

        # tmp = val[0]
        arr = np.array(val)

        subfigs = None
        steps = np.arange(arr.shape[0])
        if warmup is not None and warmup > 0:
            drop = int(warmup * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        if len(arr.shape) == 2:
            import seaborn as sns

            _ = subplots_kwargs.pop('constrained_layout', True)
            figsize = (3 * figsize[0], 1.5 * figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)

            gs_kw = {'width_ratios': [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(
                1, 2, sharey=True, gridspec_kw=gs_kw
            )
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get('color', None)
            label = r'$\langle$' + f' {key} ' + r'$\rangle$'
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
            ax1.set_xlabel('')
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
                cmap = plt.get_cmap('viridis')
                nlf = arr.shape[1]
                for idx in range(nlf):
                    # y = arr[:, idx, :].mean(-1)
                    # pkwargs = {
                    #     'color': cmap(idx / nlf),
                    #     'label': f'{idx}',
                    # }
                    # ax.plot(steps, y, **pkwargs)
                    label = plot_kwargs.pop('label', None)
                    if label is not None:
                        label = f'{label}-{idx}'
                    y = arr[:, idx, :]
                    color = cmap(idx / y.shape[1])
                    plot_kwargs['color'] = cmap(idx / y.shape[1])
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
                raise ValueError('Unexpected shape encountered')

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

        ax.set_xlabel('draw')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f'{key}.svg')
            if outfile.is_file():
                tstamp = ezpz.get_timestamp()
                pngdir = Path(outdir).joinpath('pngs')
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f'{key}-{tstamp}.png')
                svgfile = Path(outdir).joinpath(f'{key}-{tstamp}.svg')
                plt.savefig(pngfile, dpi=400, bbox_inches='tight')
                plt.savefig(svgfile, dpi=400, bbox_inches='tight')

        return fig, subfigs, axes

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
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        ezplot.set_plot_style()
        plt.rcParams['axes.labelcolor'] = '#bdbdbd'
        figsize = subplots_kwargs.get('figsize', ezplot.set_size())
        subplots_kwargs.update({'figsize': figsize})
        subfigs = None
        # if key == 'dt':
        #     warmup = 0.2
        arr = val.values  # shape: [nchains, ndraws]
        # steps = np.arange(len(val.coords['draw']))
        steps = val.coords['draw']
        if warmup is not None and warmup > 0.0:
            drop = int(warmup * arr.shape[0])
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
                        logger.error(f'Unable to plot {key}! Continuing')
                _ = ax.grid(True, alpha=0.2)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = ezplot.subplots(**subplots_kwargs)
                cmap = plt.get_cmap('viridis')
                y = val.mean('chain')
                for idx in range(len(val.coords['leapfrog'])):
                    pkwargs = {
                        'color': cmap(idx / len(val.coords['leapfrog'])),
                        'label': f'{idx}',
                    }
                    ax.plot(steps, y[idx], **pkwargs)
                axes = ax
            else:
                raise ValueError('Unexpected shape encountered')
            ax = plt.gca()
            assert isinstance(ax, plt.Axes)
            _ = ax.set_ylabel(key)
            _ = ax.set_xlabel('draw')
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
                [f'{logfreq * int(i)}' for i in xticks]  # type: ignore
            )
        if outdir is not None:
            dirs = {
                'png': Path(outdir).joinpath('pngs/'),
                'svg': Path(outdir).joinpath('svgs/'),
            }
            _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
            # from l2hmc.configs import PROJECT_DIR
            # from ezpz
            if verbose:
                logger.info(f'Saving {key} plot to: {Path(outdir).resolve()}')
            for ext, d in dirs.items():
                outfile = d.joinpath(f'{key}.{ext}')
                plt.savefig(outfile, dpi=400, bbox_inches='tight')
        return (fig, subfigs, axes)

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
        import seaborn as sns

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        assert len(xarr.shape) == 2
        assert 'draw' in xarr.coords and 'chain' in xarr.coords
        num_chains = len(xarr.chain) if num_chains is None else num_chains
        # _ = subplots_kwargs.pop('constrained_layout', True)
        figsize = plt.rcParams.get('figure.figsize', (8, 6))
        figsize = (3 * figsize[0], 1.5 * figsize[1])
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        subfigs = fig.subfigures(1, 2)
        gs_kw = {'width_ratios': [1.33, 0.33]}
        (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
        ax.grid(alpha=0.2)
        ax1.grid(False)
        color = plot_kwargs.get('color', f'C{np.random.randint(6)}')
        label = r'$\langle$' + f' {label} ' + r'$\rangle$'
        ax.plot(
            xarr.draw.values,
            xarr.mean('chain'),
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
        ax1.set_xlabel('')
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
        ax.set_xlabel('draw')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            assert label is not None
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f'{label}.svg')
            if outfile.is_file():
                tstamp = ezpz.get_timestamp('%Y-%m-%d-%H%M%S')
                pngdir = Path(outdir).joinpath('pngs')
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f'{label}-{tstamp}.png')
                svgfile = Path(outdir).joinpath(f'{label}-{tstamp}.svg')
                plt.savefig(pngfile, dpi=400, bbox_inches='tight')
                plt.savefig(svgfile, dpi=400, bbox_inches='tight')

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

        outdir = Path(os.getcwd()) if outdir is None else Path(outdir)
        logger.info(f'Saving tplots to {outdir.as_posix()}')
        for _, (key, val) in enumerate(dataset.items()):
            if (xkey is not None and key == xkey) or xkey in ['iter', 'draw']:
                continue
            if len(val.values) > 0:
                self._tplot(
                    y=val.values,
                    x=None,
                    xlabel='iter',
                    plot_type=plot_type,
                    ylabel=str(key),
                    append=append,
                    title=f'{key} [{ezpz.get_timestamp()}]',
                    verbose=verbose,
                    outfile=outdir.joinpath(f'{key}.txt').as_posix(),
                    logfreq=logfreq,
                )
            else:
                logger.warning(
                    f'No data found in {key=}: {len(val.values)=} <= 0'
                )

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
        import seaborn as sns

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

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
            cmap='viridis',
            save_plot=(outdir is not None),
        )

        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            color = f'C{idx % 9}'
            plot_kwargs['color'] = color

            fig, subfigs, ax = self.plot(
                val=val.values.T.real,
                key=str(key),
                title=title,
                outdir=outdir,
                warmup=warmup,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            if fig is not None:
                _ = sns.despine(
                    fig, top=True, right=True, bottom=True, left=True
                )

            # _ = plt.grid(True, alpha=0.4)
            if subfigs is not None:
                # edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.facecolor']
                ax = subfigs[0].subplots(1, 1)
                # ax = fig[1].subplots(constrained_layout=True)
                _ = xplt.pcolormesh(
                    val, 'draw', 'chain', ax=ax, robust=True, add_colorbar=True
                )
                # im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)
                # im.colorbar.set_label(f'{key}')  # , labelpad=1.25)
                sns.despine(
                    subfigs[0], top=True, right=True, left=True, bottom=True
                )
            if outdir is not None:
                dirs = {
                    'png': Path(outdir).joinpath('pngs/'),
                    'svg': Path(outdir).joinpath('svgs/'),
                }
                _ = [
                    i.mkdir(exist_ok=True, parents=True) for i in dirs.values()
                ]
                # if verbose:
                logger.info(f'Saving {key} plot to: {Path(outdir).resolve()}')
                for ext, d in dirs.items():
                    outfile = d.joinpath(f'{key}.{ext}')
                    if outfile.is_file():
                        outfile = d.joinpath(f'{key}-subfig.{ext}')
                    # logger.info(f"Saving {key}.ext to: {outfile}")
                    if verbose:
                        logger.info(
                            f'Saving {key} plot to: {outfile.resolve()}'
                        )
                    plt.savefig(outfile, dpi=400, bbox_inches='tight')
            if is_interactive():
                plt.show()

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
            logger.info(f'len(x): {len(x)}')
            logger.info(f'x[0].shape: {x[0].shape}')
            logger.info(f'arr.shape: {arr.shape}')
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
            dims = ['draw']
            coords = [np.arange(len(arr))]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        if len(arr.shape) == 2:  # [nchains, ndraws]
            arr = arr.T
            nchains, ndraws = arr.shape
            dims = ('chain', 'draw')
            coords = [np.arange(nchains), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        if len(arr.shape) == 3:  # [nchains, nlf, ndraws]
            arr = arr.T
            nchains, nlf, ndraws = arr.shape
            dims = ('chain', 'leapfrog', 'draw')
            coords = [np.arange(nchains), np.arange(nlf), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        else:
            print(f'arr.shape: {arr.shape}')
            raise ValueError('Invalid shape encountered')

    def get_dataset(
        self,
        data: Optional[dict[str, Union[list, np.ndarray, torch.Tensor]]] = None,
        warmup: Optional[float] = 0.0,
    ):
        data = self.history_to_dict() if data is None else data
        data_vars = {}
        for key, val in data.items():
            name = key.replace('/', '_')
            try:
                data_vars[name] = self.to_DataArray(val, warmup)
            except ValueError:
                logger.error(f'Unable to create DataArray for {key}! Skipping!')
                logger.error(f'{key}.shape= {np.stack(val).shape}')  # type:ignore
        return xr.Dataset(data_vars)

    def save_dataset(
        self,
        outdir: PathLike,
        fname: str = 'dataset',
        use_hdf5: bool = True,
        data: Optional[dict[str, Union[list, np.ndarray, torch.Tensor]]] = None,
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
        data: Optional[dict[str, Union[list, np.ndarray, torch.Tensor]]] = None,
        dataset: Optional[xr.Dataset] = None,
        xkey: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        tplot_type: Optional[str] = None,
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
            f'History-{ezpz.get_timestamp()}' if run_name is None else run_name
        )
        fallback_outdir = Path(os.getcwd()).joinpath('outputs')
        if run_name is not None:
            fallback_outdir = fallback_outdir.joinpath(
                run_name, get_timestamp()
            )
        outdir = (
            # Path(os.getcwd()).joinpath('outputs')
            fallback_outdir if outdir is None else Path(outdir)
        )
        outdir = outdir.joinpath(run_name)
        if plot:
            plotdir = outdir.joinpath('plots')
            tplotdir = plotdir.joinpath('tplot')
            mplotdir = plotdir.joinpath('mplot')
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
            fname = 'dataset' if dataset_fname is None else dataset_fname
            _ = self.save_dataset(dataset=dataset, outdir=outdir, fname=fname)
        return dataset
