"""
ezpz/utils.py
"""

import sys
import pdb
import os
import h5py
import logging
import xarray as xr
import numpy as np
from typing import Optional, Union, Any

from ezpz.configs import ScalarLike, PathLike

import torch
import torch.distributed as tdist

from ezpz.dist import get_rank
from pathlib import Path


# logger = ezpz.get_logger(__name__)
RANK = get_rank()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
_ = logger.setLevel(LOG_LEVEL) if RANK == 0 else logger.setLevel('CRITICAL')


class DistributedPdb(pdb.Pdb):
    """
    Supports using PDB from inside a multiprocessing child process.

    Usage:
    DistributedPdb().set_trace()
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def breakpoint(rank: int = 0):
    """
    Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
    done with the breakpoint before continuing.

    Args:
        rank (int): Which rank to break on.  Default: ``0``
    """
    if get_rank() == rank:
        pdb = DistributedPdb()
        pdb.message(
            '\n!!! ATTENTION !!!\n\n'
            f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
        )
        pdb.set_trace()
    tdist.barrier()


def grab_tensor(
    x: Any, force: bool = False
) -> Union[np.ndarray, ScalarLike, None]:
    if x is None:
        return None
    if isinstance(x, (int, float, bool, np.floating)):
        return x
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        if isinstance(x[0], np.ndarray):
            return np.stack(x)
        if isinstance(x[0], (int, float, bool, np.floating)):
            return np.array(x)
        else:
            raise ValueError(f'Unable to convert list: \n {x=}\n to array')
        # else:
        #     try:
        #         import tensorflow as tf  # type:ignore
        #     except (ImportError, ModuleNotFoundError) as exc:
        #         raise exc
        #     if isinstance(x[0], tf.Tensor):
        #         return grab_tensor(tf.stack(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.numpy(force=force)
        # return x.detach().cpu().numpy()
    elif callable(getattr(x, 'numpy', None)):
        assert callable(getattr(x, 'numpy'))
        return x.numpy(force=force)
    breakpoint(0)
    # raise ValueError


def save_dataset(
    dataset: xr.Dataset,
    outdir: PathLike,
    use_hdf5: Optional[bool] = True,
    fname: Optional[str] = None,
    **kwargs,
) -> Path:
    if use_hdf5:
        fname = 'dataset.h5' if fname is None else f'{fname}_dataset.h5'
        outfile = Path(outdir).joinpath(fname)
        Path(outdir).mkdir(exist_ok=True, parents=True)
        try:
            dataset_to_h5pyfile(outfile, dataset=dataset, **kwargs)
        except TypeError:
            logger.warning(
                'Unable to save as `.h5` file, falling back to `netCDF4`'
            )
            save_dataset(
                dataset, outdir=outdir, use_hdf5=False, fname=fname, **kwargs
            )
    else:
        fname = 'dataset.nc' if fname is None else f'{fname}_dataset.nc'
        outfile = Path(outdir).joinpath(fname)
        mode = 'a' if outfile.is_file() else 'w'
        logger.info(f'Saving dataset to: {outfile.as_posix()}')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(outfile.as_posix(), mode=mode)

    return outfile


def dataset_to_h5pyfile(hfile: PathLike, dataset: xr.Dataset, **kwargs):
    logger.info(f'Saving dataset to: {hfile}')
    f = h5py.File(hfile, 'a')
    for key, val in dataset.data_vars.items():
        arr = val.values
        if len(arr) == 0:
            continue
        if key in list(f.keys()):
            shape = f[key].shape[0] + arr.shape[0]  # type: ignore
            f[key].resize(shape, axis=0)  # type: ignore
            f[key][-arr.shape[0] :] = arr  # type: ignore
        else:
            maxshape = (None,)
            if len(arr.shape) > 1:
                maxshape = (None, *arr.shape[1:])
            f.create_dataset(key, data=arr, maxshape=maxshape, **kwargs)

    f.close()


def dict_from_h5pyfile(hfile: PathLike) -> dict:
    f = h5py.File(hfile, 'r')
    data = {key: f[key] for key in list(f.keys())}
    f.close()
    return data


def dataset_from_h5pyfile(hfile: PathLike) -> xr.Dataset:
    f = h5py.File(hfile, 'r')
    data = {key: f[key] for key in list(f.keys())}
    f.close()

    return xr.Dataset(data)
