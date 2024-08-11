"""
ezpz/utils.py
"""

import os
import h5py
import xarray as xr
import numpy as np
from typing import Optional, Union, Any
import ezpz as ez

from ezpz.configs import PathLike
from pathlib import Path

import logging

RANK = ez.get_rank()
log = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
_ = log.setLevel(LOG_LEVEL) if RANK == 0 else log.setLevel("CRITICAL")


def grab_tensor(x: Any) -> Union[np.ndarray, ez.configs.ScalarLike, None]:
    import torch

    if x is None:
        return None
    if isinstance(x, (int, float, bool, np.floating)):
        return x
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        elif isinstance(x[0], np.ndarray):
            return np.stack(x)
        else:
            try:
                import tensorflow as tf  # type:ignore
            except (ImportError, ModuleNotFoundError) as exc:
                raise exc
            if isinstance(x[0], tf.Tensor):
                return grab_tensor(tf.stack(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif callable(getattr(x, "numpy", None)):
        assert callable(getattr(x, "numpy"))
        return x.numpy()
    raise ValueError


def save_dataset(
    dataset: xr.Dataset,
    outdir: PathLike,
    use_hdf5: Optional[bool] = True,
    fname: Optional[str] = None,
    **kwargs,
) -> Path:
    if use_hdf5:
        fname = "dataset.h5" if fname is None else f"{fname}_dataset.h5"
        outfile = Path(outdir).joinpath(fname)
        Path(outdir).mkdir(exist_ok=True, parents=True)
        try:
            dataset_to_h5pyfile(outfile, dataset=dataset, **kwargs)
        except TypeError:
            log.warning("Unable to save as `.h5` file, falling back to `netCDF4`")
            save_dataset(dataset, outdir=outdir, use_hdf5=False, fname=fname, **kwargs)
    else:
        fname = "dataset.nc" if fname is None else f"{fname}_dataset.nc"
        outfile = Path(outdir).joinpath(fname)
        mode = "a" if outfile.is_file() else "w"
        log.info(f"Saving dataset to: {outfile.as_posix()}")
        outfile.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(outfile.as_posix(), mode=mode)

    return outfile


def dataset_to_h5pyfile(hfile: PathLike, dataset: xr.Dataset, **kwargs):
    log.info(f"Saving dataset to: {hfile}")
    f = h5py.File(hfile, "a")
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
    f = h5py.File(hfile, "r")
    data = {key: f[key] for key in list(f.keys())}
    f.close()
    return data


def dataset_from_h5pyfile(hfile: PathLike) -> xr.Dataset:
    f = h5py.File(hfile, "r")
    data = {key: f[key] for key in list(f.keys())}
    f.close()

    return xr.Dataset(data)
