"""
ezpz/utils/__init__.py
"""

from __future__ import annotations

import logging
import os
import pdb
import re
from typing import Any
import sys
from dataclasses import asdict, dataclass

import ezpz
# from ezpz import get_rank
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import torch
# import torch.distributed
import xarray as xr
from torchinfo import ModelStatistics

from ezpz.configs import PathLike, ScalarLike, ZeroConfig
from ezpz.utils.dummies import DummyMPI, DummyTorch

import math

# import numpy as np

# ScalarLike = Any  # keep your existing alias if you already have one

# import torch.distributed as tdist


__all__ = [
    "Color",
    "NoColor",
    "DistributedPdb",
    "DummyMPI",
    "DummyTorch",
    "breakpoint",
    "get_timestamp",
    "format_pair",
    "summarize_dict",
    "model_summary",
    "normalize",
    "get_max_memory_allocated",
    "get_max_memory_reserved",
    "grab_tensor",
    "check_for_tarball",
    "make_tarfile",
    "create_tarball",
    "save_dataset",
    "dataset_to_h5pyfile",
    "dataset_from_h5pyfile",
    "dict_from_h5pyfile",
    "get_deepspeed_zero_config_json",
    "write_generic_deepspeed_config",
    "get_deepspeed_adamw_optimizer_config_json",
    "get_deepspeed_warmup_decay_scheduler_config_json",
    "get_deepspeed_config_json",
    "write_deepspeed_zero12_auto_config",
    "write_deepspeed_zero3_auto_config",
]


# try:
#     import intel_extension_for_pytorch as ipex
# except (ImportError, ModuleNotFoundError):
#     ipex = None

# logger = ezpz.get_logger(__name__)
# RANK = get_rank()
logger = logging.getLogger(__name__)
# LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
# _ = logger.setLevel(LOG_LEVEL) if RANK == 0 else logger.setLevel("CRITICAL")

# logger = ezpz.get_logger(__name__)
#
#

@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2,180;60,0m"
    turquoise = "\033[38;2,54,234;195m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""
    orange = ""
    turquoise = ""




class DistributedPdb(pdb.Pdb):
    """
    Supports using PDB from inside a multiprocessing child process.

    Usage:
    DistributedPdb().set_trace()
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
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
    if ezpz.get_rank() == rank:
        pdb = DistributedPdb()
        pdb.message(
            "\n!!! ATTENTION !!!\n\n"
            f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
        )
        pdb.set_trace()
    # torch.distributed.barrier()
    ezpz.dist.barrier()


def get_timestamp(fstr: Optional[str] = None) -> str:
    """Get formatted timestamp.

    Returns the current date and time as a formatted string. By default, returns
    a timestamp in the format 'YYYY-MM-DD-HHMMSS'. A custom format string can
    be provided to change the output format.

    Args:
        fstr (str, optional): Format string for strftime. If None, uses default
            format '%Y-%m-%d-%H%M%S'. Defaults to None.

    Returns:
        str: Formatted timestamp string.

    Example:
        >>> get_timestamp()  # Returns something like '2023-12-01-143022'
        >>> get_timestamp("%Y-%m-%d")  # Returns something like '2023-12-01'
    """
    import datetime

    now = datetime.datetime.now()
    return (
        now.strftime("%Y-%m-%d-%H%M%S") if fstr is None else now.strftime(fstr)
    )


def format_pair(k: str, v: Any, precision: int = 6) -> str:
    """Format a key-value pair (supports nested dict/list/tuple/set).

    Nested dicts become dotted keys:  key.subkey=value
    Sequences become indexed keys:    key[0]=value

    Returns a newline-joined string if multiple leaf pairs are produced.
    """

    def _is_int_like(x: Any) -> bool:
        return (
            isinstance(x, (bool, int, np.integer))
            and not isinstance(x, (bool,)) is False
        )  # keep bool distinct below

    def _is_bool_like(x: Any) -> bool:
        return isinstance(x, (bool, np.bool_))

    def _is_float_like(x: Any) -> bool:
        return isinstance(x, (float, np.floating))

    def _scalar_str(key: str, val: Any) -> str:
        # numpy scalar -> python scalar (helps consistent isinstance checks)
        if isinstance(val, np.generic):
            val = val.item()

        if _is_bool_like(val):
            return f"{key}={bool(val)}"

        if isinstance(val, (int, np.integer)):
            return f"{key}={int(val)}"

        if isinstance(val, float):
            # be explicit for non-finite floats (avoids ValueError with format spec)
            if not math.isfinite(val):
                return f"{key}={val}"
            return f"{key}={val:.{precision}f}"

        # fallback: strings, None, objects, etc.
        return f"{key}={val}"

    def _flatten(key: str, val: Any) -> list[str]:
        # numpy scalar -> python scalar early
        if isinstance(val, np.generic):
            val = val.item()

        if isinstance(val, dict):
            out: list[str] = []
            for kk, vv in val.items():
                out.extend(_flatten(f"{key}.{kk}", vv))
            return out

        if isinstance(val, (list, tuple)):
            out: list[str] = []
            for i, vv in enumerate(val):
                out.extend(_flatten(f"{key}[{i}]", vv))
            return out

        if isinstance(val, set):
            # sets are unordered; make deterministic
            out: list[str] = []
            for i, vv in enumerate(sorted(val, key=lambda x: repr(x))):
                out.extend(_flatten(f"{key}[{i}]", vv))
            return out

        return [_scalar_str(key, val)]

    return "\n".join(_flatten(k, v))


# def format_pair1(k: str, v: ScalarLike, precision: int = 6) -> str:
#     """Format a key-value pair as a string.
#
#     Formats a key-value pair where the value can be an integer, boolean, or float.
#     Integers and booleans are formatted without decimal places, while floats are
#     formatted with the specified precision.
#
#     Args:
#         k (str): The key/name of the parameter.
#         v (ScalarLike): The value to format (int, bool, float, or numpy scalar).
#         precision (int, optional): Number of decimal places for float values.
#             Defaults to 6.
#
#     Returns:
#         str: Formatted key-value pair string in the format "key=value".
#
#     Example:
#         >>> format_pair("lr", 0.001)
#         'lr=0.001000'
#         >>> format_pair("epochs", 10)
#         'epochs=10'
#         >>> format_pair("verbose", True)
#         'verbose=True'
#     """
#     # handle the case when v is a (potentially nested) {list, dict, ...}
#     if isinstance(v, (list, dict)):
#
#
#     if isinstance(v, (int, bool, np.integer)):
#         # return f'{k}={v:<3}'
#         return f"{k}={v}"
#     # return f'{k}={v:<3.4f}'
#     return f"{k}={v:<.{precision}f}"


def summarize_dict(
    d: dict,
    precision: int = 6,
    keys_to_skip: Iterable | None = None,
) -> str:
    """
    Summarize a dictionary into a string with formatted key-value pairs.

    Args:
        d (dict): The dictionary to summarize.
        precision (int): The precision for floating point values. Default: ``6``.

    Returns:
        str: A string representation of the dictionary with formatted key-value pairs.
    """
    keys_to_skip = [] if keys_to_skip is None else keys_to_skip
    return " ".join(
        [
            format_pair(k, v, precision=precision)
            for k, v in d.items()
            if k not in keys_to_skip
        ]
    )


def model_summary(
    model: Any,
    verbose: bool = False,
    depth: int = 1,
    input_size: Optional[Sequence[int]] = None,
) -> ModelStatistics | None:
    """
    Print a summary of the model using torchinfo.

    Args:
        model: The model to summarize.
        verbose (bool): Whether to print the summary. Default: ``False``.
        depth (int): The depth of the summary. Default: ``1``.
        input_size (Optional[Sequence[int]]): The input size for the model. Default: ``None``.

    Returns:
        ModelStatistics | None: The model summary if torchinfo is available, otherwise None.
    """
    try:
        from torchinfo import summary

        return summary(
            model,
            input_size=input_size,
            depth=depth,
            verbose=verbose,
        )
        # logger.info(f'\n{summary_str}')

    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "torchinfo not installed, unable to print model summary!"
        )


def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def get_max_memory_allocated(device: torch.device) -> float:
    """
    Get the maximum memory allocated on the specified device.

    Args:
        device (torch.device): The device to check memory allocation for.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device)
    elif torch.xpu.is_available():  #  and ipex is not None:
        try:
            import intel_extension_for_pytorch as ipex

            return ipex.xpu.max_memory_allocated(device)
        except ImportError:
            return -1.0
    raise RuntimeError(f"Memory allocation not available for {device=}")


def get_max_memory_reserved(device: torch.device) -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved(device)
    elif torch.xpu.is_available():  #  and ipex is not None:
        try:
            import intel_extension_for_pytorch as ipex

            return ipex.xpu.max_memory_reserved(device)
        except ImportError:
            return -1.0
    raise RuntimeError(f"Memory allocation not available for {device=}")


# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, MI325X, MI355X and Intel PVC
def get_peak_flops(device_name: str) -> float:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 2.25e15
    elif "MI355X" in device_name:
        # MI355X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
        return 2500e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


def grab_tensor(
    x: Any, force: bool = False
) -> Union[np.ndarray, ScalarLike, None]:
    """Convert various tensor/array-like objects to numpy arrays.

    This function converts different types of array-like objects (tensors, lists, etc.)
    to numpy arrays for consistent handling. Supports PyTorch tensors, numpy arrays,
    and nested lists.

    Args:
        x (Any): The object to convert to a numpy array. Can be None, scalar values,
            lists, numpy arrays, or PyTorch tensors.
        force (bool, optional): Force conversion even if it requires copying data.
            Defaults to False.

    Returns:
        Union[np.ndarray, ScalarLike, None]: Numpy array representation of the input,
            or the original scalar value, or None if input was None.

    Raises:
        ValueError: If unable to convert a list to array.

    Example:
        >>> import torch
        >>> import numpy as np
        >>> grab_tensor([1, 2, 3])
        array([1, 2, 3])
        >>> grab_tensor(torch.tensor([1, 2, 3]))
        array([1, 2, 3])
        >>> grab_tensor(np.array([1, 2, 3]))
        array([1, 2, 3])
    """
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
            raise ValueError(f"Unable to convert list: \n {x=}\n to array")
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
    elif callable(getattr(x, "numpy", None)):
        assert callable(getattr(x, "numpy"))
        return x.numpy(force=force)
    # breakpoint(0)
    # raise ValueError


def check_for_tarball(
    env_prefix: Optional[str | os.PathLike | Path] = None,
    overwrite: Optional[bool] = False,
):
    if env_prefix is None:
        # NOTE:
        # - `sys.executable` looks like:
        #   `/path/to/some/envs/env_name/bin/python`
        fpl = sys.executable.split("/")
        # `env_prefix` looks like `/path/to/some/envs/env_name`
        env_prefix = "/".join(fpl[:-2])
        # `env_name` looks like `env_name`
        env_name = fpl[-3]
    else:
        env_name = Path(env_prefix).name
    # tarball will be `env_name.tar.gz`
    tarball = f"{env_name}.tar.gz"
    tar_on_tmp = Path("/tmp") / tarball
    if overwrite and tar_on_tmp.exists():
        logger.info(f"Removing existing tarball at {tar_on_tmp}")

    if not tar_on_tmp.exists():
        if not os.path.exists(tarball):
            logger.info(f"Creating tarball {tarball} from {env_prefix}")
            make_tarfile(tarball, env_prefix)
        else:
            logger.info(
                f"Tarball {tarball} already exists in current directory"
            )
    else:
        logger.info(f"Tarball {tarball} already exists, skipping creation")
    return tar_on_tmp if tar_on_tmp.exists() else Path(tarball)


def make_tarfile(
    output_filename: str,
    source_dir: str | os.PathLike | Path,
) -> str:
    output_filename = (
        output_filename.replace(".tar", "").replace(".gz", "") + ".tar.gz"
    )
    srcfp = Path(source_dir).absolute().resolve()
    dirname = srcfp.name
    logger.info(f"Creating tarball at {output_filename} from {source_dir}")
    logger.info(
        f"Executing: 'tar -cvf {output_filename} --directory  {srcfp.parent} {dirname}'"
    )
    os.system(
        f"tar -cvf {output_filename} --directory  {srcfp.parent} {dirname}"
    )

    return output_filename


def create_tarball(src: str | os.PathLike) -> Path:
    src_dir = Path(src).resolve().absolute()
    root_dir = Path(src).parent.resolve().absolute()
    assert root_dir.exists(), f"{root_dir} does not exist"
    fpname = f"{src_dir.name}"
    dst_fp = make_tarfile(fpname, src_dir.as_posix())
    return Path(dst_fp)


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
            logger.warning(
                "Unable to save as `.h5` file, falling back to `netCDF4`"
            )
            save_dataset(
                dataset, outdir=outdir, use_hdf5=False, fname=fname, **kwargs
            )
    else:
        fname = "dataset.nc" if fname is None else f"{fname}_dataset.nc"
        outfile = Path(outdir).joinpath(fname)
        mode = "a" if outfile.is_file() else "w"
        logger.info(f"Saving dataset to: {outfile.as_posix()}")
        outfile.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(outfile.as_posix(), mode=mode)

    return outfile


def dataset_to_h5pyfile(hfile: PathLike, dataset: xr.Dataset, **kwargs):
    try:
        import h5py
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "h5py is not installed. Please install h5py to use this function."
        )

    logger.info(f"Saving dataset to: {hfile}")
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
    try:
        import h5py
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "h5py is not installed. Please install h5py to use this function."
        )
    f = h5py.File(hfile, "r")
    data = {key: f[key] for key in list(f.keys())}
    f.close()
    return data


def dataset_from_h5pyfile(hfile: PathLike) -> xr.Dataset:
    try:
        import h5py
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "h5py is not installed. Please install h5py to use this function."
        )
    f = h5py.File(hfile, "r")
    data = {key: f[key] for key in list(f.keys())}
    f.close()

    return xr.Dataset(data)


def get_deepspeed_zero_config_json(zero_config: ZeroConfig) -> dict:
    """Return the DeepSpeed zero config as a dict."""
    return asdict(zero_config)


def write_generic_deepspeed_config(
    gradient_accumulation_steps: int = 1,
    gradient_clipping: str | float = "auto",
    steps_per_print: int = 10,
    train_batch_size: str = "auto",
    train_micro_batch_size_per_gpu: str = "auto",
    wall_clock_breakdown: bool = False,
    wandb: Optional[dict] = None,
    bf16: Optional[dict] = None,
    fp16: Optional[dict] = None,
    flops_profiler: Optional[dict] = None,
    optimizer: Optional[dict] = None,
    scheduler: Optional[dict] = None,
    zero_optimization: Optional[dict] = None,
):
    """
    Write a generic deepspeed config to the output directory.
    """
    ds_config = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": steps_per_print,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "wall_clock_breakdown": wall_clock_breakdown,
        "wandb": wandb,
        "bf16": bf16,
        "fp16": fp16,
        "flops_profiler": flops_profiler,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "zero_optimization": zero_optimization,
    }
    return ds_config


def get_deepspeed_adamw_optimizer_config_json(
    auto_config: Optional[bool] = True,
) -> dict:
    """
    Get the deepspeed adamw optimizer config json.

    Args:
        auto_config (bool): Whether to use the auto config. Default: ``True``.

    Returns:
        dict: Deepspeed adamw optimizer config.
    """
    return (
        {"type": "AdamW"}
        if not auto_config
        else {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "weight_decay": "auto",
                "torch_adam": True,
                "adam_w_mode": True,
            },
        }
    )


def get_deepspeed_warmup_decay_scheduler_config_json(
    auto_config: Optional[bool] = True,
) -> dict:
    """
    Get the deepspeed warmup decay scheduler config json.

    Args:
        auto_config (bool): Whether to use the auto config. Default: ``True``.

    Returns:
        dict: Deepspeed warmup decay scheduler config.
    """
    return (
        {"type": "WarmupDecayLR"}
        if not auto_config
        else {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        }
    )


def get_flops_profiler_config_json(
    enabled: bool = True,
    profile_step: int = 1,
    module_depth: int = -1,
    top_modules: int = 1,
    detailed: bool = True,
) -> dict:
    """
    Get the deepspeed flops profiler config json.

    Args:
        enabled (bool): Whether to use the flops profiler. Default: ``True``.
        profile_step (int): The step to profile. Default: ``1``.
        module_depth (int): The depth of the module. Default: ``-1``.
        top_modules (int): The number of top modules to show. Default: ``1``.
        detailed (bool): Whether to show detailed profiling. Default: ``True``.

    Returns:
        dict: Deepspeed flops profiler config.
    """
    return {
        "enabled": enabled,
        "profile_step": profile_step,
        "module_depth": module_depth,
        "top_modules": top_modules,
        "detailed": detailed,
    }


def get_bf16_config_json(
    enabled: bool = True,
) -> dict:
    """
    Get the deepspeed bf16 config json.

    Args:
        enabled (bool): Whether to use bf16. Default: ``True``.

    Returns:
        dict: Deepspeed bf16 config.
    """
    return {"enabled": enabled}


def get_fp16_config_json(
    enabled: bool = True,
) -> dict[str, bool]:
    """
    Get the deepspeed fp16 config json.

    Args:
        enabled (bool): Whether to use fp16. Default: ``True``.

    Returns:
        dict: Deepspeed fp16 config.
    """
    return {"enabled": enabled}


def get_deepspeed_config_json(
    auto_config: Optional[bool] = True,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: Optional[str | float] = "auto",
    steps_per_print: Optional[int] = 10,
    train_batch_size: str = "auto",
    train_micro_batch_size_per_gpu: str = "auto",
    wall_clock_breakdown: bool = False,
    wandb: bool = True,  # NOTE: Opinionated, W&B is enabled by default
    bf16: bool = True,  # NOTE: Opinionated, BF16 is enabled by default
    fp16: Optional[bool] = None,
    flops_profiler: Optional[dict] = None,
    optimizer: Optional[dict] = None,
    scheduler: Optional[dict] = None,
    zero_optimization: Optional[dict] = None,
    stage: Optional[int] = 0,
    allgather_partitions: Optional[bool] = None,
    allgather_bucket_size: Optional[int] = int(5e8),
    overlap_comm: Optional[bool] = None,
    reduce_scatter: Optional[bool] = True,
    reduce_bucket_size: Optional[int] = int(5e8),
    contiguous_gradients: Optional[bool] = None,
    offload_param: Optional[dict] = None,
    offload_optimizer: Optional[dict] = None,
    stage3_max_live_parameters: Optional[int] = int(1e9),
    stage3_max_reuse_distance: Optional[int] = int(1e9),
    stage3_prefetch_bucket_size: Optional[int] = int(5e8),
    stage3_param_persistence_threshold: Optional[int] = int(1e6),
    sub_group_size: Optional[int] = None,
    elastic_checkpoint: Optional[dict] = None,
    stage3_gather_16bit_weights_on_model_save: Optional[bool] = None,
    ignore_unused_parameters: Optional[bool] = None,
    round_robin_gradients: Optional[bool] = None,
    zero_hpz_partition_size: Optional[int] = None,
    zero_quantized_weights: Optional[bool] = None,
    zero_quantized_gradients: Optional[bool] = None,
    log_trace_cache_warnings: Optional[bool] = None,
    save_config: bool = True,
    output_file: Optional[str] = None,
    output_dir: Optional[PathLike] = None,
) -> dict[str, Any]:
    """
    Write a deepspeed config to the output directory.
    """
    import json

    wandb_config = {"enabled": wandb}
    bf16_config = {"enabled": bf16}
    fp16_config = {"enabled": fp16}
    flops_profiler_config = (
        get_flops_profiler_config_json()
        if flops_profiler is None
        else flops_profiler
    )

    optimizer = (
        get_deepspeed_adamw_optimizer_config_json()
        if optimizer is None
        else optimizer
    )
    scheduler = (
        get_deepspeed_warmup_decay_scheduler_config_json()
        if scheduler is None
        else scheduler
    )

    if stage is not None and int(stage) > 0:
        zero_optimization = (
            get_deepspeed_zero_config_json(
                stage=stage,
                allgather_partitions=allgather_partitions,
                allgather_bucket_size=allgather_bucket_size,
                overlap_comm=overlap_comm,
                reduce_scatter=reduce_scatter,
                reduce_bucket_size=reduce_bucket_size,
                contiguous_gradients=contiguous_gradients,
                offload_param=offload_param,
                offload_optimizer=offload_optimizer,
                stage3_max_live_parameters=stage3_max_live_parameters,
                stage3_max_reuse_distance=stage3_max_reuse_distance,
                stage3_prefetch_bucket_size=stage3_prefetch_bucket_size,
                stage3_param_persistence_threshold=stage3_param_persistence_threshold,
                sub_group_size=sub_group_size,
                elastic_checkpoint=elastic_checkpoint,
                stage3_gather_16bit_weights_on_model_save=stage3_gather_16bit_weights_on_model_save,
                ignore_unused_parameters=ignore_unused_parameters,
                round_robin_gradients=round_robin_gradients,
                zero_hpz_partition_size=zero_hpz_partition_size,
                zero_quantized_weights=zero_quantized_weights,
                zero_quantized_gradients=zero_quantized_gradients,
                log_trace_cache_warnings=log_trace_cache_warnings,
            )
            if zero_optimization is None
            else zero_optimization
        )
    else:
        zero_optimization = None
    ds_config = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": steps_per_print,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "wall_clock_breakdown": wall_clock_breakdown,
        "wandb": wandb,
        "bf16": bf16,
        "fp16": fp16,
        "flops_profiler": flops_profiler,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "zero_optimization": zero_optimization,
    }
    if save_config:
        if output_file is None:
            if output_dir is None:
                output_dir = Path(os.getcwd()).joinpath("ds_configs")
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            outfile = output_dir.joinpath("deepspeed_config.json")
        else:
            outfile = Path(output_file)
        logger.info(f"Saving DeepSpeed config to: {outfile.as_posix()}")
        logger.info(json.dumps(ds_config, indent=4))
        with outfile.open("w") as f:
            json.dump(
                ds_config,
                fp=f,
                indent=4,
            )

    return ds_config


def write_deepspeed_zero12_auto_config(
    zero_stage: int = 1, output_dir: Optional[PathLike] = None
) -> dict:
    """
    Write a deepspeed zero1 auto config to the output directory.
    """
    import json

    ds_config = {
        "gradient_accumulation_steps": 1,
        "gradient_clipping": "auto",
        "steps_per_print": 1,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": True,
        "wandb": {"enabled": True},
        "bf16": {"enabled": True},
        "flops_profiler": {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "weight_decay": "auto",
                "torch_adam": True,
                "adam_w_mode": True,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
        },
    }
    if output_dir is None:
        output_dir = Path(os.getcwd()).joinpath("ds_configs")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    outfile = output_dir.joinpath(
        f"deepspeed_zero{zero_stage}_auto_config.json"
    )
    logger.info(
        f"Saving DeepSpeed ZeRO Stage {zero_stage} "
        f"auto config to: {outfile.as_posix()}"
    )
    with outfile.open("w") as f:
        json.dump(
            ds_config,
            fp=f,
            indent=4,
        )

    return ds_config


def write_deepspeed_zero3_auto_config(
    zero_stage: int = 3, output_dir: Optional[PathLike] = None
) -> dict:
    """
    Write a deepspeed zero1 auto config to the output directory.
    """
    import json

    ds_config = {
        "gradient_accumulation_steps": 1,
        "gradient_clipping": "auto",
        "steps_per_print": 1,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": True,
        "wandb": {"enabled": True},
        "bf16": {"enabled": True},
        "flops_profiler": {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "weight_decay": "auto",
                "torch_adam": True,
                "adam_w_mode": True,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
        },
    }
    if output_dir is None:
        output_dir = Path(os.getcwd()).joinpath("ds_configs")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    outfile = output_dir.joinpath(
        f"deepspeed_zero{zero_stage}_auto_config.json"
    )
    logger.info(
        f"Saving DeepSpeed ZeRO Stage {zero_stage} "
        f"auto config to: {outfile.as_posix()}"
    )
    with outfile.open("w") as f:
        json.dump(
            ds_config,
            fp=f,
            indent=4,
        )

    return ds_config
