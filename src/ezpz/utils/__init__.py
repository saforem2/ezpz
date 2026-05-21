"""
ezpz/utils/__init__.py
"""

from __future__ import annotations

import logging
import os
import pdb
import re
import subprocess
import sys
import tqdm
from typing import Any
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

import math

# import numpy as np

# ScalarLike = Any  # keep your existing alias if you already have one

# import torch.distributed as tdist


__all__ = [
    "Color",
    "NoColor",
    "DistributedPdb",
    "breakpoint",
    "get_timestamp",
    "format_pair",
    "summarize_dict",
    "model_summary",
    "normalize",
    "get_max_memory_allocated",
    "get_max_memory_reserved",
    "get_current_memory_allocated",
    "get_current_memory_reserved",
    "reset_peak_memory_stats",
    "get_memory_metrics",
    "format_memory_summary",
    "format_compact_summary",
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
    "ForkedPdb",
    "DummyTqdmFile",
]

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
    ezpz.distributed.barrier()


class ForkedPdb(pdb.Pdb):
    """PDB subclass for debugging multi-processed code."""

    def interaction(self, *args, **kwargs):  # pragma: no cover - interactive
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class DummyTqdmFile:
    """Dummy file-like wrapper that forwards writes to tqdm."""

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, text):
        if len(text.rstrip()) > 0:
            tqdm.tqdm.write(text, file=self.file, end="\n")

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


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

    Examples:
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


def _device_type(device: "torch.device | int | str | None") -> str:
    """Return the canonical device type for routing memory APIs.

    Accepts the same input shapes as torch.device(). Maps:
        - int           → "cuda" or "xpu" based on availability (ambiguous;
                          we prefer CUDA when both are present, matching torch's
                          own convention).
        - "cuda:0", torch.device("cuda", 0), etc. → "cuda"
        - "cpu", "cpu:0", torch.device("cpu") → "cpu"
        - "mps" → "mps"
        - None → resolved via ezpz.get_torch_device()

    Branching on this string lets us call the right backend instead of
    routing every call to torch.cuda.* whenever CUDA happens to be
    available — which previously raised when a CPU/MPS device was passed
    on a CUDA-capable box (issue caught in PR #134 review).
    """
    if device is None:
        import ezpz
        device = ezpz.get_torch_device()
    if isinstance(device, int):
        # Bare int is ambiguous between cuda:N and xpu:N. Mirror torch's
        # default by preferring CUDA when available; otherwise XPU.
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    return torch.device(device).type


def get_max_memory_allocated(device: "torch.device | int | str") -> float:
    """Peak allocated memory in bytes on ``device``. 0.0 on CPU/MPS.

    Routes to the backend matching ``device``'s type, not whichever
    accelerator happens to be globally available. So
    ``get_max_memory_allocated("cpu")`` returns 0.0 even on a CUDA box.
    """
    dtype = _device_type(device)
    if dtype == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if dtype == "xpu":
        try:
            return torch.xpu.max_memory_allocated(device)
        except (ImportError, AttributeError):
            return 0.0
    return 0.0


def get_max_memory_reserved(device: "torch.device | int | str") -> float:
    """Peak reserved memory in bytes on ``device``. 0.0 on CPU/MPS."""
    dtype = _device_type(device)
    if dtype == "cuda":
        return torch.cuda.max_memory_reserved(device)
    if dtype == "xpu":
        try:
            return torch.xpu.max_memory_reserved(device)
        except (ImportError, AttributeError):
            return 0.0
    return 0.0


def get_current_memory_allocated(device: "torch.device | int | str") -> float:
    """Currently allocated memory in bytes on ``device``. 0.0 on CPU/MPS."""
    dtype = _device_type(device)
    if dtype == "cuda":
        return torch.cuda.memory_allocated(device)
    if dtype == "xpu":
        try:
            return torch.xpu.memory_allocated(device)
        except (ImportError, AttributeError):
            return 0.0
    return 0.0


def get_current_memory_reserved(device: "torch.device | int | str") -> float:
    """Currently reserved memory in bytes on ``device``. 0.0 on CPU/MPS."""
    dtype = _device_type(device)
    if dtype == "cuda":
        return torch.cuda.memory_reserved(device)
    if dtype == "xpu":
        try:
            return torch.xpu.memory_reserved(device)
        except (ImportError, AttributeError):
            return 0.0
    return 0.0


def reset_peak_memory_stats(device: "torch.device | int | str") -> None:
    """Reset peak-memory counters on ``device``. No-op on CPU/MPS."""
    dtype = _device_type(device)
    if dtype == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return
    if dtype == "xpu":
        try:
            torch.xpu.reset_peak_memory_stats(device)
        except (ImportError, AttributeError):
            pass


def get_memory_metrics(
    device: "torch.device | int | str | None" = None,
    *,
    reset_peak: bool = True,
    prefix: str = "",
) -> dict[str, float]:
    """Return device memory metrics in GiB.

    Returns 4 keys when supported (CUDA, XPU):

        {prefix}mem_alloc          currently allocated
        {prefix}mem_peak_alloc     peak allocated since last reset
        {prefix}mem_reserved       currently reserved by the allocator
        {prefix}mem_peak_reserved  peak reserved since last reset

    Returns ``{}`` on CPU / MPS (silent — caller's metrics dict simply
    doesn't gain these keys), and unconditionally when the env var
    ``EZPZ_TRACK_MEMORY=0`` is set.

    Args:
        device: device to query. If None, uses ``ezpz.get_torch_device()``.
        reset_peak: if True (default), reset peak counters AFTER reading.
            Next call's ``mem_peak_*`` then reflect only what happened
            between calls — the standard per-step pattern.
        prefix: optional string prepended to every key. Useful for the
            examples that namespace their metrics (e.g. ``"train/"``).
    """
    if os.environ.get("EZPZ_TRACK_MEMORY", "1") == "0":
        return {}

    # Lazy default device resolution — only pay the cost when caller
    # didn't pass one explicitly.
    if device is None:
        import ezpz
        device = ezpz.get_torch_device()
    assert device is not None  # type narrowing for pyright

    # On CPU/MPS, all four helpers return 0.0. Short-circuit to avoid
    # emitting a row of zeros. Route via the canonical device type so
    # the check works for torch.device('cpu'), 'cpu', 'cpu:0', etc.
    if _device_type(device) not in ("cuda", "xpu"):
        return {}

    _GIB = 1024 ** 3
    metrics = {
        f"{prefix}mem_alloc": get_current_memory_allocated(device) / _GIB,
        f"{prefix}mem_peak_alloc": get_max_memory_allocated(device) / _GIB,
        f"{prefix}mem_reserved": get_current_memory_reserved(device) / _GIB,
        f"{prefix}mem_peak_reserved": get_max_memory_reserved(device) / _GIB,
    }
    if reset_peak:
        reset_peak_memory_stats(device)
    return metrics


def format_memory_summary(
    metrics: dict[str, float],
    *,
    device: "torch.device | int | str | None" = None,
    prefix: "str | None" = None,
) -> str:
    """Condense the four mem_* keys into a single console-friendly string.

    Input: a dict that contains (some subset of) the keys produced by
    :func:`get_memory_metrics` — ``{prefix}mem_alloc``,
    ``{prefix}mem_peak_alloc``, ``{prefix}mem_reserved``,
    ``{prefix}mem_peak_reserved``.

    Output: ``"X.XX/Y.YYGiB (Z%)"`` where X is current alloc, Y is peak
    alloc, and Z is current alloc as a percent of device total memory
    (omitted when device total isn't available — e.g. unknown XPU, CPU
    fallback).

    Args:
        metrics: dict that may contain mem_* keys.
        device: optional device for total-memory lookup. ``int`` index
            or ``torch.device('cuda:N')`` honored; ``None`` uses the
            local rank's device.
        prefix: explicit prefix (e.g. ``"train/"``). When ``None``
            (default), the prefix is inferred by scanning ``metrics`` for
            ``*mem_alloc`` / ``*mem_peak_alloc`` keys — so callers that
            don't know whether their metrics are namespaced don't have
            to probe twice.

    Returns an empty string if no mem_* keys are present (CPU/MPS) — so
    callers can ``" ".join(filter(None, [...]))`` without checking.
    """
    if prefix is None:
        prefix = ""
        for key in metrics:
            if key.endswith("mem_alloc") or key.endswith("mem_peak_alloc"):
                # Strip the suffix to recover whatever namespace the
                # caller used (e.g. "train/", "eval/", or "").
                if key.endswith("mem_peak_alloc"):
                    prefix = key[: -len("mem_peak_alloc")]
                else:
                    prefix = key[: -len("mem_alloc")]
                break
    alloc = metrics.get(f"{prefix}mem_alloc")
    peak = metrics.get(f"{prefix}mem_peak_alloc")
    if alloc is None and peak is None:
        return ""
    # Total device VRAM for the percentage. Lazy-resolve to avoid the
    # import cost when caller has no memory keys to format anyway.
    # Normalize `device` → int index where possible, so callers passing
    # `torch.device('cuda:1')` get the right device's total (not rank 0's
    # device by way of get_local_rank()).
    pct_str = ""
    try:
        import ezpz
        idx: int | None
        if isinstance(device, int):
            idx = device
        elif device is None:
            idx = None
        else:
            # torch.device('cuda:1').index == 1; torch.device('cuda').index is None
            try:
                idx = torch.device(device).index
            except (TypeError, RuntimeError):
                idx = None
        props = ezpz.distributed.get_device_properties(idx)
        total_bytes = props.get("total_memory", -1)
        if total_bytes and total_bytes > 0 and alloc is not None:
            total_gib = total_bytes / (1024 ** 3)
            pct = 100.0 * alloc / total_gib
            pct_str = f" ({pct:.0f}%)"
    except Exception:
        # Any failure resolving total memory: omit the percentage rather
        # than break logging. Raw numbers still print.
        pct_str = ""

    if alloc is not None and peak is not None:
        return f"{alloc:.2f}/{peak:.2f}GiB{pct_str}"
    if alloc is not None:
        return f"{alloc:.2f}GiB{pct_str}"
    # Only peak is present (rare — caller passed `mem_peak_alloc` without
    # `mem_alloc`). Format matches the alloc-only branch for consistency.
    return f"{peak:.2f}GiB{pct_str}"


# Base names for the 4 memory metrics produced by get_memory_metrics().
_MEMORY_METRIC_BASES = (
    "mem_alloc", "mem_peak_alloc", "mem_reserved", "mem_peak_reserved",
)
# History._compute_distributed_metrics appends these suffixes to every
# numeric key it aggregates across ranks. We need to strip the aggregated
# forms too — not just the raw bases — or the console summary still
# shows 16 noisy `mem_alloc/mean=...` style lines.
_AGGREGATION_SUFFIXES = ("", "/mean", "/max", "/min", "/std", "/avg")


def _format_std(std: float, *, precision: int) -> "str | None":
    """Format a std value for the inline ``(±X)`` console suffix.

    Returns:
        - ``None`` when the std rounds to 0 at the chosen precision —
          the caller should drop the parenthetical entirely (``(±0)``
          adds no signal; e.g. LR with `lr/std=0`).
        - A trimmed string otherwise. We use lower precision than the
          base value (std is a noise-band hint, not a measurement)
          and strip trailing zeros: `0.157124` → `0.16`, `0.000` → None.

    The std-precision policy: ``min(precision, 2)`` keeps the visual
    weight of `(±0.16)` proportional to `loss=0.289993` without
    drowning the actual value in trailing digits.
    """
    if std == 0:
        return None
    std_precision = min(precision, 2)
    formatted = f"{std:.{std_precision}f}"
    # After truncation, the value might be 0.00 — same "no signal" case.
    if float(formatted) == 0:
        # Try one more precision step before giving up, in case caller
        # asked for precision=6 and we have std=1e-5: 0.00 → 0.00001.
        if precision > std_precision:
            formatted = f"{std:.{precision}f}"
            if float(formatted) == 0:
                return None
        else:
            return None
    # Strip trailing zeros (`0.10` → `0.1`, `1.00` → `1`). Keep at
    # least one digit after the decimal if there is one.
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
        if not formatted or formatted == "-":
            formatted = "0"
    return formatted


_DEFAULT_MIN_WIDTHS: dict[str, int] = {
    # Counters whose value width grows over the run. Pad so the eye can
    # scan down the left edge — `iter=8     ` aligns under `iter=12000`.
    # Widths include the `iter=` prefix; chosen for typical training-run
    # upper bounds (5-digit iters/steps, 4-digit epochs). Override
    # per-call via the `min_widths` kwarg if you need more or less.
    "iter": 10,   # supports up to 99999 iters
    "step": 10,
    "epoch": 8,   # supports up to 999 epochs
    "batch": 10,
    "idx": 10,
    "bidx": 10,
}


def format_compact_summary(
    metrics: dict[str, float],
    precision: int = 6,
    keys_to_skip: Iterable | None = None,
    min_widths: "dict[str, int] | None" = None,
) -> str:
    """Render *metrics* as a compact ``key=value(±std)`` summary line.

    Designed to replace the noisy per-step summary that looks like:

        loss=0.047 loss/mean=0.030 loss/max=0.120 loss/min=0.011 loss/std=0.022
        accuracy=1.000 accuracy/mean=0.997 accuracy/max=1.000 ...

    with the much tighter:

        loss=0.047(±0.022) accuracy=1.000(±0.006) ...

    Rules:
      - For each base metric ``X``, look up ``X/std`` in *metrics* and
        append it inline as ``X=val(±std)``. Other aggregation suffixes
        (``/mean``, ``/min``, ``/max``, ``/avg``) are dropped entirely
        from the console line — trackers still get them.
      - Memory keys (handled by :func:`format_memory_summary`) are
        skipped here. The caller is expected to append the compact
        memory token separately at the end of the line.
      - Counter-like base names (``iter``, ``step``, ``epoch``, ``batch``,
        ``idx``) suppress the ``(±std)`` suffix even if std is present
        — a counter's std is meaningless noise.
      - Counter tokens are right-padded so successive lines align at
        the left edge: ``iter=8     loss=...`` lines up under
        ``iter=180   loss=...``. Override widths via ``min_widths``.

    Aggregation values (``X/mean``, ``X/std``, etc.) that have NO
    corresponding base value in *metrics* are still emitted as
    standalone keys, so we don't silently lose data.
    """
    skip = set(keys_to_skip or ())
    # Merge caller-supplied widths over the defaults.
    widths: dict[str, int] = dict(_DEFAULT_MIN_WIDTHS)
    if min_widths:
        widths.update(min_widths)

    def _pad(base_name: str, token: str) -> str:
        """Right-pad ``token`` so each line's counter aligns with prior
        lines. ``base_name`` strips any namespace prefix (``train/iter``
        → ``iter``) before looking up the configured width."""
        leaf = base_name.rsplit("/", 1)[-1]
        target = widths.get(leaf)
        if target is None or len(token) >= target:
            return token
        return token + " " * (target - len(token))
    # Pre-build a lookup of std values keyed by base name so we can
    # match them onto bases in a single pass.
    std_lookup: dict[str, float] = {}
    aggregation_suffixes = ("/mean", "/min", "/max", "/std", "/avg")
    aggregation_keys: set[str] = set()
    for k, v in metrics.items():
        for suffix in aggregation_suffixes:
            if k.endswith(suffix):
                aggregation_keys.add(k)
                if suffix == "/std":
                    std_lookup[k[: -len(suffix)]] = float(v)
                break

    # Counter-like bases for which (±std) is meaningless.
    _counter_bases = ("iter", "step", "epoch", "batch", "idx")

    def _is_counter(base: str) -> bool:
        # Match exact name AND prefixed forms (e.g. "train/iter").
        return base.rsplit("/", 1)[-1] in _counter_bases

    tokens: list[str] = []
    seen_bases: set[str] = set()
    for k, v in metrics.items():
        if k in skip:
            continue
        if is_memory_metric_key(k):
            continue
        if k in aggregation_keys:
            continue  # handled inline (via std_lookup) or skipped
        seen_bases.add(k)
        base_token = format_pair(k, v, precision=precision)
        std = std_lookup.get(k)
        if std is not None and not _is_counter(k):
            std_token = _format_std(std, precision=precision)
            if std_token is None:
                # std rounds to zero at the chosen precision (e.g.
                # `lr/std=1e-12` with precision=2). `(±0)` adds no
                # signal; drop it.
                tokens.append(base_token)
            else:
                tokens.append(f"{base_token}(±{std_token})")
        else:
            # Pad counter tokens so the next field aligns across rows.
            tokens.append(_pad(k, base_token))

    # Emit aggregation keys whose base wasn't present in the dict — so
    # we don't silently drop them. (Rare; happens when caller passes
    # only an aggregated metric, e.g. ``loss/mean`` without ``loss``.)
    for k in aggregation_keys:
        base = next(
            k[: -len(s)] for s in aggregation_suffixes if k.endswith(s)
        )
        if base in seen_bases or k in skip:
            continue
        # Memory-metric aggregations are handled by format_memory_summary;
        # never emit them as standalone tokens here.
        if is_memory_metric_key(k):
            continue
        # /std for a base we already showed inline → drop. Other
        # aggregations without a base → keep (visible debug info).
        if k.endswith("/std") and base in std_lookup and base in seen_bases:
            continue
        tokens.append(format_pair(k, metrics[k], precision=precision))

    return " ".join(tokens)


def is_memory_metric_key(key: str) -> bool:
    """True if *key* is one of the 4 mem_* metrics (raw OR aggregated).

    Matches both:
      - raw: ``mem_alloc``, ``train/mem_peak_reserved``, etc.
      - aggregated: ``mem_alloc/mean``, ``train/mem_alloc/max``, etc.
        (History._compute_distributed_metrics emits these per rank.)

    Does NOT match unrelated keys that happen to contain ``mem_`` —
    e.g. ``mem_loss`` or ``memo_field`` — because we anchor on the
    full base name + aggregation suffix.
    """
    for base in _MEMORY_METRIC_BASES:
        for suffix in _AGGREGATION_SUFFIXES:
            target = f"{base}{suffix}"
            # endswith() catches both 'mem_alloc' and 'train/mem_alloc';
            # the explicit base+suffix list keeps 'mem_loss' from matching.
            if key == target or key.endswith(f"/{target}"):
                return True
    return False


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
        logger.warning(
            f"Error running lspci: {e}, fallback to use device_name"
        )
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
        max_comp_units = torch.xpu.get_device_properties(
            "xpu"
        ).max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(
            f"Peak flops undefined for: {device_name}, fallback to A100"
        )
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

    Examples:
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
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        if len(x) == 0:
            return np.array([])
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        if isinstance(x[0], np.ndarray):
            return np.stack(x)
        if isinstance(x[0], (int, float, bool, np.floating)):
            return np.array(x)
        if isinstance(x[0], (tuple, list)):
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
    """Locate or create a `.tar.gz` of *env_prefix*; return its absolute path.

    Search order (first hit wins, unless ``overwrite=True``):

    1. ``<env_prefix.parent>/<env_name>.tar.gz`` — alongside the env.
       This is where ``_suggest_tarball_if_present`` (in
       ``ezpz.utils.yeet_env``) looks first, so co-locating here means
       a subsequent ``ezpz yeet`` (no args) will see and suggest it.
    2. ``/tmp/<env_name>.tar.gz`` — node-local fallback.
    3. ``<cwd>/<env_name>.tar.gz`` — current-directory fallback.

    If none exist (or ``overwrite=True``), creates a new tarball at
    location #1 (next to the env).
    """
    if env_prefix is None:
        # NOTE:
        # - `sys.executable` looks like:
        #   `/path/to/some/envs/env_name/bin/python`
        fpl = sys.executable.split("/")
        # `env_prefix` looks like `/path/to/some/envs/env_name`
        env_prefix = "/".join(fpl[:-2])
    env_path = Path(env_prefix).resolve()
    env_name = env_path.name
    tarball_name = f"{env_name}.tar.gz"

    candidates = [
        env_path.parent / tarball_name,    # next to the venv (canonical)
        Path("/tmp") / tarball_name,
        Path.cwd() / tarball_name,
    ]
    if overwrite:
        for c in candidates:
            if c.exists():
                logger.info(f"Removing existing tarball at {c}")
                c.unlink()
    else:
        for c in candidates:
            if c.exists():
                logger.info(f"Tarball {c} already exists, skipping creation")
                return c

    target = candidates[0]
    logger.info(f"Creating tarball {target} from {env_prefix}")
    make_tarfile(str(target), str(env_prefix))
    return target


def make_tarfile(
    output_filename: str,
    source_dir: str | os.PathLike | Path,
) -> str:
    """Create a gzipped tar archive of *source_dir* at *output_filename*.

    Normalizes the output to end in `.tar.gz`, then runs
    ``tar -czvf <out> -C <parent> <dirname>``. Uses subprocess (not
    os.system + f-string) so paths with spaces or shell-meta characters
    don't break or get reinterpreted.
    """
    output_filename = (
        output_filename.replace(".tar", "").replace(".gz", "") + ".tar.gz"
    )
    srcfp = Path(source_dir).absolute().resolve()
    dirname = srcfp.name
    cmd = [
        "tar", "-czvf", output_filename,
        "--directory", str(srcfp.parent), dirname,
    ]
    logger.info(f"Creating tarball at {output_filename} from {source_dir}")
    logger.info("Executing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
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
