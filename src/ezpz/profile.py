"""
profile.py

Sam Foreman
[2024-06-21]

Contains implementation of:

- `get_context_manager`
- `PyInstrumentProfiler`

which can be used as a context manager to profile a block of code, e.g.


```python
# test.py


def main():
    print("Hello!")
    from ezpz.profile import get_context_manager

    # NOTE:
    # 1. if `rank` is passed to `get_context_manager`:
    #        - it will ONLY be instantiated if rank == 0,
    #          otherwise, it will return a contextlib.nullcontext() instance.
    # 2. if `strict=True`:
    #        - only run if "PYINSTRUMENT_PROFILER=1" in environment
    cm = get_context_manager(rank=RANK, strict=False)
    with cm:
        main()


if __name__ == "__main__":
    main()
```

"""

import os
import time
import logging
import datetime

import ezpz
import torch
import torch.profiler
from torch.profiler import (
    ProfilerAction,
    profile,
    # record_function,
    ProfilerActivity,
)

from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from contextlib import nullcontext, AbstractContextManager


def get_profiling_context(
    profiler_type: str,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    rank_zero_only: bool,
    record_shapes: bool = True,
    with_stack: bool = True,
    with_flops: bool = True,
    with_modules: bool = True,
    acc_events: bool = False,
    profile_memory: bool = False,
    outdir: Optional[str | Path | os.PathLike] = None,
    strict: Optional[bool] = True,
) -> AbstractContextManager:
    """
    Returns a context manager for profiling code blocks using either
    PyTorch Profiler or PyInstrument.

    Args:
        profiler_type (str): The type of profiler to use.
            Must be one of ['torch', 'pyinstrument'].
        wait (int): The number of steps to wait before starting profiling.
        warmup (int): The number of warmup steps before profiling starts.
        active (int): The number of active profiling steps.
        repeat (int): The number of times to repeat the profiling schedule.
        rank_zero_only (bool): If True, the profiler will only run on rank 0.
            Defaults to True.
        record_shapes (bool): If True, shapes of tensors are recorded.
            Defaults to True.
        with_stack (bool): If True, stack traces are recorded.
            Defaults to True.
        with_flops (bool): If True, FLOPs are recorded.
            Defaults to True.
        with_modules (bool): If True, module information is recorded.
            Defaults to True.
        acc_events (bool): If True, accumulated events are recorded.
            Defaults to False.
        profile_memory (bool): If True, memory profiling is enabled.
            Defaults to False.
        outdir (Optional[str | Path | os.PathLike]): The output directory
            for saving profiles. Defaults to `ezpz.OUTPUTS_DIR`.
        strict (Optional[bool]): If True, the profiler will only run if
            "PYINSTRUMENT_PROFILER" is set in the environment. Defaults to True.
    Returns:
        AbstractContextManager: A context manager that starts and stops
            the profiler.
    """
    if profiler_type not in {"pt", "pytorch", "torch", "pyinstrument"}:
        raise ValueError(
            f"Invalid profiling type: {profiler_type}. "
            "Must be one of ['torch', 'pyinstrument']"
        )
    outdir_fallback = Path(os.getcwd()).joinpath("ezpz", "torch_profiles")
    outdir = outdir_fallback if outdir is None else outdir
    _ = Path(outdir).mkdir(parents=True, exist_ok=True)
    if profiler_type in {"torch", "pytorch", "pt"}:

        def trace_handler(p: torch.profiler.profile):
            """
            Callback function to handle the trace when it is ready.
            """
            logger.info(
                "\n"
                + p.key_averages().table(
                    sort_by=(f"self_{ezpz.get_torch_device_type()}_time_total"),
                    row_limit=-1,
                )
            )
            fname: str = "-".join(
                [
                    "torch-profiler",
                    f"rank{ezpz.get_rank()}",
                    f"step{p.step_num}",
                    f"{ezpz.get_timestamp()}",
                ]
            )
            trace_output = Path(outdir).joinpath(f"{fname}.json")
            logger.info(f"Saving torch profiler trace to: {trace_output.as_posix()}")
            p.export_chrome_trace(trace_output.as_posix())

        schedule = torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        return get_torch_profiler(
            rank=ezpz.get_rank(),
            schedule=schedule,
            on_trace_ready=trace_handler,
            rank_zero_only=rank_zero_only,
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            acc_events=acc_events,
        )

    if profiler_type == "pyinstrument":
        return get_context_manager(rank=ezpz.get_rank(), strict=strict)

    raise ValueError(
        f"Invalid profiling type: {profiler_type}. "
        "Must be one of ['torch', 'pyinstrument']"
    )


def get_context_manager(
    rank: Optional[int] = None,
    outdir: Optional[str] = None,
    strict: Optional[bool] = True,
) -> AbstractContextManager:
    """
    Returns a context manager for profiling code blocks using PyInstrument.

    Args:
        rank (Optional[int]): The rank of the process (default: None).
            If provided, the profiler will only run if rank is 0.
        outdir (Optional[str]): The output directory for saving profiles.
            Defaults to `ezpz.OUTPUTS_DIR`.
        strict (Optional[bool]): If True, the profiler will only run if
            "PYINSTRUMENT_PROFILER" is set in the environment.
            Defaults to True.

    Returns:
        AbstractContextManager: A context manager that starts and stops
            the PyInstrument profiler.
    """
    d = ezpz.OUTPUTS_DIR if outdir is None else outdir
    fp = Path(d)
    fp = fp.joinpath("ezpz", "pyinstrument_profiles")

    if strict:
        if os.environ.get("PYINSTRUMENT_PROFILER", None) is not None:
            return PyInstrumentProfiler(rank=rank, outdir=fp.as_posix())
        return nullcontext()
    if rank is None or rank == 0:
        return PyInstrumentProfiler(rank=rank, outdir=fp.as_posix())
    # if rank == 0:
    #     return PyInstrumentProfiler(rank=rank, outdir=outdir)
    return nullcontext()


def get_torch_profiler(
    rank: Optional[int] = None,
    schedule: Optional[Callable[[int], ProfilerAction]] = None,
    on_trace_ready: Optional[Callable] = None,
    rank_zero_only: bool = True,
    profile_memory: bool = False,
    record_shapes: bool = True,
    with_stack: bool = True,
    with_flops: bool = True,
    with_modules: bool = True,
    acc_events: bool = False,
):
    """
    A thin wrapper around `torch.profiler.profile` that:

    1. Supports automatic device detection {CPU, CUDA, XPU}
    2. Runs on rank 0 only (by default)
       - To run from _all_ ranks, set `rank_zero_only=False`

    Args:
        rank (Optional[int]): The rank of the process (default: None).
            If provided, the profiler will only run if rank is 0.
        schedule (Optional[Callable[[int], ProfilerAction]]): A callable
            that returns a `ProfilerAction` for the profiler schedule.
        on_trace_ready (Optional[Callable]): A callback function that is
            called when the trace is ready.
        rank_zero_only (bool): If True, the profiler will only run on rank 0.
            Defaults to True.
        profile_memory (bool): If True, memory profiling is enabled.
            Defaults to False.
        record_shapes (bool): If True, shapes of tensors are recorded.
            Defaults to True.
        with_stack (bool): If True, stack traces are recorded.
            Defaults to True.
        with_flops (bool): If True, FLOPs are recorded.
            Defaults to True.
        with_modules (bool): If True, module information is recorded.
            Defaults to True.
        acc_events (bool): If True, accumulated events are recorded.
            Defaults to False.
    Returns:
        torch.profiler.profile: A profiler context manager that can be used
            to profile code blocks.
    """
    if rank_zero_only and (rank is None or rank != 0):
        return nullcontext()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        activities.append(ProfilerActivity.XPU)
    return profile(
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_flops=with_flops,
        with_modules=with_modules,
        # acc_events=acc_events,
    )


class PyInstrumentProfiler:

    def __init__(
        self,
        rank: Optional[int] = None,
        outdir: Optional[str] = None,
    ):
        try:
            import pyinstrument  # pyright: ignore
        except ImportError:
            pyinstrument = None  # type:ignore
        if pyinstrument is None:
            self.profiler = None
            logger.critical("Unable to import 'pyinstrument', not running profiles!!")
            logger.error(
                "To run with 'pyinstrument',"
                "run: 'python3 -m pip install pyinstrument'"
            )
        else:
            self.profiler = (
                pyinstrument.Profiler()
                if (rank is None or (rank is not None and rank == 0))
                else None
            )
        self._start = time.perf_counter_ns()
        # outdir = os.getcwd() if outdir is None else outdir
        outdir = ezpz.OUTPUTS_DIR.as_posix() if outdir is None else outdir
        self.outdir = Path(outdir).joinpath("ezpz_pyinstrument_profiles")
        # self.outdir = Path(outdir) if outdir is None else Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

    def __enter__(self):
        self._start = time.perf_counter_ns()
        if self.profiler is not None:
            self.profiler.start()

    def __exit__(self, type, value, traceback):  # noqa
        dtpyinstrument = (time.perf_counter_ns() - self._start) / (10**9)
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler.print(color=True, timeline=True)  # , time='percent_of_total')
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            html_fp = Path(self.outdir).joinpath(f"pyinstrument-profile-{now}.html")
            text_fp = Path(self.outdir).joinpath(f"pyinstrument-profile-{now}.txt")
            logger.info(
                " ".join(
                    [
                        "Saving pyinstrument profile output to:",
                        f"{self.outdir.as_posix()}",
                    ]
                )
            )
            logger.info(
                " ".join(
                    [
                        "PyInstrument profile saved (as html) to: ",
                        f"{html_fp.as_posix()}",
                    ]
                )
            )
            logger.info(
                " ".join(
                    [
                        "PyInstrument profile saved (as text) to: ",
                        f"{text_fp.as_posix()}",
                    ]
                )
            )
            self.profiler.write_html(html_fp)
            ptext = self.profiler.output_text(unicode=True, color=True)
            with text_fp.open("w") as f:
                f.write(ptext)
            logger.info(
                " ".join(
                    [
                        "Finished with pyinstrument profiler.",
                        f"Took: {dtpyinstrument:.5f}s",
                    ]
                )
            )
