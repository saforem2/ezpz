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
    record_function,
    ProfilerActivity,
)

from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

from contextlib import nullcontext, AbstractContextManager


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


def get_pytorch_profiler(
    rank: Optional[int] = None,
    schedule: Optional[Callable[[int], ProfilerAction]] = None,
    on_trace_ready: Optional[Callable] = None,
    outdir: Optional[str] = None,
    rank_zero_only: bool = True,
    profile_memory: bool = False,
    record_shapes: bool = True,
    with_stack: bool = True,
    with_flops: bool = True,
    with_modules: bool = True,
    acc_events: bool = False,
):
    """
    Returns a PyTorch profiler context manager if profiling is enabled.
    """
    # from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    if torch.xpu.is_available():
        activities.append(ProfilerActivity.XPU)
    if not rank_zero_only or (rank_zero_only and rank == 0):
        return profile(
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            activities=activities,
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            acc_events=acc_events,
        )
    else:
        return nullcontext()


def get_torch_profiler_context_manager(
    rank: Optional[int] = None,
    outdir: Optional[str] = None,
    rank_zero_only: bool = True,
    profile_memory: bool = False,
    record_shapes: bool = True,
    with_stack: bool = True,
    with_flops: bool = True,
    with_modules: bool = True,
    acc_events: bool = False,
    # use_cuda: bool = True,
    # experimental_config: Optional[
    #     torch.profiler._ExperimentalConfig  # noqa
    # ] = None,  # noqa
):
    """
    Returns a context manager for profiling code blocks using PyTorch Profiler.

    Args:
        rank (Optional[int]): The rank of the process (default: None).
            If provided, the profiler will only run if rank is 0.
        outdir (Optional[str]): The output directory for saving profiles.
            Defaults to `ezpz.OUTPUTS_DIR`.
        rank_zero_only (Optional[bool]): If True, the profiler will only run
            if rank is 0. Defaults to True.
        **kwargs: Additional keyword arguments for the PyTorch Profiler.

    Returns:
        AbstractContextManager: A context manager that starts and stops
            the PyTorch Profiler.
    """
    # if os.environ.get("TORCH_PROFILER", None) is not None:
    if not rank_zero_only or (rank_zero_only and rank == 0):
        # self.profiler = self.build_profiler(
        #     profile_memory=profile_memory,
        #     record_shapes=record_shapes,
        #     with_stack=with_stack,
        #     with_flops=with_flops,
        #     with_modules=with_modules,
        #     experimental_config=experimental_config,
        #     acc_events=acc_events,
        #     # use_cuda=use_cuda,
        # )
        return PyTorchProfiler(
            rank=ezpz.get_rank(),
            outdir=outdir,
            rank_zero_only=rank_zero_only,
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            acc_events=acc_events,
            # use_cuda=use_cuda,
            # experimental_config=experimental_config,
        )
    return nullcontext()
    # return PyTorchProfiler(
    #     rank=rank, outdir=outdir, rank_zero_only=rank_zero_only, **kwargs
    # )


class PyTorchProfiler:
    def __init__(
        self,
        rank: int,
        outdir: Optional[str] = None,
        rank_zero_only: bool = True,
        profile_memory: bool = False,
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
        acc_events: bool = False,
        # use_cuda: bool = True,
        # experimental_config: Optional[
        #     torch.profiler._ExperimentalConfig  # noqa
        # ] = None,  # noqa
    ):
        """PyTorch Profiler context manager."""
        self._start = time.perf_counter_ns()
        self.rank = rank
        self.rank_zero_only = rank_zero_only
        self.outdir = Path(os.getcwd()) if outdir is None else Path(outdir)

        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        # self.experimental_config = experimental_config
        self.acc_events = acc_events
        # self.use_cuda = use_cuda
        # profile,
        # record_function,
        # ProfilerActivity,
        # if (
        #     profile is None
        #     or record_function is None
        #     or ProfilerActivity is None
        # ):
        #     self.profiler = None
        #     log.critical(
        #         "Unable to import 'torch.profiler', not running profiles!!"
        #     )
        #     log.error(
        #         "To run with 'torch.profiler',"
        #         "run: 'python3 -m pip install torch'"
        #     )
        # else:
        self.activities = [
            ProfilerActivity.CPU,
        ]
        if torch.cuda.is_available():
            self.activities.append(ProfilerActivity.CUDA)
        if torch.xpu.is_available():
            self.activities.append(ProfilerActivity.XPU)
        # if not rank_zero_only or (rank_zero_only and rank == 0):
        self.profiler = self.build_profiler(
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            acc_events=acc_events,
            # experimental_config=experimental_config,
            # use_cuda=use_cuda,
        )
        # if rank_zero_only:
        #     if rank == 0:
        #         self.profiler = self.build_profiler(
        #             profile_memory=profile_memory,
        #             record_shapes=record_shapes,
        #             with_stack=with_stack,
        #             with_flops=with_flops,
        #             with_modules=with_modules,
        #             experimental_config=experimental_config,
        #             acc_events=acc_events,
        #             use_cuda=use_cuda,
        #         )
        #     else:
        #         self.profiler = None
        # else:

    def build_profiler(
        self,
        profile_memory: bool = False,
        record_shapes: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        with_modules: bool = True,
        acc_events: bool = False,
        # use_cuda: bool = True,
        # experimental_config: Optional[
        #     torch.profiler._ExperimentalConfig  # noqa
        # ] = None,  # noqa
    ):
        return torch.profiler.profile(
            activities=self.activities,
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            acc_events=acc_events,
            # use_cuda=use_cuda,
            # experimental_config=experimental_config,
        )

    def __enter__(self):
        self._start = time.perf_counter_ns()
        if self.profiler is not None:
            self.profiler.__enter__()

    def __exit__(self, type, value, traceback):  # noqa
        dt = (time.perf_counter_ns() - self._start) / (10**9)
        log.info(
            " ".join(
                [
                    "Starting torch profiler...",
                    f"Rank: {self.rank}",
                    f"Output directory: {self.outdir.as_posix()}",
                ]
            )
        )
        if self.profiler is not None:
            self.profiler.__exit__(type, value, traceback)
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            fp = Path(self.outdir).joinpath(
                f"torch-profile-{self.rank}-{now}.json"
            )
            log.info(
                " ".join(
                    [
                        f"Saving torch profiler output to: {fp}",
                    ]
                )
            )
            self.profiler.export_chrome_trace(fp)
            log.info(
                " ".join(
                    [
                        "Finished with torch profiler.",
                        f"Trace saved to: {trace_output.as_posix()}",
                        f"Took: {dt:.5f}s",
                    ]
                )
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
            log.critical(
                "Unable to import 'pyinstrument', not running profiles!!"
            )
            log.error(
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
            self.profiler.print(
                color=True, timeline=True
            )  # , time='percent_of_total')
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            html_fp = Path(self.outdir).joinpath(
                f"pyinstrument-profile-{now}.html"
            )
            text_fp = Path(self.outdir).joinpath(
                f"pyinstrument-profile-{now}.txt"
            )
            log.info(
                " ".join(
                    [
                        "Saving pyinstrument profile output to:",
                        f"{self.outdir.as_posix()}",
                    ]
                )
            )
            log.info(
                " ".join(
                    [
                        "PyInstrument profile saved (as html) to: ",
                        f"{html_fp.as_posix()}",
                    ]
                )
            )
            log.info(
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
            log.info(
                " ".join(
                    [
                        "Finished with pyinstrument profiler.",
                        f"Took: {dtpyinstrument:.5f}s",
                    ]
                )
            )
