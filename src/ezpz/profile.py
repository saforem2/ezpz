"""
profile.py

Contains helper functions for using Profilers.
"""
import os
import time
import logging
import datetime

from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# try:
#     from scalene import scalene_profiler
#     HAS_SCALENE = True
# except Exception:
#     HAS_SCALENE = False

    # import datetime
    # now = datetime.datetime.now()
    # if fstr is None:
    #     return now.strftime('%Y-%m-%d-%H%M%S')
    # return now.strftime(fstr)


from contextlib import nullcontext, AbstractContextManager

def get_context_manager(
        rank: Optional[int] = None,
        outdir: Optional[str] = None,
        strict: Optional[bool] = True,
) -> AbstractContextManager:
    if strict:
        if os.environ.get("PYINSTRUMENT_PROFILER", None) is not None:
            return PyInstrumentProfiler(rank=rank, outdir=outdir)
        return nullcontext()
    if rank is None:
        return PyInstrumentProfiler(rank=rank, outdir=outdir)
    if rank == 0:
        return PyInstrumentProfiler(rank=rank, outdir=outdir)
    return nullcontext()

class PyInstrumentProfiler:
    def __init__(
            self,
            rank: Optional[int] = None,
            outdir: Optional[str] = None,
            # rank_zero_only: Optional[bool] = True,
    ):
        try:
            import pyinstrument
        except (ImportError, ModuleNotFoundError):
            pyinstrument = None
        assert pyinstrument is not None
        self.profiler = (
            pyinstrument.Profiler()
            if (
                rank is None or (rank is not None and rank == 0)
            )
            else None
        )
        self._start = time.perf_counter_ns()
        self.outdir = Path(os.getcwd()) if outdir is None else Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)
        # profiler = None
        # if (
        #         rank is None
        #         or (rank is not None and rank == 0)
        # ):
        #     profiler = pyinstrument.Profiler()
        # if rank is None:
        #     profiler = pyinstrument.Profiler()
        # elif rank is not None and rank == 0:
        #     profiler = pyinstrument.Profiler()
        #
        # if rank is not None and rank != 0:
        # profiler = None
        # if not rank_zero_only:
        #     profiler = pyinstrument.Profiler()
        # elif (rank_zero_only and rank == 0):
        #     profiler = pyinstrument.Profiler()
        # else:
        #
        # if rank_zero_only and rank == 0:
        #     profiler = pyinstrument.Profiler()
        # elif not rank_zero_only:
        #     profiler = pyinstrument.Profiler()
        #
        # if (
        #             (rank_zero_only and rank == 0)
        #             or (not rank_zero_only)
        # ):
        #     self.profiler = pyinstrument.Profiler()

    def __enter__(self):
        self._start = time.perf_counter_ns()
        if self.profiler is not None:
            self.profiler.start()

    def __exit__(self, type, value, traceback):
        dtpyinstrument = (time.perf_counter_ns() - self._start) / (10 ** 9)
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler.print(color=True)
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            html_fp = Path(self.outdir).joinpath(
                    f'pyinstrument-profile-{now}.html'
            )
            text_fp = Path(self.outdir).joinpath(
                f'pyinstrument-profile-{now}.txt'
            )
            log.info(
                ' '.join(
                    [
                        "Saving pyinstrument profile output to:",
                        f"{self.outdir.as_posix()}",
                    ]
                )
            )
            log.info(
                ' '.join(
                    [
                        'PyInstrument profile saved (as html) to: ',
                        f"{html_fp.relative_to(os.getcwd()).as_posix()}",
                    ]
                )
            )
            log.info(
                ' '.join(
                    [
                        'PyInstrument profile saved (as text) to: ',
                        f"{text_fp.relative_to(os.getcwd()).as_posix()}",
                    ]
                )
            )
            self.profiler.write_html(html_fp)
            ptext = self.profiler.output_text(unicode=True, color=True)
            with text_fp.open('w') as f:
                f.write(ptext)
            log.info(' '.join([
                'Finished with pyinstrument profiler.',
                f'Took: {dtpyinstrument:.5f}s',
            ]))



# class File(object):
#     def __init__(self, file_name, method):
#         self.file_obj = open(file_name, method)
#     def __enter__(self):
#         return self.file_obj
#     def __exit__(self, type, value, traceback):
#         self.file_obj.close()

# class ScaleneProfiler:
#     def __init__(self):
#         import scalene
#
#     def __enter__(self):
#         scalene_profiler.start()
#
#     def __exit__(self, type, value, traceback):
#         sca
