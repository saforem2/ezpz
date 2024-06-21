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

if __name__ == '__main__':
    main()
```

"""
import os
import time
import logging
import datetime

from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

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
    ):
        try:
            import pyinstrument  # pyright: ignore
        except (ImportError, ModuleNotFoundError):
            pyinstrument = None
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
                if (
                    rank is None or (rank is not None and rank == 0)
                )
                else None
            )
        self._start = time.perf_counter_ns()
        outdir = os.getcwd() if outdir is None else outdir
        self.outdir = Path(outdir).joinpath("ezpz_pyinstrument_profiles")
        self.outdir = Path(os.getcwd()) if outdir is None else Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

    def __enter__(self):
        self._start = time.perf_counter_ns()
        if self.profiler is not None:
            self.profiler.start()

    def __exit__(self, type, value, traceback):  # pyright: ignore
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
                        f"{html_fp.as_posix()}",
                    ]
                )
            )
            log.info(
                ' '.join(
                    [
                        'PyInstrument profile saved (as text) to: ',
                        f"{text_fp.as_posix()}",
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
