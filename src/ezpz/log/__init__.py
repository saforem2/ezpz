"""
ezpz/log/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional
import logging.config
from ezpz.log.config import STYLES

#
# # os.environ['PYTHONIOENCODING'] = 'utf-8'
#
#
# #
# # if __name__ == "__main__":  # pragma: no cover
# #     print_styles()
from ezpz.dist import get_rank, get_world_size

RANK = get_rank()
WORLD_SIZE = get_world_size()


def get_file_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    rank_zero_only: bool = True,
    fname: Optional[str] = None,
    # rich_stdout: bool = True,
) -> logging.Logger:
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    import logging

    fname = "output" if fname is None else fname
    log = logging.getLogger(name)
    if rank_zero_only:
        fh = logging.FileHandler(f"{fname}.log")
        if RANK == 0:
            log.setLevel(level)
            fh.setLevel(level)
        else:
            log.setLevel("CRITICAL")
            fh.setLevel("CRITICAL")
    else:
        fh = logging.FileHandler(f"{fname}-{RANK}.log")
        log.setLevel(level)
        fh.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_active_enrich_handlers(logger: logging.Logger) -> list:
    from ezpz.log.handler import RichHandler as EnrichHandler

    return [
        (idx, h)
        for idx, h in enumerate(logger.handlers)
        if isinstance(h, EnrichHandler)
    ]


def print_styles():
    import argparse

    parser = argparse.ArgumentParser()
    from rich.text import Text
    from ezpz.log.console import Console

    parser.add_argument("--html", action="store_true", help="Export as HTML table")
    args = parser.parse_args()
    html: bool = args.html
    from rich.table import Table

    console = Console(record=True, width=120) if html else Console()
    table = Table("Name", "Styling")
    for style_name, style in STYLES.items():
        table.add_row(Text(style_name, style=style), str(style))

    console.print(table)
    if html:
        outfile = "enrich_styles.html"
        print(f"Saving to `{outfile}`")
        with open(outfile, "w") as f:
            f.write(console.export_html(inline_styles=True))


def print_styles_alt(
    html: bool = False,
    txt: bool = False,
):
    from pathlib import Path
    from rich.text import Text
    from ezpz.log.style import DEFAULT_STYLES
    from rich.table import Table
    from ezpz.log.console import get_console

    console = get_console(record=html, width=150)
    table = Table("Name", "Styling")
    styles = DEFAULT_STYLES
    styles |= STYLES
    for style_name, style in styles.items():
        table.add_row(Text(style_name, style=style), str(style))
    console.print(table)
    if html:
        outfile = "ezpz_styles.html"
        print(f"Saving to `{outfile}`")
        with open(outfile, "w") as f:
            f.write(console.export_html(inline_styles=True))
    if txt:
        file1 = "ezpz_styles.txt"
        text = console.export_text()
        # with open(file1, "w") as file:
        with Path(file1).open("w") as file:
            file.write(text)


def get_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    rank_zero_only: bool = True,
) -> logging.Logger:
    import logging
    import logging.config
    from ezpz.configs import get_logging_config

    logging.config.dictConfig(get_logging_config())
    log = logging.getLogger(name if name is not None else __name__)
    if rank_zero_only:
        if RANK == 0:
            log.setLevel(level)
        else:
            log.setLevel("CRITICAL")
    else:
        log.setLevel(level)
    return log


def _get_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    markup: Optional[bool] = True,
    redirect: Optional[bool] = False,
    **kwargs,
) -> logging.Logger:
    from ezpz.log.console import get_console
    from ezpz.log.handler import RichHandler as EnrichHandler
    import logging

    log = logging.getLogger(name)
    log.setLevel(level)
    console = get_console(markup=markup, redirect=redirect, **kwargs)
    if console.is_jupyter:
        console.is_jupyter = False
    log.addHandler(
        EnrichHandler(
            omit_repeated_times=False,
            level=level,
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=markup,
            enable_link_path=False,
        )
    )
    if len(log.handlers) > 1 and all([i == log.handlers[0] for i in log.handlers]):
        log.handlers = [log.handlers[0]]
    enrich_handlers = get_active_enrich_handlers(log)
    found_handlers = 0
    if len(enrich_handlers) > 1:
        for h in log.handlers:
            if isinstance(h, EnrichHandler):
                if found_handlers > 1:
                    log.warning(
                        "More than one `EnrichHandler` in current logger: "
                        f"{log.handlers}"
                    )
                    log.removeHandler(h)
                found_handlers += 1
    if len(get_active_enrich_handlers(log)) > 1:
        log.warning(
            "More than one `EnrichHandler` in current logger: " f"{log.handlers}"
        )
    #     log.warning(f'Using {enrich_handlers[-1][1]}')
    #     log.removeHandler(log.handlers[enrich_handlers[-1][0]])
    #     # log.handlers = enrich_handlers[-1]
    # # assert (
    #     len() == 1
    # # )

    return log


def get_logger1(
    name: Optional[str] = None,
    level: str = "INFO",
    rank_zero_only: bool = True,
    **kwargs,
) -> logging.Logger:
    log = logging.getLogger(name)
    from ezpz.log.handler import RichHandler
    from ezpz.log.console import get_console
    from ezpz.log.console import is_interactive
    from ezpz.log.handler import RichHandler as EnrichHandler
    from rich.logging import RichHandler as OriginalRichHandler

    _ = (
        log.setLevel("CRITICAL")
        if (RANK == 0 and rank_zero_only)
        else log.setLevel(level)
    )
    # if rank_zero_only:
    #     if RANK != 0:
    #         log.setLevel('CRITICAL')
    #     else:
    #         log.setLevel(level)
    if RANK == 0:
        console = get_console(
            markup=True,  # (WORLD_SIZE == 1),
            redirect=(WORLD_SIZE > 1),
            **kwargs,
        )
        if console.is_jupyter:
            console.is_jupyter = False
        # log.propagate = True
        # log.handlers = []
        use_markup = WORLD_SIZE == 1 and not is_interactive()
        log.addHandler(
            OriginalRichHandler(
                omit_repeated_times=False,
                level=level,
                console=console,
                show_time=True,
                show_level=True,
                show_path=True,
                markup=use_markup,
                enable_link_path=use_markup,
            )
        )
        log.setLevel(level)
    # if (
    #         len(log.handlers) > 1
    #         and all([i == log.handlers[0] for i in log.handlers])
    # ):
    #     log.handlers = [log.handlers[0]]
    if len(log.handlers) > 1 and all([i == log.handlers[0] for i in log.handlers]):
        log.handlers = [log.handlers[0]]
    enrich_handlers = get_active_enrich_handlers(log)
    found_handlers = 0
    if len(enrich_handlers) > 1:
        for h in log.handlers:
            if isinstance(h, EnrichHandler):
                if found_handlers > 1:
                    log.warning(
                        "More than one `EnrichHandler` in current logger: "
                        f"{log.handlers}"
                    )
                    log.removeHandler(h)
                found_handlers += 1
    if len(get_active_enrich_handlers(log)) > 1:
        log.warning(
            "More than one `EnrichHandler` in current logger: " f"{log.handlers}"
        )
    return log
