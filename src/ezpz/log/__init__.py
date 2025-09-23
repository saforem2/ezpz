"""
ezpz/log/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function

import logging
import logging.config
import os
from typing import Optional

from ezpz.configs import get_logging_config
# from ezpz.dist import get_rank, get_world_size
from ezpz.log.config import STYLES, use_colored_logs
from ezpz.log.console import (Console, get_console, get_theme, get_width,
                              is_interactive, should_do_markup, to_bool)
from ezpz.log.handler import FluidLogRender, RichHandler
from ezpz.log.style import (BEAT_TIME, COLORS, CustomLogging, add_columns,
                            build_layout, flatten_dict, make_layout,
                            nested_dict_to_df, print_config, printarr)

__all__ = [
    "BEAT_TIME",
    "COLORS",
    "CustomLogging",
    "Console",
    "FluidLogRender",
    "RichHandler",
    "STYLES",
    "add_columns",
    "build_layout",
    "flatten_dict",
    "get_console",
    "get_theme",
    "get_width",
    "is_interactive",
    "make_layout",
    "nested_dict_to_df",
    "print_config",
    "printarr",
    "should_do_markup",
    "to_bool",
    "use_colored_logs",
]

#
# # os.environ['PYTHONIOENCODING'] = 'utf-8'
#
#
# #
# # if __name__ == "__main__":  # pragma: no cover
# #     print_styles()
# from ezpz.dist import get_rank, get_world_size
#
# RANK = get_rank()
# WORLD_SIZE = get_world_size()


def get_file_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    rank_zero_only: bool = True,
    fname: Optional[str] = None,
    # rich_stdout: bool = True,
) -> logging.Logger:
    """Create a file-backed logger, optionally emitting only on rank zero."""
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    import logging

    from ezpz.dist import get_rank

    fname = "output" if fname is None else fname
    log = logging.getLogger(name)
    if rank_zero_only:
        fh = logging.FileHandler(f"{fname}.log")
        if get_rank() == 0:
            log.setLevel(level)
            fh.setLevel(level)
        else:
            log.setLevel("CRITICAL")
            fh.setLevel("CRITICAL")
    else:
        fh = logging.FileHandler(f"{fname}-{get_rank()}.log")
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
    """Return ``(index, handler)`` pairs for active ``RichHandler`` instances."""
    from ezpz.log.handler import RichHandler as EnrichHandler

    return [
        (idx, h)
        for idx, h in enumerate(logger.handlers)
        if isinstance(h, EnrichHandler)
    ]


def print_styles():
    """Print the configured logging styles (optionally exporting to HTML)."""
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
    """Variant of :func:`print_styles` with HTML and plain-text exports."""
    from pathlib import Path

    from rich.table import Table
    from rich.text import Text

    from ezpz.log.console import get_console
    from ezpz.log.style import DEFAULT_STYLES

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
    level: Optional[str] = None,
    rank_zero_only: bool = True,
    rank: Optional[int | str] = None,
    colored_logs: Optional[bool] = True,
) -> logging.Logger:
    """Return a logger initialised with the project's logging configuration."""
    if rank is None and rank_zero_only:
        from ezpz.dist import get_rank

        rank = get_rank()
    assert rank is not None
    # if is_interactive():
    #     return get_rich_logger(name=name, level=level)
    ezpz_log_level = (
        os.environ.get("EZPZ_LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO"))
        if level is None
        else level
    )
    # level = os.environ.get("LOG_LEVEL", "INFO") if level is None else level
    # if colored_logs and use_colored_logs():
    if not colored_logs:
        os.environ["NO_COLOR"] = "1"
    logging.config.dictConfig(get_logging_config())
    logger = logging.getLogger(name if name is not None else __name__)
    if rank_zero_only:
        if int(rank) == 0:
            logger.setLevel(ezpz_log_level)
        else:
            logger.setLevel("CRITICAL")
    else:
        logger.setLevel(ezpz_log_level)
    return logger


# def _get_logger(
#     name: Optional[str] = None,
#     level: str = "INFO",
#     markup: Optional[bool] = True,
#     redirect: Optional[bool] = False,
#     **kwargs,
# ) -> logging.Logger:
#     from ezpz.log.console import get_console
#     from ezpz.log.handler import RichHandler as EnrichHandler
#     import logging
#
#     log = logging.getLogger(name)
#     log.setLevel(level)
#     console = get_console(markup=markup, redirect=redirect, **kwargs)
#     if console.is_jupyter:
#         console.is_jupyter = False
#     log.addHandler(
#         EnrichHandler(
#             omit_repeated_times=False,
#             level=level,
#             console=console,
#             show_time=True,
#             show_level=True,
#             show_path=True,
#             markup=markup,
#             enable_link_path=False,
#         )
#     )
#     if len(log.handlers) > 1 and all(
#         [i == log.handlers[0] for i in log.handlers]
#     ):
#         log.handlers = [log.handlers[0]]
#     enrich_handlers = get_active_enrich_handlers(log)
#     found_handlers = 0
#     if len(enrich_handlers) > 1:
#         for h in log.handlers:
#             if isinstance(h, EnrichHandler):
#                 if found_handlers > 1:
#                     log.warning(
#                         "More than one `EnrichHandler` in current logger: "
#                         f"{log.handlers}"
#                     )
#                     log.removeHandler(h)
#                 found_handlers += 1
#     if len(get_active_enrich_handlers(log)) > 1:
#         log.warning(
#             f"More than one `EnrichHandler` in current logger: {log.handlers}"
#         )
#     #     log.warning(f'Using {enrich_handlers[-1][1]}')
#     #     log.removeHandler(log.handlers[enrich_handlers[-1][0]])
#     #     # log.handlers = enrich_handlers[-1]
#     # # assert (
#     #     len() == 1
#     # # )
#
#     return log


def get_console_from_logger(logger: logging.Logger) -> Console:
    """Return the ``Console`` attached to *logger* or synthesise a new one."""
    from ezpz.log.handler import RichHandler as EnrichHandler

    for handler in logger.handlers:
        if isinstance(handler, (RichHandler, EnrichHandler)):
            return handler.console  # type: ignore
    from ezpz.log.console import get_console

    return get_console()


def get_rich_logger(
    name: Optional[str] = None, level: Optional[str] = None
) -> logging.Logger:
    """Return a logger backed by a single :class:`RichHandler`."""
    from ezpz.dist import get_world_size
    from ezpz.log.handler import RichHandler

    level = "INFO" if level is None else level
    # log: logging.Logger = get_logger(name=name, level=level)
    log = logging.getLogger(name)
    log.handlers = []
    console = get_console(
        markup=True,
        redirect=(get_world_size() > 1),
    )
    handler = RichHandler(
        level,
        rich_tracebacks=False,
        console=console,
        show_path=False,
        enable_link_path=False,
    )
    log.handlers = [handler]
    log.setLevel(level)
    return log


# def _get_file_logger_old(
#         name: Optional[str] = None,
#         level: str = 'INFO',
#         rank_zero_only: bool = True,
#         fname: Optional[str] = None,
# ) -> logging.Logger:
#     import logging
#     fname = 'output' if fname is None else fname
#     log = logging.getLogger(name)
#     fh = logging.FileHandler(f"{fname}.log")
#     log.setLevel(level)
#     fh.setLevel(level)
#     # create formatter and add it to the handlers
#     formatter = logging.Formatter(
#         "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
#     )
#     fh.setFormatter(formatter)
#     log.addHandler(fh)
#     return log


def get_enrich_logging_config_as_yaml(name: str = "enrich", level: str = "INFO") -> str:
    """Render the Enrich logging YAML snippet with the requested name/level."""
    return rf"""
    ---
    # version: 1
    handlers:
      {name}:
        (): ezpz.log.handler.RichHandler
        show_time: true
        show_level: true
        enable_link_path: false
        level: {level.upper()}
    root:
      handlers: [{name}]
    disable_existing_loggers: false
    ...
    """


def get_logger_new(
    name: str,
    level: str = "INFO",
):
    """Return a logger configured solely via the Enrich YAML template."""
    import yaml

    config = yaml.safe_load(
        get_enrich_logging_config_as_yaml(name=name, level=level),
    )
    logging.config.dictConfig(config)
    log = logging.getLogger(name=name)
    log.setLevel(level)
    return log


def get_logger1(
    name: Optional[str] = None,
    level: str = "INFO",
    rank_zero_only: bool = True,
    **kwargs,
) -> logging.Logger:
    """Legacy helper retained for compatibility; prefer :func:`get_logger`."""
    from ezpz.dist import get_rank, get_world_size

    log = logging.getLogger(name)
    # from ezpz.log.handler import RichHandler
    from rich.logging import RichHandler as OriginalRichHandler

    from ezpz.log.console import get_console, is_interactive
    from ezpz.log.handler import RichHandler as EnrichHandler

    _ = (
        log.setLevel("CRITICAL")
        if (get_rank() == 0 and rank_zero_only)
        else log.setLevel(level)
    )
    # if rank_zero_only:
    #     if RANK != 0:
    #         log.setLevel('CRITICAL')
    #     else:
    #         log.setLevel(level)
    if get_rank() == 0:
        console = get_console(
            markup=True,  # (WORLD_SIZE == 1),
            redirect=(get_world_size() > 1),
            **kwargs,
        )
        # if console.is_jupyter:
        #     console.is_jupyter = False
        # log.propagate = True
        # log.handlers = []
        use_markup = get_world_size() == 1 and not is_interactive()
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
        log.warning(f"More than one `EnrichHandler` in current logger: {log.handlers}")
    return log
