"""
Implements enriched RichHandler

Based on:

https://github.com/willmcgugan/rich/blob/master/rich/_log_render.py
"""

import re
from datetime import datetime
from logging import LogRecord
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.console import ConsoleRenderable
from rich.logging import RichHandler as OriginalRichHandler
from rich.style import Style
from rich.text import Span, Text, TextType

from ezpz.distributed import get_rank
from ezpz.log.config import (
    EZPZ_LOG_DAY_TIME_SEPARATOR,
    EZPZ_LOG_SHOW_LEVEL,
    EZPZ_LOG_SHOW_PATH,
    EZPZ_LOG_SHOW_RANK,
    EZPZ_LOG_SHOW_TIME,
    EZPZ_LOG_TIME_FORMAT,
    EZPZ_LOG_USE_BRACKETS,
    EZPZ_LOG_USE_COLORED_PREFIX,
    EZPZ_LOG_USE_SINGLE_BRACKET,
    use_colored_logs,
)
from ezpz.log.console import Console, get_console

# Define the regex pattern for ANSI escape codes
ANSI_ESCAPE_PATTERN = re.compile(r"\x1B\[[0-9;]*m")


def get_styles(colorized: bool = True) -> dict:
    styles = {}
    if colorized and use_colored_logs():
        from rich.default_styles import DEFAULT_STYLES

        styles |= {k: v for k, v in DEFAULT_STYLES.items()}

        from ezpz.log.config import STYLES

        styles |= {k: v for k, v in STYLES.items()}
    return styles


# if not COLOR:
# else:
#     import ezpz.log.config as logconfig
# from ezpz.log.config import STYLES as logstyle
#
#
# STYLES = logconfig.STYLES


class RichHandler(OriginalRichHandler):
    """Enriched handler that does not wrap."""

    def __init__(
        self, rank: Optional[int | str] = None, *args: Any, **kwargs: Any
    ) -> None:
        if "console" not in kwargs:
            console = get_console(
                redirect=False,
                width=9999,
                markup=use_colored_logs(),
                soft_wrap=False,
            )
            kwargs["console"] = console
            self.__console = console
        else:
            self.__console = kwargs["console"]
        super().__init__(*args, **kwargs)
        # RichHandler constructor does not allow custom renderer
        # https://github.com/willmcgugan/rich/issues/438
        self._log_render = FluidLogRender(
            show_time=kwargs.get("show_time", True),
            show_level=kwargs.get("show_level", True),
            show_path=kwargs.get("show_path", True),
            link_path=kwargs.get("enable_link_path", False),
            rank=rank,
        )

    def render(
        self,
        *,
        record: LogRecord,
        traceback: Optional[Any],
        message_renderable: "ConsoleRenderable",
    ) -> "ConsoleRenderable":
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.

        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        fp = getattr(record, "pathname", None)
        parent = Path(fp).parent.as_posix().split("/")[-1] if fp else None
        module = getattr(record, "module", None)
        name = getattr(record, "name", None)
        # funcName = getattr(record, "funcName", None)
        parr = []
        if fp is not None:
            fp = Path(fp)
            parent = fp.parent.as_posix().split("/")[-1]
            parr.append(parent)
        if module is not None:
            parr.append(module)

        if (
            name is not None
            and parent is not None
            and f"{parent}.{module}" != name
        ):
            parr.append(name)
        pstr = "/".join([parr[0], ".".join(parr[1:])])
        level = self.get_level_text(record)
        time_format = (
            None if self.formatter is None else self.formatter.datefmt
        )
        # default_time_fmt = "%Y%m%d@%H:%M:%S,%f"
        # default_time_fmt = "%Y-%m-%d %H:%M:%S"  # .%f'
        time_format = time_format if time_format else EZPZ_LOG_TIME_FORMAT
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.__console,
            (
                [message_renderable]
                if not traceback
                else [message_renderable, traceback]
            ),
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=pstr,  # getattr(record, "pathname", None),
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
            funcName=record.funcName,
        )
        return log_renderable


class FluidLogRender:  # pylint: disable=too-few-public-methods
    """Renders log by not using columns and avoiding any wrapping."""

    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = True,
        show_path: bool = True,
        time_format: str | None = None,
        link_path: Optional[bool] = False,
        rank: Optional[int | str] = None,
    ) -> None:
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.link_path = link_path
        self.rank = rank
        self._last_time: Optional[str] = None
        self.colorized = use_colored_logs()
        self.styles = get_styles(colorized=self.colorized)
        self.colored_prefix = EZPZ_LOG_USE_COLORED_PREFIX

    def _ps(self, key: str) -> str:
        """Return the prefix style for *key*, or empty if prefix color is off."""
        return self.styles.get(key, "") if self.colored_prefix else ""

    # Helpers for bracket modes
    def _open(self, result: Text) -> None:
        result.append(Text("[", style=self._ps("log.brace")))

    def _close(self, result: Text) -> None:
        result.append(Text("]", style=self._ps("log.brace")))

    def __call__(  # pylint: disable=too-many-arguments
        self,
        console: Console,  # type: ignore
        renderables: Iterable[ConsoleRenderable],
        log_time: Optional[datetime] = None,
        time_format: str | None = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
        funcName: Optional[str] = None,
    ) -> Text:
        # Resolve bracket mode once:
        #   "full"   -> each component wrapped: [time][level][path]
        #   "single" -> one wrapper:            [time level path]
        #   "none"   -> no brackets:            time level path -- msg
        if EZPZ_LOG_USE_BRACKETS:
            bmode = "full"
        elif EZPZ_LOG_USE_SINGLE_BRACKET:
            bmode = "single"
        else:
            bmode = "none"

        result = Text()
        has_prefix = False

        # -- Rank --
        if self.rank is not None or EZPZ_LOG_SHOW_RANK:
            rank_val = self.rank if self.rank is not None else get_rank()
            self._open(result)
            result += Text(f"{rank_val}", style=self._ps("log.rank"))
            self._close(result)
            has_prefix = True

        # Open single bracket (wraps everything until the final close)
        if bmode == "single":
            self._open(result)

        # -- Time --
        if self.show_time and EZPZ_LOG_SHOW_TIME:
            log_time = datetime.now() if log_time is None else log_time
            log_time_display = log_time.strftime(
                time_format or self.time_format or EZPZ_LOG_TIME_FORMAT
            )
            if bmode == "full":
                self._open(result)
            if EZPZ_LOG_DAY_TIME_SEPARATOR in log_time_display:
                d, t = log_time_display.split(EZPZ_LOG_DAY_TIME_SEPARATOR)
                result += Text(d, style=self._ps("log.day_color"))
                result += Text(
                    EZPZ_LOG_DAY_TIME_SEPARATOR,
                    style=self._ps("repr.colon"),
                )
                result += Text(t, style=self._ps("log.time_color"))
            else:
                result += Text(
                    log_time_display,
                    style=self._ps("log.time_color"),
                )
            if bmode == "full":
                self._close(result)
            else:
                result += Text(" ")
            self._last_time = log_time_display
            has_prefix = True

        # -- Level --
        if self.show_level and EZPZ_LOG_SHOW_LEVEL:
            if isinstance(level, Text):
                lstr = level.plain.rstrip(" ")[0]
                lstyle = (
                    level.spans[0].style
                    if self.colorized and self.colored_prefix and level.spans
                    else Style.null()
                )
                level.spans = [Span(0, len(lstr), lstyle)]
            elif isinstance(level, str):
                lstr = level.rstrip(" ")[0]
                lstyle = (
                    f"logging.level.{lstr}"
                    if self.colorized and self.colored_prefix
                    else Style.null()
                )
            else:
                lstr = str(level)
                lstyle = Style.null()
            show_path = (self.show_path and EZPZ_LOG_SHOW_PATH) and path
            if bmode == "full":
                ltext = (
                    Text("[", style=self._ps("log.brace"))
                    + Text(lstr, style=lstyle)
                    + Text("]", style=self._ps("log.brace"))
                )
            elif show_path:
                ltext = Text(lstr, style=lstyle) + Text(" ")
            else:
                ltext = Text(lstr, style=lstyle)
            result += ltext
            has_prefix = True

        # -- Path --
        if (self.show_path and EZPZ_LOG_SHOW_PATH) and path:
            path_text = Text()
            if bmode == "full":
                self._open(path_text)
            text_arr = []
            parent, remainder = path.split("/")
            if "." in remainder:
                module, *fn = remainder.split(".")
                fn = ".".join(fn)
            else:
                module = remainder
                fn = None
            if funcName is not None:
                fn = funcName
            text_arr += [
                Text(parent, style=self._ps("log.parent")),
                Text("/"),
                Text(module, style=self._ps("log.path")),
            ]
            if line_no:
                text_arr += [
                    Text(":", style=self._ps("repr.colon")),
                    Text(
                        f"{line_no}",
                        style=self._ps("log.linenumber"),
                    ),
                ]
            if fn is not None:
                text_arr += [
                    Text(":"),
                    Text(fn, style=self._ps("repr.function")),
                ]
            path_text.append(Text.join(Text(""), text_arr))
            if bmode == "full":
                self._close(path_text)
            result += path_text
            has_prefix = True

        # -- Close wrapper / separator before message --
        if has_prefix:
            if bmode == "single":
                self._close(result)
                result += Text(" ")
            elif bmode == "none":
                result += Text(" -- ", style=self._ps("repr.dash"))
            else:
                result += Text(" ")

        # -- Message --
        for elem in renderables:
            if ANSI_ESCAPE_PATTERN.search(str(elem)):
                result += Text.from_ansi(str(elem))
            else:
                result += elem

        return result
