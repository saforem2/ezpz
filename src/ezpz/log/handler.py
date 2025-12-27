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

# from ezpz.log.config import NO_COLOR
from ezpz.log.config import use_colored_logs
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        )  # type: ignore

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
        funcName = getattr(record, "funcName", None)
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
        default_time_fmt = "%Y-%m-%d %H:%M:%S,%f"
        # default_time_fmt = "%Y-%m-%d %H:%M:%S"  # .%f'
        time_format = time_format if time_format else default_time_fmt
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
        time_format: str = "%Y-%m-%d %H:%M:%S.%f",
        link_path: Optional[bool] = False,
    ) -> None:
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.link_path = link_path
        self._last_time: Optional[str] = None
        self.colorized = use_colored_logs()
        self.styles = get_styles()

    def __call__(  # pylint: disable=too-many-arguments
        self,
        console: Console,  # type: ignore
        renderables: Iterable[ConsoleRenderable],
        log_time: Optional[datetime] = None,
        time_format: str = "%Y-%m-%d %H:%M:%S.%f",
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
        funcName: Optional[str] = None,
    ) -> Text:
        result = Text()
        if self.show_time:
            log_time = datetime.now() if log_time is None else log_time
            log_time_display = log_time.strftime(
                time_format or self.time_format
            )
            d, t = log_time_display.split(" ")
            result += Text("[", style=self.styles.get("log.brace", ""))
            result += Text(f"{d} ")
            result += Text(t)
            result += Text("]", style=self.styles.get("log.brace", ""))
            self._last_time = log_time_display
        if self.show_level:
            if isinstance(level, Text):
                lstr = level.plain.rstrip(" ")[0]
                if self.colorized:
                    style = level.spans[0].style
                else:
                    style = Style.null()
                level.spans = [Span(0, len(lstr), style)]
                ltext = Text("[", style=self.styles.get("log.brace", ""))
                ltext.append(Text(f"{lstr}", style=style))
                ltext.append(Text("]", style=self.styles.get("log.brace", "")))
            elif isinstance(level, str):
                lstr = level.rstrip(" ")[0]
                style = (
                    f"logging.level.{str(lstr)}"
                    if self.colorized
                    else Style.null()
                )
                ltext = Text("[", style=self.styles.get("log.brace", ""))
                ltext = Text(f"{lstr}", style=style)
                ltext.append(Text("]", style=self.styles.get("log.brace", "")))
            result += ltext
        if self.show_path and path:
            path_text = Text("[", style=self.styles.get("log.brace", ""))
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
                Text(
                    f"{parent}", style="log.parent"
                ),  # self.styles.get('log.pa', '')),
                Text("/"),
                Text(f"{module}", style="log.path"),
            ]
            if line_no:
                text_arr += [
                    Text(":", style=self.styles.get("log.colon", "")),
                    Text(
                        f"{line_no}",
                        style=self.styles.get("log.linenumber", ""),
                    ),
                ]
            if fn is not None:
                text_arr += [
                    Text(":", style="log.colon"),
                    Text(
                        f"{fn}",
                        style="repr.function",  # self.styles.get('repr.inspect.def', 'json.key'),
                    ),
                ]
            path_text.append(Text.join(Text(""), text_arr))
            path_text.append("]", style=self.styles.get("log.brace", ""))

            result += path_text
        result += Text(" ", style=self.styles.get("repr.dash", ""))
        for elem in renderables:
            if ANSI_ESCAPE_PATTERN.search(str(elem)):
                # If the element is already ANSI formatted, append it directly
                result += Text.from_ansi(str(elem))
            else:
                result += elem
        return result
