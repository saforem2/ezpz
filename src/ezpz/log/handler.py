"""
Implements enriched RichHandler

Based on:

https://github.com/willmcgugan/rich/blob/master/rich/_log_render.py
"""

from datetime import datetime
from typing import Any, Iterable, Optional

from pathlib import Path
from rich.style import Style

from rich.logging import RichHandler as OriginalRichHandler
from rich.text import Text, TextType, Span

# from ezpz.log.config import NO_COLOR
from ezpz.log.config import use_colored_logs
from ezpz.log.console import get_console, Console
from rich.console import ConsoleRenderable

from logging import LogRecord


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
                redirect=False, width=9999, markup=use_colored_logs()
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
        fp = Path(record.pathname)
        parent = fp.parent.as_posix().split("/")[-1]
        module = getattr(record, "module", None)
        name = getattr(record, "name", None)

        parr = [parent]
        if module is not None:
            parr.append(module)
        if name is not None and f"{parent}.{module}" != name:
            parr.append(name)
        pstr = "/".join([parr[0], ".".join(parr[1:])])

        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        # default_time_fmt = '%Y-%m-%d %H:%M:%S.%f'
        default_time_fmt = "%Y-%m-%d %H:%M:%S"  # .%f'
        time_format = time_format if time_format else default_time_fmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.__console,
            [message_renderable]
            if not traceback
            else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=pstr,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
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
    ) -> Text:
        result = Text()
        if self.show_time:
            log_time = datetime.now() if log_time is None else log_time
            log_time_display = log_time.strftime(
                time_format or self.time_format
            )
            d, t = log_time_display.split(" ")
            result += Text("[", style=self.styles.get("log.brace", ""))
            result += Text(f"{d} ", style=self.styles.get("logging.date", ""))
            result += Text(t, style=self.styles.get("logging.time", ""))
            result += Text("]", style=self.styles.get("log.brace", ""))
            # result += Text(log_time_display, style=self.styles['logging.time'])
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
                # ltext = Text(f'[{lstr}]', style=style)
            elif isinstance(level, str):
                lstr = level.rstrip(" ")[0]
                style = (
                    f"logging.level.{str(lstr)}"
                    if self.colorized
                    else Style.null()
                )
                ltext = Text("[", style=self.styles.get("log.brace", ""))
                ltext = Text(
                    f"{lstr}", style=style
                )  # f"logging.level.{str(lstr)}")
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
                # text_arr.append(Text(':', style=self.styles.get('log.colon', '')))
                # text_arr.append(
                #     Text(f'{line_no}', style=self.styles.get('log.linenumber', '')),
                # )
            if fn is not None:
                text_arr += [
                    Text(":", style="log.colon"),
                    Text(
                        f"{fn}",
                        style="repr.function",  # self.styles.get('repr.inspect.def', 'json.key'),
                    ),
                ]
            path_text.append(Text.join(Text(""), text_arr))

            # for t in text_arr:
            #     path_text.append(t)
            # path_text.append(
            #     Text(f'{parent}', style='cyan'),
            # )
            # path_text.append(Text('/'))
            # path_text.append(
            #     remainder,
            #     style=STYLES.get('log.path', ''),
            # )
            # if fn is not None:
            #     path_text += [Text('.'), Text(f'{fn}', style='cyan')]

            path_text.append("]", style=self.styles.get("log.brace", ""))
            result += path_text
        result += Text(" ", style=self.styles.get("repr.dash", ""))
        for elem in renderables:
            result += elem
        return result
