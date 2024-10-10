"""
Implements enriched RichHandler

Based on:

https://github.com/willmcgugan/rich/blob/master/rich/_log_render.py
"""
from datetime import datetime
from typing import Any, Iterable, Optional

from rich.style import Style
# from rich.style import Style
from rich.logging import RichHandler as OriginalRichHandler
from rich.text import Text, TextType, Span

from ezpz.log.config import NO_COLOR
from ezpz.log.console import get_console, Console
from rich.console import ConsoleRenderable

# from datetime import datetime
# from typing import Any, Iterable, Optional

# from rich.logging import RichHandler as OriginalRichHandler
# from rich.text import Text, TextType, Span
COLOR = (not NO_COLOR)
if not COLOR:
    STYLES = {}
else:
    from ezpz.log.config import STYLES



class RichHandler(OriginalRichHandler):
    """Enriched handler that does not wrap."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if 'console' not in kwargs:
            kwargs['console'] = get_console(
                redirect=False,
                width=9999,
                markup=COLOR
            )
        super().__init__(*args, **kwargs)
        # RichHandler constructor does not allow custom renderer
        # https://github.com/willmcgugan/rich/issues/438
        self._log_render = FluidLogRender(
            show_time=kwargs.get("show_time", True),
            show_level=kwargs.get("show_level", True),
            show_path=kwargs.get("show_path", True),
        )  # type: ignore



# class RichHandler(OriginalRichHandler):
#     """Enriched handler that does not wrap."""
#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         # RichHandler constructor does not allow custom renderer
#         # https://github.com/willmcgugan/rich/issues/438
#         self._log_render = FluidLogRender(
#             show_time=kwargs.get("show_time", False),
#             show_level=kwargs.get("show_level", True),
#             show_path=kwargs.get("show_path", False),
#         )  # type: ignore


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

    def __call__(  # pylint: disable=too-many-arguments
            self,
            console: Console,  # type: ignore
            renderables: Iterable[ConsoleRenderable],
            log_time: Optional[datetime] = None,
            time_format: str = '%Y-%m-%d %H:%M:%S.%f',
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
            d, t = log_time_display.split(' ')
            result += Text("[", style=STYLES.get('log.brace', ''))
            result += Text(f'{d} ', style=STYLES.get('logging.date', ''))
            result += Text(t, style=STYLES.get('logging.time', ''))
            result += Text("]", style=STYLES.get('log.brace', ''))
            # result += Text(log_time_display, style=STYLES['logging.time'])
            self._last_time = log_time_display
        if self.show_level:
            if isinstance(level, Text):
                lstr = level.plain.rstrip(' ')
                if COLOR:
                    style = level.spans[0].style
                else:
                    style = Style.null()
                level.spans = [Span(0, len(lstr), style)]
                ltext = Text('[', style=STYLES.get('log.brace', ''))
                ltext.append(Text(f'{lstr}', style=style))
                ltext.append(Text(']', style=STYLES.get('log.brace', '')))
                # ltext = Text(f'[{lstr}]', style=style)
            elif isinstance(level, str):
                lstr = level.rstrip(' ')
                style = f"logging.level.{str(lstr)}" if COLOR else Style.null()
                ltext = Text('[', style=STYLES.get('log.brace', ''))
                ltext = Text(f"{lstr}", style=style)  # f"logging.level.{str(lstr)}")
                ltext.append(Text(']', style=STYLES.get('log.brace', '')))
            else:
                raise TypeError('Unexpected type for level')
            result += ltext
        if self.show_path and path:
            path_text = Text("[", style=STYLES.get('log.brace', ''))
            # path_text.append( /)
            path_text.append(
                # path.rstrip('.py'),
                path,
                style=STYLES.get('log.path', ''),
                # style=STYLES.get('log.path', Style(color='black')),
                # style=(
                #     f"link file://{link_path}" + " underline"
                #     if link_path else ""
                #     # + STYLES["repr.url"]
                # )
            )
            if line_no:
                path_text.append(Text(":", style=STYLES.get('log.colon', '')))
                path_text.append(
                    f"{line_no}",
                    style=STYLES.get('log.linenumber', ''),
                    # style=(
                    #     f"link file://{link_path}#{line_no}"
                    #     if link_path else ""
                    # ),
                )
            path_text.append("]", style=STYLES.get('log.brace', ''))
            result += path_text
        result += Text(' - ', style=STYLES.get('repr.dash', ''))
        for elem in renderables:
            result += elem
        return result
