"""
src/ezpz/logging/console.py

Module that helps integrating with rich library.
"""

import os
import sys
from typing import Any, TextIO

from rich.ansi import AnsiDecoder
import rich.console as rich_console
from rich.file_proxy import FileProxy
from rich.theme import Theme
from ezpz.log.config import STYLES
# from enrich.logging import RichHandler


def get_theme():
    return Theme(STYLES)


def is_interactive() -> bool:
    # from IPython.core.getipython import get_ipython
    # # from IPython import get_ipython
    # eval = os.environ.get('INTERACTIVE', None) is not None
    # bval = get_ipython() is not None
    # return (eval or bval)
    return hasattr(sys, "ps1")


def get_width():
    import shutil

    width = os.environ.get("COLUMNS", os.environ.get("WIDTH", 255))
    if width is not None:
        return int(width)
    size = shutil.get_terminal_size()
    os.environ["COLUMNS"] = str(size.columns)
    return size.columns


class Console(rich_console.Console):
    """Extends rich Console class."""

    def __init__(
        self, *args: str, redirect: bool = True, **kwargs: Any
    ) -> None:
        """
        enrich console does soft-wrapping by default and this diverge from
        original rich console which does not, creating hard-wraps instead.
        """
        self.redirect = redirect

        if "soft_wrap" not in kwargs:
            kwargs["soft_wrap"] = True

        if "theme" not in kwargs:
            kwargs["theme"] = get_theme()

        if "markup" not in kwargs:
            kwargs["markup"] = True

        if "width" not in kwargs:
            kwargs["width"] = 55510

        # Unless user already mentioning terminal preference, we use our
        # heuristic to make an informed decision.
        if "force_terminal" not in kwargs:
            kwargs["force_terminal"] = should_do_markup(
                stream=kwargs.get("file", sys.stdout)
            )

        super().__init__(*args, **kwargs)
        self.extended = True

        if self.redirect:
            if not hasattr(sys.stdout, "rich_proxied_file"):
                sys.stdout = FileProxy(self, sys.stdout)  # type: ignore
            if not hasattr(sys.stderr, "rich_proxied_file"):
                sys.stderr = FileProxy(self, sys.stderr)  # type: ignore

    # https://github.com/python/mypy/issues/4441
    def print(self, *args, **kwargs) -> None:  # type: ignore
        """Print override that respects user soft_wrap preference."""
        # Currently rich is unable to render ANSI escapes with print so if
        # we detect their presence, we decode them.
        # https://github.com/willmcgugan/rich/discussions/404
        if args and isinstance(args[0], str) and "\033" in args[0]:
            text = format(*args) + "\n"
            decoder = AnsiDecoder()
            args = list(decoder.decode(text))  # type: ignore
        super().print(*args, **kwargs)


# Based on Ansible implementation
def to_bool(value: Any) -> bool:
    """Return a bool for the arg."""
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        value = value.lower()
    if value in ("yes", "on", "1", "true", 1):
        return True
    return False


def should_do_markup(stream: TextIO = sys.stdout) -> bool:
    """Decide about use of ANSI colors."""
    py_colors = None

    # https://xkcd.com/927/
    for env_var in [
        "PY_COLORS",
        "CLICOLOR",
        "FORCE_COLOR",
        "ANSIBLE_FORCE_COLOR",
    ]:
        value = os.environ.get(env_var, None)
        if value is not None:
            py_colors = to_bool(value)
            break

    # If deliverately disabled colors
    if os.environ.get("NO_COLOR", None):
        return False

    # User configuration requested colors
    if py_colors is not None:
        return to_bool(py_colors)

    term = os.environ.get("TERM", "")
    if "xterm" in term:
        return True

    if term.lower() == "dumb":
        return False

    # Use tty detection logic as last resort because there are numerous
    # factors that can make isatty return a misleading value, including:
    # - stdin.isatty() is the only one returning true, even on a real terminal
    # - stderr returting false if user user uses a error stream coloring solution
    return stream.isatty()


def get_console(**kwargs) -> Console:
    # interactive = is_interactive()
    from rich.theme import Theme

    # theme = Theme(STYLES)
    # if "width" not in kwargs:
    #     kwargs['width'] = 9999
    if "log_path" not in kwargs:
        kwargs["log_path"] = True
    if "soft_wrap" not in kwargs:
        kwargs["soft_wrap"] = False
    if "theme" not in kwargs:
        kwargs["theme"] = Theme(STYLES)
    # if "force_jupyter" not in kwargs:
    #     kwargs["force_jupyter"] = is_interactive()
    console = Console(
        color_system="truecolor",
        # force_jupyter=interactive,
        # log_path=False,
        # theme=theme,
        # soft_wrap=False,
        **kwargs,
    )
    return console


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    from rich.text import Text

    parser.add_argument(
        "--html", action="store_true", help="Export as HTML table"
    )
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
