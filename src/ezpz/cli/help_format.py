"""Version-independent colorized argparse help formatters.

Prefers ``rich-argparse`` when installed; otherwise falls back to stock
``argparse`` formatters. The stock formatters self-colorize on Python 3.14+
(argparse gained native ``color=True``) and render plain text below that, so
this module never regresses help output on any supported interpreter.

Exports
-------
DefaultsFormatter
    Colorized, auto-appends ``(default: ...)``. Drop-in replacement for
    ``argparse.ArgumentDefaultsHelpFormatter``.
ColorFormatter
    Colorized only (no defaults appended). Drop-in for the argparse default.
RawDescAndDefaultsFormatter
    Raw multi-line description + appended defaults, colorized. Used by the
    ``ezpz launch`` parser whose description contains hand-formatted examples.
HAVE_RICH_ARGPARSE
    ``True`` when rich-argparse supplied the formatters.
"""

from __future__ import annotations

import argparse

try:
    from rich_argparse import (
        ArgumentDefaultsRichHelpFormatter as _Defaults,
        RawDescriptionRichHelpFormatter as _RawDesc,
        RichHelpFormatter as _Color,
    )

    class RawDescAndDefaultsFormatter(_Defaults, _RawDesc):
        """Raw multi-line description + appended defaults, colorized."""

    DefaultsFormatter: type[argparse.HelpFormatter] = _Defaults
    ColorFormatter: type[argparse.HelpFormatter] = _Color
    HAVE_RICH_ARGPARSE = True
except ImportError:  # pragma: no cover - exercised only without the extra

    class RawDescAndDefaultsFormatter(  # type: ignore[no-redef]
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        """Raw multi-line description + appended defaults."""

    DefaultsFormatter = argparse.ArgumentDefaultsHelpFormatter
    ColorFormatter = argparse.HelpFormatter
    HAVE_RICH_ARGPARSE = False


__all__ = [
    "DefaultsFormatter",
    "ColorFormatter",
    "RawDescAndDefaultsFormatter",
    "HAVE_RICH_ARGPARSE",
]
