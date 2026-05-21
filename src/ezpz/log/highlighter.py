"""
ezpz/log/highlighter.py

Custom rich highlighter extending ``ReprHighlighter`` with patterns
specific to ezpz log output. Currently adds:

  - ``ezpz_std``: matches the ``(±NUM)`` parenthetical produced by
    :func:`ezpz.utils.format_compact_summary` so the std token can be
    styled separately from regular numeric values.

The class is wired into the ezpz console via
:func:`ezpz.log.console.get_console` (which is what
:class:`ezpz.log.handler.RichHandler` uses to render log records).
"""

from __future__ import annotations

from rich.highlighter import ReprHighlighter


class EzpzReprHighlighter(ReprHighlighter):
    """ReprHighlighter + ezpz-specific patterns.

    Subclass to add patterns *after* rich's defaults so they win the
    final span priority (rich applies regex matches sequentially and
    later spans override earlier ones in the overlapping region).
    """

    # Inherit rich's base highlights and append our own.
    # Pattern explanation for ezpz_std:
    #   \(±        literal opening paren + plus-minus sign
    #   \s*        any optional leading whitespace (right-aligned std
    #              tokens are padded with spaces inside the parens)
    #   [^)]+      the std value itself (number, may include `e-`, `.`,
    #              `-`, etc.) — anything up to the closing paren
    #   \)         literal closing paren
    # The whole match is captured in a named group `ezpz_std` so the
    # style key becomes `repr.ezpz_std` (rich appends the group name
    # to the highlighter's `base_style` = "repr.").
    highlights = ReprHighlighter.highlights + [
        r"(?P<ezpz_std>\(±\s*[^)]+\))",
    ]
