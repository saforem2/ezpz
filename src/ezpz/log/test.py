"""
Log demo utilities: print style table and smoke-test log output.
"""
from __future__ import absolute_import, annotations, division, print_function

import argparse

from ezpz.log import get_logger
from ezpz.log.console import Console
from ezpz.log.style import STYLES

log = get_logger(__name__)


def print_styles(html: bool = False):
    from rich.table import Table
    from rich.text import Text

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


def smoke_logs():
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
    log.info("Long lines are automatically wrapped by the terminal")
    log.info(250 * "-")


def main(argv: list[str] | None = None):  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--styles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the style table",
    )
    parser.add_argument(
        "--logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit a short log smoke test",
    )
    parser.add_argument("--html", action="store_true", help="Export styles as HTML")
    args = parser.parse_args(argv)

    if args.styles:
        print_styles(html=args.html)
    if args.logs:
        smoke_logs()


if __name__ == "__main__":  # pragma: no cover
    main()
