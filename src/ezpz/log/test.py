"""
src/ezpz/logging/test.py
"""
from __future__ import absolute_import, annotations, division, print_function

# from typing import Dict
# from rich.style import Style
from ezpz.log.style import STYLES


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


if __name__ == "__main__":  # pragma: no cover
    print_styles()
