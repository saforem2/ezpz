"""
ezpz/log/dataframe.py
"""

from typing import Any

import pandas as pd
from rich.box import MINIMAL, SIMPLE, SIMPLE_HEAD, SQUARE
from rich.columns import Columns
from rich.live import Live
from rich.measure import Measurement
from rich.table import Table

from ezpz.log.console import get_console
from ezpz.log.style import COLORS, beat


class DataFramePrettify:
    """Create animated and pretty Pandas DataFrame.

    Modified from: https://github.com/khuyentran1401/rich-dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The data you want to prettify
    row_limit : int, optional
        Number of rows to show, by default 20
    col_limit : int, optional
        Number of columns to show, by default 10
    first_rows : bool, optional
        Whether to show first n rows or last n rows, by default True.
        If this is set to False, show last n rows.
    first_cols : bool, optional
        Whether to show first n columns or last n columns, by default True.
        If this is set to False, show last n rows.
    delay_time : int, optional
        How fast is the animation, by default 5.
        Increase this to have slower animation.
    clear_console: bool, optional
         Clear the console before printing the table, by default True.
         If this is set to False the previous console
         input/output is maintained
    """

    def __init__(
        self,
        df: pd.DataFrame,
        row_limit: int = 20,
        col_limit: int = 10,
        first_rows: bool = True,
        first_cols: bool = True,
        delay_time: int = 5,
        clear_console: bool = True,
    ) -> None:
        self.df = df.reset_index().rename(columns={"index": ""})
        self.table = Table(show_footer=False)
        self.table_centered = Columns((self.table,), align="center", expand=True)
        self.num_colors = len(COLORS)
        self.delay_time = delay_time
        self.row_limit = row_limit
        self.first_rows = first_rows
        self.col_limit = col_limit
        self.first_cols = first_cols
        self.clear_console = clear_console
        self.console = get_console()
        if first_cols:
            self.columns = self.df.columns[:col_limit]
        else:
            self.columns = list(self.df.columns[-col_limit:])
            self.columns.insert(0, "index")
        if first_rows:
            self.rows = self.df.values[:row_limit]
        else:
            self.rows = self.df.values[-row_limit:]
        if self.clear_console:
            self.console.clear()

    def _add_columns(self):
        for col in self.columns:
            with beat(self.delay_time):
                self.table.add_column(str(col))

    def _add_rows(self):
        for row in self.rows:
            with beat(self.delay_time):
                row = (
                    row[: self.col_limit] if self.first_cols else row[-self.col_limit :]
                )

                row = [str(item) for item in row]
                self.table.add_row(*list(row))

    def _move_text_to_right(self):
        for i in range(len(self.table.columns)):
            with beat(self.delay_time):
                self.table.columns[i].justify = "right"

    def _add_random_color(self):
        for i in range(len(self.table.columns)):
            with beat(self.delay_time):
                self.table.columns[i].header_style = COLORS[i % self.num_colors]

    def _add_style(self):
        for i in range(len(self.table.columns)):
            with beat(self.delay_time):
                self.table.columns[i].style = "bold " + COLORS[i % self.num_colors]

    def _adjust_box(self):
        for box in [SIMPLE_HEAD, SIMPLE, MINIMAL, SQUARE]:
            with beat(self.delay_time):
                self.table.box = box

    def _dim_row(self):
        with beat(self.delay_time):
            self.table.row_styles = ["none", "dim"]

    def _adjust_border_color(self):
        with beat(self.delay_time):
            self.table.border_style = "bright_yellow"

    def _change_width(self):
        original_width = Measurement.get(
            console=self.console,
            options=self.console.options,
            renderable=self.table,
        ).maximum
        width_ranges = [
            [original_width, self.console.width, 2],
            [self.console.width, original_width, -2],
            [original_width, 90, -2],
            [90, original_width + 1, 2],
        ]

        for width_range in width_ranges:
            for width in range(*width_range):
                with beat(self.delay_time):
                    self.table.width = width

            with beat(self.delay_time):
                self.table.width = None

    def _add_caption(self):
        row_text = "first" if self.first_rows else "last"
        col_text = "first" if self.first_cols else "last"

        with beat(self.delay_time):
            self.table.caption = (
                f"Only the {row_text} "
                f"{self.row_limit} rows "
                f"and the {col_text} "
                f"{self.col_limit} columns "
                "is shown here."
            )
        with beat(self.delay_time):
            self.table.caption = (
                f"Only the [bold green] {row_text} "
                "{self.row_limit} rows[/bold green] and the "
                "[bold red]{self.col_limit} {col_text} "
                "columns[/bold red] is shown here."
            )
        with beat(self.delay_time):
            self.table.caption = (
                f"Only the [bold magenta not dim] "
                f"{row_text} {self.row_limit} rows "
                f"[/bold magenta not dim] and the "
                f"[bold green not dim]{col_text} "
                f"{self.col_limit} columns "
                f"[/bold green not dim] "
                f"are shown here."
            )

    def prettify(self):
        with Live(
            self.table_centered,
            console=self.console,
            refresh_per_second=self.delay_time,
            vertical_overflow="ellipsis",
        ):
            self._add_columns()
            self._add_rows()
            self._move_text_to_right()
            self._add_random_color()
            self._add_style()
            self._adjust_border_color()
            self._add_caption()

        return self.table


def prettify(
    df: Any,
    row_limit: int = 20,
    col_limit: int = 10,
    first_rows: bool = True,
    first_cols: bool = True,
    delay_time: int = 5,
    clear_console: bool = True,
):
    """Create animated and pretty Pandas DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        The data you want to prettify
    row_limit : int, optional
        Number of rows to show, by default 20
    col_limit : int, optional
        Number of columns to show, by default 10
    first_rows : bool, optional
        Whether to show first n rows or last n rows, by default True. If this is set to False, show last n rows.
    first_cols : bool, optional
        Whether to show first n columns or last n columns, by default True. If this is set to False, show last n rows.
    delay_time : int, optional
        How fast is the animation, by default 5. Increase this to have slower animation.
    clear_console: bool, optional
        Clear the console before printing the table, by default True. If this is set to false the previous console input/output is maintained
    """
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        DataFramePrettify(
            df,
            row_limit,
            col_limit,
            first_rows,
            first_cols,
            delay_time,
            clear_console,
        ).prettify()

    else:
        # In case users accidentally pass a non-datafame input, use rich's print instead
        print(df)
