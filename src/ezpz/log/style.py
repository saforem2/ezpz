"""
src/ezpz/log/style.py
"""

from __future__ import absolute_import, annotations, division, print_function

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import time
from typing import Optional
from typing import Any
from typing import Generator

import logging

# from ezpz.log.config import STYLES, DEFAULT_STYLES
# from ezpz.log.console import get_console

from omegaconf import DictConfig, OmegaConf
import rich
from rich import print

# from rich.box import MINIMAL, SIMPLE, SIMPLE_HEAD, SQUARE
# from rich.columns import Columns
from rich.layout import Layout

# from rich.live import Live
# from rich.measure import Measurement
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import rich.syntax
from rich.table import Table
import rich.tree


def make_layout(ratio: int = 4, visible: bool = True) -> Layout:
    """Define the layout."""
    layout = Layout(name="root", visible=visible)
    layout.split_row(
        Layout(name="main", ratio=ratio, visible=visible),
        Layout(name="footer", visible=visible),
    )
    return layout


def build_layout(
    steps: Any,
    visible: bool = True,
    job_type: Optional[str] = "train",
) -> dict:
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn("dots"),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    tasks = {}
    border_style = "white"
    if job_type == "train":
        border_style = "green"
        tasks["step"] = job_progress.add_task(
            "[blue]Total",
            total=(steps.nera * steps.nepoch),
        )
        # tasks['era'] = job_progress.add_task(
        #     "[blue]Era",
        #     total=steps.nera
        # )
        tasks["epoch"] = job_progress.add_task(
            "[cyan]Epoch", total=steps.nepoch
        )
    elif job_type == "eval":
        border_style = "green"
        tasks["step"] = job_progress.add_task(
            "[green]Eval",
            total=steps.test,
        )
    elif job_type == "hmc":
        border_style = "yellow"
        tasks["step"] = job_progress.add_task(
            "[green]HMC",
            total=steps.test,
        )
    else:
        raise ValueError(
            "Expected job_type to be one of train, eval, or HMC,\n"
            f"Received: {job_type}"
        )

    # total = sum(task.total for task in job_progress.tasks)
    # overall_progress = Progress()
    # overall_task = overall_progress.add_task("All jobs", total=int(total))

    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        Panel.fit(
            job_progress,
            title=f"[b]{job_type}",
            border_style=border_style,
            # padding=(1, 1),
        )
    )
    layout = make_layout(visible=visible)
    if visible:
        layout["root"]["footer"].update(progress_table)
    # layout['root']['right']['top'].update(Panel.fit(' '))
    # if columns is not None:
    #     layout['root']['main'].update(Panel.fit(columns))
    #     # add_row(Panel.fit(columns))
    # layout['root']['footer']['bottom'].update(avgs_table)

    return {
        "layout": layout,
        "tasks": tasks,
        "progress_table": progress_table,
        "job_progress": job_progress,
    }


def add_columns(
    avgs: dict,
    table: Table,
    skip: Optional[str | list[str]] = None,
    keep: Optional[str | list[str]] = None,
) -> Table:
    for key in avgs.keys():
        if skip is not None and key in skip:
            continue
        if keep is not None and key not in keep:
            continue

        if key == "loss":
            table.add_column(str(key), justify="center", style="green")
        elif key == "dt":
            table.add_column(str(key), justify="center", style="red")

        elif key == "acc":
            table.add_column(str(key), justify="center", style="magenta")
        elif key == "dQint":
            table.add_column(str(key), justify="center", style="cyan")
        elif key == "dQsin":
            table.add_column(str(key), justify="center", style="yellow")
        else:
            table.add_column(str(key), justify="center")

    return table


def flatten_dict(d) -> dict:
    res = {}
    if isinstance(d, dict):
        for k in d:
            if k == "_target_":
                continue

            dflat = flatten_dict(d[k])
            for key, val in dflat.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = d

    return res


def nested_dict_to_df(d):
    import pandas as pd

    dflat = flatten_dict(d)
    df = pd.DataFrame.from_dict(dflat, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


def print_config(
    config: DictConfig | dict | Any,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config
            components are printed.
        resolve (bool, optional): Whether to resolve reference fields of
            DictConfig.
    """
    import pandas as pd

    tree = rich.tree.Tree("CONFIG")  # , style=style, guide_style=style)
    quee = []
    for f in config:
        if f not in quee:
            quee.append(f)
    dconfig = {}
    for f in quee:
        branch = tree.add(f)  # , style=style, guide_style=style)
        config_group = config[f]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
            cfg = OmegaConf.to_container(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
            cfg = str(config_group)
        dconfig[f] = cfg
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    outfile = Path(os.getcwd()).joinpath("config_tree.log")
    from rich.console import Console

    with outfile.open("wt") as f:
        console = Console(file=f)
        console.print(tree)
    with open("config.json", "w") as f:
        f.write(json.dumps(dconfig))
    cfgfile = Path("config.yaml")
    OmegaConf.save(config, cfgfile, resolve=True)
    cfgdict = OmegaConf.to_object(config)
    logdir = Path(os.getcwd()).resolve().as_posix()
    if not config.get("debug_mode", False):
        dbfpath = Path(os.getcwd()).joinpath("logdirs.csv")
    else:
        dbfpath = Path(os.getcwd()).joinpath("logdirs-debug.csv")
    if dbfpath.is_file():
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True
    df = pd.DataFrame({logdir: cfgdict})
    df.T.to_csv(dbfpath.resolve().as_posix(), mode=mode, header=header)
    os.environ["LOGDIR"] = logdir


@dataclass
class CustomLogging:
    version: int = 1
    formatters: dict[str, Any] = field(
        default_factory=lambda: {
            "simple": {
                "format": (
                    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
                )
            }
        }
    )
    handlers: dict[str, Any] = field(
        default_factory=lambda: {
            "console": {
                "class": "rich.logging.RichHandler",
                "formatter": "simple",
                "rich_tracebacks": "true",
            },
            "file": {
                "class": "logging.FileHander",
                "formatter": "simple",
                "filename": "${hydra.job.name}.log",
            },
        }
    )
    root: dict[str, Any] = field(
        default_factory=lambda: {
            "level": "INFO",
            "handlers": ["console", "file"],
        }
    )
    disable_existing_loggers: bool = False


def printarr(*arrs, float_width=6):
    """
    Print a pretty table giving name, shape, dtype, type, and content
    information for input tensors or scalars.

    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a
    variable number of arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None

    It may also work with other array-like types, but they have not been tested

    Use the `float_width` option specify the precision to which floating point
    types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source:
        https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also
    released into the public domain. Please retain this docstring as a
    reference.
    """
    import inspect

    frame_ = inspect.currentframe()
    assert frame_ is not None
    frame = frame_.f_back
    # if frame_ is not None:
    #     frame = frame_.f_back
    # else:
    #     frame = inspect.getouterframes()
    default_name = "[temporary]"

    # helpers to gather data about each array

    def name_from_outer_scope(a):
        if a is None:
            return "[None]"
        name = default_name
        if frame_ is not None:
            for k, v in frame_.f_locals.items():
                if v is a:
                    name = k
                    break
        return name

    def dtype_str(a):
        if a is None:
            return "None"
        if isinstance(a, int):
            return "int"
        if isinstance(a, float):
            return "float"
        return str(a.dtype)

    def shape_str(a):
        if a is None:
            return "N/A"
        if isinstance(a, int):
            return "scalar"
        if isinstance(a, float):
            return "scalar"
        return str(list(a.shape))

    def type_str(a):
        # TODO this is is weird... what's the better way?
        return str(type(a))[8:-2]

    def device_str(a):
        if hasattr(a, "device"):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want,
                # ignore it
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ("N/A", "N/A", "N/A")
        if isinstance(a, int) or isinstance(a, float):
            return (format_float(a), format_float(a), format_float(a))

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try:
            min_str = format_float(a.min())
        except Exception:
            pass
        max_str = "N/A"
        try:
            max_str = format_float(a.max())
        except Exception:
            pass
        mean_str = "N/A"
        try:
            mean_str = format_float(a.mean())
        except Exception:
            pass

        return (min_str, max_str, mean_str)

    try:
        props = [
            "name",
            "dtype",
            "shape",
            "type",
            "device",
            "min",
            "max",
            "mean",
        ]

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append(
                {
                    "name": name_from_outer_scope(a),
                    "dtype": dtype_str(a),
                    "shape": shape_str(a),
                    "type": type_str(a),
                    "device": device_str(a),
                    "min": minmaxmean[0],
                    "max": minmaxmean[1],
                    "mean": minmaxmean[2],
                }
            )

        # for each property, compute its length
        maxlen = {}
        for p in props:
            maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings,
        # don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # print a header
        header_str = ""
        for p in props:
            prefix = "" if p == "name" else " | "
            fmt_key = ">" if p == "name" else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-" * len(header_str))
        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix = "" if p == "name" else " | "
                fmt_key = ">" if p == "name" else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end="")
            print("")

    finally:
        del frame


# console = Console()

BEAT_TIME = 0.008

COLORS = ["cyan", "magenta", "red", "green", "blue", "purple"]

# log = get_logger(__name__)

log = logging.getLogger(__name__)
# handlers = log.handlers
# if (
#         len(handlers) > 0
#         and isinstance(handlers[0], RichHandler)
# ):
#     console = handlers[0].console
# else:
#     console = get_console(markup=True)


@contextmanager
def beat(length: int = 1) -> Generator:
    import rich.console

    console = rich.console.Console()
    with console:
        yield
    time.sleep(length * BEAT_TIME)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    from rich.text import Text

    # from ezpz.log.console import Console
    parser.add_argument(
        "--html", action="store_true", help="Export as HTML table"
    )
    args = parser.parse_args()
    html: bool = args.html
    # from rich.table import Table
    # console = Console(record=True, width=120) if html else Console()
    from ezpz.log.console import get_console
    from ezpz.log.config import STYLES, DEFAULT_STYLES

    console = get_console(record=html, width=150)
    table = Table("Name", "Styling")
    styles = DEFAULT_STYLES
    styles |= STYLES
    for style_name, style in styles.items():
        table.add_row(Text(style_name, style=style), str(style))
    console.print(table)
    if html:
        outfile = "enrich_styles.html"
        print(f"Saving to `{outfile}`")
        with open(outfile, "w") as f:
            f.write(console.export_html(inline_styles=True))
