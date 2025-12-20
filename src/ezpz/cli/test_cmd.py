"""Click-based test CLI to keep ezpz test and ezpz-test in sync."""

from __future__ import annotations

from typing import Iterable, Sequence

import click


def _ensure_sequence(args: Iterable[str]) -> Sequence[str]:
    return tuple(args)


def _print_help(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from ezpz.test_dist_args import build_arg_parser

    test_parser = build_arg_parser()
    test_help = test_parser.format_help().rstrip()
    launch_help = "\n".join(
        [
            "Launcher options (ezpz test wraps ezpz launch):",
            "  --filter FILTER [FILTER ...]",
            "                        Filter output lines by these strings.",
            "  -n NPROC, -np NPROC, --n NPROC, --np NPROC, --nproc NPROC, --world_size NPROC, --nprocs NPROC",
            "                        Number of processes.",
            "  -ppn NPROC_PER_NODE, --ppn NPROC_PER_NODE, --nproc_per_node NPROC_PER_NODE",
            "                        Processes per node.",
            "  -nh NHOSTS, --nh NHOSTS, --nhost NHOSTS, --nnode NHOSTS, --nnodes NHOSTS, --nhosts NHOSTS, --nhosts NHOSTS",
            "                        Number of nodes to use.",
            "  --hostfile HOSTFILE   Hostfile to use for launching.",
        ]
    )
    click.echo(f"{test_help}\n\n{launch_help}\n")
    ctx.exit()


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
    add_help_option=False,
)
@click.option(
    "--help",
    "-h",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_help,
    help="Show this message and exit.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def test_cmd(args: tuple[str, ...]) -> None:
    """Run the distributed smoke test."""
    from ezpz import test as test_module

    rc = test_module.main(list(_ensure_sequence(args)))
    if rc:
        raise click.exceptions.Exit(rc)


def main() -> None:
    test_cmd()
