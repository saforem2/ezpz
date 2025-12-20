"""Click-based launcher CLI to keep ezpz launch and ezpz-launch in sync."""

from __future__ import annotations

from typing import Iterable, Sequence

import click


def _ensure_sequence(args: Iterable[str]) -> Sequence[str]:
    return tuple(args)


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "help_option_names": ["-h", "--help"],
    }
)
@click.option(
    "--filter",
    "filters",
    multiple=True,
    help="Filter output lines by these strings.",
)
@click.option(
    "-n",
    "-np",
    "--n",
    "--np",
    "--nproc",
    "--world_size",
    "--nprocs",
    "nproc",
    type=int,
    default=-1,
    help="Number of processes.",
)
@click.option(
    "-ppn",
    "--ppn",
    "--nproc_per_node",
    "nproc_per_node",
    type=int,
    default=-1,
    help="Processes per node.",
)
@click.option(
    "-nh",
    "--nh",
    "--nhost",
    "--nnode",
    "--nnodes",
    "--nhosts",
    "nhosts",
    type=int,
    default=-1,
    help="Number of nodes to use.",
)
@click.option(
    "--hostfile",
    type=str,
    default=None,
    help="Hostfile to use for launching.",
)
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def launch_cmd(
    command: tuple[str, ...],
    *,
    filters: tuple[str, ...],
    nproc: int,
    nproc_per_node: int,
    nhosts: int,
    hostfile: str | None,
) -> None:
    """Launch a command across the active scheduler."""
    argv: list[str] = []
    if filters:
        argv.extend(["--filter", *filters])
    if nproc > -1:
        argv.extend(["--nproc", str(nproc)])
    if nproc_per_node > -1:
        argv.extend(["--nproc_per_node", str(nproc_per_node)])
    if nhosts > -1:
        argv.extend(["--nhosts", str(nhosts)])
    if hostfile:
        argv.extend(["--hostfile", hostfile])
    argv.extend(command)

    from ezpz import launch as launch_module

    rc = launch_module.run(_ensure_sequence(argv))
    if rc:
        raise click.exceptions.Exit(rc)


def main() -> None:
    launch_cmd()
