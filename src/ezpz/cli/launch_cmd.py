"""Click-based launcher CLI to keep ezpz launch and ezpz-launch in sync."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from ezpz.cli.flags import build_launch_parser


DEPRECATION_NOTICE = "ezpz-launch is deprecated! use `ezpz launch` as a drop in replacement"

def _invoked_as_ezpz_launch() -> bool:
    return Path(sys.argv[0]).name == "ezpz-launch"


def _maybe_warn_deprecated() -> None:
    if _invoked_as_ezpz_launch():
        click.secho(f"\n{DEPRECATION_NOTICE}\n", fg="red", err=True)


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    }
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def launch_cmd(ctx: click.Context, args: tuple[str, ...]) -> None:
    """Launch a command across the active scheduler."""
    _maybe_warn_deprecated()
    argv = list(args)
    if any(arg in ("-h", "--help") for arg in argv):
        parser = build_launch_parser(
            prog="ezpz-launch" if _invoked_as_ezpz_launch() else "ezpz launch"
        )
        click.echo(parser.format_help().rstrip())
        ctx.exit(0)
    from ezpz import launch as launch_module

    rc = launch_module.run(argv)
    if rc:
        raise click.exceptions.Exit(rc)


def main() -> None:
    launch_cmd()
