"""Click-based test CLI to keep ezpz test and ezpz-test in sync."""

from __future__ import annotations

from typing import Iterable, Sequence

import click


def _ensure_sequence(args: Iterable[str]) -> Sequence[str]:
    return tuple(args)


def _print_help(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from ezpz.cli.flags import build_launch_parser, build_test_parser

    test_parser = build_test_parser(prog="ezpz test")
    test_help = test_parser.format_help().rstrip()
    launch_parser = build_launch_parser(prog="ezpz launch")
    launch_help = launch_parser.format_help().rstrip()
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
