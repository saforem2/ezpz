"""Unified Click-based command line entry point for ezpz."""

from __future__ import annotations

from typing import Iterable, Sequence

import click

from ezpz.__about__ import __version__
from ezpz.cli.launch_cmd import launch_cmd
from ezpz.cli.test_cmd import test_cmd

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


def _ensure_sequence(args: Iterable[str]) -> Sequence[str]:
    return tuple(args)


def _handle_exit_code(return_code: int) -> None:
    if return_code:
        raise click.exceptions.Exit(return_code)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def main() -> None:
    """ezpz distributed utilities."""


main.add_command(test_cmd, name="test")


main.add_command(launch_cmd, name="launch")


@main.command(
    name="tar-env", context_settings={"ignore_unknown_options": True}
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def tar_env_cmd(args: tuple[str, ...]) -> None:
    """Create (or locate) a tarball for the current environment."""
    from ezpz.utils import tar_env as tar_env_module
    rc = tar_env_module.main()

    # rc = tar_env_module.main(_ensure_sequence(args))
    # _handle_exit_code(rc)


@main.command(
    name="yeet-env", context_settings={"ignore_unknown_options": True}
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def yeet_env_cmd(args: tuple[str, ...]) -> None:
    """Distribute an environment tarball across worker nodes."""
    # from ezpz.utils import yeet_env as yeet_env_module
    from ezpz.utils import yeet_env as yeet_env_module

    rc = yeet_env_module.run(_ensure_sequence(args))
    _handle_exit_code(rc)


@main.command(name="doctor", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def doctor_cmd(args: tuple[str, ...]) -> None:
    """Inspect the environment for ezpz launch readiness."""
    from ezpz import doctor as doctor_module

    rc = doctor_module.run(_ensure_sequence(args))
    _handle_exit_code(rc)
