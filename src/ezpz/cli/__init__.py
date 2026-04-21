"""Unified Click-based command line entry point for ezpz."""

from __future__ import annotations

from typing import Iterable, Sequence

import click

from ezpz.__about__ import __version__
from ezpz.cli.launch_cmd import launch_cmd
from ezpz.cli.submit_cmd import submit_cmd
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


main.add_command(submit_cmd, name="submit")


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
    """Distribute a Python environment to worker nodes via parallel rsync.

    By default, rsyncs the active venv/conda env to /tmp/<env-name>/
    on all nodes in the current job allocation.

    \b
    Examples:
      ezpz yeet-env                        # sync active env to all nodes
      ezpz yeet-env --src /path/to/env     # sync a specific environment
      ezpz yeet-env --dst /local/scratch    # custom destination
      ezpz yeet-env --dry-run              # preview without syncing

    \b
    Options (passed through):
      --src PATH       Source environment (default: active venv/conda)
      --dst PATH       Destination on workers (default: /tmp/<env-name>/)
      --hostfile PATH  Hostfile for node list (default: auto-detect)
      --dry-run        Show what would be synced
    """
    from ezpz.utils import yeet_env as yeet_env_module

    rc = yeet_env_module.run(list(args) if args else None)
    _handle_exit_code(rc)


@main.command(
    name="benchmark",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def benchmark_cmd(args: tuple[str, ...]) -> None:
    """Run ezpz examples sequentially and generate a benchmark report.

    Runs all (or selected) examples with timing, captures logs, and
    produces a summary report with per-example metrics.

    \b
    Examples:
      ezpz benchmark                        # run all examples
      ezpz benchmark --run test,fsdp        # run specific examples
      ezpz benchmark --model debug          # use debug model size
      ezpz benchmark --outdir ./my-results  # custom output directory

    \b
    Options (passed through to the benchmark runner):
      --run NAME[,NAME,...]   Examples to run (default: all)
      --model SIZE            Model size: debug, small, medium, large
      --outdir PATH           Output directory for logs and report
    """
    from ezpz.examples.run_all import main as run_all_main

    run_all_main(list(args))


@main.command(name="doctor", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def doctor_cmd(args: tuple[str, ...]) -> None:
    """Inspect the environment for ezpz launch readiness."""
    from ezpz import doctor as doctor_module

    rc = doctor_module.run(_ensure_sequence(args))
    _handle_exit_code(rc)
