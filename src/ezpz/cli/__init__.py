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
    name="yeet", context_settings={"ignore_unknown_options": True}
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def yeet_cmd(args: tuple[str, ...]) -> None:
    """Distribute files (envs, models, datasets, etc.) to worker nodes via parallel rsync.

    By default (no args), rsyncs the active venv/conda env to
    /tmp/<env-name>/ on all nodes in the current job allocation.
    Pass any path (positional or via --src) to yeet arbitrary content.

    \b
    Examples:
      ezpz yeet                            # sync active env to all nodes
      ezpz yeet .venv.tar.gz               # positional shorthand for --src
      ezpz yeet --src /path/to/env         # sync a specific environment
      ezpz yeet --src /path/to/dataset     # sync a dataset / model / etc.
      ezpz yeet --dst /local/scratch       # custom destination
      ezpz yeet --dry-run                  # preview without syncing

    \b
    Options (passed through):
      --src PATH       Source path (default: active venv/conda)
      --dst PATH       Destination on workers (default: /tmp/<basename>/)
      --hostfile PATH  Hostfile for node list (default: auto-detect)
      --copy           Use cp -a for local copy (faster on Lustre)
      --compress       tar.gz → copy → extract (least Lustre I/O)
      --dry-run        Show what would be synced
    """
    from ezpz.utils import yeet_env as yeet_env_module

    rc = yeet_env_module.run(list(args) if args else None)
    _handle_exit_code(rc)


@main.command(
    name="yeet-env", context_settings={"ignore_unknown_options": True},
    hidden=True,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def yeet_env_cmd(args: tuple[str, ...]) -> None:
    """Deprecated alias for ``ezpz yeet``."""
    click.secho(
        "ezpz yeet-env is deprecated; use 'ezpz yeet' as a drop-in replacement",
        fg="yellow", err=True,
    )
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


@main.command(name="kill", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def kill_cmd(args: tuple[str, ...]) -> None:
    """Kill ezpz-launched python processes (or any matching pattern).

    Without arguments, kills processes on the local node whose
    environment contains EZPZ_RUN_COMMAND (set automatically by
    `ezpz launch`).

    \b
    Examples:
      ezpz kill                       # local node, ezpz-launched procs only
      ezpz kill train.py              # local node, anything matching `train.py`
      ezpz kill --all-nodes           # fan out across the job's hostfile
      ezpz kill --dry-run             # list matches, don't kill
      ezpz kill --signal KILL train   # SIGKILL anything matching `train`

    \b
    Options (passed through):
      --all-nodes      SSH into every node in the hostfile and kill there too
      --hostfile PATH  Hostfile for --all-nodes (default: auto-detect)
      --signal NAME    Signal to send (TERM, KILL, INT, HUP, QUIT)
      --dry-run        List matches without signaling
    """
    from ezpz.utils import kill as kill_module

    rc = kill_module.run(list(args) if args else None)
    _handle_exit_code(rc)
