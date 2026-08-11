"""Unified Click-based command line entry point for ezpz."""

from __future__ import annotations

from pathlib import Path
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

    # Always pass a list (even empty) so argparse doesn't fall back to
    # sys.argv[1:] — which on `ezpz yeet` (no args) would contain
    # ["yeet"], picked up as a positional SRC == "yeet".
    rc = yeet_env_module.run(list(args))
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

    rc = yeet_env_module.run(list(args))
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
      --model SIZE            Model size preset passed through to each example
                              (debug/small/medium/large + xl/xxl/xxxl with
                              long-form aliases like xlarge/extra-large).
                              agpt-2b/agpt-20b are fsdp_tp-only. Each example
                              validates the preset against its own MODEL_PRESETS.
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


@main.command(name="export-amsc")
@click.argument(
    "rundir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--config",
    required=True,
    help="Benchmark configuration label, e.g. agpt-2b/bs1/seq2048/tp1.",
)
@click.option("--system", default=None, help="Facility name (default: MACHINE).")
@click.option("--nodes", type=int, default=None, help="Node count (default: NUM_NODES).")
@click.option("--gpus", type=int, default=None, help="GPU count (default: NGPUS).")
@click.option(
    "--status",
    type=click.Choice(["pass", "fail"]),
    default="pass",
    help="Run outcome. NOT inferable from a run dir -- set it yourself.",
)
@click.option("--error", default="", help="Error note for a failed run.")
@click.option(
    "--warmup",
    type=int,
    default=None,
    help="Leading steps to drop before reducing throughput (default: 1).",
)
@click.option(
    "--reducer",
    type=click.Choice(["median", "mean", "max", "min", "last"]),
    default="median",
    help="How to reduce the per-step series (default: median).",
)
@click.option(
    "--wall-time-sec",
    type=float,
    default=None,
    help="Override wall time with the scheduler's figure (recommended).",
)
@click.option(
    "--append",
    "append_to",
    type=click.Path(path_type=Path),
    default=None,
    help="Append to this runs.csv (header written only if new).",
)
@click.option("--no-header", is_flag=True, help="Omit the header on stdout.")
def export_amsc_cmd(
    rundir: Path,
    config: str,
    system: str | None,
    nodes: int | None,
    gpus: int | None,
    status: str,
    error: str,
    warmup: int | None,
    reducer: str,
    wall_time_sec: float | None,
    append_to: Path | None,
    no_header: bool,
) -> None:
    """Export a finished run as one AmSC at-scale benchmarks CSV row.

    Reads a run directory's metrics JSONL, reduces it to the schema
    used by the AmSC benchmarks dashboard, and writes one CSV row.

    \b
    Examples:
      ezpz export-amsc outputs/ezpz.examples.fsdp_tp/2026-08-10-201958 \\
          --config agpt-2b/bs1/seq2048/tp1
      ezpz export-amsc <rundir> --config agpt-2b/... \\
          --append benchmarks/training/llm-finetuning/results/runs.csv

    \b
    Notes:
      * `mfu` is a PERCENTAGE (0-100), not a fraction.
      * `mfu`/`tflops` are per-GPU; `throughput_tokens_per_sec` is global.
      * wall_time_sec is the sum of measured step times and EXCLUDES
        setup; pass --wall-time-sec for the scheduler's real figure.
      * `status` cannot be inferred from a run directory.
    """
    from ezpz.export.amsc import (
        AMSC_COLUMNS,
        AmscExportError,
        DEFAULT_WARMUP,
        format_csv,
        load_run_metrics,
        load_run_provenance,
        summarize_run,
        to_amsc_row,
    )

    try:
        records = load_run_metrics(rundir)
        summary = summarize_run(
            records,
            warmup=DEFAULT_WARMUP if warmup is None else warmup,
            reducer=reducer,
        )
        if wall_time_sec is not None:
            summary["wall_time_sec"] = wall_time_sec
        row = to_amsc_row(
            summary,
            config=config,
            system=system,
            nodes=nodes,
            gpus=gpus,
            provenance=load_run_provenance(rundir),
            status=status,
            error=error,
        )
    except AmscExportError as exc:
        click.echo(f"error: {exc}", err=True)
        _handle_exit_code(1)
        return

    if append_to is not None:
        exists = append_to.exists() and append_to.stat().st_size > 0
        if exists:
            # Refuse to append under a header we do not match. We write
            # values positionally, so a file whose columns differ (or are
            # merely ordered differently) would silently take our numbers
            # into the wrong fields -- and a corrupted results file is
            # worse than a failed export.
            import csv as _csv

            with append_to.open(encoding="utf-8", newline="") as fh:
                existing = next(_csv.reader(fh), [])
            expected = list(AMSC_COLUMNS)
            if existing != expected:
                click.echo(
                    f"error: {append_to} has a different header, so "
                    "appending would put values in the wrong columns.\n"
                    f"  file:     {','.join(existing)}\n"
                    f"  exporter: {','.join(expected)}\n"
                    "Reconcile the columns, or write to stdout and merge "
                    "by hand.",
                    err=True,
                )
                _handle_exit_code(1)
                return
        append_to.parent.mkdir(parents=True, exist_ok=True)
        with append_to.open("a", encoding="utf-8") as fh:
            fh.write(format_csv([row], header=not exists))
        click.echo(f"appended 1 row to {append_to}", err=True)
    else:
        click.echo(format_csv([row], header=not no_header), nl=False)
