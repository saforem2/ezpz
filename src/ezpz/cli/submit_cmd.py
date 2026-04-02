"""Click command for ``ezpz submit``."""

from __future__ import annotations

from pathlib import Path

import click


@click.command(
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "-N", "--nodes", type=int, default=1, help="Number of compute nodes."
)
@click.option(
    "-t", "--time", "walltime", default="01:00:00",
    help="Walltime in HH:MM:SS format.",
)
@click.option(
    "-q", "--queue", default="debug",
    help="Queue (PBS) or partition (SLURM).",
)
@click.option(
    "-A", "--account", default=None,
    help="Project/account for billing.  Falls back to PBS_ACCOUNT / PROJECT.",
)
@click.option(
    "--filesystems", default="home",
    help="PBS filesystems directive (ignored for SLURM).",
)
@click.option("--job-name", default=None, help="Job name.")
@click.option(
    "--scheduler", default=None, type=click.Choice(["PBS", "SLURM"]),
    help="Override scheduler auto-detection.",
)
@click.option(
    "--env", "env_setup", default=None,
    help="Environment setup: a script path or inline shell commands.",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Print the generated script without submitting.",
)
@click.option(
    "--launch", is_flag=True, default=False,
    help="Wrap the command with 'ezpz launch'.",
)
def submit_cmd(
    args: tuple[str, ...],
    nodes: int,
    walltime: str,
    queue: str,
    account: str | None,
    filesystems: str,
    job_name: str | None,
    scheduler: str | None,
    env_setup: str | None,
    dry_run: bool,
    launch: bool,
) -> None:
    """Submit a job to the active scheduler (PBS/SLURM).

    \b
    Two modes:
      1. Wrap a command:  ezpz submit -N2 -q debug -- python3 -m my.module
      2. Submit a script: ezpz submit job.sh -N4 --time 02:00:00
    """
    from ezpz.submit import submit

    # Determine if first arg is an existing script file
    script_path = None
    command = None

    if args:
        candidate = Path(args[0])
        if candidate.is_file() and candidate.suffix in (".sh", ".bash", ".job"):
            script_path = candidate
            # Remaining args are currently ignored for script mode
        else:
            command = list(args)

    if script_path is None and command is None:
        raise click.UsageError(
            "Provide a command after '--' or a script file.\n\n"
            "Examples:\n"
            "  ezpz submit -N2 -q debug -- python3 -m ezpz.examples.test\n"
            "  ezpz submit job.sh --nodes 4"
        )

    # Resolve --env: if it's a file path, source it; otherwise use verbatim
    resolved_env: str | None = None
    if env_setup is not None:
        if Path(env_setup).is_file():
            resolved_env = f"source {env_setup}"
        else:
            resolved_env = env_setup

    submit(
        command=command,
        script=script_path,
        nodes=nodes,
        time=walltime,
        queue=queue,
        account=account,
        filesystems=filesystems,
        job_name=job_name,
        scheduler=scheduler,
        wrap_with_launch=launch,
        dry_run=dry_run,
        env_setup=resolved_env,
    )
