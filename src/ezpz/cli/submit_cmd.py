"""Click command for ``ezpz submit``."""

from __future__ import annotations

from pathlib import Path

import pathlib

import click


@click.command(
    context_settings={"ignore_unknown_options": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "-N", "--nodes", type=int, default=1, help="Number of compute nodes."
)
@click.option(
    "-t",
    "--time",
    "walltime",
    default="01:00:00",
    help="Walltime in HH:MM:SS format.",
)
@click.option(
    "-q",
    "--queue",
    default="debug",
    help="Queue (PBS) or partition (SLURM).",
)
@click.option(
    "-A",
    "--account",
    default=None,
    help="Project/account for billing.  Falls back to PBS_ACCOUNT / PROJECT.",
)
@click.option(
    "--filesystems",
    default="home",
    help="PBS filesystems directive (ignored for SLURM).",
)
@click.option("--job-name", default=None, help="Job name.")
@click.option(
    "--scheduler",
    default=None,
    type=click.Choice(["PBS", "SLURM"]),
    help="Override scheduler auto-detection.",
)
@click.option(
    "--env",
    "env_setup",
    default=None,
    help="Environment setup: a script path or inline shell commands.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the generated script without submitting.",
)
@click.option(
    "--remote",
    default=None,
    metavar="HOST",
    help=(
        "Submit to another machine over SSH instead of locally, e.g. "
        "--remote aurora. Uses your existing SSH config and auth."
    ),
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "ssh", "iri"]),
    help=(
        "Remote backend (with --remote). 'auto' tries SSH and falls back "
        "to the ALCF IRI API only if SSH itself cannot connect; a "
        "scheduler rejection is never retried. 'iri' serves Aurora, "
        "Polaris and Crux only."
    ),
)
@click.option(
    "--workdir",
    default=None,
    metavar="DIR",
    help="Remote working directory to submit from (with --remote).",
)
@click.option(
    "--launch/--no-launch",
    default=True,
    help="Wrap the command with 'ezpz launch' (default: on).",
)
@click.option(
    "--gpus-per-node",
    type=int,
    default=None,
    help=(
        "SLURM --gpus-per-node (ignored for PBS). Required on Perlmutter: "
        "without it a GPU job is allocated none."
    ),
)
@click.option(
    "--ntasks-per-node",
    type=int,
    default=None,
    help="SLURM --ntasks-per-node (ignored for PBS).",
)
@click.option(
    "-C",
    "--constraint",
    default=None,
    help="SLURM --constraint, e.g. 'gpu' on Perlmutter (ignored for PBS).",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    help=(
        "Emit 'set -eo pipefail' (default: on). Use --no-strict for "
        "multi-arm experiment scripts where one arm timing out must not "
        "abort the rest. 'set -u' is never emitted -- it breaks Lmod."
    ),
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
    gpus_per_node: int | None,
    ntasks_per_node: int | None,
    constraint: str | None,
    strict: bool,
    remote: str | None,
    backend: str,
    workdir: str | None,
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
        if candidate.is_file() and candidate.suffix in (
            ".sh",
            ".bash",
            ".job",
        ):
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

    if remote:
        _submit_remote(
            remote=remote,
            backend=backend,
            workdir=workdir,
            command=command,
            script_path=script_path,
            nodes=nodes,
            walltime=walltime,
            queue=queue,
            account=account,
            filesystems=filesystems,
            job_name=job_name,
            scheduler=scheduler,
            launch=launch,
            dry_run=dry_run,
            env_setup=resolved_env,
            gpus_per_node=gpus_per_node,
            ntasks_per_node=ntasks_per_node,
            constraint=constraint,
            strict=strict,
        )
        return

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
        gpus_per_node=gpus_per_node,
        ntasks_per_node=ntasks_per_node,
        constraint=constraint,
        strict=strict,
    )


def _submit_remote(
    *,
    remote: str,
    backend: str,
    workdir: str | None,
    command: list[str] | None,
    script_path: str | None,
    nodes: int,
    walltime: str,
    queue: str,
    account: str | None,
    filesystems: str,
    job_name: str | None,
    scheduler: str | None,
    launch: bool,
    dry_run: bool,
    env_setup: str | None,
    gpus_per_node: int | None,
    ntasks_per_node: int | None,
    constraint: str | None,
    strict: bool,
) -> None:
    """Build the script locally, then submit it on *remote*.

    The script is generated here rather than on the far side so
    ``--dry-run`` shows exactly what would run, without needing a
    connection at all.
    """
    import sys

    from ezpz.remote import (
        RemoteSubmitError,
        scheduler_for,
        submit_remote,
    )
    from ezpz.submit import generate_pbs_script, generate_slurm_script

    sched = (scheduler or scheduler_for(remote)).upper()
    if script_path:
        script_text = pathlib.Path(script_path).read_text()
    else:
        cmd = " ".join(command or [])
        gen = generate_pbs_script if sched == "PBS" else generate_slurm_script
        kwargs: dict = {
            "nodes": nodes,
            "time": walltime,
            "queue": queue,
            "account": account,
            "job_name": job_name,
            # Never leak the LOCAL cwd into a remote script: the
            # generator defaults working_dir to os.getcwd(), which does
            # not exist on the target machine. Default to the remote
            # $HOME instead and let --workdir override.
            "working_dir": workdir
            or "",  # "" -> no cd; scheduler lands in $HOME
            # Same for env setup: a local utils.sh path is meaningless
            # remotely, so the network form is correct here -- but it
            # only works from a login node, and submission happens on
            # one, so this is fetched before the job is queued.
            "env_setup": env_setup,
            "wrap_with_launch": launch,
        }
        if sched == "PBS":
            kwargs["filesystems"] = filesystems
        else:
            kwargs["gpus_per_node"] = gpus_per_node
            kwargs["ntasks_per_node"] = ntasks_per_node
            kwargs["constraint"] = constraint
            kwargs["strict"] = strict
        script_text = gen(cmd, **kwargs)

    if dry_run:
        click.echo(
            f"# would submit to {remote} (scheduler={sched}, "
            f"backend={backend})"
        )
        click.echo(script_text)
        return

    iri_kwargs = {
        "command": " ".join(command or []),
        "nodes": nodes,
        "duration_seconds": _walltime_seconds(walltime),
        "queue": queue,
        "account": account or "",
        "workdir": workdir,
        "filesystems": filesystems,
    }
    try:
        result = submit_remote(
            remote,
            script_text,
            backend=backend,
            scheduler=sched,
            workdir=workdir,
            iri_kwargs=iri_kwargs,
        )
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if result.job_id:
        click.echo(f"{result.job_id}  (via {result.backend})")
        return
    click.echo(
        f"error: submission to {remote} failed via {result.backend} "
        f"(exit {result.returncode}): {result.stderr}",
        err=True,
    )
    sys.exit(result.returncode or 1)


def _walltime_seconds(walltime: str) -> int:
    """Convert ``HH:MM:SS`` (or ``MM:SS``, or ``SS``) to seconds.

    The IRI API takes a duration in seconds, while every scheduler flag
    here uses the colon form.
    """
    parts = [int(x) for x in walltime.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return h * 3600 + m * 60 + sec
