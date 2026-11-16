"""Click command for ``ezpz submit``."""

from __future__ import annotations

from pathlib import Path

import pathlib
import shlex

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
    "env_path",
    default=None,
    metavar="PATH",
    help="Target-side venv/conda/Python prefix, Python executable, or shell script. A .sh path is auto-routed as --env-setup-script.",
)
@click.option(
    "--env-setup-script",
    default=None,
    metavar="PATH",
    help="Shell script to source on the job machine before selecting --env. Mutually-exclusive with an auto-routed --env script.",
)
@click.option(
    "--module-load",
    "--module",
    "--ml",
    multiple=True,
    metavar="MOD",
    help="Module to `module load` in the job shell (repeatable: --ml MOD1 --ml MOD2). Composable with --env; mutually exclusive with --env-setup-script.",
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
        "Remote backend (with --remote). 'auto' tries IRI first on "
        "Aurora/Polaris/Crux, with SSH fallback; other hosts use SSH. "
        "Choose 'ssh' or 'iri' to disable automatic fallback."
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
    env_path: str | None,
    env_setup_script: str | None,
    module_load: tuple[str, ...],
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

    # --env sugar: a .sh is a script, not an env. Auto-route it to
    # --env-setup-script so `ezpz submit --env ./setup.sh` DTRT.
    # Use the raw value for suffix detection (stripping would change a
    # valid relative path like ' setup.sh' → './setup.sh').
    if env_path is not None and env_path.strip():
        if env_path.strip().endswith(".sh") or env_path.strip().endswith(".bash"):
            candidate_env = env_path.strip()
            if env_setup_script is not None:
                raise click.UsageError(
                    "--env looks like a shell script (*.sh/*.bash) — "
                    "use --env and --env-setup-script together correctly, "
                    "or pass the script only via --env-setup-script."
                )
            if module_load:
                raise click.UsageError(
                    "--module-load/--ml is not valid with an --env that is a shell script. "
                    "Put `module load` lines inside that script, or use "
                    "`--ml MOD --env /path/to/venv` instead."
                )
            env_setup_script = candidate_env
            env_path = None

    for option, value in (
        ("--env", env_path),
        ("--env-setup-script", env_setup_script),
    ):
        if value is not None and not value.strip():
            raise click.BadParameter(
                "expected a non-empty target-side path", param_hint=option
            )

    # --env-setup-script and --module-load sugar the same source line;
    # require one or the other, not both, so there's a single source of
    # truth for module initialization in the generated job script.
    if env_setup_script is not None and module_load:
        raise click.UsageError(
            "Use either --env-setup-script or --module-load/--ml/--module, not both. "
            "Put your `module load` lines in the script, or use "
            "`--ml MOD --env /path/to/venv` to synthesize one."
        )
    for mod in module_load or ():
        if not mod or not mod.strip():
            raise click.BadParameter(
                "expected a non-empty module name", param_hint="--module-load"
            )

    # Positional remote sugar: ezpz submit aurora -- ...  ≡  --remote aurora
    if args and not remote:
        from ezpz.remote import looks_like_host

        if looks_like_host(args[0]):
            remote = args[0]
            args = args[1:]

    # Determine if first arg is an existing script file
    script_path = None
    command = None

    if args:
        candidate = Path(args[0])
        # .pbs and .sbatch are the conventional names for hand-written
        # scheduler scripts -- this repo alone has 17 and 10 of them --
        # and omitting them meant `ezpz submit run.pbs` silently treated
        # the filename as a COMMAND instead of a script.
        if candidate.is_file() and candidate.suffix in (
            ".sh",
            ".bash",
            ".job",
            ".pbs",
            ".sbatch",
            ".slurm",
        ):
            script_path = candidate
            # Remaining args are currently ignored for script mode
        else:
            command = list(args)

    if script_path is not None and (
        env_path is not None or env_setup_script is not None or module_load
    ):
        raise click.UsageError(
            "--env, --env-setup-script, and --module-load/--ml apply only to "
            "generated commands; put environment setup inside an existing job script."
        )

    if script_path is None and command is None:
        raise click.UsageError(
            "Provide a command after '--' or a script file.\n\n"
            "Examples:\n"
            "  ezpz submit -N2 -q debug -- python3 -m ezpz.examples.test\n"
            "  ezpz submit job.sh --nodes 4"
        )

    # submit already wraps with `ezpz launch` — silently strip a redundant prefix
    # (CXI LE-pool exhaustion on Sunspot if double-wrapped). Warn so the caller
    # knows it happened, but don't fail — the intent is unambiguous.
    if command and launch:
        joined = (
            " ".join(command) if isinstance(command, list) else str(command)
        )
        if joined.lstrip().startswith("ezpz launch "):
            click.echo(
                "warning: stripping redundant `ezpz launch` — `ezpz submit` already wraps with it "
                "(use --no-launch if you need custom launch flags)",
                err=True,
            )
            stripped = joined.lstrip()[len("ezpz launch ") :]
            command = (
                stripped.split() if isinstance(command, list) else stripped
            )
        elif "ezpz launch" in joined:
            # Non-prefix occurrence (e.g. in a quoted arg) — still likely a mistake, but
            # don't mangle it; just warn and leave as-is so --no-launch remains the escape hatch.
            click.echo(
                "warning: command contains `ezpz launch` inside `ezpz submit` which already wraps with it; "
                "if this was intentional, pass --no-launch",
                err=True,
            )

    # Synthesize raw shell for --module-load, then hand it directly to
    # resolved_env so we bypass the "source $path" interpretation in
    # build_env_setup.  --module-load is sugar for module-load lines, not a
    # file path, and should compose with --env (load modules, then select env).
    synthesized_env = None
    if module_load:
        if env_setup_script is not None:
            raise click.UsageError(
                "Use either --env-setup-script or --module-load/--ml/--module, not both."
            )
        from ezpz.remote_env import _strict_setup

        _mod_lines = [
            f"module load {shlex.quote(m.strip())}"
            for m in module_load
            if m and m.strip()
        ]
        # --ml alone replaces managed setup that set CCL_OP_SYNC=1 on
        # Aurora/Sunspot (see utils.sh:1175-1195). Without it FSDP2+TP
        # can hang. Preserve the default when no explicit env is given.
        if env_path is None and _mod_lines:
            _mod_lines.append(
                'export CCL_OP_SYNC="${CCL_OP_SYNC:-1}"  # Aurora/Sunspot default; override if needed'
            )
        synthesized_env = _strict_setup(_mod_lines) if _mod_lines else None

    from ezpz.remote_env import build_env_setup

    resolved_env = build_env_setup(
        env_path,
        env_setup_script,
        require_ezpz=launch
        or bool(command and Path(command[0]).name == "ezpz"),
    )
    # Prepend synthesized module-load shell when requested; it should run
    # before env selection (login shell: modules → then env) and compose
    # correctly with --env.  --env-setup-script + --ml is already rejected
    # above, so this is --ml [ + --env] only.
    if synthesized_env is not None:
        resolved_env = (
            f"{synthesized_env}\n{resolved_env}"
            if resolved_env
            else synthesized_env
        )

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
    script_path: Path | None,
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
    from ezpz.remote_env import build_remote_env_setup

    sched = (scheduler or scheduler_for(remote)).upper()
    if script_path:
        script_text = pathlib.Path(script_path).read_text()
    else:
        cmd = shlex.join(command or [])
        gen = generate_pbs_script if sched == "PBS" else generate_slurm_script
        kwargs: dict = {
            "nodes": nodes,
            "time": walltime,
            "queue": queue,
            "account": account,
            "job_name": job_name,
            # Never copy the local cwd into a remote script. The shared
            # per-job or explicit workdir is selected inside env_setup,
            # independently of either environment override.
            "working_dir": "",
            # Explicit setup avoids managed network/bootstrap steps.
            "env_setup": build_remote_env_setup(
                workdir,
                env_setup=env_setup,
                alcf_proxies=remote.lower().startswith(
                    (
                        "aurora",
                        "sunspot",
                        "polaris",
                        "crux",
                        "sophia",
                        "sirius",
                    )
                ),
            ),
            "wrap_with_launch": launch,
        }
        if sched == "PBS":
            kwargs["filesystems"] = filesystems
            kwargs["strict"] = strict
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

    # Run the SAME script the ssh backend would. Passing the bare command
    # here meant an IRI fallback silently dropped the environment setup
    # and the `ezpz launch` wrapping -- a different job from the one the
    # user asked for, which is worse than failing.
    iri_kwargs = {
        "command": script_text,
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
