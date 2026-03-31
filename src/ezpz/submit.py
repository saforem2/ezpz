"""Job submission helpers for PBS (``qsub``) and SLURM (``sbatch``).

Generates scheduler-specific job scripts from a command and resource
requirements, then submits them.  Supports two modes:

1. **Wrap a command** — auto-generates a job script that calls
   ``ezpz launch <command>``.
2. **Submit an existing script** — passes a user-written script to the
   scheduler, optionally overriding resource directives.

Usage::

    from ezpz.submit import submit

    job_id = submit(
        command=["python3", "-m", "ezpz.examples.test", "--model", "small"],
        nodes=2,
        queue="debug",
        time="01:00:00",
    )
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "detect_env_setup",
    "generate_pbs_script",
    "generate_slurm_script",
    "submit",
    "submit_job",
]


# ── Environment detection ────────────────────────────────────────────────────


def detect_env_setup() -> str:
    """Return shell commands that reproduce the current Python environment.

    Checks (in order):

    1. ``EZPZ_SETUP_ENV`` — sourced verbatim if set.
    2. ``VIRTUAL_ENV`` — activates the venv.
    3. ``CONDA_PREFIX`` — activates the conda environment.

    Returns:
        A (possibly multi-line) string of shell commands, or ``""`` if no
        environment was detected.
    """
    lines: list[str] = []

    setup_env = os.environ.get("EZPZ_SETUP_ENV")
    if setup_env and Path(setup_env).is_file():
        lines.append(f"source {shlex.quote(setup_env)}")

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        activate = Path(venv) / "bin" / "activate"
        lines.append(f"source {shlex.quote(str(activate))}")
        return "\n".join(lines)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_name = Path(conda_prefix).name
        lines.append(f"conda activate {shlex.quote(env_name)}")
        return "\n".join(lines)

    return "\n".join(lines)


# ── Script generation ────────────────────────────────────────────────────────


def generate_pbs_script(
    command: str,
    *,
    nodes: int = 1,
    time: str = "01:00:00",
    queue: str = "debug",
    account: str | None = None,
    filesystems: str = "home",
    job_name: str | None = None,
    working_dir: str | None = None,
    env_setup: str | None = None,
    wrap_with_launch: bool = True,
) -> str:
    """Generate a PBS job script.

    Args:
        command: The command to execute (already formatted as a string).
        nodes: Number of compute nodes.
        time: Walltime in ``HH:MM:SS`` format.
        queue: PBS queue name.
        account: PBS account/project.  Falls back to ``PBS_ACCOUNT`` or
            ``PROJECT`` environment variables.
        filesystems: Comma-separated or colon-separated filesystem list.
        job_name: Job name (defaults to ``"ezpz"``).
        working_dir: Directory to ``cd`` into (defaults to cwd).
        env_setup: Shell commands for environment activation.
        wrap_with_launch: If ``True``, prefix *command* with ``ezpz launch``.

    Returns:
        The complete job script as a string.
    """
    account = account or os.environ.get(
        "PBS_ACCOUNT", os.environ.get("PROJECT", "")
    )
    job_name = job_name or "ezpz"
    working_dir = working_dir or os.getcwd()
    if env_setup is None:
        env_setup = detect_env_setup()

    fs = filesystems.replace(",", ":")

    lines = [
        "#!/bin/bash --login",
        f"#PBS -l select={nodes}",
        f"#PBS -l walltime={time}",
        f"#PBS -l filesystems={fs}",
    ]
    if account:
        lines.append(f"#PBS -A {account}")
    lines += [
        f"#PBS -q {queue}",
        f"#PBS -N {job_name}",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(working_dir)}",
    ]
    if env_setup:
        lines += ["", "# ── Environment setup ──", env_setup]

    run_cmd = f"ezpz launch {command}" if wrap_with_launch else command
    lines += ["", run_cmd, ""]
    return "\n".join(lines)


def generate_slurm_script(
    command: str,
    *,
    nodes: int = 1,
    time: str = "01:00:00",
    queue: str = "debug",
    account: str | None = None,
    job_name: str | None = None,
    working_dir: str | None = None,
    env_setup: str | None = None,
    wrap_with_launch: bool = True,
) -> str:
    """Generate a SLURM job script.

    Args:
        command: The command to execute (already formatted as a string).
        nodes: Number of compute nodes.
        time: Walltime in ``HH:MM:SS`` format.
        queue: SLURM partition name.
        account: SLURM account.  Falls back to ``SLURM_ACCOUNT`` or
            ``PROJECT`` environment variables.
        job_name: Job name (defaults to ``"ezpz"``).
        working_dir: Directory to ``cd`` into (defaults to cwd).
        env_setup: Shell commands for environment activation.
        wrap_with_launch: If ``True``, prefix *command* with ``ezpz launch``.

    Returns:
        The complete job script as a string.
    """
    account = account or os.environ.get(
        "SLURM_ACCOUNT", os.environ.get("PROJECT", "")
    )
    job_name = job_name or "ezpz"
    working_dir = working_dir or os.getcwd()
    if env_setup is None:
        env_setup = detect_env_setup()

    lines = [
        "#!/bin/bash --login",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --time={time}",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines += [
        f"#SBATCH --partition={queue}",
        f"#SBATCH --job-name={job_name}",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(working_dir)}",
    ]
    if env_setup:
        lines += ["", "# ── Environment setup ──", env_setup]

    run_cmd = f"ezpz launch {command}" if wrap_with_launch else command
    lines += ["", run_cmd, ""]
    return "\n".join(lines)


# ── Submission ───────────────────────────────────────────────────────────────


def submit_job(script_path: str | Path, scheduler: str) -> str | None:
    """Submit a job script and return the job ID.

    Args:
        script_path: Path to the job script.
        scheduler: ``"PBS"`` or ``"SLURM"``.

    Returns:
        The job ID string, or ``None`` if submission failed.
    """
    script_path = str(script_path)
    if scheduler.upper() == "PBS":
        cmd = ["qsub", script_path]
    elif scheduler.upper() == "SLURM":
        cmd = ["sbatch", script_path]
    else:
        logger.error("Unknown scheduler %r — cannot submit", scheduler)
        return None

    logger.info("Submitting: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip()
        return job_id
    except FileNotFoundError:
        logger.error(
            "%s not found — is the scheduler available on this system?",
            cmd[0],
        )
        return None
    except subprocess.CalledProcessError as exc:
        logger.error("Submission failed (exit %d): %s", exc.returncode, exc.stderr.strip())
        return None


def submit(
    command: list[str] | None = None,
    script: str | Path | None = None,
    *,
    nodes: int = 1,
    time: str = "01:00:00",
    queue: str = "debug",
    account: str | None = None,
    filesystems: str = "home",
    job_name: str | None = None,
    working_dir: str | None = None,
    scheduler: str | None = None,
    wrap_with_launch: bool = True,
    dry_run: bool = False,
) -> str | None:
    """Submit a job to the active scheduler.

    Either *command* (a list of args to wrap) or *script* (path to an
    existing job script) must be provided.

    Args:
        command: Command to wrap in a generated job script.
        script: Path to an existing job script to submit directly.
        nodes: Number of compute nodes.
        time: Walltime in ``HH:MM:SS`` format.
        queue: Queue or partition name.
        account: Project/account for billing.
        filesystems: PBS filesystems directive (ignored for SLURM).
        job_name: Job name.
        working_dir: Working directory for the job.
        scheduler: Override scheduler detection (``"PBS"`` or ``"SLURM"``).
        wrap_with_launch: Wrap *command* with ``ezpz launch``.
        dry_run: Print the script but do not submit.

    Returns:
        The job ID string, or ``None`` on failure / dry-run.
    """
    from ezpz.configs import get_scheduler

    if scheduler is None:
        scheduler = get_scheduler()

    if scheduler.upper() not in ("PBS", "SLURM"):
        print(
            f"No supported scheduler detected (got {scheduler!r}).\n"
            "Set PBS_JOBID, SLURM_JOB_ID, or pass --scheduler explicitly.",
            file=sys.stderr,
        )
        return None

    # ── Mode 2: submit existing script ───────────────────────────────────
    if script is not None:
        script_path = Path(script)
        if not script_path.is_file():
            print(f"Script not found: {script_path}", file=sys.stderr)
            return None
        if dry_run:
            print(script_path.read_text())
            print(f"[dry-run] Would submit {script_path} via {scheduler}")
            return None
        job_id = submit_job(script_path, scheduler)
        if job_id:
            print(f"Submitted job {job_id}")
        return job_id

    # ── Mode 1: generate script from command ─────────────────────────────
    if command is None or not command:
        print("No command or script provided.", file=sys.stderr)
        return None

    cmd_str = shlex.join(command)

    if job_name is None:
        # Derive from command: "python3 -m ezpz.examples.test" → "ezpz.examples.test"
        for i, arg in enumerate(command):
            if arg == "-m" and i + 1 < len(command):
                job_name = command[i + 1]
                break
        if job_name is None:
            job_name = Path(command[0]).stem

    if scheduler.upper() == "PBS":
        script_text = generate_pbs_script(
            cmd_str,
            nodes=nodes,
            time=time,
            queue=queue,
            account=account,
            filesystems=filesystems,
            job_name=job_name,
            working_dir=working_dir,
            wrap_with_launch=wrap_with_launch,
        )
    else:
        script_text = generate_slurm_script(
            cmd_str,
            nodes=nodes,
            time=time,
            queue=queue,
            account=account,
            job_name=job_name,
            working_dir=working_dir,
            wrap_with_launch=wrap_with_launch,
        )

    # Print for transparency
    print("Generated job script:")
    print("-" * 60)
    print(script_text)
    print("-" * 60)

    if dry_run:
        print(f"[dry-run] Would submit via {scheduler.lower()}")
        return None

    # Write to a file so the user can inspect/resubmit later
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(working_dir or os.getcwd())
    script_path = script_dir / f".ezpz-submit-{timestamp}.sh"
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(0o755)

    job_id = submit_job(script_path, scheduler)
    if job_id:
        print(f"Submitted job {job_id}")
        print(f"Script saved to {script_path}")
    return job_id
