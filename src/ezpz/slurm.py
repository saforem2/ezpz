"""
ezpz/slurm.py

SLURM scheduler utilities: job discovery, nodefile generation, and launch
command construction.

Prefers ``SLURM_JOB_ID`` / ``SLURM_NODELIST`` environment variables (always
set inside a SLURM allocation) and falls back to ``sacct`` / ``scontrol``
shell commands only when the env vars are absent (e.g. on a login node).
"""

import os
import re
from pathlib import Path
from typing import Optional, Union

import ezpz
from ezpz.distributed import _expand_slurm_nodelist

logger = ezpz.get_logger(__name__)


# ── job discovery ──────────────────────────────────────────────────────────


def get_slurm_running_jobs() -> list[str] | None:
    """Get the running jobs from ``sacct``.

    Returns a deduplicated list of base job IDs (strings) that are
    currently in ``RUNNING`` state, or ``None`` if ``sacct`` is
    unavailable.
    """
    try:
        from sh import sacct  # type:ignore

        return list(
            {
                i.replace(".", " ").split(" ")[0]
                for i in [j for j in sacct().split("\n") if " RUNNING " in j]
            }
        )
    except Exception as e:
        logger.error("Error getting running jobs from sacct: %s", e)
        raise e


def get_nodelist_from_slurm_jobid(jobid: str | int) -> list[str]:
    """Get the expanded nodelist for *jobid*.

    Checks ``SLURM_NODELIST`` first (instant, no subprocess).  Falls back
    to ``scontrol show job <jobid>`` when the env var is absent.
    """
    # Fast path: env var is set inside every SLURM allocation.
    nodelist_str = os.environ.get("SLURM_NODELIST")
    if nodelist_str:
        return _expand_slurm_nodelist(nodelist_str)

    # Slow path: query scontrol for an arbitrary job ID.
    try:
        from sh import scontrol  # type:ignore
    except Exception as e:
        logger.error("Error importing sh.scontrol: %s", e)
        raise e
    try:
        output = scontrol("show", "job", str(jobid)).split("\n")
        node_line = next((i for i in output if " NodeList=" in i), None)
        if not node_line:
            raise ValueError("NodeList not found in scontrol output")
        match = re.search(r"NodeList=([^\s]+)", node_line)
        if not match:
            raise ValueError("NodeList value not found")
        return _expand_slurm_nodelist(match.group(1))
    except Exception as e:
        logger.error(f"Error getting nodelist for job {jobid}: {e}")
        raise e


def get_slurm_running_jobs_for_user() -> dict[str, list[str]]:
    """Get all running jobs for the current user.

    Returns a dict mapping job-ID strings to their expanded nodelists.
    """
    running_jobs = get_slurm_running_jobs()
    jobs = {}
    if running_jobs is not None:
        for job in running_jobs:
            jobs[job] = get_nodelist_from_slurm_jobid(job)
    return jobs


def get_slurm_jobid_of_active_job() -> str | None:
    """Get the job ID of the currently active SLURM job.

    Checks ``SLURM_JOB_ID`` / ``SLURM_JOBID`` env vars first (instant).
    Falls back to ``sacct`` + hostname matching only when the env vars
    are absent.
    """
    # Fast path: env var is always set inside a SLURM allocation.
    jobid = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if jobid:
        return str(jobid)

    # Slow path: query sacct and match hostname.
    import socket

    hostname = socket.getfqdn().split("-")[0]
    running_jobs = get_slurm_running_jobs_for_user()
    for jid, nodelist in running_jobs.items():
        logger.info(f"Checking jobid {jid} for hostname {hostname}...")
        if hostname in nodelist:
            logger.info(f"Found {hostname} in nodelist for {jid}")
            return str(jid)
    return None


# ── nodefile generation ────────────────────────────────────────────────────


def get_slurm_nodefile_from_jobid(jobid: int | str) -> str:
    """Write a nodefile for *jobid* and return its path.

    The file is written to ``./nodefile-<jobid>`` in the current
    working directory with one hostname per line.
    """
    assert jobid is not None, "Job ID must be provided."
    nodelist = get_nodelist_from_slurm_jobid(jobid)
    nodefile = Path(os.getcwd()).joinpath(f"nodefile-{jobid}")
    logger.info(f"Writing {nodelist} to {nodefile}")
    with nodefile.open("w") as f:
        for hn in nodelist:
            f.write(f"{hn}\n")

    return nodefile.absolute().resolve().as_posix()


def get_slurm_nodefile_of_active_job() -> str | None:
    """Get the nodefile of the currently active job."""
    jobid = get_slurm_jobid_of_active_job()
    if jobid is not None:
        return get_slurm_nodefile_from_jobid(jobid)
    return None


# ── launch command ─────────────────────────────────────────────────────────


def build_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[Union[str, Path, os.PathLike]] = None,
    cpu_bind: Optional[str] = None,
) -> str:
    """Build an ``srun`` command to launch a distributed job on SLURM.

    Resolution order for node count:
      1. Explicit *nhosts* argument.
      2. Line count from *hostfile*.
      3. ``SLURM_NNODES`` env var.
      4. Active job discovery via ``sacct`` / ``scontrol`` (slow fallback).

    Parameters
    ----------
    ngpus : int, optional
        Total number of tasks (GPUs). If ``None``, computed as
        ``nhosts * ngpu_per_host``.
    nhosts : int, optional
        Number of nodes. If ``None``, inferred from *hostfile*, env vars,
        or the active job.
    ngpu_per_host : int, optional
        GPUs per node. If ``None``, detected via
        ``ezpz.get_gpus_per_node()``.
    hostfile : path-like, optional
        Path to a hostfile (one hostname per line).
    cpu_bind : str, optional
        Accepted for interface parity with the PBS launcher; not currently
        appended to the ``srun`` invocation.
    """
    if ngpu_per_host is None:
        ngpu_per_host = ezpz.get_gpus_per_node()

    if nhosts is not None:
        num_nodes = nhosts
    elif hostfile is not None and Path(hostfile).is_file():
        with open(hostfile) as f:
            num_nodes = len([ln for ln in f if ln.strip()])
    else:
        # Try SLURM_NNODES env var before expensive sacct/scontrol calls.
        slurm_nnodes = os.environ.get("SLURM_NNODES")
        if slurm_nnodes is not None:
            num_nodes = int(slurm_nnodes)
        else:
            running_jobid = get_slurm_jobid_of_active_job()
            if running_jobid is None:
                raise ValueError(
                    "No running SLURM job found for current user."
                )
            nodelist = get_nodelist_from_slurm_jobid(running_jobid)
            if not nodelist:
                raise ValueError(
                    f"No nodelist found for jobid {running_jobid}"
                )
            num_nodes = len(nodelist)

    total_gpus = ngpus if ngpus is not None else num_nodes * ngpu_per_host

    return f"srun -u --verbose -N{num_nodes} -n{total_gpus}"
