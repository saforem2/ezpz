"""
ezpz/slurm.py
"""

import os
from pathlib import Path

import ezpz

logger = ezpz.get_logger(__name__)


"""
1. Get all running jobs for user.
2. For each running job:
   1. Get the list of nodes assigned to that job
   2. Check if $(hostname) in list of nodes
   3. If so, I belong to that jobid.

1. For a given host need to identify:
"""


def get_slurm_running_jobs() -> list[int] | None:
    """Get the running jobs from sacct."""
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
    """Get the nodelist for a given jobid."""
    try:
        from sh import scontrol  # type:ignore
    except Exception as e:
        logger.error("Error importing sh.scontrol: %s", e)
        raise e
    try:
        output = scontrol("show", "job", str(jobid)).split("\n")
        # Find the line containing NodeList=
        node_line = next((i for i in output if " NodeList=" in i), None)
        if not node_line:
            raise ValueError("NodeList not found in scontrol output")
        # Extract the NodeList value
        import re

        match = re.search(r"NodeList=([^\s]+)", node_line)
        if not match:
            raise ValueError("NodeList value not found")
        nodelist_str = match.group(1)
        nodelist = (
            nodelist_str.replace("[", "")
            .replace("]", "")
            .replace("-", ",nid")
            .split(",")
        )
        return nodelist
    except Exception as e:
        logger.error(f"Error getting nodelist for job {jobid}: {e}")

        return []


def get_slurm_running_jobs_for_user() -> dict[int, list[str]]:
    """Get all running jobs for the current user."""
    running_jobs = get_slurm_running_jobs()
    jobs = {}
    if running_jobs is not None:
        for job in running_jobs:
            jobs[job] = get_nodelist_from_slurm_jobid(job)
    return jobs


def get_slurm_jobid_of_active_job() -> str | None:
    """Get the jobid of the currently active job."""
    import socket

    hostname = socket.getfqdn().split("-")[0]
    running_jobs = get_slurm_running_jobs_for_user()
    for jobid, nodelist in running_jobs.items():
        logger.info(f"Checking jobid {jobid} for hostname {hostname}...")
        if hostname in nodelist:
            logger.info(f"Found {hostname} in nodelist for {jobid}")
            return str(jobid)
    return None


def get_slurm_nodefile_from_jobid(jobid: int | str) -> str:
    """Get the nodefile for a given jobid.

    Args:
        jobid (int | str): The job ID to get the nodefile for.

    Returns:
        str: The path to the nodefile.
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


def build_launch_cmd() -> str:
    """Build command to launch a job on SLURM."""
    running_jobid = get_slurm_jobid_of_active_job()
    if running_jobid is not None:
        nodelist = get_nodelist_from_slurm_jobid(running_jobid)
        if nodelist is not None and len(nodelist) > 0:
            num_nodes = len(nodelist)
            num_gpus_per_node = ezpz.get_gpus_per_node()
            total_gpus = num_nodes * num_gpus_per_node
            cmd_to_launch = f"srun -u --verbose -N{num_nodes} -n{total_gpus}"
            return cmd_to_launch
        else:
            raise ValueError(f"No nodelist found for jobid {running_jobid}")
    else:
        raise ValueError("No running SLURM job found for current user.")
