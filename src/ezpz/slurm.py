"""
ezpz/slurm.py
"""

import os
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
        nodelist = [
            i.replace(" ", "")
            .replace("NodeList=", "")
            .replace("[", "")
            .replace("-", "\nnid")
            .replace("]", "")
            .split("\n")
            for i in output if "   NodeList=" in i
        ][0]
        return nodelist
    except Exception as e:
        logger.error(f"Error getting nodelist for job {jobid}: {e}")
        return []


def get_slurm_running_jobs_for_user() -> dict[int, list[str]]:
    """Get all running jobs for the current user."""
    try:
        from sh import scontrol  # type:ignore
    except Exception as e:
        print("Error importing sh.squeue:", e)
        raise e

    running_jobs = get_slurm_running_jobs()
    jobs = {}
    if running_jobs is not None:
        for job in running_jobs:
            job_nodes = get_nodelist_from_slurm_jobid(job)
            # job_nodes = [
            #     i.replace(" ", "")
            #     .replace("NodeList=", "")
            #     .replace("[", "")
            #     .replace("-", "\nnid")
            #     .replace("]", "")
            #     .split("\n")
            #     # for i in scontrol("show", "job", f"{running_jobs[0]}").split(
            #     for i in scontrol("show", "job", f"{job}").split(
            #         "\n"
            #     )
            #     if "   NodeList=" in i
            #     # if " NodeList=" in i
            # ][0]
            jobs[job] = job_nodes
    return jobs


def get_slurm_jobid_of_active_job() -> str | None:
    """Get the jobid of the currently active job."""
    import socket
    hostname = socket.getfqdn().split('_')[0]
    running_jobs = get_slurm_running_jobs_for_user()
    for jobid, nodelist in running_jobs.items():
        if hostname in nodelist:
            logger.info(f'Found {hostname} in nodelist for {jobid}')
            return str(jobid)

def get_slurm_nodefile_from_jobid(jobid: int | str) -> str:
    """Get the nodefile for a given jobid.

    Args:
        jobid (int | str): The job ID to get the nodefile for.

    Returns:
        str: The path to the nodefile.
    """
    assert jobid is not None, "Job ID must be provided."


