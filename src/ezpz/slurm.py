"""
ezpz/slurm.py
"""

import os
import ezpz


logger = ezpz.get_logger(__name__)


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
            job_nodes = [
                i.replace(" ", "")
                .replace("NodeList=", "")
                .replace("[", "")
                .replace("-", "\nnid")
                .replace("]", "")
                .split("\n")
                for i in scontrol("show", "job", f"{running_jobs[0]}").split(
                    "\n"
                )
                if " NodeList=" in i
            ][0]
            jobs[job] = job_nodes
    return jobs
