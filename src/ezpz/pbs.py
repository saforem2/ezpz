"""
pbs.py

Contains helper functions for working with the PBS Pro scheduler @ ALCF

```markdown

## Determine Details of Currently Active Job


1. Find all currently running[^semantics] jobs owned by the user.
2. For each of these running jobs, build a dictionary of the form:

    ```python
    >>> jobs = ezpz.pbs.get_users_running_pbs_jobs()
    >>> jobs
    {
        jobid_A: [host_A0, host_A1, host_A2, ..., host_AN],
        jobid_B: [host_B0, host_B1, host_B2, ..., host_BN],
        ...,
    }
    ```

3. Look for _our_ hostname in the list of hosts for each job.
   - If found, we know we are participating in that job.

4. Once we have the `PBS_JOBID` of the job containing our hostname,
   we can find the hostfile for that job.
   - The hostfile is located in `/var/spool/pbs/aux/`.
   - The filename is of the form `jobid.hostname`.

5. ✅ Done!

   Example:

   ```python
   jobid = ezpz.pbs.get_pbs_jobid_of_active_job()
   num_nodes = len(jobs[jobid])
   world_size = num_nodes * ezpz.get_gpus_per_node()
   ```


[^semantics]: |
    Semantics:

    - **Running**: Can have _multiple_ PBS jobs running at the same time
    - **Active**: Can only have _one_ active PBS job at a time
        - This is the job that we are **currently running on**
```
"""

import os
from pathlib import Path
from typing import Optional
import ezpz

import sys
import socket


logger = ezpz.get_logger(__name__)


def get_pbs_running_jobs_for_user():
    """Get all running jobs for the current user."""
    try:
        from sh import qstat  # type:ignore
    except Exception as e:
        print("Error importing sh.qstat:", e)
        raise e

    jobarr = [
        i for i in qstat(f"-fn1wru {os.getlogin()}").split("\n") if " R " in i
    ]
    jobs = {}
    for row in jobarr:
        jstr = [i for i in row.split(" ") if len(i) > 0]
        jobid = jstr[0].split(".")[0]
        nodelist = [h.split("/")[0] for h in jstr[-1].split("+")]
        jobs[jobid] = nodelist

    return jobs


def get_pbs_jobid_of_active_job() -> str | None:
    """Get the jobid of the currently active job."""
    # 1. Find all of users' currently running jobs
    #
    #    ```python
    #    jobs = {
    #        jobid_A: [host_A0, host_A1, host_A2, ..., host_AN],
    #        jobid_B: [host_B0, host_B0, host_B2, ..., host_BN],
    #        ...,
    #    }
    #    ```
    #
    # 2. Loop over {jobid, [hosts]} dictionary.
    #    At each iteration, look and see if _our_ `hostname` is anywhere in the `[hosts]` list.
    #    If so, then we know that we are currently participating in the `jobid` of that entry.
    jobs = get_pbs_running_jobs_for_user()
    for jobid, nodelist in jobs.items():
        # NOTE:
        # - `socket.fqdn()` (fully qualified domain name):
        #   - This will be of the form `x[0-9]+.cm.aurora.alcf.anl.gov`
        #     We only need the part before the first '.'
        if socket.getfqdn().split(".")[0] in nodelist:
            return jobid
    return None


def get_pbs_nodefile_from_jobid(jobid: int | str) -> str:
    """Get the nodefile for a given jobid."""
    assert jobid is not None, "No jobid provided and no active job found."

    pbs_parent = Path("/var/spool/pbs/aux")
    pfiles = [
        Path(pbs_parent).joinpath(f)
        for f in os.listdir(pbs_parent)
        if str(jobid) in f
    ]
    assert len(pfiles) == 1, (
        f"Found {len(pfiles)} files matching {jobid} in {pbs_parent}"
    )
    pbs_nodefile = pfiles[0]
    assert pbs_nodefile.is_file(), f"Nodefile {pbs_nodefile} does not exist."
    return pbs_nodefile.absolute().resolve().as_posix()


def get_pbs_nodefile_of_active_job() -> str | None:
    """Get the nodefile for the currently active job."""
    jobid = get_pbs_jobid_of_active_job()
    if jobid is None:
        return None
    pbs_nodefile = get_pbs_nodefile_from_jobid(jobid)
    return pbs_nodefile


def get_pbs_nodefile(jobid: Optional[int | str] = None) -> str | None:
    """Get the nodefile for a given jobid.

    Args:
        jobid (int | str, optional): The jobid to get the nodefile for. Defaults to None.

    Returns:
        str: The path to the nodefile.
    """
    if jobid is None:
        jobid = get_pbs_jobid_of_active_job()
    if jobid is None:
        logger.warning("No active job found.")
        return None
    return get_pbs_nodefile_from_jobid(jobid)


def build_launch_cmd():
    """Build the launch command for the current job.

    Returns:
        str: The launch command.
    """
    return f"{ezpz.get_jobenv()['LAUNCH_CMD']} {sys.executable}"


if __name__ == "__main__":
    jobenv = ezpz.get_jobenv()
    os.system(jobenv["LAUNCH_CMD"])
