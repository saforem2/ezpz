"""
pbs.py

Contains helper functions for working with the PBS Pro scheduler @ ALCF

See:
[docs/pbs.md](https://github.com/saforem2/ezpz/blob/main/docs/pbs.md)
for more information.
"""

import os
from getpass import getuser
from pathlib import Path
from typing import Optional, Union

import ezpz

import socket

from ezpz.dist import get_hostfile_with_fallback


logger = ezpz.get_logger(__name__)


def get_pbs_running_jobs_for_user():
    """Get all running jobs for the current user."""
    try:
        from sh import qstat  # type:ignore
    except Exception as e:
        print("Error importing sh.qstat:", e)
        raise e

    jobarr = [
        i for i in qstat(f"-fn1wru {getuser()}").split("\n") if " R " in i
    ]
    jobs = {}
    for row in jobarr:
        jstr = [i for i in row.split(" ") if len(i) > 0]
        jobid = jstr[0].split(".")[0]
        nodelist = [h.split("/")[0] for h in jstr[-1].split("+")]
        jobs[jobid] = nodelist

    return jobs


def get_pbs_nodelist_from_jobid(jobid: int | str) -> list[str]:
    """Get the nodelist for a given jobid."""
    assert jobid is not None, "No jobid provided and no active job found."
    jobs = get_pbs_running_jobs_for_user()
    assert str(jobid) in jobs, (
        f"Job ID {jobid} not found in running jobs for user {getuser()}"
    )
    return jobs[str(jobid)]


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
    return None if jobid is None else get_pbs_nodefile_from_jobid(jobid)


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


def get_pbs_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
) -> str:
    """Get the PBS launch command"""

    # ngpus_available = (
    #     ezpz.get_world_size(total=True) if ngpus is None else ngpus
    # )
    ngpus_max = ezpz.get_world_size(total=True)
    ngpu_per_host_max = ezpz.get_gpus_per_node()
    hostfile_fallback = get_hostfile_with_fallback(hostfile)
    hostfile = hostfile or hostfile_fallback
    nhosts_max = ezpz.get_num_nodes(hostfile=Path(hostfile_fallback))
    if nhosts is not None:
        assert 0 < nhosts <= nhosts_max, (
            f"`nhosts` must be > 0 and <= {nhosts_max}: {nhosts=}"
        )
    else:
        nhosts = nhosts_max

    #  ---- [`ngpus`, `nhosts`, `ngpu_per_host` logic] -------------------------
    #  ```bash
    # 1. [ ❌, ✅, ❌ ] -> [ ngpus_max, nhosts_max, ngpu_per_host_max ]
    # 2. [ ❌, ✅, ✅ ] -> [ ngpu_per_host * nhosts_max, nhosts_max, ngpu_per_host ]
    # 3. [ ❌, ✅, ❌ ] -> [ ngpus_max, nhosts, ngpus_max // nhosts ]
    # 4. [ ✅, ✅, ❌ ] -> [ ngpus, ngpus // ngpu_per_host_max, ngpu_per_host_max ]
    # 5. [ ❌, ✅, ✅ ] -> [ ngpu_per_host * nhosts, nhosts, ngpu_per_host ]
    # 6. [ ✅, ✅, ✅ ] -> [ ngpus, ngpus // ngpu_per_host, ngpu_per_host ]
    # 7. [ ✅, ✅, ❌ ] -> [ ngpus, nhosts, ngpus // nhosts ]
    # 8. [ ✅, ✅, ✅ ] -> [ ngpus, nhosts, ngpu_per_host ]
    # ```
    #
    if ngpus is None:
        if ngpu_per_host is None and nhosts is not None:
            # [3.] [ ❌, ✅, ❌ ]
            ngpus = nhosts * ngpu_per_host_max
            ngpu_per_host = ngpus // nhosts
        else:
            # [5.] [ ❌, ✅, ✅ ]
            assert nhosts is not None and ngpu_per_host is not None
            ngpus = nhosts * ngpu_per_host
    else:
        if nhosts is not None and ngpu_per_host is None:
            # [7.] [ ✅, ✅, ❌ ]
            assert ngpus % nhosts == 0, (
                f"`ngpus` must be divisible by `nhosts`: {ngpus=} vs. {nhosts=}"
            )
            ngpu_per_host = ngpus // nhosts
        else:
            # [8.] [ ✅, ✅, ✅ ]
            assert nhosts is not None and ngpu_per_host is not None
            assert ngpus == (nhosts * ngpu_per_host), (
                f"Mismatch in `ngpus` and `nhosts * ngpu_per_host`: "
                f"{ngpus=} vs. {nhosts=} * {ngpu_per_host=}"
            )

    assert 0 < ngpus <= ngpus_max, (
        f"`ngpus` must be > 0 and <= {ngpus_max}: {ngpus=}"
    )
    emoji = "✅" if ngpus == ngpus_max else "⚠️"
    logger.info(
        f"{emoji} Using [{ngpus:>}/{ngpus_max:<}] GPUs [{nhosts} hosts] x [{ngpu_per_host} GPU/host]"
    )
    if ngpus != ngpus_max:
        logger.warning(
            f"[🚧 WARNING] Using only [{ngpus:>}/{ngpus_max:<}] available GPUs!!"
        )
    if ezpz.get_machine().lower() == "sophia":
        return " ".join(
            [
                "mpirun",
                f"-n={ngpus}",
                f"-N={ngpu_per_host}",
                f"--hostfile={hostfile}",
                "-x PATH",
                "-x LD_LIBRARY_PATH",
            ]
        )
    else:
        cmd_list = [
            "mpiexec",
            "--verbose",
            "--envall",
            f"--np={ngpus}",
            f"--ppn={ngpu_per_host}",
            f"--hostfile={hostfile}",
        ]
        machine_name = ezpz.get_machine()
        cpu_bind = os.environ.get("CPU_BIND", None)
        if cpu_bind is not None:
            logger.warning(f"Detected CPU_BIND from environment: {cpu_bind}")
            if ngpus < 1024:
                cmd_list.append(
                    f"--cpu-bind=verbose,{cpu_bind.replace('--cpu-bind=', '')}"
                )
            else:
                cmd_list.append(
                    f"--cpu-bind={cpu_bind.replace('--cpu-bind=', '')}"
                )
        else:
            if machine_name.lower() in {"aurora", "sunspot"}:
                CPU_BIND_INTEL_XPU: str = "list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
                if ngpus < 1024:
                    cmd_list.extend(
                        [
                            "--no-vni",
                            f"--cpu-bind=verbose,{CPU_BIND_INTEL_XPU}",
                        ]
                    )
                else:
                    cmd_list.extend(
                        [
                            "--no-vni",
                            f"--cpu-bind={CPU_BIND_INTEL_XPU}",
                        ]
                    )

            else:
                cmd_list.extend(
                    [
                        "--cpu-bind=depth",
                        "--depth=8",
                    ]
                )

    return " ".join(cmd_list)


def get_running_jobs_from_qstat() -> list[int]:
    """Get the running jobs from qstat"""
    try:
        from sh import qstat as shqstat  # type: ignore
    except Exception as e:
        raise e
    return [
        int(i.split(".")[0])
        for i in shqstat("-u", os.environ.get("USER")).split("\n")[2:-1]
        if " R " in i
    ]


def get_pbs_launch_info(
    hostfile: Optional[str | Path] = None,
    jobid: Optional[int | str] = None,
) -> dict[str, str]:
    """Get the PBS launch info"""
    from ezpz.configs import get_scheduler

    assert get_scheduler() == "PBS"
    if hostfile is None:
        hostfile = get_pbs_nodefile(jobid=jobid)
    assert hostfile is not None
    hfp = Path(hostfile)
    hosts = ezpz.dist.get_nodes_from_hostfile(hfp)
    hosts = [h.split(".")[0] for h in hosts]
    nhosts = len(hosts)
    ngpu_per_host = ezpz.dist.get_gpus_per_node()
    ngpus_available = ezpz.dist.get_world_size(total=True)
    ngpus = nhosts * ngpu_per_host
    world_size_total = ezpz.dist.get_world_size_total()
    launch_cmd = get_pbs_launch_cmd(hostfile=hostfile)
    return {
        "HOSTFILE": hfp.as_posix(),
        "HOSTS": (
            f"[{', '.join(hosts)}]"
            if nhosts < 1000
            else "[truncated (>1000 nodes)]"
        ),
        "NHOSTS": f"{nhosts}",
        "NGPU_PER_HOST": f"{ngpu_per_host}",
        "NGPUS": f"{ngpus}",
        "NGPUS_AVAILABLE": f"{ngpus_available}",
        "MACHINE": ezpz.dist.get_machine(),
        "DEVICE": ezpz.dist.get_torch_device_type(),
        "BACKEND": ezpz.dist.get_torch_backend(),
        "LAUNCH_CMD": launch_cmd,
        "world_size_total": f"{world_size_total}",
    }


def get_pbs_env(
    hostfile: Optional[Union[str, Path]] = None,
    jobid: Optional[Union[int, str]] = None,
    verbose: Optional[bool] = None,
) -> dict[str, str]:
    """Get the PBS environment variables"""
    from ezpz.configs import get_scheduler

    assert get_scheduler() == "PBS"
    pbsenv = {k: v for k, v in dict(os.environ).items() if "PBS" in k}
    if hostfile is None:
        hostfile = get_pbs_nodefile(jobid=jobid)

    assert hostfile is not None
    if (hfp := Path(hostfile)).is_file():
        pbsenv |= {
            f"{k.upper()}": f"{v}" for k, v in get_pbs_launch_info(hfp).items()
        }
        pbsenv |= {"LAUNCH_CMD": get_pbs_launch_cmd(hostfile=hostfile)}
    os.environ |= pbsenv
    if verbose and ezpz.dist.get_rank() == 0:
        ezpz.dist.log_dict_as_bulleted_list(pbsenv, name="pbsenv")
    return pbsenv


def build_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[Union[str, Path, os.PathLike]] = None,
) -> str:
    """Build the launch command for the current job.

    Returns:
        str: The launch command.
    """
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        return get_pbs_launch_cmd(
            ngpus=ngpus,
            nhosts=nhosts,
            ngpu_per_host=ngpu_per_host,
            hostfile=hostfile,
        )

    elif scheduler == "slurm":
        import ezpz.slurm

        return ezpz.slurm.build_launch_cmd()
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")


if __name__ == "__main__":
    jobenv = ezpz.get_jobenv()
    os.system(jobenv["LAUNCH_CMD"])
