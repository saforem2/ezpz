"""
pbs.py

Contains helper functions for working with the PBS Pro scheduler @ ALCF

See:
[docs/pbs.md](https://github.com/saforem2/ezpz/blob/main/docs/pbs.md)
for more information.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, Tuple

import socket

# LCMD1: 1.43 ms +- 8.66 us per loop (mean +- std. dev. of 7 runs, 1,000 loops each)
# LCMD2: 1.35 ms +- 7.27 us per loop (mean +- std. dev. of 7 runs, 1,000 loops each)
from getpass import getuser

import ezpz
from ezpz.dist import get_hostfile_with_fallback

Pathish = Union[str, os.PathLike, Path]

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


def _normalize_hostfile(hostfile: Optional[str | os.PathLike]) -> Path:
    """Return a concrete hostfile path, falling back if needed."""
    if hostfile is not None:
        return Path(hostfile)
    return get_hostfile_with_fallback(None)


def _resolve_topology(
    ngpus: Optional[int],
    nhosts: Optional[int],
    ngpu_per_host: Optional[int],
    *,
    ngpus_max: int,
    nhosts_max: int,
    ngpu_per_host_max: int,
) -> tuple[int, int, int]:
    """Resolve (ngpus, nhosts, ngpu_per_host) from partially specified inputs.

    Rules:
      - If nhosts is not provided, default to nhosts_max.
      - If nothing is specified, use the full machine.
      - Otherwise infer the missing quantity and validate consistency.
    """

    if nhosts is None:
        nhosts = nhosts_max

    # Case 1: nothing specified -> use full machine
    if ngpus is None and ngpu_per_host is None:
        ngpu_per_host = ngpu_per_host_max
        ngpus = nhosts * ngpu_per_host

    # Case 2: only ngpu_per_host specified
    elif ngpus is None and ngpu_per_host is not None:
        ngpus = nhosts * ngpu_per_host

    # Case 3: only ngpus specified
    elif ngpus is not None and ngpu_per_host is None:
        q, r = divmod(ngpus, nhosts)
        if r:
            raise ValueError(
                f"`ngpus` must be divisible by `nhosts`: ngpus={ngpus}, nhosts={nhosts}"
            )
        ngpu_per_host = q

    # Case 4: all specified: just validate
    else:
        if ngpus != nhosts * ngpu_per_host:  # type: ignore[operator]
            raise ValueError(
                "Inconsistent topology: "
                f"{ngpus=} vs {nhosts=} * {ngpu_per_host=}"
            )

    # Final validation
    if ngpus <= 0 or ngpus > ngpus_max:  # type: ignore[operator]
        raise ValueError(f"`ngpus` must be in (0, {ngpus_max}]: got {ngpus}")

    return int(ngpus), int(nhosts), int(ngpu_per_host)  # type: ignore[arg-type]


def _infer_topology(
    *,
    ngpus: Optional[int],
    nhosts: Optional[int],
    ngpu_per_host: Optional[int],
    hostfile: Path,
) -> Tuple[int, int, int]:
    """
    Infer (ngpus, nhosts, ngpu_per_host) from user inputs and machine limits.

    Rules:
      - If nothing is specified, use all resources.
      - If combinations are inconsistent (e.g. ngpus not divisible by nhosts),
        raise ValueError instead of relying on `assert`.
    """
    ngpus_max = ezpz.get_world_size(total=True)
    ngpu_per_host_max = ezpz.get_gpus_per_node()
    nhosts_max = ezpz.get_num_nodes(hostfile=hostfile)

    if nhosts is not None and not (0 < nhosts <= nhosts_max):
        raise ValueError(
            f"`nhosts` must be > 0 and <= {nhosts_max}, got {nhosts}."
        )

    if nhosts is None:
        nhosts = nhosts_max

    # Case A: ngpus not provided -> infer from nhosts / ngpu_per_host.
    if ngpus is None:
        assert nhosts is not None
        if ngpu_per_host is None:
            # Use all GPUs on all hosts
            ngpu_per_host = ngpu_per_host_max
            assert ngpu_per_host is not None
            ngpus = nhosts * ngpu_per_host
        else:
            # ngpu_per_host given, nhosts known (we just set it)
            ngpus = nhosts * ngpu_per_host
    else:
        assert ngpus is not None
        # Case B: ngpus provided
        if ngpu_per_host is None:
            # Deduce ngpu_per_host from ngpus and nhosts
            assert nhosts is not None
            if ngpus % nhosts != 0:
                raise ValueError(
                    f"`ngpus` must be divisible by `nhosts`: ngpus={ngpus}, nhosts={nhosts}"
                )
            ngpu_per_host = ngpus // nhosts
        else:
            # All three specified: check consistency
            assert nhosts is not None
            if ngpus != nhosts * ngpu_per_host:
                raise ValueError(
                    "Mismatch in `ngpus` and `nhosts * ngpu_per_host`: "
                    f"ngpus={ngpus}, nhosts={nhosts}, ngpu_per_host={ngpu_per_host}"
                )

    if not (0 < ngpus <= ngpus_max):
        raise ValueError(
            f"`ngpus` must be > 0 and <= {ngpus_max}, got {ngpus}."
        )

    return ngpus, nhosts, ngpu_per_host


def _maybe_add_cpu_bind(
    cmd: list[str], ngpus: int, machine_name: str
) -> list[str]:
    """Add CPU binding flags to cmd in-place based on env and machine."""
    cpu_bind = os.environ.get("CPU_BIND")
    if cpu_bind:
        logger.warning(f"Detected CPU_BIND from environment: {cpu_bind}")
        cpu_bind_val = cpu_bind.replace("--cpu-bind=", "")
        if ngpus < 1024:
            cmd.append(f"--cpu-bind=verbose,{cpu_bind_val}")
        else:
            cmd.append(f"--cpu-bind={cpu_bind_val}")
        return cmd

    # No explicit CPU_BIND -> set sensible defaults by machine
    machine_name_l = machine_name.lower()
    if machine_name_l in {"aurora", "sunspot"}:
        cpu_bind_intel_xpu = (
            "list:2-4:10-12:18-20:26-28:"
            "34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
        )
        if ngpus < 1024:
            cmd.extend(
                ["--no-vni", f"--cpu-bind=verbose,{cpu_bind_intel_xpu}"]
            )
        else:
            cmd.extend(["--no-vni", f"--cpu-bind={cpu_bind_intel_xpu}"])
    else:
        cmd.extend(["--cpu-bind=depth", "--depth=8"])

    return cmd


def get_pbs_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[Pathish] = None,
    *,
    verbose: bool = False,
) -> str:
    """Get the PBS launch command.

    Parameters
    ----------
    ngpus : int, optional
        Total number of GPUs to use. If None, inferred from other args or max.
    nhosts : int, optional
        Number of hosts (nodes). If None, defaults to max available.
    ngpu_per_host : int, optional
        GPUs per host. If None, inferred when possible.
    hostfile : path-like, optional
        Hostfile to use. If None, uses a fallback from `get_hostfile_with_fallback`.
    verbose : bool, keyword-only
        If True, log more and pass `--verbose` (where applicable).
    """

    ngpus_max = ezpz.get_world_size(total=True)
    ngpu_per_host_max = ezpz.get_gpus_per_node()

    hostfile = hostfile or get_hostfile_with_fallback(hostfile)
    hostfile_path = Path(hostfile)
    nhosts_max = ezpz.get_num_nodes(hostfile=hostfile_path)

    ngpus, nhosts, ngpu_per_host = _resolve_topology(
        ngpus=ngpus,
        nhosts=nhosts,
        ngpu_per_host=ngpu_per_host,
        ngpus_max=ngpus_max,
        nhosts_max=nhosts_max,
        ngpu_per_host_max=ngpu_per_host_max,
    )

    emoji = "✅" if ngpus == ngpus_max else "⚠️"
    logger.info(
        f"{emoji} Using [{ngpus}/{ngpus_max}] GPUs "
        f"[{nhosts} hosts] x [{ngpu_per_host} GPU/host]"
    )

    if ngpus != ngpus_max:
        logger.warning(
            f"[🚧 WARNING] Using only [{ngpus}/{ngpus_max}] available GPUs!!"
        )

    machine_name = ezpz.get_machine().lower()
    hostfile_str = str(hostfile_path)

    if machine_name == "sophia":
        # mpirun style
        cmd_list = [
            "mpirun",
            f"-n={ngpus}",
            f"-N={ngpu_per_host}",
            f"--hostfile={hostfile_str}",
            "-x PATH",
            "-x LD_LIBRARY_PATH",
        ]
        if verbose:
            cmd_list.append("--verbose")
        return " ".join(cmd_list)

    # Default: mpiexec
    cmd_list = [
        "mpiexec",
        "--envall",
        f"--np={ngpus}",
        f"--ppn={ngpu_per_host}",
        f"--hostfile={hostfile_str}",
    ]
    if verbose:
        cmd_list.append("--verbose")

    cpu_bind_env = os.environ.get("CPU_BIND")
    use_verbose_cpu_bind = ngpus < 1024
    cpu_bind_prefix = (
        "--cpu-bind=verbose," if use_verbose_cpu_bind else "--cpu-bind="
    )

    if cpu_bind_env:
        logger.warning(f"Detected CPU_BIND from environment: {cpu_bind_env}")
        bind_value = cpu_bind_env.replace("--cpu-bind=", "")
        cmd_list.append(f"{cpu_bind_prefix}{bind_value}")
    else:
        is_intel_xpu_machine = machine_name in {"aurora", "sunspot"}
        if is_intel_xpu_machine:
            CPU_BIND_INTEL_XPU = (
                "list:2-4:10-12:18-20:26-28:34-36:42-44:"
                "54-56:62-64:70-72:78-80:86-88:94-96"
            )
            cmd_list.extend(
                [
                    "--no-vni",
                    f"{cpu_bind_prefix}{CPU_BIND_INTEL_XPU}",
                ]
            )
        else:
            # generic CPU binding
            cmd_list.extend(["--cpu-bind=depth", "--depth=8"])

    return " ".join(cmd_list)


def get_pbs_launch_cmd1(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike] = None,
    verbose: bool = False,
) -> str:
    """
    Construct the PBS launch command for the current machine.

    Parameters
    ----------
    ngpus : int, optional
        Total number of GPUs to use. If None, inferred from nhosts/ngpu_per_host.
    nhosts : int, optional
        Number of hosts to use. If None, defaults to all available hosts.
    ngpu_per_host : int, optional
        GPUs per host. If None, inferred from ngpus/nhosts or machine defaults.
    hostfile : path-like, optional
        PBS hostfile to use. If None, a reasonable fallback is inferred.
    verbose : bool, optional
        If True, add verbose flags where available.
    """
    hostfile_path = _normalize_hostfile(hostfile)

    ngpus_inf, nhosts_inf, ngpu_per_host_inf = _infer_topology(
        ngpus=ngpus,
        nhosts=nhosts,
        ngpu_per_host=ngpu_per_host,
        hostfile=hostfile_path,
    )

    ngpus_max = ezpz.get_world_size(total=True)
    emoji = "✅" if ngpus_inf == ngpus_max else "⚠️"
    logger.info(
        f"{emoji} Using [{ngpus_inf}/{ngpus_max}] GPUs "
        f"[{nhosts_inf} hosts] x [{ngpu_per_host_inf} GPU/host]"
    )
    if ngpus_inf != ngpus_max:
        logger.warning(
            f"[🚧 WARNING] Using only [{ngpus_inf}/{ngpus_max}] available GPUs!!"
        )

    machine_name = ezpz.get_machine()
    hostfile_str = str(hostfile_path)

    if machine_name.lower() == "sophia":
        # PBS + OpenMPI
        cmd_list = [
            "mpirun",
            f"-n={ngpus_inf}",
            f"-N={ngpu_per_host_inf}",
            f"--hostfile={hostfile_str}",
            "-x PATH",
            "-x LD_LIBRARY_PATH",
        ]
        if verbose:
            cmd_list.insert(1, "--verbose")
    else:
        # PBS + MPICH
        cmd_list = [
            "mpiexec",
            "--envall",
            f"--np={ngpus_inf}",
            f"--ppn={ngpu_per_host_inf}",
            f"--hostfile={hostfile_str}",
        ]
        if verbose:
            cmd_list.insert(1, "--verbose")

        cmd_list = _maybe_add_cpu_bind(cmd_list, ngpus_inf, machine_name)

    return " ".join(cmd_list)


def get_pbs_launch_cmd_deprecated(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
    verbose: Optional[bool] = False,
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
            assert ngpus is not None
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
        hostfile = os.environ.get("PBS_NODEFILE")
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
