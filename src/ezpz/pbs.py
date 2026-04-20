"""
pbs.py

Contains helper functions for working with the PBS Pro scheduler @ ALCF

See:
[docs/pbs.md](https://github.com/saforem2/ezpz/blob/main/docs/pbs.md)
for more information.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union, Tuple

import socket

# LCMD1: 1.43 ms +- 8.66 us per loop (mean +- std. dev. of 7 runs, 1,000 loops each)
# LCMD2: 1.35 ms +- 7.27 us per loop (mean +- std. dev. of 7 runs, 1,000 loops each)
from getpass import getuser

import ezpz
from ezpz.distributed import get_hostfile_with_fallback

Pathish = Union[str, os.PathLike, Path]

logger = ezpz.get_logger(__name__)

# Suppress the sh library's INFO-level "process started" noise
import logging as _logging
_logging.getLogger("sh").setLevel(_logging.WARNING)

_QSTAT_MAX_RETRIES = 5
_QSTAT_RETRY_DELAY = 2  # seconds
_PBS_JOBS_CACHE_TTL = 30  # seconds
_pbs_jobs_cache: tuple[float, dict[str, list[str]]] | None = None


def _run_qstat_with_retry(qstat_fn, *args, **kwargs) -> str:
    """Run a qstat command with retries on transient PBS server errors."""
    last_exc = None
    for attempt in range(_QSTAT_MAX_RETRIES):
        try:
            return str(qstat_fn(*args, **kwargs))
        except Exception as e:
            last_exc = e
            if "Communication failure" in str(e) or "cannot connect to server" in str(e):
                logger.warning(
                    "qstat failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _QSTAT_MAX_RETRIES, e, _QSTAT_RETRY_DELAY,
                )
                time.sleep(_QSTAT_RETRY_DELAY)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def get_pbs_running_jobs_for_user() -> dict[str, list[str]]:
    """Get all running jobs for the current user.

    Results are cached for up to 30 s to avoid redundant qstat calls
    during a single launch sequence.  Only rank 0 runs qstat; other
    ranks use the cached result.
    """
    global _pbs_jobs_cache
    if _pbs_jobs_cache is not None:
        ts, cached = _pbs_jobs_cache
        if time.monotonic() - ts < _PBS_JOBS_CACHE_TTL:
            return cached

    # Only rank 0 should call qstat — all ranks get the same answer
    # and spawning N qstat processes hammers the PBS server.
    if ezpz.distributed.get_rank() != 0:
        return {}

    try:
        from sh import qstat  # type:ignore
    except Exception as e:
        logger.debug("Error importing sh.qstat: %s", e)
        raise

    output = _run_qstat_with_retry(qstat, f"-fn1wru {getuser()}")
    jobarr = [
        i for i in output.split("\n") if " R " in i
    ]
    jobs: dict[str, list[str]] = {}
    for row in jobarr:
        jstr = [i for i in row.split(" ") if len(i) > 0]
        jobid = jstr[0].split(".")[0]
        nodelist = [h.split("/")[0] for h in jstr[-1].split("+")]
        jobs[jobid] = nodelist

    _pbs_jobs_cache = (time.monotonic(), jobs)
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
        # When both ngpus and ngpu_per_host are given, infer nhosts from them
        # rather than defaulting to the max available nodes.  This lets users
        # request a subset of their allocation (e.g. -n 2 -ppn 2 on a 2-node
        # job to use only 1 node).
        if ngpus is not None and ngpu_per_host is not None:
            if ngpu_per_host <= 0 or ngpus % ngpu_per_host != 0:
                raise ValueError(
                    f"`ngpus` must be divisible by `ngpu_per_host`: "
                    f"ngpus={ngpus}, ngpu_per_host={ngpu_per_host}"
                )
            nhosts = ngpus // ngpu_per_host
        else:
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

    assert (
        ngpus is not None and nhosts is not None and ngpu_per_host is not None
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


def _normalize_cpu_bind_value(cpu_bind: str) -> str:
    """Normalize user-provided CPU bind values to a launcher value."""
    return cpu_bind.strip().removeprefix("--cpu-bind=").strip()


def get_pbs_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[Pathish] = None,
    cpu_bind: Optional[str] = None,
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

    hostfile = hostfile or get_hostfile_with_fallback(hostfile)
    hostfile_path = Path(hostfile)

    ngpus, nhosts, ngpu_per_host = _infer_topology(
        ngpus=ngpus,
        nhosts=nhosts,
        ngpu_per_host=ngpu_per_host,
        hostfile=hostfile_path,
    )

    ngpus_max = ezpz.get_world_size(total=True)
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
        "--line-buffer",
        f"--np={ngpus}",
        f"--ppn={ngpu_per_host}",
        f"--hostfile={hostfile_str}",
    ]
    if verbose:
        cmd_list.append("--verbose")

    cpu_bind_env = os.environ.get("CPU_BIND")
    cpu_bind_cli = (
        _normalize_cpu_bind_value(cpu_bind)
        if cpu_bind is not None and cpu_bind.strip()
        else None
    )
    cpu_bind_env_value = (
        _normalize_cpu_bind_value(cpu_bind_env)
        if cpu_bind_env is not None and cpu_bind_env.strip()
        else None
    )
    use_verbose_cpu_bind = ngpus < 1024
    cpu_bind_prefix = (
        "--cpu-bind=verbose," if use_verbose_cpu_bind else "--cpu-bind="
    )
    selected_cpu_bind = cpu_bind_cli or cpu_bind_env_value

    if selected_cpu_bind:
        if cpu_bind_cli is not None:
            logger.info("Using cpu bind from --cpu-bind: %s", cpu_bind_cli)
        else:
            logger.warning(
                "Detected CPU_BIND from environment: %s", cpu_bind_env
            )
        cmd_list.append(f"{cpu_bind_prefix}{selected_cpu_bind}")
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


def get_running_jobs_from_qstat() -> list[int]:
    """Get the running jobs from qstat"""
    try:
        from sh import qstat as shqstat  # type: ignore
    except Exception as e:
        raise e
    output = _run_qstat_with_retry(shqstat, "-u", os.environ.get("USER"))
    return [
        int(i.split(".")[0])
        for i in output.split("\n")[2:-1]
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
    hosts = ezpz.distributed.get_nodes_from_hostfile(hfp)
    hosts = [h.split(".")[0] for h in hosts]
    nhosts = len(hosts)
    ngpu_per_host = ezpz.distributed.get_gpus_per_node()
    ngpus_available = ezpz.distributed.get_world_size(total=True)
    ngpus = nhosts * ngpu_per_host
    world_size_total = ezpz.distributed.get_world_size_total()
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
        "MACHINE": ezpz.distributed.get_machine(),
        "DEVICE": ezpz.distributed.get_torch_device_type(),
        "BACKEND": ezpz.distributed.get_torch_backend(),
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
    if verbose and ezpz.distributed.get_rank() == 0:
        ezpz.distributed.log_dict_as_bulleted_list(pbsenv, name="pbsenv")
    return pbsenv


def build_launch_cmd(
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[Union[str, Path, os.PathLike]] = None,
    cpu_bind: Optional[str] = None,
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
            cpu_bind=cpu_bind,
        )

    elif scheduler == "slurm":
        import ezpz.slurm

        return ezpz.slurm.build_launch_cmd(
            ngpus=ngpus,
            nhosts=nhosts,
            ngpu_per_host=ngpu_per_host,
            hostfile=hostfile,
            cpu_bind=cpu_bind,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")


if __name__ == "__main__":
    jobenv = ezpz.get_jobenv()
    os.system(jobenv["LAUNCH_CMD"])
