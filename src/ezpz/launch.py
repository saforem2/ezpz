#!/usr/bin/env python

"""
ezpz/launch.py

Launch a command on the current PBS or SLURM job.

By default, the command to be executed will be launched across _all_ nodes.
"""

import os
import sys
import subprocess
import shlex
import time
from typing import Optional
from pathlib import Path

import ezpz

logger = ezpz.get_logger(__name__)

EZPZ_LOG_LEVEL: str = os.environ.get("EZPZ_LOG_LEVEL", "INFO").upper()


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch a command on the current PBS job."
    )
    parser.add_argument(
        "command",
        type=str,
        help="The command to run on the current PBS job.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        help="Filter output lines by these strings.",
    )
    parser.add_argument(
        "-n",
        "--nproc",
        "--world_size",
        "--nprocs",
        type=int,
        default=-1,
        help="Number of processes.",
    )
    parser.add_argument(
        "-np",
        "--nproc_per_node",
        type=int,
        default=-1,
        help="Processes per node.",
    )
    parser.add_argument(
        "-nh",
        "--nnode",
        "--nhost",
        "--nhosts",
        type=int,
        default=-1,
        help="Number of nodes to use.",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        help="Hostfile to use for launching.",
    )
    # parser.add_argument(
    #     "--world_size",
    #     required=False,
    #     type=int,
    #     default=-1,
    #     help="Number of processes to launch.",
    # )
    return parser.parse_args()


def run_command(command: list | str, filters: Optional[list] = None) -> int:
    """Run a command and print its output line by line.

    Args:

    - command (str or list): The command to run. If a string, it will be split
      into a list
    - filters (list, optional): A list of strings to filter the output
      lines.
    """
    # XXX: Replace `subprocess.Popen`
    # with `subprocess.run` for better error handling ??
    # <https://docs.python.org/3.10/library/subprocess.html#subprocess.run>
    if filters is not None and len(filters) > 0:
        logger.info(f"Caught {len(filters)} filters")
    logger.info(
        " ".join(
            [
                "Running command:\n",
                shlex.join(command) if isinstance(command, list) else command,
            ]
        )
    )
    with subprocess.Popen(
        shlex.join(command),
        stdout=subprocess.PIPE,
        shell=True,
        stderr=subprocess.STDOUT,
        bufsize=0,
        close_fds=True,
        # executable="/bin/bash" if isinstance(command, str) else None,
    ) as process:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, b""):
            decoded = line.decode("utf-8")
            if (
                filters is None
                or len(filters) == 0
                or not any(f in decoded for f in filters)
            ):
                print(decoded.rstrip())
    return process.returncode


def get_command_to_launch_from_argv() -> Optional[str | list[str]]:
    assert len(sys.argv) > 1, "No command to run."
    # cmd_to_launch = shlex.join(sys.argv[1:])
    cmd_to_launch = " ".join(sys.argv[1:])

    return cmd_to_launch


def configure_warnings():
    import os
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:__main__"


def get_aurora_filters(additional_filters: Optional[list] = None) -> list:
    mn = ezpz.get_machine()
    filters = [*additional_filters] if additional_filters else []
    if mn.lower() == "aurora":
        if EZPZ_LOG_LEVEL == "DEBUG":
            filters = []
        else:
            filters += [
                "cuda",
                "CUDA",
                "cuDNN",
                "cuBLAS",
                "[W501",
                "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
                "  Overriding a previously registered kernel",
                "operator: aten::_cummax_helper",
                "    registered at build",
                "dispatch key: XPU",
                "previous kernel: registered at",
                "pkg_resources is deprecated as an API",
                "import pkg_resources",
                "UserWarning: pkg_resources",
                "new kernel: registered at",
                "/build/pytorch/build/aten/src/ATen/RegisterSchema.cpp",
                "Setting ds_accelerator to xpu",
                "Trying to register 2 metrics with the same name",
                "TF-TRT Warning",
                "Warning only once",
                "measureDifference between two events",
                "AttributeError",
                "Initialized with serialization",
                "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
                # "operator: aten::geometric"
            ]
        logger.info(
            " ".join(
                [
                    "Filtering for Aurora-specific messages.",
                    "To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG",
                ]
            )
        )
    logger.debug(f"Filters: {filters}")
    return filters


def kill_existing_processes(
    filters: Optional[list] = None,
    additional_filters: Optional[list] = None,
) -> int:
    """Kill existing processes that match the filters."""
    # TODO: Run this as preamble to launching
    filters = [] if filters is None else filters
    if ezpz.get_machine().lower() == "aurora":
        filters += get_aurora_filters(additional_filters=additional_filters)

    logger.info(f"Killing existing processes with filters: {filters}")
    cmd = f"pkill -f {' '.join(filters)}"
    return run_command(cmd, filters=filters)


def get_active_jobid() -> str | None:
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        return ezpz.pbs.get_pbs_jobid_of_active_job()
    elif scheduler == "slurm":
        import ezpz.slurm

        return ezpz.slurm.get_slurm_jobid_of_active_job()
    else:
        return None


def get_nodelist_of_active_job() -> list[str] | None:
    """Get nodelist of active job."""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        jobid = ezpz.pbs.get_pbs_jobid_of_active_job()
        if jobid is not None:
            return ezpz.pbs.get_pbs_nodelist_from_jobid(jobid)
    elif scheduler == "slurm":
        import ezpz.slurm

        jobid = ezpz.slurm.get_slurm_jobid_of_active_job()
        if jobid is not None:
            return ezpz.slurm.get_nodelist_from_slurm_jobid(jobid)
    return None


def get_hostfile_of_active_job():
    """Get hostfile of active job."""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        return ezpz.pbs.get_pbs_nodefile_of_active_job()
    elif scheduler == "slurm":
        import ezpz.slurm

        # jobid = ezpz.slurm.get_slurm_jobid_of_active_job()
        # if jobid is not None:
        #     return ezpz.slurm.get_slurm_nodefile_from_jobid(jobid)
        return ezpz.slurm.get_slurm_nodefile_of_active_job()
    return None


def build_executable(
    launch_cmd: Optional[str] = None,
    cmd_to_launch: Optional[str | list[str]] = None,
    include_python: Optional[bool] = True,
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
) -> list:
    """Build the full executable command to launch.

    Args:
        launch_cmd (str, optional): The command to launch the job. If None,
            will be built using `build_launch_cmd()`.
        cmd_to_launch (str or list, optional): The command to run on the job.
            If None, will be taken from `sys.argv`.
        include_python (bool, optional): Whether to include the python
            executable in the command. Default is True.

    Returns:
        str: The full command to launch the job.
    """
    from ezpz.pbs import build_launch_cmd

    launch_cmd = (
        build_launch_cmd(
            ngpus=ngpus,
            nhosts=nhosts,
            ngpu_per_host=ngpu_per_host,
            hostfile=hostfile,
        )
        if launch_cmd is None
        else launch_cmd
    )
    cmd_to_launch = (
        get_command_to_launch_from_argv()
        if cmd_to_launch is None
        else cmd_to_launch
    )
    cmd_to_launch_list: list[str] = (
        shlex.split(cmd_to_launch)
        if isinstance(cmd_to_launch, str)
        else (cmd_to_launch if cmd_to_launch is not None else [])
    )
    if include_python:
        # and "python" not in str(cmd_to_launch_list[0]):
        found_python = False
        for part in cmd_to_launch_list:
            if "python" in str(part):
                found_python = True
        if not found_python:
            cmd_to_launch_list.insert(0, sys.executable)
        # cmd_to_launch_list = [sys.executable] + cmd_to_launch_list

    cmd_to_launch_str = shlex.join(cmd_to_launch_list)
    logger.info("Building command to execute by piecing together:")
    logger.info(f"(1.) launch_cmd: {launch_cmd}")
    logger.info(f"(2.) cmd_to_launch: {cmd_to_launch_str}")
    executable = [*shlex.split(launch_cmd), *cmd_to_launch_list]
    # executable = [
    #     shlex.join(launch_cmd.split(' ')), *cmd_to_launch_list
    # ]
    # return shlex.split(shlex.join(executable))
    return executable


def launch(
    launch_cmd: Optional[str] = None,
    cmd_to_launch: Optional[str | list[str]] = None,
    include_python: bool = True,
    ngpus: Optional[int] = None,
    nhosts: Optional[int] = None,
    ngpu_per_host: Optional[int] = None,
    hostfile: Optional[str | os.PathLike | Path] = None,
    filters: Optional[list[str]] = None,
) -> int:
    """Launch a command on the current {PBS, SLURM} job."""
    start = time.perf_counter()
    print("\n") if ezpz.get_rank() == 0 else None
    logger.info(f"----[🍋 ezpz.launch][started][{ezpz.get_timestamp()}]----")
    jobid = get_active_jobid()
    assert jobid is not None, "No active job found."
    nodelist = get_nodelist_of_active_job()
    hostfile = get_hostfile_of_active_job()
    logger.info(f"Job ID: {jobid}")
    logger.info(f"nodelist: {nodelist}")
    logger.info(f"hostfile: {hostfile}")
    cmd_list = build_executable(
        launch_cmd=launch_cmd,
        cmd_to_launch=cmd_to_launch,
        ngpus=ngpus,
        ngpu_per_host=ngpu_per_host,
        nhosts=nhosts,
        include_python=include_python,
    )
    # cmd_list = shlex.split(cmd)
    cmd_str = shlex.join([f"{i}" for i in cmd_list])
    cmd = shlex.split(cmd_str)

    logger.info(
        f"Took: {time.perf_counter() - start:.2f} seconds to build command."
    )
    logger.info("Executing:\n" + "\n  ".join([f"{i}" for i in cmd_list]))
    t0 = time.perf_counter()

    filters = [] if filters is None else filters
    if ezpz.get_machine().lower() in {"aurora", "sunspot"}:
        filters += get_aurora_filters()

    logger.info(f"Execution started @ {ezpz.get_timestamp()}...")
    logger.info(f"----[🍋 ezpz.launch][stop][{ezpz.get_timestamp()}]----")
    cmd_start = time.perf_counter()
    retcode = run_command(command=cmd, filters=filters)
    cmd_finish = time.perf_counter()
    logger.info(f"Execution finished with {retcode}.")
    logger.info(f"Executing finished in {cmd_finish - cmd_start:.2f} seconds.")
    logger.info(
        f"Took {time.perf_counter() - t0:.2f} seconds to run. Exiting."
    )
    return retcode


def main():
    import ezpz.dist

    # import shlex
    # argv = shlex.split(" ".join(sys.argv[1:]))
    # if "python" in
    # args = parse_args()
    # print(f"{args=}")
    configure_warnings()
    launch(cmd_to_launch=" ".join(sys.argv[1:]))
    ezpz.dist.cleanup()
    exit(0)


if __name__ == "__main__":
    main()
