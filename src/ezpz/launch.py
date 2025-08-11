"""
ezpz/launch.py
"""

import sys
import subprocess
import shlex
import time
from typing import Optional

import ezpz

logger = ezpz.get_logger(__name__)


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
    # add argument for controlling WORLD_SIZE
    parser.add_argument(
        "--world_size",
        required=False,
        type=int,
        default=-1,
        help="Number of processes to launch.",
    )
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
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True,
        stderr=subprocess.STDOUT,
        bufsize=0,
        close_fds=True,
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
    if not cmd_to_launch.startswith("python"):
        cmd_to_launch = f"{sys.executable} {cmd_to_launch}"

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
        filters += [
            "cuda",
            "CUDA",
            "cuDNN",
            "cuBLAS",
            "[W501",
            "  Overriding a previously registered kernel",
            "operator: aten::_cummax_helper",
            "    registered at build",
            "dispatch key: XPU",
            "previous kernel: registered at",
            "new kernel: registered at",
            "/build/pytorch/build/aten/src/ATen/RegisterSchema.cpp",
            "Setting ds_accelerator to xpu",
            "Trying to register 2 metrics with the same name",
            "TF-TRT Warning",
            "Warning only once",
            "measureDifference between two events",
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


def build_launch_cmd() -> str:
    """Build command to launch a job on {PBS, SLURM}."""
    from ezpz.configs import get_scheduler

    scheduler = get_scheduler().lower()
    if scheduler == "pbs":
        import ezpz.pbs

        return ezpz.pbs.build_launch_cmd()
    elif scheduler == "slurm":
        import ezpz.slurm

        return ezpz.slurm.build_launch_cmd()
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")


def launch(
    cmd_to_launch: Optional[str | list[str]] = None,
    filters: Optional[list[str]] = None,
) -> int:
    """Launch a command on the current {PBS, SLURM} job."""
    start = time.perf_counter()
    print("\n") if ezpz.get_rank() == 0 else None
    logger.info("======== [ezpz.launch: START] ========")
    jobid = get_active_jobid()
    assert jobid is not None, "No active job found."
    nodelist = get_nodelist_of_active_job()
    hostfile = get_hostfile_of_active_job()
    logger.info(f"Job ID: {jobid}")
    logger.info(f"nodelist: {nodelist}")
    logger.info(f"hostfile: {hostfile}")
    # TODO: Add mechanism for specifying hostfile
    # - Initial experiments using argparse were giving me trouble, WIP
    # hostfile = ezpz.pbs.get_pbs_nodefile_of_active_job()
    # logger.info(f"Node file: {hostfile}")

    # launch_cmd = ezpz.pbs.build_launch_cmd()
    launch_cmd = build_launch_cmd()

    if cmd_to_launch is not None:
        if isinstance(cmd_to_launch, str):
            cmd_to_launch = shlex.join(shlex.split(cmd_to_launch))
        elif isinstance(cmd_to_launch, list):
            cmd_to_launch = shlex.join(cmd_to_launch)
    else:
        cmd_to_launch = get_command_to_launch_from_argv()

    assert cmd_to_launch is not None
    if isinstance(cmd_to_launch, list):
        cmd_to_launch = shlex.join(cmd_to_launch)
    assert isinstance(cmd_to_launch, str)
    logger.info(
        "Building command to execute by piecing together:\n\n"
        "\t(1.) ['launch_cmd'] + (2.) ['python'] + (3.) ['cmd_to_launch']\n"
    )
    logger.info(f"(1.) ['launch_cmd']: {launch_cmd}")
    logger.info(f"(2.) ['python']: {sys.executable}")
    logger.info(
        f"(3.) ['cmd_to_launch']: {cmd_to_launch.replace(sys.executable, '')}"
    )
    cmd = shlex.join(shlex.split(" ".join([launch_cmd, cmd_to_launch])))

    logger.info(
        f"Took: {time.perf_counter() - start:.2f} seconds to build command."
    )
    split_cmd = shlex.split(cmd)
    logger.info("Executing:\n\t" + "\n\t".join(split_cmd))
    # logger.info(f"Executing: \n\t{\n - {i}.join(cmd.split())}\n")
    t0 = time.perf_counter()

    filters = [] if filters is None else filters
    if ezpz.get_machine().lower() == "aurora":
        filters += get_aurora_filters()

    logger.info(f"Execution started @ {ezpz.get_timestamp()}...")
    logger.info("======== [ezpz.launch: STOP] ========\n")
    retcode = run_command(cmd, filters=filters)
    logger.info(f"Execution finished @ {ezpz.get_timestamp()}")
    logger.info(
        f"Command took {time.perf_counter() - t0:.2f} seconds to run. Exiting."
    )
    return retcode


def main():
    import ezpz.dist

    configure_warnings()
    launch()
    ezpz.dist.cleanup()
    exit(0)


if __name__ == "__main__":
    main()
