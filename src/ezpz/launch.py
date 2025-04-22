"""
ezpz/launch.py
"""

import subprocess
import time
from typing import Optional

from rich.text import Text

import ezpz

logger = ezpz.get_logger(__name__)


# def verbosecmd(command, filters: Optional[list] = None):
#     with subprocess.Popen(
#         command,
#         stdout=subprocess.PIPE,
#         shell=True,
#         stderr=subprocess.STDOUT,
#         bufsize=0,
#         close_fds=True,
#     ) as process:
#         assert process.stdout is not None
#         for line in iter(process.stdout.readline, b""):
#             if filters is not None:
#                 for filter in filters:
#                     if filter in line.decode("utf-8"):
#                         break
#                 else:
#                     continue
#             print(line.rstrip().decode("utf-8"))


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


def run_command(command, filters: Optional[list] = None):
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
            if filters and not any(f in decoded for f in filters):
                continue
            print(decoded.rstrip())


def launch():
    """Launch a command on the current PBS job."""
    start = time.perf_counter()
    import ezpz.pbs

    jobid = ezpz.pbs.get_pbs_jobid_of_active_job()
    logger.info(f"Job ID: {jobid}")

    assert jobid is not None, "No active job found."
    # TODO: Add mechanism for specifying hostfile
    # - Initial experiments using argparse were giving me trouble, WIP
    hostfile = ezpz.pbs.get_pbs_nodefile_of_active_job()
    logger.info(f"Node file: {hostfile}")

    launch_cmd = ezpz.pbs.build_launch_cmd()
    import sys

    assert len(sys.argv) > 1, "No command to run."
    cmd_to_launch = " ".join(sys.argv[1:])
    # if "python" not in cmd_to_launch:
    if not cmd_to_launch.startswith("python"):
        cmd_to_launch = f"{sys.executable} {cmd_to_launch}"

    # lcmd_str = Text("launch_cmd", style="blue")
    # pystr = Text("python", style="blue")
    # cmdstr = Text("cmd_to_launch", style="blue")
    logger.info(
        "\n".join(
            [
                "Building command to execute by piecing together:",
                "\t(1) ['launch_cmd'] + (2) ['python'] + (3) ['cmd_to_launch']",
                "",
                f"1. ['launch_cmd']:\n\t{launch_cmd}",
                "",
                f"2. ['python']:\n\t{sys.executable}",
                "",
                f"3. ['cmd_to_launch']:\n\t{cmd_to_launch.replace(sys.executable, '')}",
                "",
            ]
        )
    )
    cmd = f"{launch_cmd} {cmd_to_launch}"

    logger.info(
        f"Took: {time.perf_counter() - start:.2f} seconds to build command."
    )
    logger.info(f"Evaluating:\n\t{cmd}")
    t0 = time.perf_counter()

    _ = run_command(cmd)
    logger.info(f"Command took {time.perf_counter() - t0:.2f} seconds to run.")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    launch()
