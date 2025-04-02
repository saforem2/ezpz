"""
ezpz/launch.py
"""

import subprocess
import time
from typing import Optional

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


def main():
    """Launch a command on the current PBS job."""
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
    cmdlist = sys.argv[1:]
    # if 'python' not in cmdlist[0]:
    #     cmdlist = [sys.executable] + cmdlist
    cmd_to_launch = " ".join(sys.argv[1:])
    logger.info(
        "\n".join(
            [
                "Building command to execute from: '{launch_cmd}' + '{python}' + '{cmd_to_launch}'",
                "",
                f"launch_cmd={launch_cmd}",
                f"python={sys.executable}",
                f"cmd_to_launch={cmd_to_launch}",
                "",
            ]
        )
    )

    # if "python" not in cmd_to_launch:
    if not cmd_to_launch.startswith("python"):
        cmd_to_launch = f"{sys.executable} {cmd_to_launch}"

    cmd = f"{launch_cmd} {cmd_to_launch}"

    logger.info(f"Evaluating:\n{cmd}")
    t0 = time.perf_counter()
    _ = run_command(cmd)
    logger.info(f"Command took {time.perf_counter() - t0:.2f} seconds to run.")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
