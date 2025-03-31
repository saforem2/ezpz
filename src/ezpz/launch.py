"""
ezpz/launch.py
"""

import subprocess
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
    # _ = os.system("clear")
    cmd = f"{launch_cmd} {' '.join(sys.argv[1:])}"
    logger.info(f"Evaluating:\n'{cmd}'")
    _ = run_command(cmd)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
