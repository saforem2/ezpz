"""
ezpz/launch.py
"""

import subprocess
from typing import Optional


def verbosecmd(command, filters: Optional[list] = None):
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
            if filters is not None:
                for filter in filters:
                    if filter in line.decode("utf-8"):
                        break
                else:
                    continue
            print(line.rstrip().decode("utf-8"))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    import os
    import ezpz

    logger = ezpz.get_logger(__name__)

    import ezpz.pbs

    # Test the functions
    jobid = ezpz.pbs.get_pbs_jobid_of_active_job()
    logger.info(f"Job ID: {jobid}")

    assert jobid is not None, "No active job found."
    nodefile = ezpz.pbs.get_pbs_nodefile_from_jobid(str(jobid))
    logger.info(f"Node file: {nodefile}")

    launch_cmd = ezpz.pbs.build_launch_cmd()
    import sys

    assert len(sys.argv) > 1, "No command to run."
    _ = os.system("clear")
    cmd = f"{launch_cmd} {' '.join(sys.argv[1:])}"
    logger.info(f"Evaluating:\n'{cmd}'")
    result = verbosecmd(
        cmd,
        # filters=[
        #     "itex/core",
        #     "Trying to register 2 metrics",
        #     "cuda",
        # ],
    )
