"""CLI entry point for running the distributed smoke test locally."""

import shlex
import subprocess
import sys
import time
from typing import List, Optional

import ezpz
from ezpz.configs import get_scheduler
from ezpz.launch import launch


def _build_test_command(argv: List[str]) -> List[str]:
    """Normalize user-provided arguments into a test command list."""

    if not argv:
        return [sys.executable, "-m", "ezpz.examples.test"]

    cmd_args = list(argv)
    if cmd_args[0] in {"python", "python3"}:
        cmd_args[0] = sys.executable

    if (
        len(cmd_args) >= 3
        and cmd_args[0] == sys.executable
        and cmd_args[1] == "-m"
        and cmd_args[2] == "ezpz.examples.test"
    ):
        extra = cmd_args[3:]
    else:
        extra = cmd_args

    return [sys.executable, "-m", "ezpz.examples.test", *extra]


# def run_test(args: Optional[Any] = None) -> None:
#     args = sys.argv[1:] if args is None else args
#     command = _build_test_command(args)
#     scheduler = get_scheduler().lower()
#     if scheduler in {"pbs", "slurm"}:
#         cmd_str = " ".join(shlex.quote(part) for part in command)
#         return launch(cmd_to_launch=cmd_str)


def main(args: Optional[List[str]] = None) -> int:
    """Run the distributed smoke test via the scheduler or ``mpirun`` fallback."""
    args = sys.argv[1:] if args is None else args
    command = _build_test_command(args)
    scheduler = get_scheduler().lower()

    if scheduler in {"pbs", "slurm"}:
        cmd_str = " ".join(shlex.quote(part) for part in command)
        return launch(cmd_to_launch=cmd_str)

    fallback_cmd = ["mpirun", "-np", "2", *command]
    # result = subprocess.run(fallback_cmd, check=False)
    # ezpz.cleanup()
    # return result.returncode
    return launch(cmd_to_launch=fallback_cmd)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Took {time.perf_counter() - t0:.2f} seconds")
