"""CLI entry point for running the distributed smoke test locally."""

import shlex
import subprocess
import sys
from typing import List

from ezpz.configs import get_scheduler
from ezpz.launch import launch


def _build_test_command(argv: List[str]) -> List[str]:
    """Normalize user-provided arguments into a test command list."""

    if not argv:
        return [sys.executable, "-m", "ezpz.test_dist"]

    cmd_args = list(argv)
    if cmd_args[0] in {"python", "python3"}:
        cmd_args[0] = sys.executable

    if (
        len(cmd_args) >= 3
        and cmd_args[0] == sys.executable
        and cmd_args[1] == "-m"
        and cmd_args[2] == "ezpz.test_dist"
    ):
        extra = cmd_args[3:]
    else:
        extra = cmd_args

    return [sys.executable, "-m", "ezpz.test_dist", *extra]


def main() -> int:
    """Run the distributed smoke test via the scheduler or ``mpirun`` fallback."""
    command = _build_test_command(sys.argv[1:])
    scheduler = get_scheduler().lower()

    if scheduler in {"pbs", "slurm"}:
        cmd_str = " ".join(shlex.quote(part) for part in command)
        return launch(cmd_to_launch=cmd_str)

    fallback_cmd = ["mpirun", "-np", "2", *command]
    result = subprocess.run(fallback_cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
