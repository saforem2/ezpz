"""CLI entry point for running the distributed smoke test locally."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence

from ezpz.configs import get_scheduler
from ezpz.launch import launch


def _build_test_command(argv: Sequence[str]) -> list[str]:
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


def run(argv: Sequence[str] | None = None) -> int:
    """Run the distributed smoke test via the scheduler or ``mpirun`` fallback."""
    argv = [] if argv is None else list(argv)
    command = _build_test_command(argv)
    scheduler = get_scheduler().lower()

    if scheduler in {"pbs", "slurm"}:
        cmd_str = " ".join(shlex.quote(part) for part in command)
        return launch(cmd_to_launch=cmd_str)

    fallback_cmd = ["mpirun", "-np", "2", *command]
    result = subprocess.run(fallback_cmd, check=False)
    return result.returncode


def main() -> int:
    """Backward-compatible console script entry point."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
