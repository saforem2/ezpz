#!/usr/bin/env python3
"""Test runner script."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests."""
    # Get the project root directory
    project_root = Path(__file__).parent

    # Run pytest
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(project_root / "tests"),
                "-v",
                "--tb=short",
            ],
            cwd=project_root,
            check=False,
        )
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
