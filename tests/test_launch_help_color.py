"""`ezpz launch --help` must preserve color through click.echo.

click.echo() strips ANSI unless color=True is passed. With rich-argparse
installed and FORCE_COLOR set, the help must contain ESC sequences.
"""
import os
import subprocess
import sys

import pytest

rich_argparse = pytest.importorskip("rich_argparse")

ESC = "\033"


def _run_launch_help(env_extra):
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    env.update(env_extra)
    # Invoke the click command module directly so the test does not depend on
    # an installed console script.
    code = (
        "from ezpz.cli.launch_cmd import launch_cmd;"
        "launch_cmd(['--help'], standalone_mode=False)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )


def test_launch_help_has_color_with_force_color():
    out = _run_launch_help({"FORCE_COLOR": "1"})
    assert out.returncode == 0, out.stderr
    assert ESC in out.stdout, "expected ANSI escapes in colorized launch help"


def test_launch_help_plain_with_no_color():
    out = _run_launch_help({"NO_COLOR": "1"})
    assert out.returncode == 0, out.stderr
    assert ESC not in out.stdout, "NO_COLOR must strip ANSI from launch help"
