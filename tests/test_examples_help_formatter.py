"""Example argparse modules must render colorized --help.

These modules import torch at module scope, so --help is rendered in a
subprocess (the test interpreter may lack torch) and skipped when torch is
absent. The behavioral check (ANSI present under FORCE_COLOR, absent under
NO_COLOR) proves the shared colorized formatter is actually wired in, not
merely imported.
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")  # modules below import torch at import time
pytest.importorskip("rich_argparse")  # color path requires the extra

ESC = "\033"
MODULES = ["fsdp", "diffusion", "vit", "fsdp_tp", "inference"]


def _render_help(modname, env_extra):
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", f"ezpz.examples.{modname}", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("modname", MODULES)
def test_example_help_colorized_with_force_color(modname):
    out = _render_help(modname, {"FORCE_COLOR": "1"})
    assert out.returncode == 0, out.stderr
    assert ESC in out.stdout, f"{modname} --help not colorized under FORCE_COLOR"


@pytest.mark.parametrize("modname", MODULES)
def test_example_help_plain_with_no_color(modname):
    out = _render_help(modname, {"NO_COLOR": "1"})
    assert out.returncode == 0, out.stderr
    assert ESC not in out.stdout, f"{modname} --help leaked ANSI under NO_COLOR"
