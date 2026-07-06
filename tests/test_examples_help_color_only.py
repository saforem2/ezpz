"""report/run_all render colorized --help WITHOUT appending defaults.

These modules are torch-free, so --help is rendered directly via subprocess in
the test interpreter (runs in both the 3.12 and 3.14 venvs).
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("rich_argparse")

ESC = "\033"
MODULES = ["report", "run_all"]


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
def test_color_only_help_colorized_with_force_color(modname):
    out = _render_help(modname, {"FORCE_COLOR": "1"})
    assert out.returncode == 0, out.stderr
    assert ESC in out.stdout, f"{modname} --help not colorized under FORCE_COLOR"


@pytest.mark.parametrize("modname", MODULES)
def test_color_only_help_has_no_injected_defaults(modname):
    # ColorFormatter must NOT auto-append "(default: ...)" the way
    # DefaultsFormatter would. "(default: None)" is the unambiguous
    # fingerprint: run_all's --run/--outdir have default=None, so the
    # defaults formatter would emit "(default: None)"; hand-written help
    # never does. (report has no default= args; assertion is trivially
    # satisfied there.)
    out = _render_help(modname, {"NO_COLOR": "1"})
    assert out.returncode == 0, out.stderr
    assert "(default: None)" not in out.stdout, (
        f"{modname} help shows auto-injected defaults — wrong formatter?"
    )
