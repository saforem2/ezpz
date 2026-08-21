"""Run the repo's bash test suites under pytest, so CI sees them.

`tests/*.sh` are self-contained bash suites (stub-based, no modules,
no accelerator, no scheduler). They were only ever documented as
"run this by hand", which meant PR CI — which runs `pytest` — never
executed them and could not catch a regression in the shell code they
cover:

  * test_ezpz_setup_venv.sh          — ezpz_setup Flow B auto-venv
  * test_failover_lib.sh             — failover.sh bad-node helpers
  * test_xpu_module_python_guard.sh  — XPU module load must not evict
  * test_setup_env_no_silent_noop.sh — ezpz_setup_env must not fake success
                                       an active python env

Each is parametrized as its own pytest case so a failure names the
suite, and the bash output is surfaced in the assertion message rather
than swallowed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SHELL_SUITES = [
    "test_ezpz_setup_venv.sh",
    "test_failover_lib.sh",
    "test_xpu_module_python_guard.sh",
    "test_setup_env_no_silent_noop.sh",
]

pytestmark = [
    pytest.mark.skipif(
        os.name != "posix", reason="bash suites assume POSIX"
    ),
    pytest.mark.skipif(
        shutil.which("bash") is None, reason="bash not available"
    ),
]


@pytest.mark.parametrize("suite", SHELL_SUITES)
def test_shell_suite_passes(suite: str) -> None:
    script = REPO_ROOT / "tests" / suite
    if not script.is_file():
        pytest.skip(f"{suite} not present")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,  # suites resolve src/ezpz/bin/utils.sh from the root
        capture_output=True,
        text=True,
        timeout=300,
        # NO_COLOR keeps the assertion message readable when it fails.
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert proc.returncode == 0, (
        f"{suite} failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


def test_every_shell_suite_is_registered() -> None:
    """A new tests/*.sh must be added to SHELL_SUITES.

    Without this, someone adds a bash suite, it silently never runs in
    CI, and we are back to the gap this file exists to close.
    """
    on_disk = {p.name for p in (REPO_ROOT / "tests").glob("test_*.sh")}
    missing = sorted(on_disk - set(SHELL_SUITES))
    assert not missing, (
        f"bash suites not registered in SHELL_SUITES (so CI skips them): "
        f"{missing}"
    )
