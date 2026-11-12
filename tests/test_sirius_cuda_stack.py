"""Tests for the Sirius CUDA module selection in utils.sh."""

import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sirius_uses_cuda13_conda_module(tmp_path):
    module_root = tmp_path / "modulefiles"
    module_root.mkdir()
    calls = tmp_path / "module-calls.txt"
    script = """
source "$EZPZ_UTILS_FILE"
module() { printf '%s\\n' "$*" >> "$MODULE_CALLS"; }
ezpz_sirius_load_cuda13_stack
"""
    env = dict(os.environ, SHELL="/bin/bash")
    env.update(
        EZPZ_UTILS_FILE=os.path.join(ROOT, "src", "ezpz", "bin", "utils.sh"),
        EZPZ_SIRIUS_MODULE_ROOT=str(module_root),
        MODULE_CALLS=str(calls),
    )
    run = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    assert run.returncode == 0, run.stdout + run.stderr
    assert calls.read_text().splitlines() == [
        f"use {module_root}",
        "load conda/2026-10-09",
    ]
