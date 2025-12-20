import os
import subprocess

import pytest


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def test_fsdp_tp_launch_tp_random_smoke():
    if os.environ.get("EZPZ_LAUNCH_TEST") != "1":
        pytest.skip("Set EZPZ_LAUNCH_TEST=1 to enable ezpz launch smoke test.")
    hostfile = os.environ.get("EZPZ_HOSTFILE") or os.environ.get("HOSTFILE")
    if not hostfile:
        pytest.skip("Set EZPZ_HOSTFILE or HOSTFILE for ezpz launch test.")

    cmd = [
        "ezpz",
        "launch",
        "-n",
        "1",
        "-ppn",
        "2",
        "--hostfile",
        hostfile,
        "python3",
        "-m",
        "ezpz.examples.fsdp_tp",
        "--tp",
        "2",
        "--dataset",
        "random",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--seq-length",
        "128",
        "--seq-len",
        "128",
        "--dim",
        "64",
        "--n-layers",
        "2",
        "--n-heads",
        "4",
        "--n-kv-heads",
        "2",
        "--vocab-size",
        "256",
        "--max-seq-len",
        "256",
        "--fp32",
    ]
    env = os.environ.copy()
    env["EZPZ_DEBUG_NAN"] = "1"
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        pytest.fail(
            "ezpz launch failed\n"
            f"stdout tail:\n{_tail(stdout)}\n"
            f"stderr tail:\n{_tail(stderr)}"
        )
    combined = f"{stdout}\n{stderr}"
    if "loss=nan" in combined.lower():
        pytest.fail(
            "NaN loss detected in ezpz launch output\n"
            f"stdout tail:\n{_tail(stdout)}\n"
            f"stderr tail:\n{_tail(stderr)}"
        )
