import ezpz

import ezpz.distributed
import os
from pathlib import Path
import subprocess

import pytest


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def test_fsdp_tp_launch_tp_random_smoke():
    if os.environ.get("EZPZ_LAUNCH_TEST") != "1":
        pytest.skip("Set EZPZ_LAUNCH_TEST=1 to enable ezpz launch smoke test.")
    # hostfile = os.environ.get("EZPZ_HOSTFILE") or os.environ.get("HOSTFILE")
    # if not hostfile:
    #     pytest.skip("Set EZPZ_HOSTFILE or HOSTFILE for ezpz launch test.")
    hostfile = Path(os.getcwd()).joinpath("hostfile-localhost")
    ezpz.distributed.write_localhost_to_hostfile(hostfile)

    cmd = [
        "ezpz",
        "launch",
        "-n",
        "2",
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
    ]
    env = os.environ.copy()
    # Scrub distributed env vars that earlier tests may have leaked into
    # the pytest process so that the mpirun children discover topology
    # from MPI rather than stale env state.
    for var in [
        "MASTER_PORT", "MASTER_ADDR",
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
        "PMI_RANK", "PMI_SIZE", "PMI_LOCAL_RANK", "PMI_LOCAL_SIZE",
        "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "PBS_JOBID", "PBS_NODEFILE", "PBS_NUM_NODES",
        "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_NNODES",
    ]:
        env.pop(var, None)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["TORCH_DEVICE"] = "cpu"
    env["TORCH_BACKEND"] = "gloo"
    env["WANDB_MODE"] = "disabled"
    env["NO_COLOR"] = "1"
    env["EZPZ_LOG_LEVEL"] = "CRITICAL"
    env["EZPZ_SCHEDULER"] = "UNKNOWN"
    env["EZPZ_DEBUG_NAN"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("ezpz launch timed out after 180s")
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
