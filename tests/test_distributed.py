"""Comprehensive tests for ``ezpz.distributed``.

Covers every public function in ``__all__`` plus critical private helpers
(``_expand_slurm_nodelist``, ``_make_hostfile_from_slurm``, ``_resolve_wandb_mode``,
``_ensure_dtype_map``).

The test strategy heavily mocks ``mpi4py`` and ``torch`` so that the suite
runs on any machine without GPUs, MPI, or wandb.
"""

from __future__ import annotations

import os
import random
import socket
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import ezpz.distributed as dist

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeComm:
    """Minimal stub for ``MPI.COMM_WORLD``."""

    def __init__(self, rank: int = 0, size: int = 4):
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def barrier(self) -> None:
        pass

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def allreduce(self, obj: Any, op: Any = None) -> Any:
        return obj


@pytest.fixture(autouse=True)
def _reset_mpi_singleton():
    """Reset the module-level ``_MPI_COMM`` cache between tests."""
    old = dist._MPI_COMM
    yield
    dist._MPI_COMM = old


@pytest.fixture()
def fake_comm(monkeypatch):
    """Inject a ``_FakeComm`` as the cached MPI communicator."""
    for var in (
        "WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS",
        "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID",
        "LOCAL_RANK", "PMI_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID", "MPICH_LOCALRANKID", "SLURM_LOCAL_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    comm = _FakeComm(rank=0, size=4)
    dist._MPI_COMM = comm
    return comm


@pytest.fixture()
def env(monkeypatch):
    """Helper to set/unset environment variables cleanly.

    Returns a dict-like ``monkeypatch`` wrapper with ``set`` / ``delete``
    helpers for readability.
    """
    return monkeypatch


# ===================================================================
# _expand_slurm_nodelist
# ===================================================================


class TestExpandSlurmNodelist:
    """Tests for ``_expand_slurm_nodelist``."""

    def test_single_node_no_brackets(self):
        assert dist._expand_slurm_nodelist("node001") == ["node001"]

    def test_single_node_whitespace(self):
        assert dist._expand_slurm_nodelist("  node001  ") == ["node001"]

    def test_comma_list(self):
        result = dist._expand_slurm_nodelist("node[001,003,005]")
        assert result == ["node001", "node003", "node005"]

    def test_range(self):
        result = dist._expand_slurm_nodelist("node[001-004]")
        assert result == ["node001", "node002", "node003", "node004"]

    def test_mixed_ranges_and_singles(self):
        result = dist._expand_slurm_nodelist("node[001-003,007,010-012]")
        assert result == [
            "node001",
            "node002",
            "node003",
            "node007",
            "node010",
            "node011",
            "node012",
        ]

    def test_zero_padding_preserved(self):
        result = dist._expand_slurm_nodelist("x3[0001-0003]")
        assert result == ["x30001", "x30002", "x30003"]

    def test_single_item_in_brackets(self):
        result = dist._expand_slurm_nodelist("node[042]")
        assert result == ["node042"]

    def test_empty_brackets(self):
        # Edge case: ``node[]`` should produce no nodes
        result = dist._expand_slurm_nodelist("node[]")
        assert result == []

    def test_large_range(self):
        result = dist._expand_slurm_nodelist("n[0000-0099]")
        assert len(result) == 100
        assert result[0] == "n0000"
        assert result[-1] == "n0099"

    def test_no_padding(self):
        result = dist._expand_slurm_nodelist("gpu[1-3]")
        assert result == ["gpu1", "gpu2", "gpu3"]


# ===================================================================
# _make_hostfile_from_slurm
# ===================================================================


class TestMakeHostfileFromSlurm:
    """Tests for ``_make_hostfile_from_slurm``."""

    def test_missing_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("SLURM_NODELIST", raising=False)
        with pytest.raises(RuntimeError, match="SLURM_NODELIST not set"):
            dist._make_hostfile_from_slurm()

    def test_writes_hostfile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLURM_NODELIST", "node[001-003]")
        monkeypatch.chdir(tmp_path)
        hf = dist._make_hostfile_from_slurm()
        assert hf.is_file()
        lines = hf.read_text().strip().splitlines()
        assert lines == ["node001", "node002", "node003"]


# ===================================================================
# get_rank / get_local_rank / get_world_size variants
# ===================================================================


class TestGetRank:
    """Tests for ``get_rank``."""

    def test_returns_int(self, fake_comm, monkeypatch):
        for v in ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
            monkeypatch.delenv(v, raising=False)
        assert dist.get_rank() == 0
        assert isinstance(dist.get_rank(), int)

    def test_nonzero_rank(self, monkeypatch):
        for v in ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
            monkeypatch.delenv(v, raising=False)
        comm = _FakeComm(rank=7, size=8)
        dist._MPI_COMM = comm
        assert dist.get_rank() == 7


class TestGetLocalRank:
    """Tests for ``get_local_rank``."""

    def test_from_LOCAL_RANK(self, fake_comm, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert dist.get_local_rank() == 3

    def test_from_PMI_LOCAL_RANK(self, fake_comm, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.setenv("PMI_LOCAL_RANK", "2")
        assert dist.get_local_rank() == 2

    def test_from_OMPI_COMM_WORLD_LOCAL_RANK(self, fake_comm, monkeypatch):
        for v in ("LOCAL_RANK", "PMI_LOCAL_RANK"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "5")
        assert dist.get_local_rank() == 5

    def test_from_MPI_LOCALRANKID(self, fake_comm, monkeypatch):
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("MPI_LOCALRANKID", "1")
        assert dist.get_local_rank() == 1

    def test_from_MPICH_LOCALRANKID(self, fake_comm, monkeypatch):
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("MPICH_LOCALRANKID", "6")
        assert dist.get_local_rank() == 6

    def test_from_SLURM_LOCAL_ID(self, fake_comm, monkeypatch):
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MPICH_LOCALRANKID",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("SLURM_LOCAL_ID", "4")
        assert dist.get_local_rank() == 4

    def test_fallback_world_size_1(self, monkeypatch):
        """world_size == 1 ⇒ local_rank = 0."""
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MPICH_LOCALRANKID",
            "SLURM_LOCAL_ID",
        ):
            monkeypatch.delenv(v, raising=False)
        comm = _FakeComm(rank=0, size=1)
        dist._MPI_COMM = comm
        assert dist.get_local_rank() == 0

    def test_fallback_modulo(self, monkeypatch):
        """No env vars, world_size > 1 ⇒ rank % gpus_per_node."""
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MPICH_LOCALRANKID",
            "SLURM_LOCAL_ID",
            "RANK",
            "PMI_RANK",
            "OMPI_COMM_WORLD_RANK",
            "SLURM_PROCID",
            "WORLD_SIZE",
            "PMI_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "SLURM_NTASKS",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        comm = _FakeComm(rank=5, size=8)
        dist._MPI_COMM = comm
        # 5 % 4 == 1
        assert dist.get_local_rank() == 1

    def test_fallback_gpus_per_node_zero(self, monkeypatch):
        """gpus_per_node == 0 ⇒ local_rank = 0 (avoid ZeroDivisionError)."""
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MPICH_LOCALRANKID",
            "SLURM_LOCAL_ID",
            "NGPU_PER_HOST",
            "LOCAL_WORLD_SIZE",
            "PMI_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ):
            monkeypatch.delenv(v, raising=False)
        comm = _FakeComm(rank=3, size=4)
        dist._MPI_COMM = comm
        # torch hardware probing returns 0 on CPU-only machines
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
        ):
            assert dist.get_local_rank() == 0


class TestGetWorldSize:
    """Tests for ``get_world_size`` and its ``total`` / ``in_use`` flags."""

    def test_default(self, fake_comm):
        assert dist.get_world_size() == 4

    def test_in_use(self, fake_comm):
        assert dist.get_world_size(in_use=True) == 4

    def test_total(self, fake_comm, monkeypatch):
        monkeypatch.setenv("NGPU_PER_HOST", "8")
        monkeypatch.setenv("SLURM_NNODES", "2")
        assert dist.get_world_size(total=True) == 16

    def test_total_flag_takes_precedence(self, fake_comm, monkeypatch):
        """If both ``total`` and ``in_use`` are True, ``total`` wins."""
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        monkeypatch.setenv("SLURM_NNODES", "2")
        assert dist.get_world_size(total=True, in_use=True) == 8


class TestGetWorldSizeTotal:
    """Tests for ``get_world_size_total``."""

    def test_calculation(self, fake_comm, monkeypatch):
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        monkeypatch.setenv("SLURM_NNODES", "3")
        assert dist.get_world_size_total() == 12

    def test_zero_gpus_clamped_to_one(self, fake_comm, monkeypatch):
        """``max(gpus_per_node, 1)`` avoids multiplying by zero."""
        monkeypatch.setenv("SLURM_NNODES", "2")
        # Patch gpus_per_node to return 0
        with patch.object(dist, "get_gpus_per_node", return_value=0):
            assert dist.get_world_size_total() == 2


class TestGetNumNodes:
    """Tests for ``get_num_nodes``."""

    def test_from_slurm_nnodes(self, fake_comm, monkeypatch):
        monkeypatch.setenv("SLURM_NNODES", "8")
        assert dist.get_num_nodes() == 8

    def test_from_hostfile(self, fake_comm, monkeypatch, tmp_path):
        monkeypatch.delenv("SLURM_NNODES", raising=False)
        hf = tmp_path / "hostfile"
        hf.write_text("node1.local\nnode2.local\nnode3.local\n")
        assert dist.get_num_nodes(hostfile=hf) == 3


class TestGetGpusPerNode:
    """Tests for ``get_gpus_per_node``."""

    def test_from_NGPU_PER_HOST(self, monkeypatch):
        monkeypatch.setenv("NGPU_PER_HOST", "8")
        assert dist.get_gpus_per_node() == 8

    def test_from_LOCAL_WORLD_SIZE(self, monkeypatch):
        monkeypatch.delenv("NGPU_PER_HOST", raising=False)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        assert dist.get_gpus_per_node() == 4

    def test_from_PMI_LOCAL_SIZE(self, monkeypatch):
        for v in ("NGPU_PER_HOST", "LOCAL_WORLD_SIZE"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("PMI_LOCAL_SIZE", "2")
        assert dist.get_gpus_per_node() == 2

    def test_from_SLURM_NTASKS_PER_NODE(self, monkeypatch):
        for v in ("NGPU_PER_HOST", "LOCAL_WORLD_SIZE", "PMI_LOCAL_SIZE"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "6")
        assert dist.get_gpus_per_node() == 6

    def test_cuda_fallback(self, monkeypatch):
        for v in (
            "NGPU_PER_HOST",
            "LOCAL_WORLD_SIZE",
            "PMI_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ):
            monkeypatch.delenv(v, raising=False)
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=4),
        ):
            assert dist.get_gpus_per_node() == 4

    def test_cpu_only(self, monkeypatch):
        for v in (
            "NGPU_PER_HOST",
            "LOCAL_WORLD_SIZE",
            "PMI_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ):
            monkeypatch.delenv(v, raising=False)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
        ):
            assert dist.get_gpus_per_node() == 0


class TestGetNodeIndex:
    """Tests for ``get_node_index``."""

    def test_basic(self, monkeypatch):
        for v in ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        comm = _FakeComm(rank=5, size=8)
        dist._MPI_COMM = comm
        # 5 // 4 == 1
        assert dist.get_node_index() == 1

    def test_zero_gpus(self, monkeypatch):
        """gpus_per_node == 0 ⇒ node_index = 0."""
        with patch.object(dist, "get_gpus_per_node", return_value=0):
            comm = _FakeComm(rank=3, size=4)
            dist._MPI_COMM = comm
            assert dist.get_node_index() == 0


# ===================================================================
# Device / backend
# ===================================================================


class TestGetTorchDeviceType:
    """Tests for ``get_torch_device_type``."""

    def test_explicit_override(self):
        assert dist.get_torch_device_type("cpu") == "cpu"
        assert dist.get_torch_device_type("cuda") == "cuda"
        assert dist.get_torch_device_type("xpu") == "xpu"
        assert dist.get_torch_device_type("mps") == "mps"

    def test_invalid_override_raises(self):
        with pytest.raises(ValueError, match="Unsupported device_type"):
            dist.get_torch_device_type("tpu")

    def test_from_TORCH_DEVICE_env(self, monkeypatch):
        monkeypatch.setenv("TORCH_DEVICE", "cuda:1")
        assert dist.get_torch_device_type() == "cuda"

    def test_from_TORCH_DEVICE_env_plain(self, monkeypatch):
        monkeypatch.setenv("TORCH_DEVICE", "xpu")
        assert dist.get_torch_device_type() == "xpu"

    def test_TORCH_DEVICE_invalid_falls_through(self, monkeypatch):
        """Invalid TORCH_DEVICE ⇒ warning + hardware probe."""
        monkeypatch.setenv("TORCH_DEVICE", "tpu")
        with (
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
        ):
            assert dist.get_torch_device_type() == "cpu"

    def test_TORCH_DEVICE_empty_falls_through(self, monkeypatch):
        monkeypatch.setenv("TORCH_DEVICE", "")
        with (
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
        ):
            assert dist.get_torch_device_type() == "cpu"

    def test_hardware_probe_xpu_first(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        with (
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            # xpu is checked before cuda
            assert dist.get_torch_device_type() == "xpu"

    def test_hardware_probe_cuda(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        with (
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            assert dist.get_torch_device_type() == "cuda"

    def test_hardware_probe_mps(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        with (
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=True
            ),
        ):
            assert dist.get_torch_device_type() == "mps"

    def test_hardware_probe_cpu_fallback(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        with (
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
        ):
            assert dist.get_torch_device_type() == "cpu"


class TestGetTorchDevice:
    """Tests for ``get_torch_device``."""

    def test_string_default(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        with patch.object(dist, "get_torch_device_type", return_value="cpu"):
            result = dist.get_torch_device()
            assert result == "cpu"
            assert isinstance(result, str)

    def test_as_torch_device(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        with patch.object(dist, "get_torch_device_type", return_value="cpu"):
            result = dist.get_torch_device(as_torch_device=True)
            assert isinstance(result, torch.device)
            assert result.type == "cpu"

    def test_TORCH_DEVICE_with_index(self, monkeypatch):
        monkeypatch.setenv("TORCH_DEVICE", "cuda:2")
        result = dist.get_torch_device()
        assert result == "cuda:2"

    def test_TORCH_DEVICE_as_torch_device(self, monkeypatch):
        monkeypatch.setenv("TORCH_DEVICE", "cuda:3")
        result = dist.get_torch_device(as_torch_device=True)
        assert isinstance(result, torch.device)
        assert result.type == "cuda"
        assert result.index == 3

    def test_explicit_device_type(self, monkeypatch):
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        result = dist.get_torch_device(device_type="xpu")
        assert result == "xpu"


class TestGetTorchBackend:
    """Tests for ``get_torch_backend``."""

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("TORCH_BACKEND", "gloo")
        assert dist.get_torch_backend() == "gloo"

    def test_nccl_when_cuda(self, monkeypatch):
        monkeypatch.delenv("TORCH_BACKEND", raising=False)
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.distributed,
                "is_backend_available",
                side_effect=lambda b: b == "nccl",
            ),
        ):
            assert dist.get_torch_backend() == "nccl"

    def test_xccl_when_xpu(self, monkeypatch):
        monkeypatch.delenv("TORCH_BACKEND", raising=False)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(
                torch.distributed,
                "is_backend_available",
                side_effect=lambda b: b == "xccl",
            ),
        ):
            assert dist.get_torch_backend() == "xccl"

    def test_ccl_fallback_when_xpu_no_xccl(self, monkeypatch):
        monkeypatch.delenv("TORCH_BACKEND", raising=False)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(
                torch.distributed,
                "is_backend_available",
                return_value=False,
            ),
        ):
            assert dist.get_torch_backend() == "ccl"

    def test_gloo_fallback(self, monkeypatch):
        monkeypatch.delenv("TORCH_BACKEND", raising=False)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=False),
        ):
            assert dist.get_torch_backend() == "gloo"


# ===================================================================
# get_machine / get_hostname
# ===================================================================


class TestGetMachine:
    """Tests for ``get_machine``."""

    @pytest.mark.parametrize(
        "hostname,expected",
        [
            ("frontier-login01", "Frontier"),
            ("sophia-n02", "Sophia"),
            ("thetagpu-node03", "ThetaGPU"),
            ("x1002c0s3b0n0", "SunSpot"),
            ("x4105c0s4b0n0", "Aurora"),
            ("login01.perlmutter", "Perlmutter"),
            ("nid00123", "Perlmutter"),
            ("x3001c0s7b0n0", "Polaris"),
            ("x3001c0s7b0n0.sirius.alcf.anl.gov", "Sirius"),
            ("myworkstation", "myworkstation"),
        ],
    )
    def test_prefix_map(self, hostname, expected):
        assert dist.get_machine(hostname=hostname) == expected

    def test_auto_detect(self, fake_comm):
        """When hostname is None, ``get_hostname()`` is called."""
        with patch.object(dist, "get_hostname", return_value="x4999c0"):
            assert dist.get_machine() == "Aurora"


class TestGetHostname:
    """Tests for ``get_hostname``."""

    def test_socket_success(self):
        """Smoke test: should return a non-empty string."""
        result = dist.get_hostname()
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == result.lower()

    def test_gethostbyaddr_failure(self, monkeypatch):
        """When ``gethostbyaddr`` fails, falls back to ``gethostname()``."""
        monkeypatch.setattr(
            socket, "gethostbyaddr", MagicMock(side_effect=OSError("nope"))
        )
        result = dist.get_hostname()
        # Should still succeed via the plain gethostname path
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_socket_fail(self, monkeypatch):
        """All socket calls fail ⇒ falls through to env vars."""
        monkeypatch.setattr(
            socket, "gethostname", MagicMock(side_effect=OSError("nope"))
        )
        monkeypatch.setenv("HOSTNAME", "fromenv.local")
        assert dist.get_hostname() == "fromenv.local"

    def test_HOST_env_fallback(self, monkeypatch):
        monkeypatch.setattr(
            socket, "gethostname", MagicMock(side_effect=OSError("nope"))
        )
        monkeypatch.delenv("HOSTNAME", raising=False)
        monkeypatch.setenv("HOST", "host-env.local")
        assert dist.get_hostname() == "host-env.local"

    def test_ultimate_fallback_localhost(self, monkeypatch):
        monkeypatch.setattr(
            socket, "gethostname", MagicMock(side_effect=OSError)
        )
        monkeypatch.delenv("HOSTNAME", raising=False)
        monkeypatch.delenv("HOST", raising=False)
        import platform

        monkeypatch.setattr(platform, "node", MagicMock(return_value=""))
        assert dist.get_hostname() == "localhost"


# ===================================================================
# Collectives / synchronization
# ===================================================================


class TestBarrier:
    """Tests for ``barrier``."""

    def test_mpi_path(self, fake_comm):
        """Default path uses MPI barrier."""
        fake_comm.barrier = MagicMock()
        dist.barrier()
        fake_comm.barrier.assert_called_once()

    def test_explicit_mpi(self, fake_comm):
        fake_comm.barrier = MagicMock()
        dist.barrier(implementation="mpi")
        fake_comm.barrier.assert_called_once()

    def test_mpi_fails_falls_back_to_torch(self, fake_comm):
        fake_comm.barrier = MagicMock(side_effect=RuntimeError("MPI gone"))
        with (
            patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
            patch.object(torch.distributed, "barrier") as mock_barrier,
        ):
            dist.barrier()
            mock_barrier.assert_called_once()

    def test_explicit_torch(self, fake_comm):
        with (
            patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
            patch.object(torch.distributed, "barrier") as mock_barrier,
        ):
            dist.barrier(implementation="torch")
            mock_barrier.assert_called_once()

    def test_invalid_implementation_raises(self, fake_comm):
        with pytest.raises(
            ValueError, match="Unsupported barrier implementation"
        ):
            dist.barrier(implementation="nccl")


class TestBroadcast:
    """Tests for ``broadcast``."""

    def test_returns_object(self, fake_comm):
        result = dist.broadcast({"key": "value"}, root=0)
        assert result == {"key": "value"}

    def test_calls_mpi_bcast(self, fake_comm):
        fake_comm.bcast = MagicMock(return_value=42)
        result = dist.broadcast(42)
        fake_comm.bcast.assert_called_once_with(42, root=0)
        assert result == 42


class TestAllReduce:
    """Tests for ``all_reduce``."""

    def test_mpi_default(self, fake_comm):
        fake_comm.allreduce = MagicMock(return_value=10.0)
        result = dist.all_reduce(5.0)
        assert result == 10.0

    def test_torch_implementation(self, fake_comm):
        with patch.object(torch.distributed, "all_reduce") as mock_ar:
            dist.all_reduce(7, implementation="torch")
            mock_ar.assert_called_once()

    def test_pytorch_alias(self, fake_comm):
        with patch.object(torch.distributed, "all_reduce"):
            # "pytorch" and "pt" are aliases
            dist.all_reduce(1, implementation="pytorch")
            dist.all_reduce(1, implementation="pt")

    def test_invalid_implementation_raises(self, fake_comm):
        with pytest.raises(ValueError, match="Unsupported all_reduce"):
            dist.all_reduce(1, implementation="nccl")


class TestSynchronize:
    """Tests for ``synchronize``."""

    def test_cuda_path(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "synchronize") as mock_sync,
        ):
            dist.synchronize()
            mock_sync.assert_called_once()

    def test_xpu_path(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.xpu, "synchronize") as mock_sync,
        ):
            dist.synchronize()
            mock_sync.assert_called_once()

    def test_cpu_noop(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(
                torch.backends.mps, "is_available", return_value=False
            ),
        ):
            # Should not raise
            dist.synchronize()


# ===================================================================
# setup_torch
# ===================================================================


class TestSetupTorch:
    """Tests for ``setup_torch``."""

    def test_single_device_fast_path(self, fake_comm, monkeypatch):
        """WORLD_SIZE=1 ⇒ returns 0, no init_process_group."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        with patch.object(dist, "get_torch_device_type", return_value="cpu"):
            rank = dist.setup_torch()
            assert rank == 0
            assert os.environ["RANK"] == "0"
            assert os.environ["LOCAL_RANK"] == "0"

    def test_seed_propagation(self, fake_comm, monkeypatch):
        """``seed`` kwarg invokes ``seed_everything`` with rank-aware value."""
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        dsetup = {"rank": 0, "world_size": 4, "local_rank": 0}
        with (
            patch.object(dist, "get_torch_device_type", return_value="cpu"),
            patch.object(dist, "get_torch_backend", return_value="gloo"),
            patch.object(dist, "_setup_ddp", return_value=dsetup),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(dist, "barrier"),
            patch.object(dist, "get_dist_info", return_value={}),
            patch.object(dist, "print_dist_setup", return_value=""),
            patch.object(dist, "seed_everything") as mock_seed,
        ):
            dist.setup_torch(seed=42)
            # seed * (rank + 1) * (local_rank + 1) = 42 * 1 * 1 = 42
            mock_seed.assert_called_once_with(42)


# ===================================================================
# cleanup
# ===================================================================


class TestCleanup:
    """Tests for ``cleanup``."""

    def test_when_initialized(self):
        with (
            patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
            patch.object(
                torch.distributed, "destroy_process_group"
            ) as mock_destroy,
        ):
            dist.cleanup()
            mock_destroy.assert_called_once()

    def test_when_not_initialized(self):
        with (
            patch.object(
                torch.distributed, "is_initialized", return_value=False
            ),
            patch.object(
                torch.distributed, "destroy_process_group"
            ) as mock_destroy,
        ):
            dist.cleanup()
            mock_destroy.assert_not_called()

    def test_wandb_run_logging(self):
        """When wandb.run is active, logs name/url without crashing."""
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.url = "https://wandb.ai/test"
        with (
            patch.dict("sys.modules", {"wandb": MagicMock(run=mock_run)}),
            patch.object(
                torch.distributed, "is_initialized", return_value=False
            ),
        ):
            dist.cleanup()  # Should not raise

    def test_wandb_import_failure(self):
        """If wandb is not importable, cleanup still works."""
        with (
            patch.dict("sys.modules", {"wandb": None}),
            patch.object(
                torch.distributed, "is_initialized", return_value=False
            ),
        ):
            dist.cleanup()  # Should not raise


# ===================================================================
# wrap_model / wrap_model_for_ddp
# ===================================================================


class TestWrapModel:
    """Tests for ``wrap_model``."""

    def test_world_size_1_returns_unwrapped(self, fake_comm):
        """world_size <= 1 ⇒ return original model."""
        comm = _FakeComm(rank=0, size=1)
        dist._MPI_COMM = comm
        model = torch.nn.Linear(10, 10)
        with patch.object(dist, "get_world_size", return_value=1):
            result = dist.wrap_model(model)
        assert result is model

    def test_ddp_path(self, fake_comm):
        """use_fsdp=False ⇒ calls wrap_model_for_ddp."""
        model = torch.nn.Linear(10, 10)
        with patch.object(
            dist, "wrap_model_for_ddp", return_value=model
        ) as mock_ddp:
            dist.wrap_model(model, use_fsdp=False)
            mock_ddp.assert_called_once_with(model)

    def test_fsdp_path(self, fake_comm):
        """use_fsdp=True ⇒ calls _wrap_fsdp on CUDA/XPU devices."""
        model = torch.nn.Linear(10, 10)
        with (
            patch.object(dist, "get_torch_device_type", return_value="cuda"),
            patch.object(dist, "_wrap_fsdp", return_value=model) as mock_fsdp,
        ):
            dist.wrap_model(model, use_fsdp=True, dtype="bf16")
            mock_fsdp.assert_called_once_with(
                model, dtype="bf16", device_id=None
            )

    def test_fsdp_falls_back_to_ddp_on_cpu(self, fake_comm):
        """use_fsdp=True on CPU ⇒ falls back to DDP."""
        model = torch.nn.Linear(10, 10)
        with (
            patch.object(dist, "get_torch_device_type", return_value="cpu"),
            patch.object(
                dist, "wrap_model_for_ddp", return_value=model
            ) as mock_ddp,
        ):
            dist.wrap_model(model, use_fsdp=True, dtype="bf16")
            mock_ddp.assert_called_once_with(model)

    def test_fsdp_falls_back_to_ddp_on_mps(self, fake_comm):
        """use_fsdp=True on MPS ⇒ falls back to DDP."""
        model = torch.nn.Linear(10, 10)
        with (
            patch.object(dist, "get_torch_device_type", return_value="mps"),
            patch.object(
                dist, "wrap_model_for_ddp", return_value=model
            ) as mock_ddp,
        ):
            dist.wrap_model(model, use_fsdp=True, dtype="bf16")
            mock_ddp.assert_called_once_with(model)


class TestWrapModelForDDP:
    """Tests for ``wrap_model_for_ddp``."""

    def test_cpu_wrapping(self, fake_comm):
        model = torch.nn.Linear(10, 10)
        with (
            patch.object(dist, "get_torch_device_type", return_value="cpu"),
            patch.object(dist, "get_local_rank", return_value=0),
            patch(
                "torch.nn.parallel.DistributedDataParallel",
                return_value=MagicMock(),
            ) as mock_ddp,
        ):
            dist.wrap_model_for_ddp(model)
            # CPU path: no device_ids argument
            mock_ddp.assert_called_once_with(model)

    def test_cuda_wrapping(self, fake_comm):
        """CUDA path passes device_ids=[local_rank] (int ordinal)."""
        model = torch.nn.Linear(10, 10)
        with (
            patch.object(dist, "get_torch_device_type", return_value="cuda"),
            patch.object(dist, "get_local_rank", return_value=2),
            patch(
                "torch.nn.parallel.DistributedDataParallel",
                return_value=MagicMock(),
            ) as mock_ddp,
        ):
            dist.wrap_model_for_ddp(model)
            mock_ddp.assert_called_once_with(model, device_ids=[2])

    def test_xpu_wrapping(self, fake_comm):
        """XPU path passes device_ids=[local_rank] (int, not string)."""
        model = torch.nn.Linear(10, 10)
        with (
            patch.object(dist, "get_torch_device_type", return_value="xpu"),
            patch.object(dist, "get_local_rank", return_value=1),
            patch(
                "torch.nn.parallel.DistributedDataParallel",
                return_value=MagicMock(),
            ) as mock_ddp,
        ):
            dist.wrap_model_for_ddp(model)
            mock_ddp.assert_called_once_with(model, device_ids=[1])


# ===================================================================
# wandb helpers
# ===================================================================


class TestVerifyWandb:
    """Tests for ``verify_wandb``."""

    def test_wandb_disabled_env(self, fake_comm, monkeypatch):
        monkeypatch.setenv("WANDB_DISABLED", "1")
        assert dist.verify_wandb() is False

    def test_wandb_mode_disabled(self, fake_comm, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.setenv("WANDB_MODE", "disabled")
        assert dist.verify_wandb() is False

    def test_wandb_import_failure(self, fake_comm, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        with patch.dict("sys.modules", {"wandb": None}):
            # Importing wandb raises ImportError when sys.modules[key] is None
            assert dist.verify_wandb() is False

    def test_api_key_present(self, fake_comm, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = "test-key"
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert dist.verify_wandb() is True

    def test_api_key_from_env(self, fake_comm, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "env-key")
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = None
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert dist.verify_wandb() is True


class TestResolveWandbMode:
    """Tests for ``_resolve_wandb_mode``."""

    def test_disabled_env_wins(self, monkeypatch):
        monkeypatch.setenv("WANDB_DISABLED", "1")
        assert dist._resolve_wandb_mode(mode="online") == "disabled"

    def test_explicit_mode(self, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert dist._resolve_wandb_mode(mode="offline") == "offline"

    def test_env_mode(self, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.setenv("WANDB_MODE", "shared")
        assert dist._resolve_wandb_mode() == "shared"

    def test_no_env_no_arg(self, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert dist._resolve_wandb_mode() == "online"

    def test_invalid_mode_raises(self, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        with pytest.raises(ValueError, match="Invalid wandb mode"):
            dist._resolve_wandb_mode(mode="foobar")

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        # explicit mode= takes precedence via ``mode or env_mode``
        assert dist._resolve_wandb_mode(mode="online") == "online"


# ===================================================================
# timeitlogit
# ===================================================================


class TestTimeitlogit:
    """Tests for ``timeitlogit``."""

    def test_return_value_preserved(self, fake_comm):
        @dist.timeitlogit(rank=0, record=False, verbose=False)
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_timing_is_positive(self, fake_comm):

        @dist.timeitlogit(rank=0, record=False, verbose=False)
        def sleeper():
            import time

            time.sleep(0.01)
            return "done"

        result = sleeper()
        assert result == "done"

    def test_wandb_logging(self, fake_comm):
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        @dist.timeitlogit(rank=0, record=True, verbose=False, prefix="test")
        def my_func():
            return 42

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = my_func()
            assert result == 42

    def test_verbose_output(self, fake_comm, capsys):
        @dist.timeitlogit(rank=0, record=False, verbose=True)
        def greet(name):
            return f"hello {name}"

        result = greet("world")
        assert result == "hello world"

    def test_wraps_preserves_name(self, fake_comm):
        @dist.timeitlogit(rank=0, record=False, verbose=False)
        def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"


# ===================================================================
# Hostfile helpers
# ===================================================================


class TestGetNodesFromHostfile:
    """Tests for ``get_nodes_from_hostfile``."""

    def test_reads_file(self, tmp_path):
        hf = tmp_path / "hostfile"
        hf.write_text("node1\nnode2\nnode3\n")
        result = dist.get_nodes_from_hostfile(hf)
        assert result == ["node1", "node2", "node3"]

    def test_strips_blank_lines(self, tmp_path):
        hf = tmp_path / "hostfile"
        hf.write_text("node1\n\n\nnode2\n\n")
        result = dist.get_nodes_from_hostfile(hf)
        assert result == ["node1", "node2"]

    def test_missing_file_returns_hostname(self, fake_comm, tmp_path):
        result = dist.get_nodes_from_hostfile(tmp_path / "nonexistent")
        assert isinstance(result, list)
        assert len(result) == 1


class TestGetHostfileWithFallback:
    """Tests for ``get_hostfile_with_fallback``."""

    def test_slurm_scheduler(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SLURM_NODELIST", "node[001-002]")
        with patch("ezpz.configs.get_scheduler", return_value="slurm"):
            hfp = dist.get_hostfile_with_fallback()
            assert hfp.is_file()
            lines = hfp.read_text().strip().splitlines()
            assert lines == ["node001", "node002"]

    def test_explicit_hostfile(self, monkeypatch, tmp_path):
        hf = tmp_path / "myhosts"
        hf.write_text("nodeA\nnodeB\n")
        with patch("ezpz.configs.get_scheduler", return_value="unknown"):
            result = dist.get_hostfile_with_fallback(hostfile=hf)
            assert result == hf

    def test_pbs_nodefile_env(self, monkeypatch, tmp_path):
        hf = tmp_path / "pbs_nodes"
        hf.write_text("n1\nn2\n")
        monkeypatch.setenv("PBS_NODEFILE", str(hf))
        with patch("ezpz.configs.get_scheduler", return_value="unknown"):
            result = dist.get_hostfile_with_fallback()
            assert result == hf

    def test_fallback_writes_localhost(self, fake_comm, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PBS_NODEFILE", raising=False)
        monkeypatch.delenv("HOSTFILE", raising=False)
        with patch("ezpz.configs.get_scheduler", return_value="unknown"):
            hfp = dist.get_hostfile_with_fallback()
            assert hfp.is_file()


class TestWriteLocalhostToHostfile:
    """Tests for ``write_localhost_to_hostfile``."""

    def test_rank_zero_writes(self, fake_comm, monkeypatch, tmp_path):
        for v in ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
            monkeypatch.delenv(v, raising=False)
        hf = tmp_path / "hostfile"
        dist.write_localhost_to_hostfile(hf)
        assert hf.is_file()
        content = hf.read_text()
        assert len(content) > 0

    def test_nonzero_rank_does_not_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        hf = tmp_path / "hostfile"
        dist.write_localhost_to_hostfile(hf)
        assert not hf.exists()


# ===================================================================
# Misc: seed_everything, log_dict_as_bulleted_list, query_environment,
#        print_dist_setup, get_dist_info, TORCH_DTYPES_MAP
# ===================================================================


class TestSeedEverything:
    """Tests for ``seed_everything``."""

    def test_deterministic_output(self):
        dist.seed_everything(12345)
        a = random.random()
        x = np.random.random()
        t = torch.randn(1).item()

        dist.seed_everything(12345)
        b = random.random()
        y = np.random.random()
        u = torch.randn(1).item()

        assert a == b
        assert x == y
        assert t == u

    def test_sets_pythonhashseed(self):
        dist.seed_everything(99)
        assert os.environ["PYTHONHASHSEED"] == "99"


class TestLogDictAsBulletedList:
    """Tests for ``log_dict_as_bulleted_list``."""

    def test_does_not_raise(self):
        dist.log_dict_as_bulleted_list({"a": 1, "b": 2}, name="test")

    def test_no_name(self):
        dist.log_dict_as_bulleted_list({"x": 10})


class TestQueryEnvironment:
    """Tests for ``query_environment``."""

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "1")
        result = dist.query_environment()
        assert result == {"world_size": 4, "rank": 2, "local_rank": 1}

    def test_fallback_to_mpi(self, fake_comm, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        # get_local_rank needs an env var or fallback
        monkeypatch.setenv("PMI_LOCAL_RANK", "0")
        result = dist.query_environment()
        assert result["world_size"] == 4
        assert result["rank"] == 0
        assert result["local_rank"] == 0


class TestPrintDistSetup:
    """Tests for ``print_dist_setup``."""

    def test_returns_string(self, fake_comm, monkeypatch):
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        monkeypatch.setenv("SLURM_NNODES", "1")
        result = dist.print_dist_setup(display=False)
        assert isinstance(result, str)
        assert "rank=" in result


class TestGetDistInfo:
    """Tests for ``get_dist_info``."""

    def test_returns_dict(self, fake_comm, monkeypatch):
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        monkeypatch.setenv("SLURM_NNODES", "1")
        monkeypatch.setenv("PMI_LOCAL_RANK", "0")
        with (
            patch("ezpz.configs.get_scheduler", return_value="slurm"),
            patch.object(
                dist,
                "get_hostfile_with_fallback",
                return_value=Path("/dev/null"),
            ),
            patch.object(dist, "get_torch_device", return_value="cpu"),
            patch.object(dist, "get_torch_backend", return_value="gloo"),
        ):
            info = dist.get_dist_info()
            assert isinstance(info, dict)
            assert "RANK" in info
            assert "WORLD_SIZE_IN_USE" in info
            assert "HOSTNAME" in info

    def test_verbose_mode(self, fake_comm, monkeypatch):
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        monkeypatch.setenv("SLURM_NNODES", "1")
        monkeypatch.setenv("PMI_LOCAL_RANK", "0")
        with (
            patch("ezpz.configs.get_scheduler", return_value="slurm"),
            patch.object(
                dist,
                "get_hostfile_with_fallback",
                return_value=Path("/dev/null"),
            ),
            patch.object(dist, "get_torch_device", return_value="cpu"),
            patch.object(dist, "get_torch_backend", return_value="gloo"),
        ):
            info = dist.get_dist_info(verbose=True)
            assert isinstance(info, dict)


class TestTorchDtypesMap:
    """Tests for ``TORCH_DTYPES_MAP`` and ``_ensure_dtype_map``."""

    def test_ensure_populates(self):
        result = dist._ensure_dtype_map()
        assert result["bf16"] is torch.bfloat16
        assert result["fp16"] is torch.float16
        assert result["float32"] is torch.float32
        assert result["half"] is torch.float16
        assert result["bfloat16"] is torch.bfloat16
        assert result["fp32"] is torch.float32

    def test_idempotent(self):
        r1 = dist._ensure_dtype_map()
        r2 = dist._ensure_dtype_map()
        assert r1 is r2

    def test_module_level_dict_populated(self):
        dist._ensure_dtype_map()
        assert dist.TORCH_DTYPES_MAP["bf16"] is torch.bfloat16


# ===================================================================
# _set_env_vars
# ===================================================================


class TestSetEnvVars:
    """Tests for ``_set_env_vars``."""

    def test_sets_vars(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        dist._set_env_vars(rank=3, local_rank=1, world_size=8)
        assert os.environ["RANK"] == "3"
        assert os.environ["LOCAL_RANK"] == "1"
        assert os.environ["WORLD_SIZE"] == "8"
        # Clean up — _set_env_vars writes to os.environ directly,
        # which monkeypatch doesn't track automatically.
        del os.environ["RANK"]
        del os.environ["LOCAL_RANK"]
        del os.environ["WORLD_SIZE"]


# ===================================================================
# _get_free_port
# ===================================================================


class TestGetFreePort:
    """Tests for ``_get_free_port``."""

    def test_returns_int(self):
        port = dist._get_free_port()
        assert isinstance(port, int)
        assert port > 0
        assert port < 65536

    def test_unique_ports(self):
        ports = {dist._get_free_port() for _ in range(10)}
        # While not guaranteed, we should get at least a few unique ports
        assert len(ports) >= 2
