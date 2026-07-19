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
        "LOCAL_RANK", "PALS_LOCAL_RANKID", "PMIX_LOCAL_RANK",
        "PMI_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
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

    def test_from_PALS_LOCAL_RANKID(self, fake_comm, monkeypatch):
        """cray-pals mpiexec (Aurora/Sunspot) sets ``PALS_LOCAL_RANKID``.

        Regression for the COMPOSITE device-index crash: without this
        var recognized, ``get_local_rank`` fell through to the modulo
        fallback and computed an out-of-range device index.
        """
        for v in ("LOCAL_RANK", "PMI_LOCAL_RANK"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("PALS_LOCAL_RANKID", "5")
        assert dist.get_local_rank() == 5

    def test_from_PMIX_LOCAL_RANK(self, fake_comm, monkeypatch):
        """PMIx layer (Aurora/Sunspot) sets ``PMIX_LOCAL_RANK``.

        ezpz's own ``affinity.sh`` reads this var, confirming it is
        exported per-rank on the ALCF XPU systems.
        """
        for v in ("LOCAL_RANK", "PMI_LOCAL_RANK", "PALS_LOCAL_RANKID"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("PMIX_LOCAL_RANK", "3")
        assert dist.get_local_rank() == 3

    def test_pals_beats_modulo_fallback_under_composite(
        self, monkeypatch
    ):
        """The real-world COMPOSITE crash: -ppn 6 on a 12-device node.

        ``NGPU_PER_HOST`` is a stale 12 (FLAT-mode constant), device
        count is 6 (COMPOSITE), and this is rank 11's process. The old
        modulo fallback would return ``11 % 12 == 11`` (out of range for
        ``xpu.set_device`` on 6 composite GPUs). With PALS_LOCAL_RANKID
        recognized, we get the launcher's authoritative local rank (5).
        """
        for v in (
            "LOCAL_RANK",
            "PMI_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MPICH_LOCALRANKID",
            "SLURM_LOCAL_ID",
        ):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("NGPU_PER_HOST", "12")  # stale FLAT constant
        monkeypatch.setenv("PALS_LOCAL_RANKID", "5")  # true local rank
        comm = _FakeComm(rank=11, size=24)
        dist._MPI_COMM = comm
        assert dist.get_local_rank() == 5

    def test_fallback_world_size_1(self, monkeypatch):
        """world_size == 1 ⇒ local_rank = 0."""
        for v in (
            "LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            "PMIX_LOCAL_RANK",
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
        """No env vars, world_size > 1 ⇒ rank % ranks_per_node."""
        for v in (
            "LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            "PMIX_LOCAL_RANK",
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
        # ranks_per_node = world_size(8) // num_nodes(2) == 4; 5 % 4 == 1
        with patch.object(dist, "get_num_nodes", return_value=2):
            assert dist.get_local_rank() == 1

    def test_fallback_ranks_per_node_not_device_count(self, monkeypatch):
        """Fallback uses ranks-per-node, not device-count.

        Regression for the COMPOSITE crash's *second* defect: when no
        local-rank env var is set and ranks-per-node (6) is less than
        the device count / NGPU_PER_HOST (12), the modulo must use the
        actual ranks-per-node so the result stays in ``[0, ppn)``.
        Rank 23 of a ``-n 24 -ppn 6`` (4-node) run is local rank 5, not
        ``23 % 12 == 11`` (out of range).
        """
        for v in (
            "LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            "PMIX_LOCAL_RANK",
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
        monkeypatch.setenv("NGPU_PER_HOST", "12")  # stale FLAT constant
        comm = _FakeComm(rank=23, size=24)
        dist._MPI_COMM = comm
        # ranks_per_node = 24 // 4 == 6; 23 % 6 == 5 (in range [0,6))
        with patch.object(dist, "get_num_nodes", return_value=4):
            assert dist.get_local_rank() == 5

    def test_fallback_cpu_single_node_uses_ranks_per_node(self, monkeypatch):
        """CPU-only, no devices: ranks-per-node still gives distinct ranks.

        ``mpirun -n 4`` on one CPU node (no accelerators, no local-rank
        env var) should yield local ranks 0..3, not 0 for everyone. The
        ranks-per-node fallback (world_size // num_nodes) delivers this
        even when device probing returns 0.
        """
        for v in (
            "LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            "PMIX_LOCAL_RANK",
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
        with patch.object(dist, "get_num_nodes", return_value=1):
            assert dist.get_local_rank() == 3  # 3 % (4 // 1)

    def test_fallback_indeterminate_returns_zero(self, monkeypatch):
        """No env var, can't determine nnodes, no devices ⇒ 0 (no crash).

        When ranks-per-node can't be derived (``get_num_nodes`` returns 0)
        and device probing yields 0, the modulo guard returns 0 rather
        than raising ``ZeroDivisionError``.
        """
        for v in (
            "LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            "PMIX_LOCAL_RANK",
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
            patch.object(dist, "get_num_nodes", return_value=0),
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

    def test_composite_clamps_stale_ngpu_per_host(self, monkeypatch):
        """Stale ``NGPU_PER_HOST=12`` (FLAT) clamps to XPU count under COMPOSITE.

        Regression for the COMPOSITE device-index skew: ezpz's shell sets
        ``NGPU_PER_HOST=12`` (FLAT tile count), but under
        ``ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`` only 6 composite GPUs
        exist. ``get_gpus_per_node`` must not over-report devices — that
        would skew ``get_node_index`` and the per-host summary line.
        """
        monkeypatch.setenv("NGPU_PER_HOST", "12")
        with (
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.xpu, "device_count", return_value=6),
        ):
            assert dist.get_gpus_per_node() == 6

    def test_flat_no_clamp_when_env_matches_devices(self, monkeypatch):
        """FLAT mode: ``NGPU_PER_HOST=12`` == device count ⇒ unchanged."""
        monkeypatch.setenv("NGPU_PER_HOST", "12")
        with (
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.xpu, "device_count", return_value=12),
        ):
            assert dist.get_gpus_per_node() == 12

    def test_under_subscription_not_bumped_up(self, monkeypatch):
        """A smaller env hint is a deliberate under-subscription — keep it.

        ``-ppn 4`` on a 12-device node sets ``NGPU_PER_HOST=4``; we must
        never bump it *up* to the device count. Only clamp down.
        """
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        with (
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.xpu, "device_count", return_value=12),
        ):
            assert dist.get_gpus_per_node() == 4

    def test_login_node_probe_error_keeps_env_hint(self, monkeypatch):
        """XPU probe errors (login node, no Level-Zero loader) ⇒ keep env.

        ``get_gpus_per_node`` runs during launch-time topology inference
        on the login node, where ``torch.xpu.is_available()`` may raise.
        The env hint must survive so we don't clamp it to 0.
        """
        monkeypatch.setenv("NGPU_PER_HOST", "12")
        with patch.object(
            torch.xpu, "is_available", side_effect=RuntimeError("no libze")
        ):
            assert dist.get_gpus_per_node() == 12

    def test_cuda_env_hint_not_clamped(self, monkeypatch):
        """Clamp is XPU-only: CUDA env-hint behavior is byte-identical.

        COMPOSITE is an Intel/XPU concept; CUDA has no equivalent split,
        so a CUDA env hint is returned verbatim even if it exceeds the
        visible CUDA device count (e.g. a single-GPU CI box).
        """
        monkeypatch.setenv("NGPU_PER_HOST", "8")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
        ):
            assert dist.get_gpus_per_node() == 8


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


class TestUseTorchcomms:
    """Tests for ``use_torchcomms`` and its availability probe."""

    def setup_method(self):
        # Each test starts with a clean probe cache.
        dist._reset_torchcomms_cache()

    def teardown_method(self):
        dist._reset_torchcomms_cache()

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("EZPZ_USE_TORCHCOMMS", raising=False)
        assert dist.use_torchcomms() is False
        assert dist._torchcomms_unavailable_reason() == ""

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_and_available(self, monkeypatch, val):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", val)
        fake_tc = MagicMock()
        fake_cfg = MagicMock()
        fake_cfg.use_torchcomms = False  # attr must EXIST
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", fake_cfg, create=True),
        ):
            assert dist.use_torchcomms() is True
            assert dist._torchcomms_unavailable_reason() == ""

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "off"])
    def test_falsy_is_false(self, monkeypatch, val):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", val)
        assert dist.use_torchcomms() is False

    def test_requested_but_package_missing(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        with patch.dict("sys.modules", {"torchcomms": None}):
            # sys.modules[name] = None makes `import torchcomms` raise ImportError
            assert dist.use_torchcomms() is False
            assert "torchcomms" in dist._torchcomms_unavailable_reason().lower()

    def test_requested_but_torch_switch_absent(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        fake_tc = MagicMock()
        cfg_without_switch = MagicMock(spec=[])  # no use_torchcomms attr
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(
                torch.distributed, "config", cfg_without_switch, create=True
            ),
        ):
            assert dist.use_torchcomms() is False
            assert dist._torchcomms_unavailable_reason() != ""

    def test_probe_is_cached(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        fake_tc = MagicMock()
        fake_cfg = MagicMock()
        fake_cfg.use_torchcomms = False
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", fake_cfg, create=True),
        ):
            assert dist.use_torchcomms() is True
        # After the patches exit, torchcomms would look unavailable — but the
        # cached True result must persist (probe ran once).
        assert dist.use_torchcomms() is True

    def test_activation_sets_flag_when_available(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        fake_tc = MagicMock()
        fake_cfg = MagicMock()
        fake_cfg.use_torchcomms = False
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", fake_cfg, create=True),
        ):
            applied = dist._maybe_enable_torchcomms(rank=0, backend="nccl")
        assert applied is True
        assert fake_cfg.use_torchcomms is True

    def test_activation_warns_when_requested_unavailable(
        self, monkeypatch, caplog
    ):
        import logging

        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        with patch.dict("sys.modules", {"torchcomms": None}):
            with caplog.at_level(logging.WARNING, logger="ezpz.distributed"):
                applied = dist._maybe_enable_torchcomms(rank=0, backend="xccl")
        assert applied is False
        assert any(
            "EZPZ_USE_TORCHCOMMS" in r.getMessage() for r in caplog.records
        )

    def test_activation_noop_when_unset(self, monkeypatch):
        monkeypatch.delenv("EZPZ_USE_TORCHCOMMS", raising=False)
        assert dist._maybe_enable_torchcomms(rank=0, backend="nccl") is False


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

    def test_xpu_set_device_called_before_setup_ddp(
        self, fake_comm, monkeypatch
    ):
        """Regression: xpu.set_device(local_rank) MUST run before _setup_ddp.

        Pre-fix the order was reversed — _setup_ddp ran first, then
        set_device. On XPU that meant init_process_group constructed
        the PG with current_device == xpu:0 on every rank (because
        set_device hadn't happened yet); subsequent collectives
        routed to different XPU queues per rank and FSDP2 deadlocked.

        Use a shared `calls` list to record the order of relevant
        events, then assert set_device fires BEFORE _setup_ddp.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "3")  # pre_local_rank == 3
        calls: list[str] = []
        dsetup = {"rank": 0, "world_size": 4, "local_rank": 3}

        def _record_setup_ddp(*args, **kwargs):
            calls.append("_setup_ddp")
            return dsetup

        def _record_xpu_set_device(rank):
            calls.append(f"xpu.set_device({rank})")

        with (
            patch.object(dist, "get_torch_device_type", return_value="xpu"),
            patch.object(dist, "get_torch_backend", return_value="ccl"),
            patch.object(dist, "_setup_ddp", side_effect=_record_setup_ddp),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(
                torch.xpu, "set_device", side_effect=_record_xpu_set_device
            ),
            patch.object(dist, "barrier"),
            patch.object(dist, "get_dist_info", return_value={}),
            patch.object(dist, "print_dist_setup", return_value=""),
            patch.object(dist, "seed_everything"),
        ):
            dist.setup_torch()

        # Exactly one set_device call (the pre-init one); local_rank
        # didn't change after _setup_ddp so the post-init re-set
        # short-circuits.
        assert calls == ["xpu.set_device(3)", "_setup_ddp"], (
            f"Wrong call order: {calls}. xpu.set_device MUST run "
            "BEFORE _setup_ddp so the process group binds to the "
            "right XPU device. Without this, FSDP2 deadlocks on the "
            "first all_gather_into_tensor (caught on Aurora job 8518207)."
        )

    def test_setup_torch_with_explicit_device_id_binds_that_device(
        self, fake_comm, monkeypatch
    ):
        """``setup_torch(device_id=N)`` must set xpu:N, not xpu:LOCAL_RANK.

        Regression for PR #149 Copilot comment: if the caller passes
        an explicit ``device_id``, ``_setup_ddp`` will bind the PG to
        that device. The pre-init ``set_device`` MUST therefore also
        use ``device_id`` (not raw ``LOCAL_RANK``); otherwise the
        current device and the PG's bound device disagree and we
        reintroduce the exact XPU FSDP2 hang this PR is fixing.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "5")  # raw local_rank
        seen: list[int] = []
        dsetup = {"rank": 0, "world_size": 8, "local_rank": 5}

        with (
            patch.object(dist, "get_torch_device_type", return_value="xpu"),
            patch.object(dist, "get_torch_backend", return_value="ccl"),
            patch.object(dist, "_setup_ddp", return_value=dsetup),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(
                torch.xpu,
                "set_device",
                side_effect=lambda i: seen.append(i),
            ),
            patch.object(dist, "barrier"),
            patch.object(dist, "get_dist_info", return_value={}),
            patch.object(dist, "print_dist_setup", return_value=""),
            patch.object(dist, "seed_everything"),
        ):
            dist.setup_torch(device_id=2)

        # Exactly one set_device call, to the caller-provided
        # device_id=2 — NOT to LOCAL_RANK=5. The post-init re-set
        # short-circuits because pre and post resolved to the same
        # index (device_id wins both times).
        assert seen == [2], (
            f"Expected [2] (caller's explicit device_id), got {seen}. "
            "Either pre-init or post-init re-set fell back to LOCAL_RANK; "
            "current device and PG-bound device will disagree and FSDP2 "
            "will hang on XPU."
        )

    def test_setup_torch_post_init_resets_only_when_resolved_changes(
        self, fake_comm, monkeypatch
    ):
        """Post-init re-set fires only when the resolved device changed.

        Regression for PR #149 Copilot comment on the post-init path:
        the comparison MUST be "did the resolved device index change
        between pre and post-init?", not "did LOCAL_RANK change?".

        Set up a case where ``_setup_ddp`` reports a different
        ``local_rank`` than ``get_local_rank()`` did pre-init.  With
        no explicit device_id, both resolutions fall back to local_rank
        — pre uses the env-var value, post uses the dsetup value — so
        the indices DIFFER and the post-init re-set must fire.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "3")  # pre_local_rank == 3
        # _setup_ddp reports local_rank=7 (mismatch — extremely rare
        # in practice, but the regression we want to pin)
        dsetup = {"rank": 0, "world_size": 12, "local_rank": 7}
        seen: list[int] = []

        with (
            patch.object(dist, "get_torch_device_type", return_value="xpu"),
            patch.object(dist, "get_torch_backend", return_value="ccl"),
            patch.object(dist, "_setup_ddp", return_value=dsetup),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(
                torch.xpu,
                "set_device",
                side_effect=lambda i: seen.append(i),
            ),
            patch.object(dist, "barrier"),
            patch.object(dist, "get_dist_info", return_value={}),
            patch.object(dist, "print_dist_setup", return_value=""),
            patch.object(dist, "seed_everything"),
        ):
            dist.setup_torch()

        assert seen == [3, 7], (
            f"Expected [3, 7] (pre-init bound to LOCAL_RANK env var, "
            f"post-init re-bound to _setup_ddp's resolved local_rank), "
            f"got {seen}. The post-init path either skipped the re-set "
            "(comparison against raw LOCAL_RANK lost the divergence) "
            "or fired unnecessarily."
        )


# ===================================================================
# _set_local_device
# ===================================================================


class TestSetLocalDevice:
    """``_set_local_device`` is the single source of truth for setting
    the per-process current accelerator device. Both pre- and post-
    ``init_process_group`` paths in ``setup_torch`` route through it.

    Contract:
      * On ``cuda`` + cuda available → ``torch.cuda.set_device(N)``.
      * On ``xpu`` + xpu attr present + xpu available →
        ``torch.xpu.set_device(N)``.
      * On any other device_type (``cpu``, ``mps``, ``hip``) → no-op.
      * On builds where ``torch.xpu`` doesn't exist (Sourcery #1
        regression) → no-op without ``AttributeError``.
    """

    def test_cuda_when_available(self, monkeypatch):
        seen: list[int] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "set_device", lambda i: seen.append(i)
        )
        dist._set_local_device("cuda", 3)
        assert seen == [3]

    def test_cuda_not_available_is_noop(self, monkeypatch):
        seen: list[int] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            torch.cuda,
            "set_device",
            MagicMock(side_effect=AssertionError("must not be called")),
        )
        dist._set_local_device("cuda", 3)
        assert seen == []

    def test_xpu_when_available(self, monkeypatch):
        seen: list[int] = []
        monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.xpu, "set_device", lambda i: seen.append(i)
        )
        dist._set_local_device("xpu", 5)
        assert seen == [5]

    def test_xpu_not_available_is_noop(self, monkeypatch):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
        monkeypatch.setattr(
            torch.xpu,
            "set_device",
            MagicMock(side_effect=AssertionError("must not be called")),
        )
        dist._set_local_device("xpu", 5)
        # No exception, no call.

    def test_xpu_attr_missing_is_noop(self, monkeypatch):
        """Sourcery #1 regression: torch.xpu may not exist on every build.

        CPU-only torch, CUDA-only nightlies, ROCm — none of these
        define ``torch.xpu``. A naive ``torch.xpu.is_available()``
        call would AttributeError before the availability check;
        ``_set_local_device`` must use ``hasattr(torch, "xpu")`` first.
        """
        monkeypatch.delattr(torch, "xpu", raising=False)
        # Must not raise.
        dist._set_local_device("xpu", 5)

    def test_non_accelerator_device_type_is_noop(self):
        """``cpu`` / ``mps`` / ``hip`` have nothing to bind.

        No env mocking needed — if the function tries to do anything,
        it'll AttributeError on ``torch.cpu.set_device`` (which doesn't
        exist).
        """
        dist._set_local_device("cpu", 0)
        dist._set_local_device("mps", 0)
        dist._set_local_device("hip", 0)


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
        """use_fsdp=True ⇒ calls _wrap_fsdp2 (FSDP2) on CUDA/XPU devices."""
        model = torch.nn.Linear(10, 10)
        mock_mesh = MagicMock()
        with (
            patch.object(dist, "get_torch_device_type", return_value="cuda"),
            patch.object(dist, "_wrap_fsdp2", return_value=model) as mock_fsdp2,
            patch(
                "torch.distributed.device_mesh.init_device_mesh",
                return_value=mock_mesh,
            ),
        ):
            dist.wrap_model(model, use_fsdp=True, dtype="bf16")
            mock_fsdp2.assert_called_once_with(
                model, dtype="bf16", device_mesh=mock_mesh,
                reshard_after_forward=True,
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

    def test_wandb_mode_offline_no_api_key(self, fake_comm, monkeypatch):
        """Regression: WANDB_MODE=offline must NOT require an API key.

        Offline runs write locally and never touch the network — that's
        the whole point of the mode, and it's the documented compute-node
        workflow (docs/troubleshooting.md). Pre-fix verify_wandb()
        returned False here, which caused setup_wandb() to silently
        skip wandb init on every compute-node offline run.
        """
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = None
        # Make sure netrc check would also fail so we're exercising
        # the offline-exception branch, not the netrc fallback.
        monkeypatch.setattr(
            "pathlib.Path.is_file", lambda self: False
        )
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

    def test_dist_info_includes_torchcomms(self, fake_comm, monkeypatch):
        monkeypatch.delenv("EZPZ_USE_TORCHCOMMS", raising=False)
        monkeypatch.setenv("NGPU_PER_HOST", "4")
        monkeypatch.setenv("SLURM_NNODES", "1")
        monkeypatch.setenv("PMI_LOCAL_RANK", "0")
        dist._reset_torchcomms_cache()
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
            assert "TORCHCOMMS" in info
            assert info["TORCHCOMMS"] is False


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


# ===================================================================
# _get_ezpz_git_sha
# ===================================================================


class TestGetEzpzGitSha:
    """Tests for the best-effort git-sha lookup used by setup_wandb."""

    def test_returns_short_sha_or_none(self):
        # Real call against the actual checkout. On a dev machine this
        # returns the short SHA; in a pip-install / non-git context
        # returns None. Either is fine — we just verify the type.
        sha = dist._get_ezpz_git_sha()
        assert sha is None or (isinstance(sha, str) and 7 <= len(sha) <= 40)
        if sha is not None:
            # short sha is hex-only (no newlines, no whitespace)
            assert sha.isalnum()

    def test_handles_subprocess_failure_safely(self, monkeypatch):
        # Simulate `git` not on PATH / non-zero exit: must return None,
        # not raise. setup_wandb is allowed to keep going even when we
        # can't compute the SHA.
        import subprocess

        def _explode(*_args, **_kwargs):
            raise FileNotFoundError("git: command not found")

        monkeypatch.setattr(subprocess, "run", _explode)
        assert dist._get_ezpz_git_sha() is None

    def test_handles_timeout_safely(self, monkeypatch):
        import subprocess

        def _timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=2.0)

        monkeypatch.setattr(subprocess, "run", _timeout)
        assert dist._get_ezpz_git_sha() is None


# ===================================================================
# _setup_ddp: device_id resolution
# ===================================================================


class TestBuildWandbSettings:
    """_build_wandb_settings must not forward ``start_method`` by default.

    Regression for the wandb deprecation noise: every default run
    used to emit
        wandb: WARNING `start_method` is deprecated and will be
        removed in a future version of wandb. This setting is
        currently non-functional and safely ignored.
    because we always passed ``start_method="fork"`` to
    ``wandb.Settings``. The fix: only forward ``start_method`` when
    the caller explicitly asked for one.
    """

    def test_omits_start_method_when_unset(self):
        from unittest.mock import MagicMock

        wandb = MagicMock()
        _ = dist._build_wandb_settings(
            wandb=wandb, init_timeout=None, start_method=None
        )
        wandb.Settings.assert_called_once()
        kwargs = wandb.Settings.call_args.kwargs
        assert "start_method" not in kwargs, (
            "_build_wandb_settings forwarded start_method=None to "
            "wandb.Settings; wandb emits a deprecation warning even "
            "for `None`, so every run gets the noise back."
        )
        # init_timeout still defaults to 60
        assert kwargs.get("init_timeout") == 60

    def test_forwards_start_method_when_set(self):
        from unittest.mock import MagicMock

        wandb = MagicMock()
        _ = dist._build_wandb_settings(
            wandb=wandb, init_timeout=120, start_method="spawn"
        )
        kwargs = wandb.Settings.call_args.kwargs
        assert kwargs["start_method"] == "spawn"
        assert kwargs["init_timeout"] == 120


class TestSetupWandbRankGate:
    """setup_wandb must short-circuit on non-zero ranks.

    Regression for the 96-rank Sunspot incident: previously setup_wandb
    ran verify_wandb() + wandb.init() + logger.info() on every rank,
    relying on _resolve_wandb_mode("disabled") for non-zero ranks to
    produce a "dummy" wandb.run. Result: 95 dummy runs + a wall of
    "Setting up wandb from rank=N" log spam on a 96-rank job.
    """

    def test_rank_nonzero_returns_none_immediately(self, monkeypatch):
        # Mock get_rank to return 17. Then mock wandb so that if
        # setup_wandb DID try to call wandb.init, we'd see it.
        monkeypatch.setattr(dist, "get_rank", lambda: 17)

        wandb_init_calls = []
        verify_calls = []

        import sys
        from unittest.mock import MagicMock

        mock_wandb = MagicMock()
        mock_wandb.init = MagicMock(
            side_effect=lambda *a, **kw: wandb_init_calls.append((a, kw))
        )
        monkeypatch.setitem(sys.modules, "wandb", mock_wandb)
        monkeypatch.setattr(
            dist,
            "verify_wandb",
            lambda: (verify_calls.append(1), True)[1],
        )

        result = dist.setup_wandb(project_name="test")

        assert result is None, "non-zero rank must return None"
        assert wandb_init_calls == [], (
            "setup_wandb called wandb.init from a non-zero rank — "
            "the rank gate regressed. This is the 96-dummy-runs bug."
        )
        assert verify_calls == [], (
            "setup_wandb called verify_wandb from a non-zero rank — "
            "the rank gate should short-circuit BEFORE verify_wandb."
        )

    def test_rank_zero_proceeds_normally(self, monkeypatch):
        # Companion: rank 0 must still call verify_wandb + wandb.init.
        monkeypatch.setattr(dist, "get_rank", lambda: 0)

        import sys
        from unittest.mock import MagicMock

        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.run = mock_wandb.init.return_value
        monkeypatch.setitem(sys.modules, "wandb", mock_wandb)
        monkeypatch.setattr(dist, "verify_wandb", lambda: True)

        result = dist.setup_wandb(project_name="test")

        assert result is not None, (
            "rank 0 should get a wandb.run, not None"
        )
        assert mock_wandb.init.called, (
            "rank 0 should call wandb.init"
        )


class TestSetupDdpDeviceId:
    """Regression for the device_id-never-set-on-CUDA bug.

    Before the fix, _setup_ddp built a torch.device("cuda:N") when the
    caller didn't pass device_id but then NEVER added it to init_kwargs
    (the guard checked the original device_id parameter, not the locally-
    constructed device). Result: every CUDA run got
        "barrier(): using the device under current context.
         You can specify `device_id` in `init_process_group` to mute
         this warning."
    on every collective op.

    These tests inspect the init_kwargs that _setup_ddp would pass to
    torch.distributed.init_process_group, without actually initing a
    real process group.
    """

    def _capture_init_kwargs(self, monkeypatch, **env):
        """Call _setup_ddp with mocked torch.distributed and return the
        kwargs it would have passed to init_process_group."""
        import torch
        from unittest.mock import MagicMock

        for k, v in env.items():
            monkeypatch.setenv(k, str(v))

        # Patch is_initialized → False so we hit the init path, and
        # init_process_group → MagicMock so we can inspect its args.
        mock_init = MagicMock()
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: False
        )
        monkeypatch.setattr(
            torch.distributed, "init_process_group", mock_init
        )
        # No-op the env-broadcasting helpers — they need real MPI.
        monkeypatch.setattr(dist, "broadcast", lambda x, root=0: x)
        # Force MASTER_ADDR/PORT so we skip the resolution branch.
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "12345")

        dist._setup_ddp(backend="gloo")
        assert mock_init.call_count == 1
        return mock_init.call_args.kwargs

    def test_cuda_default_device_id_built_from_local_rank(
        self, monkeypatch
    ):
        # CUDA + no explicit device_id → must pass device_id=cuda:LOCAL_RANK
        # so torch can bind the PG to a specific GPU and skip the warning.
        monkeypatch.setattr(dist, "get_torch_device_type", lambda: "cuda")
        kwargs = self._capture_init_kwargs(
            monkeypatch,
            RANK=0,
            LOCAL_RANK=3,
            WORLD_SIZE=4,
        )
        import torch

        assert "device_id" in kwargs, (
            "Regression: _setup_ddp dropped device_id when caller didn't pass "
            "one; init_process_group will emit barrier()-warning spam."
        )
        assert kwargs["device_id"] == torch.device("cuda:3")

    def test_xpu_auto_detect_binds_device_id_from_local_rank(
        self, monkeypatch
    ):
        """XPU + no explicit device_id → device_id=xpu:LOCAL_RANK.

        Regression for the Aurora torchtitan FSDP2 hang (job 8518207).
        Pre-fix _setup_ddp skipped device_id for xpu (citing a
        DeviceMesh._unflatten / split_group concern), but without a
        device-bound PG xccl/foreach_all_gather routed some ranks'
        collectives to xpu:0 and others to xpu:LOCAL_RANK — they
        never met up and FSDP2 silently deadlocked on the first
        all_gather_into_tensor. Caller (setup_torch) is responsible
        for calling xpu.set_device(local_rank) BEFORE _setup_ddp.
        """
        monkeypatch.setattr(dist, "get_torch_device_type", lambda: "xpu")
        kwargs = self._capture_init_kwargs(
            monkeypatch,
            RANK=0,
            LOCAL_RANK=2,
            WORLD_SIZE=4,
        )
        import torch

        assert "device_id" in kwargs, (
            "Regression: XPU dropped device_id; FSDP2 will deadlock "
            "on the first all_gather_into_tensor."
        )
        assert kwargs["device_id"] == torch.device("xpu:2")

    def test_explicit_int_device_id_resolves_to_active_device_type(
        self, monkeypatch
    ):
        """An int device_id resolves to "{get_torch_device_type()}:N".

        Pre-fix this hardcoded "cuda:N" regardless of the active device
        type — same bug class as the barrier()-warning fix above, just
        on the explicit-caller path instead of the auto-detect path.
        On a CUDA system the result is unchanged ("cuda:2"); the
        regression target is XPU (see test_explicit_int_device_id_on_xpu).
        """
        monkeypatch.setattr(dist, "get_torch_device_type", lambda: "cuda")
        import torch
        from unittest.mock import MagicMock

        mock_init = MagicMock()
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: False
        )
        monkeypatch.setattr(
            torch.distributed, "init_process_group", mock_init
        )
        monkeypatch.setattr(dist, "broadcast", lambda x, root=0: x)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "12345")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")

        dist._setup_ddp(backend="gloo", device_id=2)
        kwargs = mock_init.call_args.kwargs
        assert kwargs["device_id"] == torch.device("cuda:2")

    def test_explicit_int_device_id_on_xpu_resolves_to_xpu(
        self, monkeypatch
    ):
        """Regression: int device_id on XPU must NOT become "cuda:N".

        Pre-fix `_setup_ddp(device_id=2)` on Aurora would have built
        torch.device("cuda:2") and forwarded it to xccl
        init_process_group — would have either errored or silently
        bound to a non-existent device. This test pins the corrected
        behavior: same int, different device_type, correct device
        family.
        """
        monkeypatch.setattr(dist, "get_torch_device_type", lambda: "xpu")
        import torch
        from unittest.mock import MagicMock

        mock_init = MagicMock()
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: False
        )
        monkeypatch.setattr(
            torch.distributed, "init_process_group", mock_init
        )
        monkeypatch.setattr(dist, "broadcast", lambda x, root=0: x)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "12345")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")

        dist._setup_ddp(backend="gloo", device_id=2)
        kwargs = mock_init.call_args.kwargs
        assert kwargs["device_id"] == torch.device("xpu:2"), (
            "Regression: explicit int device_id on XPU resolved to "
            "the wrong device family. Was likely hardcoded `cuda:N`."
        )

    def test_explicit_xpu_device_id_passes_through(self, monkeypatch):
        """Explicit torch.device("xpu:N") is honored verbatim.

        Used to document a quirk where the explicit-caller path
        bypassed the "skip on xpu" guard; that quirk is now the
        correct behavior. The auto-detect XPU path ALSO binds
        device_id now (see test_xpu_auto_detect_binds_device_id_from_local_rank
        above), so both paths agree.
        """
        monkeypatch.setattr(dist, "get_torch_device_type", lambda: "xpu")
        import torch
        from unittest.mock import MagicMock

        mock_init = MagicMock()
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: False
        )
        monkeypatch.setattr(
            torch.distributed, "init_process_group", mock_init
        )
        monkeypatch.setattr(dist, "broadcast", lambda x, root=0: x)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "12345")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")

        xpu_device = torch.device("xpu", 0)
        dist._setup_ddp(backend="gloo", device_id=xpu_device)  # type: ignore[arg-type]
        kwargs = mock_init.call_args.kwargs
        assert kwargs.get("device_id") == xpu_device


# ===================================================================
# init_device_mesh_safe
# ===================================================================


class TestInitDeviceMeshSafe:
    """``init_device_mesh_safe`` round-trips ``bound_device_id`` on the
    default PG so torch's ``DeviceMesh._init_one_process_group`` takes
    the ``new_group`` fallback path (which xccl supports) instead of
    ``split_group`` (which it doesn't).

    Pre-fix, 2D meshes (``ezpz.examples.fsdp_tp``) raised
        RuntimeError: No backend for the parent process group or its
                      backend does not support splitting
    on XPU + xccl.

    These tests mock the default PG and the torch ``init_device_mesh``
    entry-point to verify (a) the workaround fires only on xpu when
    the PG actually has a ``bound_device_id``, (b) the original value
    is restored, and (c) silent disablement (e.g. a future torch making
    the attr read-only) doesn't break the call.
    """

    def _fake_pg(self, bound_device_id):
        """Build a stand-in for ``torch.distributed.ProcessGroup`` that
        carries a mutable ``bound_device_id`` like the real C++ object
        does today."""
        pg = MagicMock()
        pg.bound_device_id = bound_device_id
        return pg

    def test_xpu_clears_and_restores_bound_device_id(self, monkeypatch):
        """On xpu with a bound default PG: clear → call → restore."""
        sentinel = torch.device("xpu:3")
        pg = self._fake_pg(sentinel)

        observed: dict[str, Any] = {}

        def _fake_init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
            # During the inner call, bound_device_id must be None.
            observed["mid_call_bound_device_id"] = pg.bound_device_id
            return MagicMock(name="DeviceMesh")

        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_get_default_group",
            lambda: pg,
        )
        # Patch torch's init_device_mesh at the source the shim
        # imports it from.
        import torch.distributed.device_mesh as _dm

        monkeypatch.setattr(_dm, "init_device_mesh", _fake_init_device_mesh)

        result = dist.init_device_mesh_safe(
            "xpu", (2, 4), mesh_dim_names=("dp", "tp")
        )

        assert result is not None
        assert observed["mid_call_bound_device_id"] is None, (
            "Workaround failed: bound_device_id was not cleared "
            "during init_device_mesh. Torch's DeviceMesh._init_one_"
            "process_group will take the split_group branch and "
            "raise on xccl."
        )
        assert pg.bound_device_id == sentinel, (
            "bound_device_id was not restored after the call. "
            "Subsequent FSDP2 collectives need a device-bound PG."
        )

    def test_xpu_with_unbound_pg_is_pass_through(self, monkeypatch):
        """When ``bound_device_id`` is already None, no swap happens."""
        pg = self._fake_pg(None)

        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_get_default_group",
            lambda: pg,
        )

        import torch.distributed.device_mesh as _dm

        called = MagicMock(return_value=MagicMock(name="DeviceMesh"))
        monkeypatch.setattr(_dm, "init_device_mesh", called)

        dist.init_device_mesh_safe("xpu", (4,))

        called.assert_called_once()
        # No mutation — still None.
        assert pg.bound_device_id is None

    def test_non_xpu_skips_workaround_entirely(self, monkeypatch):
        """CUDA path must NOT touch ``bound_device_id``.

        Torch's NCCL backend supports split_group natively; this
        helper should be a pure pass-through there.
        """
        sentinel = torch.device("cuda:0")
        pg = self._fake_pg(sentinel)

        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        # If the CUDA path ever tries to fetch the default PG, blow up
        # — we want a true no-op.
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_get_default_group",
            MagicMock(side_effect=AssertionError("CUDA must not touch the default PG")),
        )

        import torch.distributed.device_mesh as _dm

        called = MagicMock(return_value=MagicMock(name="DeviceMesh"))
        monkeypatch.setattr(_dm, "init_device_mesh", called)

        dist.init_device_mesh_safe("cuda", (8,))

        called.assert_called_once_with("cuda", (8,), mesh_dim_names=None)
        # bound_device_id still untouched.
        assert pg.bound_device_id == sentinel

    def test_no_default_pg_is_pass_through(self, monkeypatch):
        """If ``torch.distributed`` isn't initialised, fall through."""
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: False
        )

        import torch.distributed.device_mesh as _dm

        called = MagicMock(return_value=MagicMock(name="DeviceMesh"))
        monkeypatch.setattr(_dm, "init_device_mesh", called)

        dist.init_device_mesh_safe("xpu", (2, 2))
        called.assert_called_once()

    def test_restores_bound_device_id_when_inner_raises(self, monkeypatch):
        """If ``init_device_mesh`` raises, the original value MUST be
        restored — otherwise downstream FSDP2 silently routes to the
        wrong device.
        """
        sentinel = torch.device("xpu:5")
        pg = self._fake_pg(sentinel)

        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_get_default_group",
            lambda: pg,
        )

        import torch.distributed.device_mesh as _dm

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated split_group failure")

        monkeypatch.setattr(_dm, "init_device_mesh", _boom)

        with pytest.raises(RuntimeError, match="simulated"):
            dist.init_device_mesh_safe("xpu", (4,))

        assert pg.bound_device_id == sentinel, (
            "Workaround leaked: bound_device_id stayed cleared after "
            "an inner failure. FSDP2 will silently route collectives "
            "to the wrong XPU device on the next all_gather."
        )

    def test_silent_disablement_when_attr_setattr_fails(self, monkeypatch):
        """If a future torch makes ``bound_device_id`` read-only, the
        shim should fail-open: the inner call still runs (and will
        raise the original ``RuntimeError`` from torch if applicable),
        rather than the shim itself crashing in an obscure place.

        This pins the try/except behavior in init_device_mesh_safe.
        """

        class _LockedAttr:
            """PG-like object where ``bound_device_id`` is read-only."""

            def __init__(self):
                # Use object.__setattr__ to bypass our own override
                # for the initial value.
                object.__setattr__(
                    self, "_bdi", torch.device("xpu:0")
                )

            @property
            def bound_device_id(self):
                return self._bdi

            @bound_device_id.setter
            def bound_device_id(self, _value):
                raise AttributeError("read-only in future torch")

        pg = _LockedAttr()

        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed.distributed_c10d,
            "_get_default_group",
            lambda: pg,
        )

        import torch.distributed.device_mesh as _dm

        called = MagicMock(return_value=MagicMock(name="DeviceMesh"))
        monkeypatch.setattr(_dm, "init_device_mesh", called)

        # Must not raise from the shim itself; inner call still fires.
        dist.init_device_mesh_safe("xpu", (4,))
        called.assert_called_once()
