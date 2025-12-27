"""Tests for PBS launch topology helpers."""
from pathlib import Path

import pytest

import ezpz.pbs as pbs


@pytest.fixture
def patch_topology(monkeypatch, tmp_path):
    """Patch topology helpers to deterministic values."""

    def _apply(
        *,
        world_size: int = 16,
        gpus_per_node: int = 4,
        num_nodes: int = 4,
        machine: str = "generic",
        hostfile: Path | None = None,
    ) -> Path:
        monkeypatch.setattr(pbs.ezpz, "get_world_size", lambda total=True: world_size)
        monkeypatch.setattr(pbs.ezpz, "get_gpus_per_node", lambda: gpus_per_node)
        monkeypatch.setattr(pbs.ezpz, "get_num_nodes", lambda hostfile=None: num_nodes)
        monkeypatch.setattr(pbs.ezpz, "get_machine", lambda: machine)
        monkeypatch.setattr(
            pbs, "get_hostfile_with_fallback", lambda _: hostfile or tmp_path / "hosts"
        )
        return hostfile or tmp_path / "hosts"

    return _apply


def test_get_pbs_launch_cmd_defaults_full_machine(patch_topology, monkeypatch):
    """Defaults consume all resources and apply generic CPU binding."""
    hostfile = patch_topology()
    monkeypatch.delenv("CPU_BIND", raising=False)

    cmd = pbs.get_pbs_launch_cmd(hostfile=hostfile)

    assert (
        cmd
        == f"mpiexec --envall --np=16 --ppn=4 --hostfile={hostfile} --cpu-bind=depth --depth=8"
    )


def test_get_pbs_launch_cmd_respects_cpu_bind_env(patch_topology, monkeypatch):
    """User-provided CPU binding is forwarded verbatim (with verbose prefix)."""
    hostfile = patch_topology(world_size=8, gpus_per_node=8, num_nodes=1)
    monkeypatch.setenv("CPU_BIND", "--cpu-bind=list:0-1")

    cmd = pbs.get_pbs_launch_cmd(ngpus=8, nhosts=1, hostfile=hostfile)

    assert cmd.endswith(f"--cpu-bind=verbose,list:0-1")


def test_get_pbs_launch_cmd_raises_on_inconsistent_topology(patch_topology):
    """Invalid topology combinations raise ValueError."""
    hostfile = patch_topology()

    with pytest.raises(ValueError):
        pbs.get_pbs_launch_cmd(ngpus=5, nhosts=2, hostfile=hostfile)


def test_get_pbs_launch_cmd_intel_cpu_binding_defaults(patch_topology, monkeypatch):
    """Intel GPU machines add vendor-specific CPU binding and no-vni flag."""
    hostfile = patch_topology(machine="aurora")
    monkeypatch.delenv("CPU_BIND", raising=False)

    cmd = pbs.get_pbs_launch_cmd(hostfile=hostfile)

    assert "--no-vni" in cmd
    assert "--cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96" in cmd
