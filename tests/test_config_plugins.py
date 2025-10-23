"""Tests for scheduler plug-in detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import pytest

import ezpz.configs as configs


@pytest.fixture(autouse=True)
def reset_plugins():
    configs._clear_scheduler_plugins()
    yield
    configs._clear_scheduler_plugins()


def test_runtime_plugin_overrides_fallback(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)

    def plugin(
        env: dict[str, str], *, hostname: str | None = None, machine: str | None = None
    ) -> str | None:
        return "custom" if env.get("EZPZ_TEST_SCHED") else None

    configs.register_scheduler_plugin(plugin)
    monkeypatch.setenv("EZPZ_TEST_SCHED", "true")
    assert configs.get_scheduler() == "CUSTOM"


@dataclass
class _FakeEntryPoint:
    loader: Callable[[], Callable[..., str | None]]

    name: str = "fake"

    def load(self) -> Callable[..., str | None]:
        return self.loader()


class _FakeEntryPoints:
    def __init__(self, items):
        self._items = items

    def select(self, group: str):
        return self._items if group == configs.SCHEDULER_ENTRYPOINT_GROUP else []


def test_entry_point_plugin_invoked(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.setenv("EZPZ_TEST_SCHED_PLUGIN", "1")

    def loader():
        def plugin(
            env: dict[str, str],
            *,
            hostname: str | None = None,
            machine: str | None = None,
        ) -> str | None:
            if env.get("EZPZ_TEST_SCHED_PLUGIN") == "1":
                return "plugin-pbs"
            return None

        return plugin

    fake_eps = _FakeEntryPoints([_FakeEntryPoint(loader=loader)])
    monkeypatch.setattr(configs, "entry_points", lambda: fake_eps)
    assert configs.get_scheduler() == "PLUGIN-PBS"
