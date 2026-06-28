"""Tests for the shared profiling CLI surface.

`add_profiling_args` (ezpz.cli.flags) is the single source of truth for the
profiler flags across all ezpz.examples.* modules, and `HfProfileArguments`
(ezpz.configs) mirrors it for the HuggingFace Trainer example. These tests
pin the flag set + key defaults so the shared surface can't silently drift.
"""

import argparse

import pytest

try:
    from ezpz.cli.flags import add_profiling_args

    FLAGS_AVAILABLE = True
except ImportError:
    FLAGS_AVAILABLE = False

# The canonical defaults add_profiling_args must produce.
EXPECTED_DEFAULTS = {
    "pytorch_profiler": False,
    "pyinstrument_profiler": False,
    "rank_zero_only": False,
    "pytorch_profiler_wait": 1,
    "pytorch_profiler_warmup": 2,
    "pytorch_profiler_active": 3,
    "pytorch_profiler_repeat": 5,
    "profile_memory": True,
    "record_shapes": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
    "acc_events": False,
}


@pytest.mark.skipif(not FLAGS_AVAILABLE, reason="ezpz.cli.flags not available")
class TestAddProfilingArgs:
    def _parse(self, argv):
        p = argparse.ArgumentParser()
        add_profiling_args(p)
        return p.parse_args(argv)

    def test_defaults(self):
        ns = self._parse([])
        for k, v in EXPECTED_DEFAULTS.items():
            assert getattr(ns, k) == v, f"{k}: {getattr(ns, k)!r} != {v!r}"

    def test_profile_short_and_long_flag(self):
        # -p and --profile both set dest=pytorch_profiler.
        assert self._parse(["-p"]).pytorch_profiler is True
        assert self._parse(["--profile"]).pytorch_profiler is True

    def test_boolean_optional_action_no_form(self):
        # BooleanOptionalAction generates the --no-* form so True-default
        # toggles can actually be disabled.
        ns = self._parse(["--no-with-stack", "--no-profile-memory"])
        assert ns.with_stack is False
        assert ns.profile_memory is False

    def test_schedule_overrides(self):
        ns = self._parse(
            ["--pytorch-profiler-active", "7", "--pytorch-profiler-wait", "0"]
        )
        assert ns.pytorch_profiler_active == 7
        assert ns.pytorch_profiler_wait == 0

    def test_returns_parser(self):
        p = argparse.ArgumentParser()
        assert add_profiling_args(p) is p


try:
    from transformers import HfArgumentParser

    from ezpz.configs import HfProfileArguments

    HF_AVAILABLE = True
except Exception:  # noqa: BLE001 - transformers optional / may fail to import
    HF_AVAILABLE = False


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers/HfProfileArguments unavailable")
class TestHfProfileArguments:
    def _parse(self, argv):
        (pr,) = HfArgumentParser((HfProfileArguments,)).parse_args_into_dataclasses(
            args=argv
        )
        return pr

    def test_defaults_match_cli(self):
        pr = self._parse([])
        # rank_zero_only must match add_profiling_args (False), not diverge.
        assert pr.rank_zero_only is False
        assert pr.pytorch_profiler is False
        assert pr.pyinstrument_profiler is False
        assert pr.pytorch_profiler_active == 3

    def test_profile_alias_exposed(self):
        # --profile / -p aliases must reach the same dest as --pytorch-profiler.
        assert self._parse(["--profile"]).pytorch_profiler is True
        assert self._parse(["-p"]).pytorch_profiler is True
        assert self._parse(["--pytorch_profiler"]).pytorch_profiler is True
