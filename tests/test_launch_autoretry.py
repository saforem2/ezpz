"""Tests for ``ezpz launch --auto-retry`` (the failover loop).

Coverage targets one row per termination-matrix case plus the
argparse validations. The classifier is a pure function so most
tests can exercise it directly without subprocess overhead. The
loop itself is tested by injecting a stub ``scrape_fn`` and using
a fake ``NodeAllocation`` so we never spawn a real child.

Why no end-to-end test that actually launches a process? The
existing ``test_launch_watchdog.py`` already exercises the
shell-out plumbing (sh -c scripts, SIGTERM handling, watchdog
exit codes). What's unique here is the *classification* and
*swap policy* — those are pure and easier to reason about with
fake inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

from ezpz.launch import (
    _auto_retry_log_dir,
    _ranks_to_hosts,
    _resolve_auto_retry_allocation,
    _resolve_auto_retry_node_pool,
    parse_args,
)
from ezpz.launch_autoretry import (
    AutoRetryConfig,
    NodeAllocation,
    TerminationReason,
    _backoff_for_attempt,
    _extract_inner_rc,
    _has_crash_patterns,
    _has_progress_markers,
    _strip_ansi,
    classify_attempt,
    derive_spare_count,
    run_with_auto_retry,
)

# Most tests don't shell out, but argparse paths and the run-loop
# tests need POSIX-y env semantics (env var inheritance, etc.).
pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="auto-retry assumes POSIX subprocess semantics",
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    """Small pure functions: ANSI stripping, inner_rc, crash detection."""

    def test_strip_ansi_removes_color_codes(self):
        assert (
            _strip_ansi("Execution finished with \x1b[1;36m127\x1b[0m")
            == "Execution finished with 127"
        )

    def test_strip_ansi_passthrough_no_codes(self):
        assert _strip_ansi("no codes") == "no codes"

    def test_extract_inner_rc_basic(self):
        assert _extract_inner_rc("Execution finished with 0\n") == 0
        assert _extract_inner_rc("Execution finished with 137\n") == 137

    def test_extract_inner_rc_ansi(self):
        # The trailer is ANSI-colored in practice; regex on the raw
        # form would pick up the "1" from "[1;36m" instead of 127.
        log = "Execution finished with \x1b[1;36m127\x1b[0m.\n"
        assert _extract_inner_rc(log) == 127

    def test_extract_inner_rc_trailing_punctuation(self):
        # Logger formats it with a period.
        assert _extract_inner_rc("Execution finished with 0.\n") == 0

    def test_extract_inner_rc_takes_last(self):
        # If multiple trailers landed (retry happened upstream),
        # the final outcome wins.
        log = "Execution finished with 0\nExecution finished with 3\n"
        assert _extract_inner_rc(log) == 3

    def test_extract_inner_rc_absent(self):
        assert _extract_inner_rc("nothing here") is None

    def test_crash_patterns_oom(self):
        assert _has_crash_patterns("torch.OutOfMemoryError: ...")

    def test_crash_patterns_gloo(self):
        msg = (
            "RuntimeError: [enforce fail at gloo/.../tcp.cc:127] "
            "Connection closed by peer [10.0.0.1]:53121"
        )
        assert _has_crash_patterns(msg)

    def test_crash_patterns_shepherd(self):
        assert _has_crash_patterns("x4502.hsn: shepherd died from signal 9")

    def test_crash_patterns_negative(self):
        assert not _has_crash_patterns("Training complete\nFinal loss: 0.5")

    def test_crash_patterns_excludes_innocent_rank_11_cascade(self):
        # `rank N died from signal 11` is a downstream cascade from a
        # primary kill on a different node — NOT a bad-node indicator.
        # Same exclusion the scraper applies (job 8466848 postmortem).
        log = (
            "rank 0 died from signal 11\n"
            "rank 1 died from signal 11\n"
            "rank 2 died from signal 11\n"
        )
        assert not _has_crash_patterns(log)

    def test_crash_patterns_excludes_innocent_rank_15_cascade(self):
        # Same for SIGTERM (15) cascading from a primary walltime kill.
        log = (
            "rank 0 died from signal 15\n"
            "rank 1 died from signal 15\n"
        )
        assert not _has_crash_patterns(log)

    def test_crash_patterns_preserves_real_death_among_cascades(self):
        # Critical regression: a real OOM mixed with innocent rank
        # cascades MUST still register as a crash. The strip should
        # only remove the cascade lines, not the real signal.
        log = (
            "rank 0 died from signal 15\n"
            "rank 1 died from signal 11\n"
            "torch.OutOfMemoryError: CUDA OOM\n"
            "rank 2 died from signal 15\n"
        )
        assert _has_crash_patterns(log)

    def test_crash_patterns_preserves_shepherd_9(self):
        # `shepherd died from signal 9` is a PALS daemon kill — real
        # hardware failure, never an innocent cascade. Must still match.
        log = "x4502.hsn: shepherd died from signal 9\n"
        assert _has_crash_patterns(log)

    def test_crash_patterns_preserves_died_from_signal_other_than_11_15(
        self,
    ):
        # Signal 6 = SIGABRT (assertion failures, etc.). Not a cascade
        # signal — still counts.
        log = "rank 4 died from signal 6\n"
        assert _has_crash_patterns(log)

    def test_crash_patterns_real_ur_oom_with_cascade_regression(self):
        # Regression for an actual Aurora torchtitan log shape (job
        # 2026-05-12): 18 successful training steps, then a real
        # level_zero UR_RESULT_ERROR_OUT_OF_RESOURCES, then mpiexec
        # tore down the job and emitted a `rank 14 died from signal 15`
        # cascade line as part of the teardown. The strip should
        # remove the cascade but preserve the OOM signal so the loop
        # correctly retries on a different node.
        log = (
            "step:  1  loss: 12.94587  ...\n"
            "step: 18  loss: 10.27772  ...\n"
            "[rank7]: RuntimeError: level_zero backend failed with "
            "error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)\n"
            "x4610c4s3b0n0.hsn.cm.aurora.alcf.anl.gov: rank 7 exited "
            "with code 1\n"
            "x4610c4s5b0n0.hsn.cm.aurora.alcf.anl.gov: rank 14 died "
            "from signal 15\n"
            "Execution finished with 143.\n"
        )
        assert _has_crash_patterns(log)

    def test_progress_markers_step_equals(self):
        assert _has_progress_markers("iter=10 step=5 loss=0.1")

    def test_progress_markers_absent(self):
        # No step=N anywhere.
        assert not _has_progress_markers("loading dataset...\nERROR: cuda")

    def test_derive_spare_count_unused(self):
        assert derive_spare_count(10, 8) == 2

    def test_derive_spare_count_exactly_full(self):
        assert derive_spare_count(8, 8) == 0

    def test_derive_spare_count_overcommitted_clamps(self):
        # Caller asked for more than available — clamp to 0 rather
        # than return negative. The loop will surface this as
        # EXHAUSTED on the first failure.
        assert derive_spare_count(5, 8) == 0

    def test_backoff_progression(self):
        # 5, 10, 20, 40, 60, 60, 60, ...
        assert _backoff_for_attempt(1) == 5.0
        assert _backoff_for_attempt(2) == 10.0
        assert _backoff_for_attempt(3) == 20.0
        assert _backoff_for_attempt(4) == 40.0
        assert _backoff_for_attempt(5) == 60.0
        assert _backoff_for_attempt(10) == 60.0


# ---------------------------------------------------------------------------
# Classifier — one test per termination-matrix row
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


class TestClassifyAttempt:
    """One test per row of the termination matrix in pr3_handoff.md."""

    def test_success_clean_exit(self, tmp_path):
        log = _write(tmp_path / "log", "Execution finished with 0\n")
        assert classify_attempt(0, log, []) is TerminationReason.SUCCESS

    def test_walltime_no_crash_patterns(self, tmp_path):
        log = _write(tmp_path / "log", "Execution finished with 0\n")
        # rc=143 (walltime SIGTERM), no crash signatures in the log
        assert classify_attempt(143, log, []) is TerminationReason.WALLTIME

    def test_walltime_with_crash_patterns_is_bad_node(self, tmp_path):
        # Real bad-node failure that races the walltime kill: we DO
        # retry, even though rc=143.
        log = _write(
            tmp_path / "log",
            "Execution finished with 0\nOutOfMemoryError on rank 5\n",
        )
        assert (
            classify_attempt(143, log, [])
            is TerminationReason.BAD_NODE_BLIND
        )

    def test_walltime_with_only_innocent_rank_cascade_is_walltime(
        self, tmp_path
    ):
        # rc=143 + log full of `rank N died from signal 15` (cascade
        # from the walltime SIGTERM raining outward). Without the
        # innocent-cascade strip, the loose `died from signal` would
        # override into a bad-node retry — burning spares for nothing.
        # Must stay WALLTIME.
        log = _write(
            tmp_path / "log",
            "Execution finished with 0\n"
            "rank 0 died from signal 15\n"
            "rank 1 died from signal 15\n"
            "rank 2 died from signal 15\n",
        )
        assert classify_attempt(143, log, []) is TerminationReason.WALLTIME

    def test_walltime_with_cascade_and_real_death_is_bad_node(
        self, tmp_path
    ):
        # Mixed log: cascade lines AND a real OOM. The real signal
        # survives the strip → bad-node retry, not walltime.
        log = _write(
            tmp_path / "log",
            "rank 0 died from signal 15\n"
            "torch.OutOfMemoryError: CUDA out of memory\n"
            "rank 1 died from signal 11\n",
        )
        assert (
            classify_attempt(143, log, [])
            is TerminationReason.BAD_NODE_BLIND
        )

    def test_watchdog_124_triggers_blind(self, tmp_path):
        log = _write(tmp_path / "log", "starting...\n")
        assert (
            classify_attempt(124, log, [])
            is TerminationReason.BAD_NODE_BLIND
        )

    def test_nonzero_with_named_bad_node(self, tmp_path):
        log = _write(
            tmp_path / "log",
            "Execution finished with 1\nx4502: shepherd died from signal 9\n",
        )
        assert (
            classify_attempt(1, log, ["x4502.hsn.cm.aurora.alcf.anl.gov"])
            is TerminationReason.BAD_NODE_KNOWN
        )

    def test_nonzero_blind_when_scraper_empty(self, tmp_path):
        log = _write(
            tmp_path / "log", "Execution finished with 1\nrandom error\n"
        )
        assert (
            classify_attempt(1, log, [])
            is TerminationReason.BAD_NODE_BLIND
        )

    def test_stuck_pre_training_when_prior_also_zero_steps(self, tmp_path):
        # Prior attempt: no progress. Current attempt: also no progress.
        # → STUCK_PRE_TRAINING regardless of rc shape.
        log = _write(
            tmp_path / "log", "Execution finished with 1\nimport error\n"
        )
        assert (
            classify_attempt(
                1, log, [], prior_attempt_had_progress=False
            )
            is TerminationReason.STUCK_PRE_TRAINING
        )

    def test_progress_overrides_stuck_guard(self, tmp_path):
        # The "step=" marker on the CURRENT attempt means training
        # started this time — even if the prior attempt had no
        # progress, we shouldn't bail.
        log = _write(
            tmp_path / "log",
            "Execution finished with 1\niter step=5 loss=0.1\n",
        )
        assert (
            classify_attempt(
                1, log, ["x4502"], prior_attempt_had_progress=False
            )
            is TerminationReason.BAD_NODE_KNOWN
        )

    def test_exhausted_when_no_spares_left_known(self, tmp_path):
        log = _write(tmp_path / "log", "Execution finished with 1\n")
        assert (
            classify_attempt(1, log, ["host"], has_spares=False)
            is TerminationReason.EXHAUSTED
        )

    def test_exhausted_when_no_spares_left_blind(self, tmp_path):
        log = _write(tmp_path / "log", "Execution finished with 1\n")
        assert (
            classify_attempt(1, log, [], has_spares=False)
            is TerminationReason.EXHAUSTED
        )

    def test_exhausted_when_no_spares_left_watchdog(self, tmp_path):
        log = _write(tmp_path / "log", "hang\n")
        assert (
            classify_attempt(124, log, [], has_spares=False)
            is TerminationReason.EXHAUSTED
        )

    def test_wrapper_lied_inner_rc_overrides_clean_shell_exit(
        self, tmp_path
    ):
        # Outer shell said 0 but the inner trailer says 7 — wrapper
        # lied. Treat as failure.
        log = _write(tmp_path / "log", "Execution finished with 7\n")
        assert (
            classify_attempt(0, log, [])
            is TerminationReason.BAD_NODE_BLIND
        )

    def test_crash_patterns_override_clean_shell_exit(self, tmp_path):
        # rc=0, no inner trailer, but log has crash signatures —
        # treat as failure (mass-traceback).
        log = _write(
            tmp_path / "log",
            "training...\nUR_RESULT_ERROR_OUT_OF_RESOURCES on rank 4\n",
        )
        assert (
            classify_attempt(0, log, [])
            is TerminationReason.BAD_NODE_BLIND
        )

    def test_missing_log_file_safe(self, tmp_path):
        # Log was never written (the child died before stdout
        # touched disk). Classifier still has to return *something*
        # sensible — should not crash.
        log = tmp_path / "nonexistent.log"
        result = classify_attempt(1, log, [])
        # rc=1, no scraper, no log → blind rotation.
        assert result is TerminationReason.BAD_NODE_BLIND


# ---------------------------------------------------------------------------
# NodeAllocation
# ---------------------------------------------------------------------------


class TestNodeAllocation:
    def _make(self, tmp_path, full=("h1", "h2", "h3", "h4", "h5"), active=3):
        hf = tmp_path / "active.hostfile"
        bf = tmp_path / "bad_nodes.txt"
        return (
            NodeAllocation.from_full_nodelist(list(full), active, hf, bf),
            hf,
            bf,
        )

    def test_split_active_and_spare(self, tmp_path):
        alloc, hf, bf = self._make(tmp_path)
        assert alloc.active == ["h1", "h2", "h3"]
        assert list(alloc.spare) == ["h4", "h5"]
        assert hf.read_text() == "h1\nh2\nh3\n"
        assert bf.read_text() == ""

    def test_split_rejects_too_many_active(self, tmp_path):
        hf = tmp_path / "active.hostfile"
        bf = tmp_path / "bad_nodes.txt"
        with pytest.raises(ValueError, match="only"):
            NodeAllocation.from_full_nodelist(["h1"], 5, hf, bf)

    def test_swap_in_replaces_named_host(self, tmp_path):
        alloc, hf, bf = self._make(tmp_path)
        swaps = alloc.swap_in(["h2"])
        assert swaps == [("h2", "h4")]
        assert alloc.active == ["h1", "h4", "h3"]
        assert list(alloc.spare) == ["h5"]
        # Persisted on disk for the next launcher attempt.
        assert hf.read_text() == "h1\nh4\nh3\n"
        # Bad nodes appended for postmortem.
        assert bf.read_text() == "h2\n"

    def test_swap_in_skips_hosts_not_in_active(self, tmp_path):
        alloc, _, _ = self._make(tmp_path)
        swaps = alloc.swap_in(["not-in-active"])
        assert swaps == []
        # No spare consumed.
        assert list(alloc.spare) == ["h4", "h5"]

    def test_swap_in_multiple_consumes_multiple_spares(self, tmp_path):
        alloc, hf, _ = self._make(tmp_path)
        swaps = alloc.swap_in(["h1", "h2"])
        assert swaps == [("h1", "h4"), ("h2", "h5")]
        assert alloc.active == ["h4", "h5", "h3"]
        assert list(alloc.spare) == []
        assert hf.read_text() == "h4\nh5\nh3\n"

    def test_swap_in_raises_when_out_of_spares(self, tmp_path):
        alloc, _, _ = self._make(
            tmp_path, full=("h1", "h2", "h3"), active=3
        )
        with pytest.raises(RuntimeError, match="out of spare"):
            alloc.swap_in(["h1"])

    def test_swap_one_blind_rotates_first_active(self, tmp_path):
        alloc, hf, bf = self._make(tmp_path)
        bad, spare = alloc.swap_one_blind()
        assert bad == "h1" and spare == "h4"
        assert alloc.active == ["h4", "h2", "h3"]
        assert list(alloc.spare) == ["h5"]
        assert hf.read_text() == "h4\nh2\nh3\n"
        assert bf.read_text() == "h1\n"

    def test_swap_one_blind_raises_when_no_spares(self, tmp_path):
        alloc, _, _ = self._make(
            tmp_path, full=("h1", "h2"), active=2
        )
        with pytest.raises(RuntimeError, match="out of spare"):
            alloc.swap_one_blind()

    def test_has_spares_property(self, tmp_path):
        alloc, _, _ = self._make(tmp_path)
        assert alloc.has_spares
        # Drain
        alloc.swap_one_blind()
        alloc.swap_one_blind()
        assert not alloc.has_spares


# ---------------------------------------------------------------------------
# run_with_auto_retry — the loop, with a stubbed scrape_fn
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Stand-in for ``_run_attempt_with_tee``.

    Each call writes ``log_text`` to the requested attempt-N.log
    path and returns the corresponding rc. The list of (rc, log)
    pairs is consumed in order; if exhausted, raises so the test
    catches an unexpected extra attempt.
    """

    def __init__(self, attempts: list[tuple[int, str]]):
        self.attempts = list(attempts)
        self.calls: list[tuple[list[str], Path, int]] = []

    def __call__(self, cmd, log_path: Path, idle_timeout_s: int) -> int:
        if not self.attempts:
            raise AssertionError(
                "ran out of canned attempts — loop made more "
                "attempts than the test expected"
            )
        rc, text = self.attempts.pop(0)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
        self.calls.append((list(cmd), log_path, idle_timeout_s))
        return rc


def _config(tmp_path: Path, **overrides) -> AutoRetryConfig:
    """Test-friendly default AutoRetryConfig (no real cmd)."""
    base = dict(
        cmd=["echo", "x"],
        hostfile=tmp_path / "active.hostfile",
        log_dir=tmp_path / "logs",
        idle_timeout_s=0,  # disable watchdog for tests
        max_failover_retries=None,
    )
    base.update(overrides)
    # mypy/dict-merge tolerance
    return AutoRetryConfig(**base)  # type: ignore[arg-type]


def _alloc(tmp_path: Path, full=("h1", "h2", "h3", "h4"), active=2):
    hf = tmp_path / "active.hostfile"
    bf = tmp_path / "bad_nodes.txt"
    return NodeAllocation.from_full_nodelist(list(full), active, hf, bf)


def _run(
    monkeypatch,
    config,
    allocation,
    fake_runner,  # duck-typed: any callable (cmd, log_path, idle_timeout_s) -> int
    scrape_fn: Callable[[Path], list[str]] = lambda _: [],
    sleep_capture: list | None = None,
):
    """Drive run_with_auto_retry with the fake runner.

    Also monkeypatches time.sleep so the backoff doesn't actually
    sleep during tests — captured into ``sleep_capture`` for
    assertions if provided.
    """
    monkeypatch.setattr(
        "ezpz.launch_autoretry._run_attempt_with_tee", fake_runner
    )

    captured = sleep_capture if sleep_capture is not None else []

    def _no_sleep(s):
        captured.append(s)

    monkeypatch.setattr("ezpz.launch_autoretry.time.sleep", _no_sleep)

    return run_with_auto_retry(config, allocation, scrape_fn=scrape_fn)


class TestRunWithAutoRetry:
    def test_success_first_attempt(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)
        runner = _FakeRunner([(0, "Execution finished with 0\n")])
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 0
        assert len(runner.calls) == 1

    def test_succeeds_on_second_attempt_after_blind_rotation(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)
        runner = _FakeRunner(
            [
                (1, "Execution finished with 1\niter step=1 loss=0.1\n"),
                (0, "Execution finished with 0\n"),
            ]
        )
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 0
        assert len(runner.calls) == 2
        # One spare consumed by the blind rotation.
        assert list(allocation.spare) == ["h4"]

    def test_named_bad_node_swap_then_success(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)
        runner = _FakeRunner(
            [
                (1, "Execution finished with 1\niter step=1\n"),
                (0, "Execution finished with 0\n"),
            ]
        )

        def scrape(_log: Path):
            # First call: name h1 as bad. Second call: log shows
            # success, but the loop calls scrape regardless.
            return ["h1"] if not allocation.bad_nodes_path.read_text() else []

        rc = _run(monkeypatch, config, allocation, runner, scrape_fn=scrape)
        assert rc == 0
        # Named swap: h1 → h3 (first spare)
        assert "h1" not in allocation.active
        assert "h3" in allocation.active
        assert allocation.bad_nodes_path.read_text() == "h1\n"

    def test_walltime_no_retry(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)
        runner = _FakeRunner([(143, "Execution finished with 0\n")])
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 143
        assert len(runner.calls) == 1

    def test_stuck_pre_training_two_zero_progress_attempts(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)
        # Both attempts have NO step= markers → STUCK guard fires
        # on the second attempt.
        runner = _FakeRunner(
            [
                (1, "Execution finished with 1\nimport error\n"),
                (1, "Execution finished with 1\nimport error\n"),
            ]
        )
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 1
        assert len(runner.calls) == 2

    def test_exhausted_when_spares_drained(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        # 1 spare → 1 swap possible, then EXHAUSTED on next failure.
        allocation = _alloc(
            tmp_path, full=("h1", "h2", "h3"), active=2
        )
        runner = _FakeRunner(
            [
                (1, "Execution finished with 1\niter step=1\n"),
                (1, "Execution finished with 1\niter step=2\n"),
            ]
        )
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 1
        # 1st attempt fails → blind swap (consumes h3). 2nd attempt
        # fails with no spares left → EXHAUSTED, no 3rd attempt.
        assert len(runner.calls) == 2
        assert list(allocation.spare) == []

    def test_max_failover_retries_cap(self, tmp_path, monkeypatch):
        # 3 spares but cap=1 → attempt 1 + 1 retry = 2 total, then bail.
        config = _config(tmp_path, max_failover_retries=1)
        allocation = _alloc(
            tmp_path, full=("h1", "h2", "h3", "h4", "h5"), active=2
        )
        runner = _FakeRunner(
            [
                (1, "Execution finished with 1\niter step=1\n"),
                (1, "Execution finished with 1\niter step=2\n"),
            ]
        )
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 1
        assert len(runner.calls) == 2

    def test_progress_clears_stuck_guard_for_next_attempt(
        self, tmp_path, monkeypatch
    ):
        # Attempt 1: no progress (would trip the guard on its own).
        # Attempt 2: progress (clears the guard). Attempt 3: no
        # progress again — but prior was YES, so guard doesn't trip.
        config = _config(tmp_path)
        allocation = _alloc(
            tmp_path, full=("h1", "h2", "h3", "h4"), active=2
        )
        runner = _FakeRunner(
            [
                (1, "Execution finished with 1\nimport error\n"),
                (1, "Execution finished with 1\niter step=3 loss=0.2\n"),
                (0, "Execution finished with 0\niter step=10\n"),
            ]
        )
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 0
        assert len(runner.calls) == 3

    def test_sigint_during_attempt_returns_interrupted(
        self, tmp_path, monkeypatch
    ):
        # Row 9 of the termination matrix: a KeyboardInterrupt raised
        # by _run_attempt_with_tee (Ctrl-C from the user, or any
        # SIGINT-equivalent during the child's lifetime) propagates
        # up through run_with_auto_retry as INTERRUPTED. Returns
        # 128 + SIGINT (130 on POSIX), logs FAILOVER STOP: interrupted,
        # does not retry.
        import signal as _signal
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)

        sigint_raiser_calls = []

        def _raises_sigint(cmd, log_path: Path, idle_timeout_s: int) -> int:
            # Simulate Ctrl-C: write a partial log (the child got
            # killed mid-attempt) then raise KeyboardInterrupt as
            # the real _run_attempt_with_tee would after SIGINT
            # reaches it.
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("starting...\nkilled by user\n")
            sigint_raiser_calls.append(log_path)
            raise KeyboardInterrupt()

        rc = _run(monkeypatch, config, allocation, _raises_sigint)
        assert rc == 128 + _signal.SIGINT
        # Exactly one attempt before the interrupt — NO retry.
        assert len(sigint_raiser_calls) == 1
        # No nodes swapped — the interrupt short-circuited before
        # the classifier ran.
        assert allocation.bad_nodes_path.read_text() == ""

    def test_sigint_on_first_attempt_doesnt_consume_spares(
        self, tmp_path, monkeypatch
    ):
        # Specifically: Ctrl-C on the very first attempt must not
        # touch the spare pool. Catches a regression where the
        # interrupt handler somehow falls through into the swap
        # logic.
        config = _config(tmp_path)
        allocation = _alloc(tmp_path)
        original_spare = list(allocation.spare)

        def _raises_sigint(cmd, log_path: Path, idle_timeout_s: int) -> int:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")
            raise KeyboardInterrupt()

        _run(monkeypatch, config, allocation, _raises_sigint)
        assert list(allocation.spare) == original_spare

    def test_unbounded_default_runs_until_exhaustion(
        self, tmp_path, monkeypatch
    ):
        # max_failover_retries=None and 3 spares → loop runs until
        # spares are drained, never hits a numeric cap.
        config = _config(tmp_path, max_failover_retries=None)
        allocation = _alloc(
            tmp_path, full=("h1", "h2", "h3", "h4", "h5"), active=2
        )
        # Each attempt makes some progress (clears STUCK guard)
        # but fails. Should exhaust 3 spares → 4 attempts total
        # (initial + 3 retries), then EXHAUSTED on the 4th's
        # classifier (has_spares=False).
        attempts = [
            (1, f"Execution finished with 1\niter step={i}\n")
            for i in range(1, 5)
        ]
        runner = _FakeRunner(attempts)
        rc = _run(monkeypatch, config, allocation, runner)
        assert rc == 1
        assert len(runner.calls) == 4


# ---------------------------------------------------------------------------
# Argparse: cross-flag validation
# ---------------------------------------------------------------------------


class TestArgparseValidation:
    def test_auto_retry_requires_explicit_nproc(self):
        with pytest.raises(SystemExit, match="requires --nproc"):
            parse_args(["--auto-retry", "--", "echo", "x"])

    def test_auto_retry_mutex_with_retries(self):
        with pytest.raises(SystemExit, match="mutually exclusive"):
            parse_args(
                ["--auto-retry", "--retries", "3", "--np", "8", "--",
                 "echo", "x"]
            )

    def test_auto_retry_with_zero_retries_is_fine(self):
        # --retries defaults to 0; not providing it shouldn't trip
        # the mutex check (only nonzero --retries conflicts).
        args = parse_args(["--auto-retry", "--np", "8", "--", "echo", "x"])
        assert args.auto_retry is True
        assert args.retries == 0

    def test_spare_nodes_accepts_auto_literal(self):
        args = parse_args(
            ["--auto-retry", "--spare-nodes", "auto", "--np", "8",
             "--", "echo", "x"]
        )
        assert args.spare_nodes == "auto"

    def test_spare_nodes_accepts_int(self):
        args = parse_args(
            ["--auto-retry", "--spare-nodes", "4", "--np", "8",
             "--", "echo", "x"]
        )
        assert args.spare_nodes == 4

    def test_spare_nodes_rejects_uppercase_auto(self):
        # Case-sensitive: only the literal lowercase "auto" works.
        # (Same case-handling lesson as machine= override on the
        # scraper — see ezpz.failover.scrape._resolve_machine_key.)
        with pytest.raises(SystemExit):
            parse_args(
                ["--auto-retry", "--spare-nodes", "AUTO", "--np", "8",
                 "--", "echo", "x"]
            )

    def test_spare_nodes_rejects_negative(self):
        with pytest.raises(SystemExit):
            parse_args(
                ["--auto-retry", "--spare-nodes", "-1", "--np", "8",
                 "--", "echo", "x"]
            )

    def test_max_failover_retries_rejects_negative(self):
        with pytest.raises(SystemExit):
            parse_args(
                ["--auto-retry", "--max-failover-retries", "-1",
                 "--np", "8", "--", "echo", "x"]
            )

    def test_back_compat_no_auto_retry(self):
        # Existing --retries / --timeout flags still work without
        # --auto-retry.
        args = parse_args(
            ["--np", "8", "--retries", "3", "--timeout", "600",
             "--", "echo", "x"]
        )
        assert args.auto_retry is False
        assert args.retries == 3
        assert args.idle_timeout_s == 600


# ---------------------------------------------------------------------------
# Helpers in launch.py — the bridge between CLI args and the loop
# ---------------------------------------------------------------------------


class TestRanksToHosts:
    """Ceiling-divide rank counts to host counts."""

    def test_exact_division(self):
        # 24 ranks on 12-GPU nodes → 2 hosts cleanly.
        assert _ranks_to_hosts(24, 12) == 2

    def test_single_host(self):
        # 8 ranks on 12-GPU nodes → 1 host (everything fits).
        assert _ranks_to_hosts(8, 12) == 1

    def test_round_up_on_overflow(self):
        # 13 ranks on 12-GPU nodes → 2 hosts. The point: a partial
        # host still needs to be reserved or rank 12 has nowhere to
        # land. The opposite (`13 // 12 = 1`) would drop a rank.
        assert _ranks_to_hosts(13, 12) == 2

    def test_polaris_4_gpu_nodes(self):
        # Realistic Polaris example: 12 ranks on 4-GPU nodes.
        assert _ranks_to_hosts(12, 4) == 3
        assert _ranks_to_hosts(13, 4) == 4

    def test_one_rank(self):
        # Edge case: 1 rank always needs 1 host regardless of ppn.
        assert _ranks_to_hosts(1, 12) == 1
        assert _ranks_to_hosts(1, 1) == 1


class TestResolveAutoRetryNodePool:
    """--hostfile beats scheduler nodelist; either-empty errors."""

    def test_hostfile_takes_priority_over_scheduler_nodelist(
        self, tmp_path
    ):
        # User pre-filtered their nodelist (e.g. dropped known-bad
        # nodes from a previous job). Silently switching to the
        # scheduler's larger list would undo that work.
        hf = tmp_path / "filtered.hostfile"
        hf.write_text("good1\ngood2\ngood3\n")
        scheduler = ["bad1", "good1", "bad2", "good2", "bad3", "good3"]
        pool = _resolve_auto_retry_node_pool(hf, scheduler)
        assert pool == ["good1", "good2", "good3"]

    def test_hostfile_strips_whitespace_and_blank_lines(self, tmp_path):
        hf = tmp_path / "messy.hostfile"
        # PBS_NODEFILE-style files can have trailing whitespace and
        # an EOF newline; we tolerate both rather than splitting and
        # producing empty hostnames.
        hf.write_text("  host1  \n\nhost2\n   \nhost3\n\n")
        assert _resolve_auto_retry_node_pool(hf, None) == [
            "host1",
            "host2",
            "host3",
        ]

    def test_falls_back_to_scheduler_nodelist_when_no_hostfile(self):
        pool = _resolve_auto_retry_node_pool(None, ["h1", "h2", "h3"])
        assert pool == ["h1", "h2", "h3"]

    def test_empty_hostfile_errors(self, tmp_path):
        hf = tmp_path / "empty.hostfile"
        hf.write_text("")
        with pytest.raises(SystemExit, match="empty"):
            _resolve_auto_retry_node_pool(hf, ["h1", "h2"])

    def test_hostfile_with_only_whitespace_errors(self, tmp_path):
        # `\n\n   \n\n` is "empty" by the same logic as a truly
        # empty file — all candidate lines were blank.
        hf = tmp_path / "blank.hostfile"
        hf.write_text("\n\n   \n\n")
        with pytest.raises(SystemExit, match="empty"):
            _resolve_auto_retry_node_pool(hf, ["h1"])

    def test_missing_hostfile_errors(self, tmp_path):
        hf = tmp_path / "nonexistent.hostfile"
        with pytest.raises(SystemExit, match="failed to read"):
            _resolve_auto_retry_node_pool(hf, ["h1"])

    def test_no_hostfile_no_scheduler_nodelist_errors(self):
        with pytest.raises(SystemExit, match="failed to read"):
            _resolve_auto_retry_node_pool(None, None)

    def test_no_hostfile_empty_scheduler_nodelist_errors(self):
        with pytest.raises(SystemExit, match="failed to read"):
            _resolve_auto_retry_node_pool(None, [])


class TestAutoRetryLogDir:
    def test_strips_pbs_jobid_suffix(self, tmp_path, monkeypatch):
        # PBS jobids on Aurora are formatted as `<num>.aurora-pbs-...`;
        # the log dir name uses only the leading numeric segment so
        # ls'ing logs/ doesn't get a wall of hostnames.
        monkeypatch.chdir(tmp_path)
        result = _auto_retry_log_dir("12345.aurora-pbs-0001.alcf.anl.gov")
        assert result == tmp_path / "logs" / "failover-12345"

    def test_handles_bare_numeric_jobid(self, tmp_path, monkeypatch):
        # SLURM-style bare numeric jobids — no `.` to split on.
        monkeypatch.chdir(tmp_path)
        result = _auto_retry_log_dir("987654")
        assert result == tmp_path / "logs" / "failover-987654"


class TestResolveAutoRetryAllocation:
    """The bridge: full PBS nodelist → NodeAllocation with active/spare split."""

    def test_auto_default_uses_full_spare_pool(self, tmp_path):
        # 10 nodes, 8 active → "auto" = 2 spares
        nodelist = [f"h{i}" for i in range(10)]
        alloc, hf = _resolve_auto_retry_allocation(
            nodelist, 8, "auto", tmp_path
        )
        assert len(alloc.active) == 8
        assert list(alloc.spare) == ["h8", "h9"]
        assert hf == tmp_path / "active.hostfile"
        # Persisted on disk so the launcher's hostfile arg picks it up.
        assert hf.read_text().splitlines() == alloc.active

    def test_none_treated_as_auto(self, tmp_path):
        # spare_nodes=None (the CLI default when --auto-retry is set
        # but --spare-nodes is omitted) behaves the same as "auto".
        nodelist = [f"h{i}" for i in range(5)]
        alloc, _ = _resolve_auto_retry_allocation(
            nodelist, 3, None, tmp_path
        )
        assert list(alloc.spare) == ["h3", "h4"]

    def test_explicit_int_caps_spare_pool(self, tmp_path):
        # 10 nodes, 8 active, --spare-nodes 1 → use only 1 spare,
        # ignore the other available host. The 10th node sits idle.
        nodelist = [f"h{i}" for i in range(10)]
        alloc, _ = _resolve_auto_retry_allocation(
            nodelist, 8, 1, tmp_path
        )
        assert len(alloc.active) == 8
        assert list(alloc.spare) == ["h8"]

    def test_explicit_int_exceeds_available_errors(self, tmp_path):
        # 10 nodes, 8 active → only 2 spare available. Asking for
        # 5 must error at parse-time-equivalent (SystemExit) rather
        # than silently using 2.
        nodelist = [f"h{i}" for i in range(10)]
        with pytest.raises(SystemExit, match="spare nodes"):
            _resolve_auto_retry_allocation(nodelist, 8, 5, tmp_path)

    def test_nproc_exceeds_allocation_errors(self, tmp_path):
        # 3 nodes, 8 active requested → impossible.
        nodelist = [f"h{i}" for i in range(3)]
        with pytest.raises(SystemExit, match="active hosts"):
            _resolve_auto_retry_allocation(nodelist, 8, "auto", tmp_path)

    def test_single_host_no_spares_succeeds(self, tmp_path):
        # 1 node, 1 active, 0 spare. Legal — the loop will just
        # surface EXHAUSTED on the first failure. We don't refuse
        # at allocation time (user might be debugging on a 1-node job).
        nodelist = ["h0"]
        alloc, _ = _resolve_auto_retry_allocation(
            nodelist, 1, "auto", tmp_path
        )
        assert alloc.active == ["h0"]
        assert list(alloc.spare) == []
        assert not alloc.has_spares

    def test_explicit_zero_spares(self, tmp_path):
        # --spare-nodes 0 explicitly: caller wants the failover loop
        # WITHOUT a swap pool. Useful for testing the classifier
        # paths in isolation. EXHAUSTED on the first failure.
        nodelist = [f"h{i}" for i in range(5)]
        alloc, _ = _resolve_auto_retry_allocation(
            nodelist, 3, 0, tmp_path
        )
        assert len(alloc.active) == 3
        assert list(alloc.spare) == []

    def test_bad_nodes_file_initialized_empty(self, tmp_path):
        nodelist = [f"h{i}" for i in range(5)]
        alloc, _ = _resolve_auto_retry_allocation(
            nodelist, 3, "auto", tmp_path
        )
        assert alloc.bad_nodes_path == tmp_path / "bad_nodes.txt"
        assert alloc.bad_nodes_path.read_text() == ""

    def test_idle_nodes_beyond_active_plus_spare_truncated(self, tmp_path):
        # 20 nodes, 8 active, --spare-nodes 2 → only 10 hosts used.
        # The other 10 sit idle. Mirrors failover_init's truncation.
        nodelist = [f"h{i}" for i in range(20)]
        alloc, _ = _resolve_auto_retry_allocation(
            nodelist, 8, 2, tmp_path
        )
        assert len(alloc.active) == 8
        assert list(alloc.spare) == ["h8", "h9"]
        # h10..h19 are NOT in the allocation.
