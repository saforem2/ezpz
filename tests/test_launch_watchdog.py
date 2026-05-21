"""Tests for the idle-output watchdog and retry loop in :mod:`ezpz.launch`.

Covers two flags exposed by ``ezpz launch``:

  * ``--timeout SECONDS`` — kill the launched process if it produces
    no output (stdout+stderr) for this many seconds (motivating case:
    xccl on XPU silently ignores ``train_timeout_seconds`` and PBS
    jobs sit hung for the full walltime).

  * ``--retries N`` — re-execute on any non-zero exit, up to N times.

Tests simulate child behavior using ``sh -c`` scripts, which targets
the POSIX environments this CLI is built for (PBS, SLURM, mpirun).
Skipped on Windows / non-POSIX platforms.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ezpz.launch import (
    _WATCHDOG_EXIT_CODE,
    _run_with_retries,
    _run_with_watchdog,
)

# `ezpz launch` is a POSIX-targeted CLI (PBS, SLURM, mpirun). The
# tests below rely on `sh`, `sleep`, and POSIX shell control flow; on
# Windows / non-POSIX they would fail with "sh not found" or
# semantically-different shell behavior. Skip the whole module rather
# than per-test, since nothing in this file is portable.
pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="watchdog tests use POSIX `sh -c` scripts",
)


class TestWatchdog:
    """``_run_with_watchdog`` — kills processes that go quiet."""

    def test_passthrough_when_timeout_none(self):
        """``idle_timeout_s=None`` bypasses the watchdog entirely; the
        process runs to completion and its exit code is returned
        unchanged."""
        rc = _run_with_watchdog(["sh", "-c", "exit 7"], idle_timeout_s=None)
        assert rc == 7

    def test_passthrough_when_timeout_zero(self):
        """``idle_timeout_s=0`` is treated the same as ``None`` — disabled.
        Avoids dividing-by-zero / instant-kill foot-guns."""
        rc = _run_with_watchdog(["sh", "-c", "exit 3"], idle_timeout_s=0)
        assert rc == 3

    def test_silent_process_is_killed(self):
        """A process that produces no stdout within ``idle_timeout_s``
        gets SIGTERMed (then SIGKILLed after grace) and returns 124."""
        t0 = time.monotonic()
        rc = _run_with_watchdog(
            ["sh", "-c", "sleep 30"], idle_timeout_s=1
        )
        elapsed = time.monotonic() - t0
        assert rc == _WATCHDOG_EXIT_CODE  # 124
        # Should kill well before the 30s sleep finishes. Allow generous
        # headroom for slow CI: 1s timeout + 10s SIGKILL grace + buffer.
        assert elapsed < 15.0, (
            f"Watchdog took {elapsed:.1f}s to kill — expected < 15s"
        )

    def test_chatty_process_runs_to_completion(self):
        """A process that emits at least one line per ``idle_timeout_s``
        is NOT killed. Verifies the activity timer actually resets."""
        # Echo a line every 0.2s for ~2 seconds total. With a 1s
        # timeout, idle never exceeds 0.2s so the watchdog stays quiet.
        script = (
            "for i in 1 2 3 4 5 6 7 8 9 10; do "
            "echo line $i; sleep 0.2; "
            "done"
        )
        rc = _run_with_watchdog(["sh", "-c", script], idle_timeout_s=1)
        assert rc == 0

    def test_zero_exit_propagates(self):
        """Clean exit with code 0 returns 0 (not the watchdog code)."""
        rc = _run_with_watchdog(
            ["sh", "-c", "echo hello; exit 0"], idle_timeout_s=5
        )
        assert rc == 0

    def test_nonzero_exit_propagates(self):
        """Process's own non-zero exit code is returned, NOT
        ``_WATCHDOG_EXIT_CODE`` — caller can distinguish "command
        failed" from "watchdog killed it"."""
        rc = _run_with_watchdog(
            ["sh", "-c", "echo about to fail; exit 42"],
            idle_timeout_s=5,
        )
        assert rc == 42

    def test_python_child_is_unbuffered(self, tmp_path: Path):
        """Regression: when stdout is piped (as it is in watchdog mode),
        CPython block-buffers stdout — a healthy Python script that uses
        ``print()`` without ``flush=True`` can emit zero bytes for many
        seconds while accumulating a 4-8 KB buffer.

        Pre-fix, the watchdog would routinely SIGTERM such healthy jobs
        with a false-positive "no output for Xs" kill. The fix sets
        ``PYTHONUNBUFFERED=1`` in the child env so each ``print()`` is
        flushed immediately.

        This test asserts the env var is set by spawning a Python
        child that reports its own ``PYTHONUNBUFFERED`` value.
        """
        # Use the same interpreter the test is running under so we
        # don't depend on `python3` being on PATH. The child writes to
        # a file we control — intercepting the watchdog's sys.stdout
        # writes from this process is fiddly.
        import sys
        out_file = tmp_path / "child_env.txt"
        rc = _run_with_watchdog(
            [
                sys.executable,
                "-c",
                f"import os; open({str(out_file)!r}, 'w').write("
                f"os.environ.get('PYTHONUNBUFFERED', '<unset>'))",
            ],
            idle_timeout_s=5,
        )
        assert rc == 0
        assert out_file.read_text() == "1", (
            f"Expected PYTHONUNBUFFERED=1 in child env, got "
            f"{out_file.read_text()!r}"
        )

    def test_python_unbuffered_can_be_overridden(self, tmp_path: Path):
        """User-set ``PYTHONUNBUFFERED`` in the calling environment
        wins — the watchdog uses ``setdefault``, not unconditional
        assignment. Lets users opt into other values (e.g. ``""``
        explicitly) for edge cases."""
        import sys
        out_file = tmp_path / "child_env.txt"
        # Set a non-"1" value to prove our setdefault doesn't clobber.
        os.environ["PYTHONUNBUFFERED"] = "custom"
        try:
            rc = _run_with_watchdog(
                [
                    sys.executable,
                    "-c",
                    f"import os; open({str(out_file)!r}, 'w').write("
                    f"os.environ.get('PYTHONUNBUFFERED', '<unset>'))",
                ],
                idle_timeout_s=5,
            )
            assert rc == 0
            assert out_file.read_text() == "custom"
        finally:
            del os.environ["PYTHONUNBUFFERED"]


class TestRetries:
    """``_run_with_retries`` — re-executes on non-zero exit."""

    def test_zero_retries_is_single_attempt(self):
        """``retries=0`` means ONE attempt total — no retry. The failure
        exit code propagates unchanged."""
        rc = _run_with_retries(
            ["sh", "-c", "exit 5"], idle_timeout_s=None, retries=0
        )
        assert rc == 5

    def test_succeeds_first_try_no_retry(self):
        """A command that succeeds on the first attempt doesn't trigger
        the retry path at all."""
        rc = _run_with_retries(
            ["sh", "-c", "exit 0"], idle_timeout_s=None, retries=3
        )
        assert rc == 0

    def test_retries_until_success(self, tmp_path: Path, monkeypatch):
        """Command that fails N-1 times then succeeds: retry loop
        should keep trying until success and return 0."""
        # Use a counter file. Each invocation increments it; succeed
        # on the 3rd run (counter=3).
        counter = tmp_path / "counter.txt"
        counter.write_text("0")
        script = (
            f'n=$(cat "{counter}"); '
            f'n=$((n + 1)); '
            f'echo "$n" > "{counter}"; '
            f'if [ "$n" -ge 3 ]; then exit 0; else exit 1; fi'
        )
        # Patch sleep so the backoff doesn't slow tests down.
        monkeypatch.setattr("ezpz.launch.time.sleep", lambda *_a, **_k: None)
        rc = _run_with_retries(
            ["sh", "-c", script], idle_timeout_s=None, retries=5
        )
        assert rc == 0
        assert counter.read_text().strip() == "3"

    def test_retries_exhausted_returns_last_rc(self, tmp_path: Path, monkeypatch):
        """When every attempt fails, returns the LAST attempt's exit
        code (not the first, in case different failures occur)."""
        counter = tmp_path / "counter.txt"
        counter.write_text("0")
        script = (
            f'n=$(cat "{counter}"); '
            f'n=$((n + 1)); '
            f'echo "$n" > "{counter}"; '
            # Different exit codes each time so we can verify which
            # one propagates. Final attempt (n=3) exits with 99.
            f'if [ "$n" -eq 3 ]; then exit 99; else exit 1; fi'
        )
        monkeypatch.setattr("ezpz.launch.time.sleep", lambda *_a, **_k: None)
        rc = _run_with_retries(
            ["sh", "-c", script], idle_timeout_s=None, retries=2
        )
        # 1 initial + 2 retries = 3 attempts total. Last one exits 99.
        assert rc == 99
        assert counter.read_text().strip() == "3"

    def test_retry_works_with_watchdog_kill(
        self, tmp_path: Path, monkeypatch
    ):
        """The watchdog's 124 exit code is a non-zero exit like any
        other, so retries should kick in. Verifies the two helpers
        compose cleanly.

        Uses a counter file (like the other retry tests) so the
        assertion is direct: after `retries=1` and two watchdog kills,
        the script must have executed exactly twice. Pre-fix this
        test inferred re-execution from elapsed time, which is brittle
        on slow CI.
        """
        counter = tmp_path / "counter.txt"
        counter.write_text("0")
        # Increment counter, then sleep long enough that the watchdog
        # always kills us. Each attempt's exit code is 124 (watchdog),
        # which is non-zero so retries kick in.
        script = (
            f'n=$(cat "{counter}"); '
            f'n=$((n + 1)); '
            f'echo "$n" > "{counter}"; '
            f'sleep 30'
        )
        monkeypatch.setattr(
            "ezpz.launch.time.sleep", lambda *_a, **_k: None
        )
        rc = _run_with_retries(
            ["sh", "-c", script], idle_timeout_s=1, retries=1
        )
        # Final exit code is still the watchdog code — both attempts
        # got killed.
        assert rc == _WATCHDOG_EXIT_CODE
        # The actual assertion that proves retry happened: counter
        # incremented exactly twice (1 initial + 1 retry).
        assert counter.read_text().strip() == "2", (
            f"Expected 2 attempts (retries=1), counter shows "
            f"{counter.read_text().strip()}"
        )
