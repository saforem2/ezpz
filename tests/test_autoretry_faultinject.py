"""``--auto-retry`` driven against real, really-failing subprocesses.

Every other test of this loop replaces `_run_attempt_with_tee` with a
fake (`test_launch_autoretry.py:623`), so `run_with_auto_retry` has never
actually spawned a process, never really killed one on the idle
watchdog, and never swapped a node in response to a genuine crash. The
loop's *decisions* are well covered; its *behavior* was not.

This file closes that gap. Each test runs `_faultinject.py` as a real
child that prints real failure signatures -- copied from
`ezpz.failover.patterns` and `_CRASH_PATTERNS_RX`, not invented -- and
asserts on what an operator would actually see afterwards: the return
code, how many attempts really ran, which host landed in
`bad_nodes.txt`, and what the active hostfile says.

Two properties are asserted throughout because they are what the feature
promises and what a mock cannot establish:

* a bad node is *removed from the hostfile the next attempt reads*, and
* the loop stops for the right reason -- several outcomes share an exit
  code, so the rc alone does not distinguish them.

Backoff (5s, 10s, 20s...) is stubbed out; without that the file would
take minutes. The idle-watchdog test is the one place real seconds are
spent, because there is no way to fake elapsed silence.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="auto-retry assumes POSIX subprocess semantics"
)

from ezpz.launch_autoretry import (  # noqa: E402
    AutoRetryConfig,
    NodeAllocation,
    run_with_auto_retry,
)

FAULT_SCRIPT = str(Path(__file__).parent / "_faultinject.py")

# Matches _faultinject.DEFAULT_HOST. Kept as a literal rather than
# imported so a rename over there fails loudly here.
BAD_HOST = "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"

WALLTIME_RC = 143
WATCHDOG_RC = 124


class Harness:
    """One auto-retry run against a real child, plus what it left behind."""

    def __init__(self, tmp_path: Path, **env: str):
        self.tmp = tmp_path
        self.counter = tmp_path / "attempts.txt"
        self.counter.write_text("0")
        self.log_dir = tmp_path / "logs"
        self.hostfile = tmp_path / "active.hostfile"
        self.bad_nodes = tmp_path / "bad_nodes.txt"
        self.env = {"FI_COUNTER": str(self.counter), **env}
        self.rc: int | None = None

    def run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        nodes: tuple[str, ...] = (BAD_HOST, "spare-1", "spare-2"),
        active: int = 1,
        machine: str | None = "sunspot",
        idle_timeout_s: int = 0,
        max_failover_retries: int | None = 3,
    ) -> int:
        for k, v in self.env.items():
            monkeypatch.setenv(k, v)
        # Zero the RETRY backoff (5/10/20s), which would otherwise
        # dominate the suite. Deliberately not `time.sleep`: that is the
        # shared stdlib function, so patching it also removes the 0.1-1s
        # poll sleep inside `_run_attempt_with_tee`, leaving the watchdog
        # tests to busy-spin a core for their whole timeout.
        monkeypatch.setattr(
            "ezpz.launch_autoretry._backoff_for_attempt", lambda _n: 0.0
        )
        if idle_timeout_s == 0:
            # With the watchdog off, the poll loop sleeps a flat 1.0s
            # per iteration, so a child that exits in 10ms still costs a
            # second per attempt. Shorten it ONLY here -- the watchdog
            # tests keep the real sleep, since shortening it there would
            # be tuning away the thing under test.
            real_sleep = time.sleep
            monkeypatch.setattr(
                "ezpz.launch_autoretry.time.sleep",
                lambda s: real_sleep(min(s, 0.02)),
            )
        cfg = AutoRetryConfig(
            cmd=[sys.executable, FAULT_SCRIPT],
            log_dir=self.log_dir,
            idle_timeout_s=idle_timeout_s,
            max_failover_retries=max_failover_retries,
            machine=machine,
        )
        alloc = NodeAllocation.from_full_nodelist(
            list(nodes), active, self.hostfile, self.bad_nodes
        )
        self.rc = run_with_auto_retry(cfg, alloc)
        return self.rc

    # -- what the run left behind -------------------------------------

    @property
    def attempts(self) -> int:
        """Attempts the CHILD actually ran, not attempt logs created."""
        return int(self.counter.read_text().strip() or 0)

    @property
    def attempt_logs(self) -> list[Path]:
        return sorted(self.log_dir.glob("attempt-*.log"))

    @property
    def bad(self) -> list[str]:
        return self.bad_nodes.read_text().split()

    @property
    def active_hosts(self) -> list[str]:
        return self.hostfile.read_text().split()

    def log_text(self) -> str:
        return "".join(p.read_text() for p in self.attempt_logs)


class TestNamedBadNode:
    """The headline promise: name a bad node, replace it, carry on."""

    def test_swaps_the_named_host_then_succeeds(self, tmp_path, monkeypatch):
        """The named host must be evicted -- not merely *a* host.

        The bad node is placed SECOND in the active set on purpose.
        Blind rotation always evicts `active[0]`, so if the scraper's
        name were ignored (or `swap_in` silently no-op'd and the loop
        fell back to blind rotation, which it does at
        `launch_autoretry.py:795`) the wrong host would be retired and
        the bad one would still be in the hostfile. With the bad host at
        index 0 this test passes either way -- it was written that way
        first, and a mutation check caught it.
        """
        h = Harness(tmp_path, FI_MODE="shepherd", FI_FAIL_ON="1")
        rc = h.run(
            monkeypatch,
            nodes=("healthy-0", BAD_HOST, "spare-1"),
            active=2,
        )

        assert rc == 0
        assert h.attempts == 2, "the child should have run exactly twice"
        assert h.bad == [BAD_HOST], (
            f"expected the SCRAPED host to be retired, got {h.bad} -- "
            "a blind rotation would have evicted 'healthy-0' instead"
        )
        assert BAD_HOST not in h.active_hosts, (
            "the bad host is STILL in the hostfile the next attempt reads "
            "-- the swap did not take effect where it matters"
        )
        assert h.active_hosts == ["healthy-0", "spare-1"], (
            "the healthy host should be untouched and the spare should "
            f"take the bad host's slot; got {h.active_hosts}"
        )

    def test_blind_rotates_when_the_machine_is_unknown(
        self, tmp_path, monkeypatch
    ):
        """Off-ALCF, auto-detect yields no patterns -> blind rotation.

        Pins real behavior rather than an aspiration: `launch.py` never
        sets `machine`, so this is the path a non-ALCF user gets. The
        run still recovers; it just cannot say *which* node was bad.
        """
        h = Harness(tmp_path, FI_MODE="shepherd", FI_FAIL_ON="1")
        rc = h.run(
            monkeypatch,
            nodes=("healthy-0", BAD_HOST, "spare-1"),
            active=2,
            machine="no-such-cluster",
        )

        assert rc == 0
        assert h.attempts == 2
        # Same layout as the named-host test, and the OPPOSITE outcome:
        # with no patterns to match, the loop evicts active[0] and the
        # genuinely-bad host stays in the pool. That is the cost of not
        # knowing the machine, and it is worth pinning explicitly.
        assert h.bad == ["healthy-0"], (
            f"expected a positional eviction of active[0], got {h.bad}"
        )
        assert BAD_HOST in h.active_hosts, (
            "without machine patterns the scraper cannot name the bad "
            "host, so it is expected to survive the rotation"
        )


class TestFailureSignatures:
    """Each mode is a real signature from a real captured log."""

    @pytest.mark.parametrize(
        "mode", ["shepherd", "ur_oom", "rank_exit", "silent_fail"]
    )
    def test_retryable_failures_are_retried(self, tmp_path, monkeypatch, mode):
        h = Harness(tmp_path, FI_MODE=mode, FI_FAIL_ON="1")
        rc = h.run(monkeypatch)

        assert rc == 0, f"{mode} was not recovered from"
        assert h.attempts == 2
        assert len(h.bad) == 1, f"{mode} did not retire a node"

    @pytest.mark.parametrize("mode", ["clean_walltime", "innocent_cascade"])
    def test_walltime_is_not_a_bad_node(self, tmp_path, monkeypatch, mode):
        """Swapping nodes cannot buy more wallclock, so do not try.

        `innocent_cascade` is the subtle one: a clean walltime kill
        SIGTERMs every rank, producing a screenful of `rank N died from
        signal 15`. Reading that as hardware death would burn a spare
        on every single walltime expiry.
        """
        h = Harness(tmp_path, FI_MODE=mode)
        rc = h.run(monkeypatch)

        assert rc == WALLTIME_RC
        assert h.attempts == 1, "walltime must not be retried"
        assert h.bad == [], "a spare was burned on a walltime kill"

    def test_rank_exit_is_retried_despite_looking_like_walltime(
        self, tmp_path, monkeypatch
    ):
        """The converse, and the reason the cascade strip is careful.

        A rank exiting non-zero under PALS also surfaces as 143. Only
        the `rank N exited with code K` line separates it from a real
        walltime kill -- and it *must* be retried.
        """
        h = Harness(tmp_path, FI_MODE="rank_exit", FI_FAIL_ON="1")
        rc = h.run(monkeypatch)

        assert rc == 0
        assert h.attempts == 2, (
            "a genuine crash that happens to exit 143 was mistaken for a "
            "walltime kill and abandoned"
        )


class TestGlooPeerScrape:
    """The gloo signature resolves an IP by shelling out to `getent`.

    Found by this harness: `reverse_resolve_ip`
    (`failover/patterns/__init__.py:141-151`) catches
    `CalledProcessError`, `TimeoutExpired` and `FileNotFoundError` -- but
    NOT `PermissionError` or any other `OSError`. Its own docstring
    promises `None` when the lookup fails "(binary missing, timeout,
    empty result)", and a `getent` that exists but is not executable
    breaks that promise.

    It matters because the default `scrape_fn`
    (`launch_autoretry.py:676-681`) also catches only
    `FileNotFoundError`, so the exception escapes `run_with_auto_retry`
    entirely: a recoverable node failure takes down the failover
    machinery instead of being retried.

    Reproduced on macOS, where `getent` does not exist as an executable
    binary. Filed as a follow-up rather than fixed here -- these tests
    were written to find bugs, not to be written against a fix.
    """

    def test_gloo_failure_is_retried_when_resolution_is_unavailable(
        self, tmp_path, monkeypatch
    ):
        """With resolution stubbed out, the gloo crash retries normally.

        The crash pattern alone is enough to trigger a blind rotation;
        the IP lookup only upgrades that to a *named* host. So the
        recovery path does not depend on `getent` -- only the
        attribution does.
        """
        # Patch where it is USED, not where it is defined: sunspot.py
        # does `from ... import reverse_resolve_ip`, so it holds its own
        # reference and patching the definition site has no effect.
        monkeypatch.setattr(
            "ezpz.failover.patterns.sunspot.reverse_resolve_ip",
            lambda *_a, **_k: None,
        )
        h = Harness(tmp_path, FI_MODE="gloo_peer", FI_FAIL_ON="1")
        rc = h.run(monkeypatch)

        assert rc == 0
        assert h.attempts == 2
        assert len(h.bad) == 1

    def test_resolver_error_currently_escapes_the_loop(self, tmp_path):
        """Pins the bug above so the fix is a deliberate edit.

        Change this to `== []` once `reverse_resolve_ip` swallows
        `OSError`.
        """
        from ezpz.failover.patterns import reverse_resolve_ip

        log = tmp_path / "a.log"
        log.write_text(
            "RuntimeError: [enforce fail at gloo/transport/tcp/pair.cc:598] "
            "Connection closed by peer [10.0.0.42]:53121\n"
        )
        import subprocess as _sp

        def _boom(*_a, **_k):
            raise PermissionError(13, "Permission denied", "getent")

        orig = _sp.check_output
        _sp.check_output = _boom
        try:
            with pytest.raises(PermissionError):
                reverse_resolve_ip("10.0.0.42")
        finally:
            _sp.check_output = orig


class TestProcessLevelFailures:
    """Deaths the classifier only sees as an exit code."""

    def test_sigkill_is_retried(self, tmp_path, monkeypatch):
        """SIGKILL gives Popen a NEGATIVE rc (-9), not 137."""
        h = Harness(tmp_path, FI_MODE="sigkill", FI_FAIL_ON="1")
        rc = h.run(monkeypatch)

        assert rc == 0
        assert h.attempts == 2

    def test_missing_executable_escapes_the_loop(self, tmp_path, monkeypatch):
        """Documents current behavior: Popen's error is NOT caught.

        The loop guards `KeyboardInterrupt` only, so a bad `cmd[0]`
        propagates out. Arguably it should be reported as a
        configuration error rather than a traceback, but that is a
        behavior change; this pins what happens today so a future fix
        is a deliberate edit and not a surprise.
        """
        monkeypatch.setattr(
            "ezpz.launch_autoretry.time.sleep", lambda *_a, **_k: None
        )
        cfg = AutoRetryConfig(
            cmd=["/nonexistent/definitely-not-a-binary"],
            log_dir=tmp_path / "logs",
            idle_timeout_s=0,
        )
        alloc = NodeAllocation.from_full_nodelist(
            ["a", "b"], 1, tmp_path / "hf", tmp_path / "bad"
        )
        with pytest.raises(FileNotFoundError):
            run_with_auto_retry(cfg, alloc)


class TestIdleWatchdog:
    """The path no test has ever executed against a real process."""

    def test_a_silent_child_is_killed_and_retried(self, tmp_path, monkeypatch):
        """A hung job produces no output and no error -- just silence.

        This is the failure the watchdog exists for, and the only one
        that costs real wall-clock to reproduce: elapsed silence cannot
        be faked. The child would sleep 120s; if the watchdog does not
        fire, this test fails by timing out rather than passing quietly.
        """
        h = Harness(
            tmp_path, FI_MODE="hang", FI_FAIL_ON="1", FI_HANG_S="120"
        )
        rc = h.run(monkeypatch, idle_timeout_s=2)

        assert rc == 0, "the loop did not recover after killing the hang"
        assert h.attempts == 2, (
            "the hung child was not killed and retried -- attempts="
            f"{h.attempts}"
        )
        assert len(h.bad) == 1, "a hang should retire the node it hung on"

    def test_watchdog_kill_returns_124_and_is_terminal_without_spares(
        self, tmp_path, monkeypatch
    ):
        """With no spare to rotate to, the watchdog kill is the answer."""
        h = Harness(tmp_path, FI_MODE="hang", FI_HANG_S="120")
        rc = h.run(
            monkeypatch,
            nodes=("only-host",),
            active=1,
            idle_timeout_s=2,
            max_failover_retries=0,
        )

        assert rc == WATCHDOG_RC, (
            f"expected the watchdog rc {WATCHDOG_RC}, got {rc}"
        )
        assert h.attempts == 1


class TestExhaustionAndCaps:
    def test_spares_are_consumed_then_the_loop_stops(
        self, tmp_path, monkeypatch
    ):
        """Two spares, an always-failing child: both get used, then it gives up."""
        h = Harness(tmp_path, FI_MODE="shepherd")  # fails every attempt
        rc = h.run(
            monkeypatch,
            nodes=(BAD_HOST, "spare-1", "spare-2"),
            active=1,
            max_failover_retries=None,  # bounded by spares, not by count
        )

        assert rc != 0
        assert h.attempts == 3, (
            "expected one attempt per available host (1 active + 2 spares), "
            f"got {h.attempts}"
        )
        assert len(h.bad) == 2, "spares were not actually consumed"

    def test_max_failover_retries_is_respected(self, tmp_path, monkeypatch):
        """The cap counts RETRIES, so 1 means at most 2 attempts."""
        h = Harness(tmp_path, FI_MODE="shepherd")
        rc = h.run(
            monkeypatch,
            nodes=(BAD_HOST, "s1", "s2", "s3", "s4"),
            active=1,
            max_failover_retries=1,
        )

        assert rc != 0
        assert h.attempts == 2, (
            f"cap of 1 retry should allow 2 attempts, got {h.attempts}"
        )


class TestProgressMarkerContract:
    """`step=N` is the ONLY thing the loop counts as progress.

    This is not a hypothetical. `_PROGRESS_MARKER_RX` is
    `\\bstep=\\d+`, and no ezpz example prints it -- `minimal.py` and
    `test.py` both emit `iter=N` via `format_compact_summary`. So a real
    ezpz job that crashes twice is filed as "never started" and
    ABANDONED rather than failed over. These two tests pin both sides of
    that behavior so the follow-up fix has something to change.
    """

    def test_progress_keeps_the_loop_failing_over(self, tmp_path, monkeypatch):
        h = Harness(tmp_path, FI_MODE="shepherd", FI_STEPS="2")
        h.run(
            monkeypatch,
            nodes=(BAD_HOST, "s1", "s2"),
            active=1,
            max_failover_retries=None,
        )

        assert h.rc != 0
        assert "step=" in h.log_text()
        assert h.attempts == 3, (
            "with visible progress the loop should keep swapping until "
            f"spares run out; got {h.attempts} attempts"
        )

    def test_real_iter_output_is_not_seen_as_progress(
        self, tmp_path, monkeypatch
    ):
        """The actual bug: `iter=N` is real output and does not count.

        `FI_MARKER=iter` emits exactly what `minimal.py` and `test.py`
        print. The log is FULL of progress; the loop still gives up at
        two attempts with spares to spare.

        An earlier version of this test used `FI_STEPS=0` -- an empty
        log -- which proved only that the guard fires on silence, and
        would have kept passing after a fix that made `iter=` count.
        Raised in review; this is the version that would actually go red
        once #224 is fixed.
        """
        h = Harness(
            tmp_path, FI_MODE="shepherd", FI_MARKER="iter", FI_STEPS="3"
        )
        h.run(
            monkeypatch,
            nodes=(BAD_HOST, "s1", "s2"),
            active=1,
            max_failover_retries=None,
        )

        log = h.log_text()
        assert "iter=" in log, "the child did not emit the real marker"
        assert "step=" not in log
        assert h.rc != 0
        assert h.attempts == 2, (
            "the loop abandoned a job that was visibly training -- see "
            f"issue #224; got {h.attempts} attempts with 2 spares free"
        )
        assert len(h.bad) == 1

    def test_no_output_at_all_aborts_after_two_attempts(
        self, tmp_path, monkeypatch
    ):
        """The genuinely-silent case: no progress lines at all.

        Distinct from the test above -- this is a job that really did
        die before training, which the guard is *right* to abandon.
        """
        h = Harness(tmp_path, FI_MODE="shepherd", FI_STEPS="0")
        rc = h.run(
            monkeypatch,
            nodes=(BAD_HOST, "s1", "s2"),
            active=1,
            max_failover_retries=None,
        )

        assert rc != 0
        assert "step=" not in h.log_text()
        assert h.attempts == 2, (
            "expected the stuck-pre-training guard to stop at 2 attempts; "
            f"got {h.attempts}"
        )
        assert len(h.bad) == 1, (
            "one spare was consumed before the guard tripped"
        )
