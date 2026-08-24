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

import logging
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="auto-retry assumes POSIX subprocess semantics"
)

from ezpz.launch_autoretry import (  # noqa: E402
    PROVENANCE_BLIND,
    PROVENANCE_SCRAPED,
    AutoRetryConfig,
    BadNodeRecord,
    NodeAllocation,
    parse_bad_nodes_file,
    run_with_auto_retry,
)
from ezpz.launch_autoretry import logger as _autoretry_logger  # noqa: E402

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
            # `AutoRetryConfig.cmd` documents that the command must
            # already carry `--hostfile=<path>`: the loop never
            # re-assembles it, it rewrites the file in place and relies
            # on the launcher re-reading it. Passing it here means the
            # child sees what a real launcher would.
            cmd=[
                sys.executable,
                FAULT_SCRIPT,
                f"--hostfile={self.hostfile}",
            ],
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
        # Column 1 only: bad_nodes.txt carries provenance columns
        # since #233, so a whole-file .split() would report 3 tokens
        # per retired host.
        return [r.host for r in self.bad_records]

    @property
    def bad_records(self) -> list[BadNodeRecord]:
        return parse_bad_nodes_file(self.bad_nodes)

    @property
    def active_hosts(self) -> list[str]:
        return self.hostfile.read_text().split()

    def log_text(self) -> str:
        return "".join(p.read_text() for p in self.attempt_logs)

    @property
    def hosts_seen(self) -> list[list[str]]:
        """Hosts each attempt's CHILD read, in attempt order.

        Distinct from `active_hosts`, which is the final on-disk state:
        this is what the running process was actually handed.
        """
        out = []
        for p in self.attempt_logs:
            for line in p.read_text(errors="replace").splitlines():
                if line.startswith("hostfile hosts="):
                    out.append(line.split("=", 1)[1].split(","))
                    break
        return out


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
        # ...and the artifact must SAY it was evidence, not a guess
        # (#233). Retiring the right host for an unrecorded reason
        # still leaves a postmortem unable to trust the file.
        assert [(r.provenance, r.attempt) for r in h.bad_records] == [
            (PROVENANCE_SCRAPED, 1)
        ]
        assert BAD_HOST not in h.active_hosts, (
            "the bad host is STILL in the hostfile the next attempt reads "
            "-- the swap did not take effect where it matters"
        )
        assert h.active_hosts == ["healthy-0", "spare-1"], (
            "the healthy host should be untouched and the spare should "
            f"take the bad host's slot; got {h.active_hosts}"
        )

    def test_the_next_attempt_is_handed_the_swapped_hostfile(
        self, tmp_path, monkeypatch
    ):
        """The swap must reach the PROCESS, not just the disk.

        `AutoRetryConfig.cmd` documents the contract: the command
        carries `--hostfile=<path>`, the loop never re-assembles it,
        and `NodeAllocation` rewrites that file in place so the
        re-spawned launcher reads the fresh contents. Asserting on the
        file after the run only shows it reached the disk -- this
        checks what each child was actually handed.

        Raised in review, and correctly: the first version of this file
        did not pass `--hostfile` at all, so it never exercised the
        contract as written.
        """
        h = Harness(tmp_path, FI_MODE="shepherd", FI_FAIL_ON="1")
        rc = h.run(
            monkeypatch,
            nodes=("healthy-0", BAD_HOST, "spare-1"),
            active=2,
        )

        assert rc == 0
        seen = h.hosts_seen
        assert len(seen) == 2, f"expected two attempts, saw {seen}"
        assert seen[0] == ["healthy-0", BAD_HOST], (
            f"attempt 1 should have been given the original set; got {seen[0]}"
        )
        assert seen[1] == ["healthy-0", "spare-1"], (
            "attempt 2's process was handed a stale hostfile: expected the "
            f"bad host replaced by the spare, got {seen[1]}"
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
        # The whole point of #233: this entry is a GUESS. `healthy-0`
        # was never implicated by anything in the log, and the file
        # has to say so or an operator will pull a healthy node.
        assert [(r.provenance, r.attempt) for r in h.bad_records] == [
            (PROVENANCE_BLIND, 1)
        ]
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

    def test_resolver_error_is_swallowed(self, tmp_path):
        """A broken `getent` must not break resolution's caller.

        Previously this caught `FileNotFoundError` only, so a
        `PermissionError` -- a `getent` that exists but is not
        executable, which is what macOS has -- escaped. This test
        asserted `pytest.raises(PermissionError)` when it was written;
        the assertion flipped when #223 was fixed.
        """
        from ezpz.failover.patterns import reverse_resolve_ip

        import subprocess as _sp

        def _boom(*_a, **_k):
            raise PermissionError(13, "Permission denied", "getent")

        orig = _sp.check_output
        _sp.check_output = _boom
        try:
            assert reverse_resolve_ip("10.0.0.42") is None
        finally:
            _sp.check_output = orig

    def test_a_failing_scraper_does_not_abort_the_run(
        self, tmp_path, monkeypatch
    ):
        """The real consequence of #223, at the loop level.

        Attribution is best-effort: it upgrades a blind rotation to a
        named one. If the scraper explodes the job should still
        recover, blindly. Before the fix this raised out of
        `run_with_auto_retry` and killed a recoverable run.
        """
        monkeypatch.setattr(
            "ezpz.failover.scrape_bad_nodes",
            lambda *_a, **_k: (_ for _ in ()).throw(
                PermissionError(13, "Permission denied", "getent")
            ),
        )
        h = Harness(tmp_path, FI_MODE="shepherd", FI_FAIL_ON="1")
        rc = h.run(monkeypatch)

        assert rc == 0, "a scraper failure aborted an otherwise fine run"
        assert h.attempts == 2
        assert len(h.bad) == 1, "it should still have rotated blindly"


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
        rc = h.run(monkeypatch, idle_timeout_s=4)

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
            idle_timeout_s=4,
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
        # Counting hosts must not be confused by the provenance
        # columns: a whole-file `.read_text().split()` reports 6
        # tokens here, not 2 (#233).
        assert len(h.bad_nodes.read_text().split()) > len(h.bad)
        assert [r.attempt for r in h.bad_records] == [1, 2]

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

    def test_real_iter_output_counts_as_progress(
        self, tmp_path, monkeypatch
    ):
        r"""`iter=N` is what every ezpz example prints, and it counts.

        This test was written to FAIL, back when the marker was
        `\bstep=\d+` and `minimal.py`/`test.py` both emitted `iter=`:
        a job visibly mid-training was filed as "never started" and
        abandoned with spares free. Fixing #224 flipped it, which is
        the strongest evidence available that the fix does what it
        claims -- the assertion changed direction, not the harness.
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
        assert "step=" not in log, "this must exercise the iter= path only"
        assert h.rc != 0
        assert h.attempts == 3, (
            "a job that is visibly training should keep failing over "
            f"until spares run out; got {h.attempts} attempts"
        )
        assert len(h.bad) == 2, "both spares should have been consumed"

    def test_no_output_at_all_aborts_after_two_attempts(
        self, tmp_path, monkeypatch
    ):
        """The genuinely-silent case: no progress lines, no named host.

        Distinct from the test above -- this is a job that really did
        die before training, which the guard is *right* to abandon.

        `silent_fail` rather than `shepherd`: since #232 the guard
        requires the scraper to have named NOBODY, and `shepherd` names
        a host. That is the point of the next test.
        """
        h = Harness(tmp_path, FI_MODE="silent_fail", FI_STEPS="0")
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


class TestBlindRotationMissesOffsetVictim:
    """What blind rotation does when the dead node is not ``active[0]``.

    Not a bug report in test form -- a pin on current behaviour, so the
    day attribution lands (issue #234) this test fails loudly and gets
    updated deliberately rather than silently continuing to pass.

    Motivation is a real run. Sunspot job 12473704 killed 13 ranks with
    ``kill -9`` and the attempt log simply STOPPED mid-training at
    ``iter=42``: no signal line, no shepherd message, no traceback. The
    scraper returns empty against it, so the loop fell to
    ``swap_one_blind``. The test killed allocation node 0, which is also
    ``active[0]`` -- exactly what blind rotation evicts -- so it looked
    like successful identification when nothing had been identified.
    """

    def test_blind_rotation_evicts_healthy_host_and_keeps_the_dead_one(
        self, tmp_path
    ):
        """Victim at ``active[1]``: the healthy host goes, the dead stays.

        This is worse than not failing over. Each attempt costs a spare
        AND a full relaunch while the actual fault stays in the active
        set, so the job burns every spare without ever touching it.
        """
        hosts = [
            f"x1921c1s{i}b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov"
            for i in range(4)
        ]
        alloc = NodeAllocation.from_full_nodelist(
            hosts, 2, tmp_path / "active.hostfile", tmp_path / "bad.txt"
        )

        victim = alloc.active[1]
        healthy = alloc.active[0]
        evicted, _spare = alloc.swap_one_blind()

        assert evicted == healthy, (
            "blind rotation is documented to evict active[0]; if this "
            "changed, the reasoning in issue #234 needs revisiting"
        )
        assert evicted != victim
        assert victim in alloc.active, (
            "the dead node is still in the active set -- the next attempt "
            "relaunches onto a host with no ranks"
        )
        # The eviction is recorded with no hint that it was a guess,
        # which is the provenance gap in issue #233.
        assert healthy in (tmp_path / "bad.txt").read_text()

    def test_blind_rotation_finds_the_victim_only_at_active_zero(
        self, tmp_path
    ):
        """The lucky case -- why job 12473704 passed.

        Same allocation, victim at index 0 instead of 1. Nothing about
        the loop differs; only where the victim happens to sit.
        """
        hosts = [
            f"x1921c1s{i}b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov"
            for i in range(4)
        ]
        alloc = NodeAllocation.from_full_nodelist(
            hosts, 2, tmp_path / "active.hostfile", tmp_path / "bad.txt"
        )

        victim = alloc.active[0]
        evicted, _spare = alloc.swap_one_blind()

        assert evicted == victim
        assert victim not in alloc.active

    def test_a_named_host_outranks_inferred_no_progress(
        self, tmp_path, monkeypatch
    ):
        r"""#232: positive evidence beats an inference from silence.

        A node that dies during a long init produces a marker-free log,
        and so does a trainer whose counter this regex does not know --
        the regex was `\bstep=\d+` while every ezpz example emits
        `iter=`, and that shipped. So "no markers" cannot carry a
        terminal verdict on its own.

        Here the scraper NAMES a host every attempt: real evidence of a
        node fault, pointing the opposite way. Before the fix the loop
        stopped at attempt 2 with a spare still free, calling a genuine
        bad-node failover a misconfigured job. It should now keep
        failing over until the spares run out.
        """
        h = Harness(tmp_path, FI_MODE="shepherd", FI_STEPS="0")
        rc = h.run(
            monkeypatch,
            nodes=(BAD_HOST, "s1", "s2"),
            active=1,
            max_failover_retries=None,
        )

        assert rc != 0
        assert "step=" not in h.log_text(), (
            "this must exercise the zero-progress path"
        )
        assert h.attempts == 3, (
            "a NAMED bad host with no progress markers should keep "
            "failing over, not be filed as stuck_pre_training; got "
            f"{h.attempts} attempts"
        )
        assert len(h.bad) == 2, (
            "both spares should have been used before the loop gave up; "
            f"got {h.bad}"
        )

    def test_stuck_verdict_says_it_is_an_inference(
        self, tmp_path, monkeypatch, caplog
    ):
        """The log line must not read as a finding (#232).

        An operator whose trainer prints an unrecognised counter can
        only spot the misfire if the verdict admits it is a guess about
        missing markers rather than an observation.
        """
        # `logger=` is load-bearing: conftest pins every ezpz logger to
        # CRITICAL, so a bare `at_level(ERROR)` only lowers the ROOT
        # level and the record is dropped at the ezpz logger before it
        # ever propagates.
        h = Harness(tmp_path, FI_MODE="silent_fail", FI_STEPS="0")
        with caplog.at_level(logging.ERROR, logger=_autoretry_logger.name):
            h.run(
                monkeypatch,
                nodes=(BAD_HOST, "s1", "s2"),
                active=1,
                max_failover_retries=None,
            )

        stop = [
            r.getMessage()
            for r in caplog.records
            if "FAILOVER STOP: stuck_pre_training" in r.getMessage()
        ]
        assert stop, "no stuck_pre_training stop line was logged"
        line = stop[0].lower()
        assert "inferred" in line, (
            "the verdict does not say it is an inference from missing "
            f"markers: {stop[0]!r}"
        )


class TestUnattributedIOFailure:
    """#231: a storage error is not evidence about any host.

    Sunspot job 12473704 died of `OSError: [Errno 28] No space left on
    device` inside a DCP checkpoint write. The PALS teardown cascade
    named a bystander host, the scraper picked it up, and a healthy node
    was retired while a spare was consumed.

    The signatures the child emits are transcribed from that log; see
    `_faultinject._die` for exactly which parts are captured and which
    (the co-occurring shepherd line in `enospc_named`) are reconstructed
    and why.
    """

    def test_enospc_retries_without_burning_a_spare(
        self, tmp_path, monkeypatch
    ):
        """The headline: retry in place, blame nobody.

        Retrying is right -- on the real incident `/lus/tegu` was 10%
        full with 2 of 4 OSTs at 99-100% and `stripe_count: 1`, so a
        reissued write has a real chance of landing on a healthy OST.
        What is wrong is doing it at the cost of a node.
        """
        h = Harness(tmp_path, FI_MODE="enospc", FI_FAIL_ON="1")
        rc = h.run(
            monkeypatch,
            nodes=("healthy-0", "spare-1", "spare-2"),
            active=1,
        )

        assert rc == 0, "the ENOSPC attempt was not retried at all"
        assert h.attempts == 2, f"expected a retry; got {h.attempts}"
        assert h.bad == [], (
            "a storage failure retired a node -- an I/O error inside a "
            f"checkpoint write is not evidence about hardware; got {h.bad}"
        )
        assert h.active_hosts == ["healthy-0"], (
            "the retry should run on the SAME hosts; got "
            f"{h.active_hosts}"
        )
        assert h.hosts_seen == [["healthy-0"], ["healthy-0"]], (
            "the second attempt was handed a different host set, so a "
            f"swap happened after all: {h.hosts_seen}"
        )

    def test_enospc_does_not_retire_the_host_the_cascade_names(
        self, tmp_path, monkeypatch
    ):
        """The exact bug from #231, with the scraper able to name a host.

        `enospc` alone leaves the scraper empty, so it only proves a
        *blind* rotation is avoided. This adds the shepherd line the
        real log must have carried for the incident to have been
        classified `BAD_NODE_KNOWN`, and asserts the named host still
        does not land in `bad_nodes.txt`.
        """
        h = Harness(tmp_path, FI_MODE="enospc_named", FI_FAIL_ON="1")
        rc = h.run(
            monkeypatch,
            nodes=(BAD_HOST, "spare-1", "spare-2"),
            active=1,
        )

        assert rc == 0
        assert h.attempts == 2
        assert h.bad == [], (
            "the host named by the teardown cascade was retired even "
            "though the log explains the death as an I/O failure; got "
            f"{h.bad}"
        )
        assert h.active_hosts == [BAD_HOST], (
            f"the named host was swapped out; got {h.active_hosts}"
        )

    def test_repeated_enospc_falls_back_to_node_swapping(
        self, tmp_path, monkeypatch
    ):
        """The bound, which is the whole safety argument.

        Believing the storage story forever would turn a real node
        fault that happens to emit an I/O line into an unbounded retry
        loop on a broken host. After the consecutive budget the loop
        reverts to normal bad-node handling and starts consuming
        spares again.

        With a child that fails every attempt: 3 free retries, then
        swaps until the 2 spares are gone.
        """
        from ezpz.launch_autoretry import _MAX_CONSECUTIVE_UNATTRIBUTED

        assert _MAX_CONSECUTIVE_UNATTRIBUTED == 3, (
            "this test's arithmetic is written against a budget of 3"
        )
        h = Harness(tmp_path, FI_MODE="enospc")  # fails every attempt
        rc = h.run(
            monkeypatch,
            nodes=("healthy-0", "spare-1", "spare-2"),
            active=1,
            # The budget + the spare pool are what should stop this. The
            # cap is a BACKSTOP so a regression that removes the bound
            # fails this test in a second instead of hanging the suite
            # forever -- a hang reads as infrastructure trouble, not as
            # the bug it actually is. Set well above the 6 expected.
            max_failover_retries=9,
        )

        assert rc != 0
        assert h.bad == ["healthy-0", "spare-1"], (
            "after the budget the loop should resume rotating hosts; "
            f"got {h.bad}"
        )
        # 3 no-blame retries (attempts 1-3), then attempt 4 swaps in
        # spare-1, attempt 5 swaps in spare-2, attempt 6 finds no
        # spares -> EXHAUSTED.
        assert h.attempts == 6, (
            "expected 3 unattributed retries then one attempt per spare; "
            f"got {h.attempts}"
        )

    def test_the_budget_resets_after_an_unrelated_failure(
        self, tmp_path, monkeypatch
    ):
        """The budget bounds a RUN of storage failures, not the job.

        A job that hits ENOSPC, recovers, trains for an hour and hits it
        again should get the full budget the second time -- otherwise a
        long job silently loses its protection partway through.

        The sequence is 2 ENOSPC, then a real shepherd kill, then 3 more
        ENOSPC. Under a resetting counter that is 2 + swap + 3 = six
        attempts and exactly one retired node. Under a counter that
        merely accumulated, the fifth ENOSPC would already be past a
        budget of 3 and would retire a second node -- so `bad` is what
        separates the two behaviours, not the attempt count alone.
        """
        h = Harness(
            tmp_path,
            FI_MODE="enospc,enospc,shepherd,enospc,enospc,enospc",
            FI_FAIL_ON="1,2,3,4,5,6",
        )
        rc = h.run(
            monkeypatch,
            nodes=(BAD_HOST, "spare-1", "spare-2"),
            active=1,
            max_failover_retries=None,
        )

        assert rc == 0, f"the run did not recover; rc={rc}"
        assert h.attempts == 7, (
            "expected 2 storage retries, one swap, 3 more storage "
            f"retries, then a clean attempt; got {h.attempts}"
        )
        assert h.bad == [BAD_HOST], (
            "exactly the shepherd-killed host should have been retired. "
            "More than one means the unattributed budget did not reset "
            f"after the intervening node failure; got {h.bad}"
        )

    def test_walltime_still_wins_over_an_enospc_line(
        self, tmp_path, monkeypatch
    ):
        """Ordering: a clean wallclock kill mid-write is still walltime.

        Swapping nodes cannot buy more wallclock, and neither can
        retrying in place. The ENOSPC check sits AFTER the walltime
        guard for exactly this reason.
        """
        h = Harness(tmp_path, FI_MODE="clean_walltime")
        rc = h.run(monkeypatch)

        assert rc == WALLTIME_RC
        assert h.attempts == 1, "a clean walltime kill must not be retried"
        assert h.bad == []
