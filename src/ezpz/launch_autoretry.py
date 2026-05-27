"""Auto-retry loop for ``ezpz launch --auto-retry``.

This is the Python counterpart to ``src/ezpz/bin/failover.sh``. Both
share the same scraper (``ezpz.failover.scrape_bad_nodes``) and the
same broad strategy (split hosts → active + spare, retry on bad-node
failures, swap a spare in for each bad host) but the in-launch path
is unbounded by default and terminates via the classifier below.

Why not just call the bash lib from Python? The classifier is the
hard part — nine outcome categories, regex-driven, sensitive to
ANSI-coloring and walltime/bad-node disambiguation. Pure-Python is
straightforward to unit-test; subprocessing into bash and parsing
its stderr would not be.

Public surface:

  * :class:`AutoRetryConfig` — caller-supplied policy
  * :class:`NodeAllocation` — active/spare tracker
  * :class:`TerminationReason` — outcomes of :func:`classify_attempt`
  * :func:`classify_attempt` — pure decision function over (rc, log)
  * :func:`run_with_auto_retry` — the loop
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import ezpz

logger = ezpz.get_logger(__name__)


# Default idle-output watchdog when --auto-retry is set. Matches
# FAILOVER_IDLE_TIMEOUT in failover.sh. 30 minutes is long enough
# for legitimate gaps (eval epochs, checkpoint saves) but short
# enough that a 5h xccl hang doesn't burn the full walltime.
DEFAULT_AUTO_RETRY_IDLE_TIMEOUT_S = 1800

# Backoff between attempts. Same shape as _run_with_retries — short
# enough to recover fast on transient failures, capped so we don't
# wait forever between attempts under a long-running --auto-retry.
_BACKOFF_BASE_S = 5.0
_BACKOFF_CAP_S = 60.0

# Crash patterns: lines whose presence in the log overrides a shell
# exit 0 (the outer wrapper sometimes exits clean even when the
# inner mpiexec child crashed) AND defeat the "exit 143 = walltime"
# heuristic (a real bad-node failure can surface as 143 when mpiexec
# teardown races the wallclock kill).
#
# Kept in sync with the same set in src/ezpz/bin/failover.sh — if
# you add a pattern there, add it here too. The bash version's
# inline comments cite the failure incidents that motivated each
# entry; see those for the postmortem context.
_CRASH_PATTERNS_RX = re.compile(
    r"RuntimeError: \[.*gloo.*\] Connection closed by peer"
    r"|RuntimeError: \[.*gloo.*\] Timed out waiting"
    r"|OutOfMemoryError"
    r"|UR_RESULT_ERROR_OUT_OF_RESOURCES"
    r"|died from signal"
    r"|EOFError: No data left in file"
)

# Innocent rank-cascade lines. These are emitted by mpiexec when a
# PRIMARY kill on one node propagates SIGTERM/SIGSEGV outward to
# every other rank — the named rank wasn't the culprit, it's just
# a downstream victim. Matching these as bad-node indicators would
# tag innocent ranks and (worse) override a clean walltime exit
# into a node-swap retry that burns spares for nothing. Strip
# these lines BEFORE the crash match runs.
#
# Mirrors the `grep -v "rank N died from signal (11|15)"` strip in
# src/ezpz/bin/failover.sh and the scraper-side exclusion
# documented in tests/test_failover_scrape.py
# ::test_innocent_rank_signal_11_not_matched (job 8466848 postmortem).
_INNOCENT_RANK_CASCADE_RX = re.compile(
    r"rank \d+ died from signal (?:11|15)"
)

# Strip ANSI color codes before parsing the "Execution finished with N"
# trailer. The trailer is logged with color (the rc digits are wrapped
# in \x1b[1;36m...\x1b[0m); a naive regex on the colored form pulls
# "1" out of the [1;36m prefix instead of the actual rc.
_ANSI_RX = re.compile(r"\x1b\[[0-9;]*m")

# Walltime exit code. PBS exit -29 surfaces as bash 143 (128 + SIGTERM).
_WALLTIME_RC = 143

# Idle-output watchdog exit code (matches GNU timeout(1), set by
# launch.py:_WATCHDOG_EXIT_CODE).
_WATCHDOG_RC = 124

# Progress marker: History.update prints `step=N` (or step=<N>) on
# its summary line. If two consecutive attempts both contain zero
# `step=` markers, the run is broken before training even started
# (bad config, missing dataset, etc.) and no amount of node swapping
# will help — bail out.
_PROGRESS_MARKER_RX = re.compile(r"\bstep=\d+", re.MULTILINE)


class TerminationReason(Enum):
    """Outcomes that drive :func:`run_with_auto_retry`'s next step.

    Each value is the verb after ``FAILOVER STOP:`` in the postmortem
    log line, so grep'ing the log for ``FAILOVER STOP`` always lands
    on the final classifier verdict.

    SUCCESS / WALLTIME / STUCK_PRE_TRAINING / EXHAUSTED are terminal:
    :func:`run_with_auto_retry` returns immediately. The two BAD_NODE
    values trigger a swap-and-retry. INTERRUPTED is produced by the
    loop's SIGINT handler — :func:`classify_attempt` never returns it,
    since the classifier never sees the interrupt path (we re-raise
    KeyboardInterrupt before reaching the classifier).
    """

    SUCCESS = "success"
    WALLTIME = "walltime"
    BAD_NODE_KNOWN = "bad_node_known"
    BAD_NODE_BLIND = "bad_node_blind"
    STUCK_PRE_TRAINING = "stuck_pre_training"
    EXHAUSTED = "exhausted"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ClassificationResult:
    """What the classifier decided + the progress flag for next-iter use.

    Returning both in one struct lets the loop avoid a second read
    of the same log to recompute ``has_progress`` (the previous
    implementation re-read the log right after the classifier had
    already parsed it once).
    """

    reason: TerminationReason
    has_progress: bool


@dataclass
class AutoRetryConfig:
    """Caller-supplied policy for :func:`run_with_auto_retry`.

    ``cmd`` is the full launcher command line *as already assembled*
    (mpiexec + topology + user command), with the ``--hostfile``
    argument already pointing at :attr:`NodeAllocation.hostfile_path`.
    The auto-retry loop does NOT re-assemble ``cmd`` between attempts
    — :class:`NodeAllocation` mutates the file at that path in place
    as nodes are swapped, and the launcher (re-spawned per attempt)
    reads the fresh contents on each re-launch.
    """

    cmd: list[str]
    """Full launcher command line (mpiexec + user command).

    Must already contain ``--hostfile=<path>`` where ``<path>``
    matches :attr:`NodeAllocation.hostfile_path`. The active hostfile
    is what mutates between attempts; this command is constant."""

    log_dir: Path
    """Directory for ``attempt-N.log`` files and ``bad_nodes.txt``."""

    idle_timeout_s: int = DEFAULT_AUTO_RETRY_IDLE_TIMEOUT_S
    """Per-attempt idle-output watchdog. 0 disables."""

    max_failover_retries: Optional[int] = None
    """Upper bound on retries. ``None`` = unbounded; loop terminates
    only via the matrix in :func:`classify_attempt`."""

    machine: Optional[str] = None
    """Scrape pattern set. ``None`` = auto-detect via
    ``ezpz.get_machine()``."""


@dataclass
class NodeAllocation:
    """In-memory active + spare hostfile tracker.

    Mirrors the on-disk state managed by ``failover_init`` /
    ``failover_swap_in`` / ``failover_swap_one_blind`` in the bash
    lib. After every mutation we re-write the active hostfile on
    disk so the launcher (which we don't re-spawn for the hostfile
    arg) picks up the new contents on the next attempt.

    ``spare`` is a deque so we can ``popleft()`` cheaply when
    rotating new hosts in. ``bad_nodes_path`` is appended-to once
    per swap for postmortem.
    """

    active: list[str]
    spare: deque[str]
    hostfile_path: Path
    bad_nodes_path: Path

    @classmethod
    def from_full_nodelist(
        cls,
        nodelist: Sequence[str],
        nproc_active_hosts: int,
        hostfile_path: Path,
        bad_nodes_path: Path,
    ) -> NodeAllocation:
        """Split a full nodelist into active + spare and persist the
        active subset to disk at ``hostfile_path``.

        ``nproc_active_hosts`` is the number of *hosts* needed for
        training (not ranks). Caller is responsible for the
        nproc/ppn → nhosts arithmetic — see :func:`derive_spare_count`.
        """
        if nproc_active_hosts > len(nodelist):
            raise ValueError(
                f"need {nproc_active_hosts} active hosts but only "
                f"{len(nodelist)} were given"
            )
        alloc = cls(
            active=list(nodelist[:nproc_active_hosts]),
            spare=deque(nodelist[nproc_active_hosts:]),
            hostfile_path=hostfile_path,
            bad_nodes_path=bad_nodes_path,
        )
        alloc._write_active()
        bad_nodes_path.write_text("")
        return alloc

    def _write_active(self) -> None:
        """Persist the active set to ``hostfile_path``.

        Trailing newline matches the convention of PBS_NODEFILE and
        the per-line tools (``wc -l``, ``head``) the rest of the
        ecosystem expects.
        """
        self.hostfile_path.write_text(
            "\n".join(self.active) + ("\n" if self.active else "")
        )

    def _append_bad(self, host: str) -> None:
        """Append one bad host to ``bad_nodes_path`` for postmortem.

        Open + append + close per call (cheap; swaps are rare and
        a leaked fd outliving the process would be worse than the
        extra syscall)."""
        with self.bad_nodes_path.open("a") as f:
            f.write(host + "\n")

    def swap_in(self, bad_hosts: Sequence[str]) -> list[tuple[str, str]]:
        """Swap each known-bad host out for a spare.

        Skips hosts not currently in the active set (already
        replaced, or scraped from an older log). Returns the list
        of ``(bad, spare)`` pairs actually swapped — caller can
        log a summary.

        Raises :class:`RuntimeError` if a swap is wanted but no
        spare is available. The caller catches that and ends the
        loop via :attr:`TerminationReason.EXHAUSTED`.
        """
        swaps: list[tuple[str, str]] = []
        for bad in bad_hosts:
            if bad not in self.active:
                logger.debug("skip swap: %s not in active set", bad)
                continue
            if not self.spare:
                raise RuntimeError(
                    f"out of spare nodes — cannot replace {bad}"
                )
            spare = self.spare.popleft()
            idx = self.active.index(bad)
            self.active[idx] = spare
            self._append_bad(bad)
            swaps.append((bad, spare))
        if swaps:
            self._write_active()
        return swaps

    def swap_one_blind(self) -> tuple[str, str]:
        """Rotate the first active host out for a spare.

        Used when the scraper can't pinpoint a culprit (silent hang,
        unknown crash pattern). Picks the first active host because
        that's also what the bash lib does — the choice is arbitrary
        but stable, so consecutive blind rotations cycle through the
        active set rather than thrashing on one host.
        """
        if not self.spare:
            raise RuntimeError("out of spare nodes — cannot blind-rotate")
        if not self.active:
            raise RuntimeError("active set is empty — nothing to rotate")
        bad = self.active[0]
        spare = self.spare.popleft()
        self.active[0] = spare
        self._append_bad(bad)
        self._write_active()
        return bad, spare

    @property
    def has_spares(self) -> bool:
        return len(self.spare) > 0


def _strip_ansi(text: str) -> str:
    return _ANSI_RX.sub("", text)


def _has_crash_patterns(log_text: str) -> bool:
    """Return True iff the log contains a real hardware-style death.

    Strips innocent rank-cascade lines BEFORE matching so a clean
    walltime kill (which fires SIGTERM at every rank, generating
    dozens of `rank N died from signal 15` lines) doesn't get
    misclassified as a bad-node failure. See _INNOCENT_RANK_CASCADE_RX
    for the postmortem context.

    The strip preserves real signals: `shepherd died from signal 9`
    (PALS shepherd kill), `died from signal 6` (SIGABRT from a real
    assert), `died from signal 9` on a process other than `rank N`,
    etc. all still match.
    """
    if not log_text:
        return False
    filtered = "\n".join(
        line
        for line in log_text.splitlines()
        if not _INNOCENT_RANK_CASCADE_RX.search(line)
    )
    return _CRASH_PATTERNS_RX.search(filtered) is not None


def _has_progress_markers(log_text: str) -> bool:
    return _PROGRESS_MARKER_RX.search(log_text) is not None


def _extract_inner_rc(log_text: str) -> Optional[int]:
    """Pull the last ``Execution finished with N`` rc from the log.

    Strips ANSI first — ezpz launch colors the trailer and naive
    regex extracts the wrong digit from the color prefix. Returns
    ``None`` if no trailer is present (e.g. the wrapper crashed
    before emitting it).
    """
    stripped = _strip_ansi(log_text)
    # rfind avoids scanning the whole log via the regex when only
    # the last hit matters.
    marker = "Execution finished with "
    idx = stripped.rfind(marker)
    if idx < 0:
        return None
    tail = stripped[idx + len(marker) :].split(None, 1)[0]
    # Strip trailing punctuation: the trailer is logged with a
    # period (`Execution finished with 0.`) which would otherwise
    # poison int().
    tail = tail.rstrip(".,;:")
    try:
        return int(tail)
    except ValueError:
        return None


def classify_attempt(
    shell_rc: int,
    log_path: Path,
    scraped_bad_nodes: Sequence[str],
    *,
    prior_attempt_had_progress: Optional[bool] = None,
    has_spares: bool = True,
) -> ClassificationResult:
    """Decide what the auto-retry loop should do after an attempt.

    Pure function — no side effects, single log read. Returns both
    the termination reason and the ``has_progress`` flag so the loop
    can thread the latter through to the next call without re-reading
    the log. The full termination matrix from PR #3's handoff doc:

    | rc       | log signals                                 | result               |
    |----------|---------------------------------------------|----------------------|
    | 0        | inner_rc=0 OR absent OR no crash pattern    | SUCCESS              |
    | 0        | inner_rc != 0 (wrapper lied about success)  | classify by inner_rc |
    | 0        | crash patterns present                      | bad-node retry       |
    | 143      | no crash patterns                           | WALLTIME             |
    | 143      | crash patterns present                      | bad-node retry       |
    | 124      | (idle-output watchdog tripped)              | BAD_NODE_BLIND       |
    | non-zero | scraper found named host(s)                 | BAD_NODE_KNOWN       |
    | non-zero | scraper empty                               | BAD_NODE_BLIND       |
    | -        | this AND prior attempt both had 0 progress  | STUCK_PRE_TRAINING   |
    | -        | bad-node verdict but no spares left         | EXHAUSTED            |

    The "two consecutive attempts with no progress" guard is the
    user's preferred substitute for a numeric cap on blind rotations.
    It catches code bugs (broken config, missing dataset) that would
    otherwise burn the entire spare pool. The current attempt's
    progress status is checked against ``prior_attempt_had_progress``;
    the caller is responsible for tracking the prior value across
    iterations (use the ``has_progress`` field of the returned
    :class:`ClassificationResult`).

    INTERRUPTED is produced by the loop itself in the SIGINT handler,
    not here — the classifier never sees the interrupt path because
    KeyboardInterrupt is re-raised before we reach it.
    """
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""

    inner_rc = _extract_inner_rc(log_text)
    crash = _has_crash_patterns(log_text)
    has_progress = _has_progress_markers(log_text)
    # Effective rc: trust the inner trailer over a clean shell exit
    # when the wrapper lied (mpiexec teardown raced a SIGTERM, etc.)
    effective_rc = shell_rc
    if shell_rc == 0 and inner_rc is not None and inner_rc != 0:
        effective_rc = inner_rc
    elif shell_rc == 0 and crash:
        # Wrapper said 0 but mass tracebacks landed in the log. Treat
        # as a generic crash — let the scraper-empty path decide
        # between named vs blind.
        effective_rc = 1

    def _result(reason: TerminationReason) -> ClassificationResult:
        return ClassificationResult(reason=reason, has_progress=has_progress)

    # Success: shell exit 0, no contrary inner_rc, no crash patterns.
    if effective_rc == 0:
        return _result(TerminationReason.SUCCESS)

    # Walltime guard. Real walltime: no point swapping nodes.
    # Walltime races: a true bad-node failure can land here when
    # mpiexec teardown races the wallclock kill. Use the crash
    # patterns to disambiguate.
    if effective_rc == _WALLTIME_RC and not crash:
        return _result(TerminationReason.WALLTIME)

    # The progress guard applies BEFORE we decide which swap path to
    # take — there's no point swapping nodes if the run is dying
    # before training starts. Note: we only check this on actual
    # failure paths; success already returned above.
    if prior_attempt_had_progress is False and not has_progress:
        return _result(TerminationReason.STUCK_PRE_TRAINING)

    # Watchdog kill: launch.py couldn't see output for `idle_timeout_s`.
    # The hang IS the silence, so the scraper rarely finds anything
    # — blind-rotate a spare.
    if effective_rc == _WATCHDOG_RC:
        if not has_spares:
            return _result(TerminationReason.EXHAUSTED)
        return _result(TerminationReason.BAD_NODE_BLIND)

    # General failure path. Scraper-named hosts win over blind.
    if scraped_bad_nodes:
        if not has_spares:
            return _result(TerminationReason.EXHAUSTED)
        return _result(TerminationReason.BAD_NODE_KNOWN)

    if not has_spares:
        return _result(TerminationReason.EXHAUSTED)
    return _result(TerminationReason.BAD_NODE_BLIND)


def _run_attempt_with_tee(
    cmd: Sequence[str],
    log_path: Path,
    idle_timeout_s: int,
) -> int:
    """Run a single attempt, tee'ing combined stdout+stderr to
    ``log_path`` while still emitting to this process's stdout.

    Returns the child's exit code, or 124 if the idle-output watchdog
    fired. SIGINT propagates: KeyboardInterrupt re-raises to the
    caller after best-effort terminating the child.

    Same buffering nudge as launch._run_with_watchdog: forces
    PYTHONUNBUFFERED=1 so Python children flush per-line. Without
    this, the watchdog kills healthy jobs that simply hadn't
    block-flushed their stdout in time.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    # Safety note (Sourcery security warning): Popen with a dynamic
    # argv is fine here because:
    #   1. shell=False (the default) — no shell metacharacter
    #      expansion, every list element becomes a direct argv slot.
    #   2. cmd originates from `ezpz launch`'s argparse REMAINDER,
    #      which is the user's own shell already past their own
    #      shell expansion. Equivalent to typing the command into
    #      a terminal — no privilege boundary is crossed here.
    # Sourcery's lint can't distinguish "user-runs-their-own-code"
    # from "untrusted-input-to-shell"; this is the former.
    proc = subprocess.Popen(  # noqa: S603 — see comment above
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )

    last_activity = time.monotonic()
    activity_lock = threading.Lock()
    reader_done = threading.Event()

    def _drain() -> None:
        nonlocal last_activity
        assert proc.stdout is not None
        try:
            with log_path.open("w") as fh:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    fh.write(line)
                    fh.flush()
                    with activity_lock:
                        last_activity = time.monotonic()
        finally:
            reader_done.set()

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    # _drain_remaining: block until the reader thread fully consumes
    # the stdout pipe and flushes attempt-N.log to disk. Caller
    # (classify_attempt) reads the log immediately after we return —
    # if we exit before the reader's last fh.write() lands, the
    # classifier sees a truncated log and can misclassify the final
    # outcome (Copilot review on PR #144).
    #
    # The reader exits naturally once the child closes its stdout
    # (which the OS guarantees on process exit), so a generous join
    # without a hard cap is safe — the child has already terminated,
    # there's no scenario where the reader runs forever. We do bound
    # it to a sane upper limit so a stuck kernel pipe doesn't hang
    # the whole loop, and log a warning if we hit it.
    _DRAIN_TIMEOUT_S = 30.0

    def _drain_remaining() -> None:
        reader_done.wait(timeout=_DRAIN_TIMEOUT_S)
        if not reader_done.is_set():
            logger.warning(
                "[auto-retry] reader thread did not drain within %.0fs "
                "after child exit; log file may be truncated",
                _DRAIN_TIMEOUT_S,
            )

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                _drain_remaining()
                return rc
            if idle_timeout_s > 0:
                with activity_lock:
                    idle_for = time.monotonic() - last_activity
                if idle_for >= idle_timeout_s:
                    logger.error(
                        "[auto-retry] watchdog: no output for %.1fs "
                        "(timeout=%ds). Sending SIGTERM to PID %d.",
                        idle_for,
                        idle_timeout_s,
                        proc.pid,
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    _drain_remaining()
                    return _WATCHDOG_RC
                sleep_for = min(1.0, max(0.1, idle_timeout_s - idle_for))
            else:
                sleep_for = 1.0
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        # Propagate Ctrl-C: try to take the child down cleanly first.
        # The bare except is intentional — we want any sigint-like
        # interrupt to surface as INTERRUPTED, not be misclassified.
        logger.warning("[auto-retry] SIGINT received; terminating child")
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        # No _drain_remaining on the SIGINT path: the caller throws
        # the partial log away and exits with INTERRUPTED. Spending
        # 30s on a doomed drain is worse than losing the tail.
        raise


def _backoff_for_attempt(attempt: int) -> float:
    """Backoff before *attempt* (where attempt 1 is the first retry).

    5, 10, 20, 40, 60, 60, ... — same shape as ``_run_with_retries``
    in launch.py so users see consistent pacing whether they're on
    ``--retries N`` or ``--auto-retry``.
    """
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** (attempt - 1)))


def derive_spare_count(
    total_nodes: int,
    active_nodes: int,
) -> int:
    """``--spare-nodes auto`` default: ``total - active``, never < 0.

    Caller has already validated that ``active_nodes > 0``; this
    helper only handles the arithmetic. A negative result (active >
    total) is clamped to zero — the loop just won't have spares to
    rotate, which surfaces as ``TerminationReason.EXHAUSTED``.
    """
    return max(0, total_nodes - active_nodes)


def run_with_auto_retry(
    config: AutoRetryConfig,
    allocation: NodeAllocation,
    scrape_fn: Optional[Callable[[Path], list[str]]] = None,
) -> int:
    """Drive attempts until the termination matrix says stop.

    ``scrape_fn`` is injected for testability — defaults to
    :func:`ezpz.failover.scrape_bad_nodes` bound with the configured
    machine. Tests pass a stub that returns canned bad-host lists
    without needing a real log file.

    Returns the final shell exit code. Always logs a single
    ``FAILOVER STOP: <reason>`` line so grep is reliable.
    """
    if scrape_fn is None:
        from ezpz.failover import scrape_bad_nodes

        machine = config.machine

        def _default_scrape(p: Path) -> list[str]:
            try:
                return scrape_bad_nodes(p, machine=machine)
            except FileNotFoundError:
                return []

        scrape_fn = _default_scrape

    attempt = 0
    last_rc = 0
    prior_attempt_had_progress: Optional[bool] = None

    while True:
        attempt += 1
        log_path = config.log_dir / f"attempt-{attempt}.log"

        # Cap check fires BEFORE backoff sleep — otherwise
        # --max-failover-retries 0 would sleep 5s only to immediately
        # exit, and even a non-zero cap would burn a backoff for the
        # final exit decision. Cap counts retries, not attempts:
        # attempt 1 is the initial run, attempt N+1 is the Nth retry.
        if (
            config.max_failover_retries is not None
            and attempt > config.max_failover_retries + 1
        ):
            logger.error(
                "[auto-retry] FAILOVER STOP: max_failover_retries=%d "
                "exhausted (rc=%d)",
                config.max_failover_retries,
                last_rc,
            )
            return last_rc

        if attempt > 1:
            backoff = _backoff_for_attempt(attempt - 1)
            logger.warning(
                "[auto-retry] attempt %d (prior rc=%d, sleeping %.0fs)...",
                attempt,
                last_rc,
                backoff,
            )
            time.sleep(backoff)

        logger.info(
            "[auto-retry] attempt %d — active=%d hosts, spare=%d hosts",
            attempt,
            len(allocation.active),
            len(allocation.spare),
        )

        try:
            last_rc = _run_attempt_with_tee(
                config.cmd,
                log_path,
                config.idle_timeout_s,
            )
        except KeyboardInterrupt:
            logger.warning(
                "[auto-retry] FAILOVER STOP: interrupted (SIGINT)"
            )
            # 128 + SIGINT is the conventional return for ^C.
            return 128 + signal.SIGINT

        scraped = scrape_fn(log_path)
        result = classify_attempt(
            last_rc,
            log_path,
            scraped,
            prior_attempt_had_progress=prior_attempt_had_progress,
            has_spares=allocation.has_spares,
        )
        reason = result.reason
        # Thread the progress flag through to the next iteration —
        # classifier already parsed the log, no need to re-read.
        prior_attempt_had_progress = result.has_progress

        if reason is TerminationReason.SUCCESS:
            logger.info(
                "[auto-retry] FAILOVER STOP: success (attempt %d)", attempt
            )
            return last_rc

        if reason is TerminationReason.WALLTIME:
            logger.warning(
                "[auto-retry] FAILOVER STOP: walltime (rc=%d, attempt %d)",
                last_rc,
                attempt,
            )
            return last_rc

        if reason is TerminationReason.STUCK_PRE_TRAINING:
            logger.error(
                "[auto-retry] FAILOVER STOP: stuck_pre_training "
                "(two consecutive attempts with zero step= markers, "
                "rc=%d)",
                last_rc,
            )
            return last_rc

        if reason is TerminationReason.EXHAUSTED:
            logger.error(
                "[auto-retry] FAILOVER STOP: exhausted "
                "(no spare nodes left, rc=%d)",
                last_rc,
            )
            return last_rc

        # Bad-node paths fall through to a swap + continue.
        try:
            if reason is TerminationReason.BAD_NODE_KNOWN:
                swaps = allocation.swap_in(scraped)
                logger.warning(
                    "[auto-retry] bad nodes: %s — swapped %d",
                    scraped,
                    len(swaps),
                )
                # swap_in() may return an empty list if all named
                # hosts were already replaced; that shouldn't normally
                # happen, but if it does, fall back to blind rotation
                # so we still make progress.
                if not swaps:
                    if not allocation.has_spares:
                        logger.error(
                            "[auto-retry] FAILOVER STOP: exhausted "
                            "(named hosts already swapped, no spares)"
                        )
                        return last_rc
                    bad, spare = allocation.swap_one_blind()
                    logger.warning(
                        "[auto-retry] blind rotation: %s -> %s",
                        bad,
                        spare,
                    )
            else:  # BAD_NODE_BLIND
                bad, spare = allocation.swap_one_blind()
                logger.warning(
                    "[auto-retry] blind rotation: %s -> %s", bad, spare
                )
        except RuntimeError as exc:
            logger.error(
                "[auto-retry] FAILOVER STOP: exhausted (%s)", exc
            )
            return last_rc
