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
from dataclasses import dataclass, field
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
    # A rank returning a NONZERO application exit code. On Aurora/Sunspot
    # PALS, one rank's `exit 1` tears the job down with SIGTERM and the
    # aggregate surfaces as bash 143 (== _WALLTIME_RC) plus a
    # `<host>: rank N exited with code K` line (K != 0). Without this
    # pattern a genuine crash is indistinguishable from a clean walltime
    # kill (which only ever emits `rank N died from signal {11,15}`,
    # stripped as an innocent cascade below) and the loop would NOT retry
    # it. Confirmed on a live Sunspot run (job 12471663): scenarios B/D/E
    # all stopped after 1-2 attempts as "walltime" for a plain `exit 1`.
    # `[1-9][0-9]*` = any nonzero code (incl. 10+, e.g. 137 = SIGKILL/OOM),
    # while excluding the clean `exited with code 0` PALS prints per rank.
    r"|rank \d+ exited with code [1-9][0-9]*"
    r"|EOFError: No data left in file"
    # SLURM's equivalent of the PALS lines above. `srun` reports a
    # SIGKILLed rank as rc=143 -- the SAME code as a clean walltime
    # expiry -- and emits none of the PALS signatures, so on SLURM
    # EVERY killed node collapsed to WALLTIME and nothing failed over
    # (#238). Confirmed on Perlmutter job 57540936: the kill landed at
    # iter=10 with no TIME LIMIT in the log, and the loop still said
    # `FAILOVER STOP: walltime (rc=143, attempt 1)`.
    #
    #     srun: error: nid001321: tasks 4-7: Killed       <- the victim
    #     srun: error: nid001320: tasks 0-3: Terminated   <- the cascade
    #
    # `Killed` (SIGKILL) ONLY. NOT `Terminated` (SIGTERM), which is
    # what every rank gets when a step is torn down normally --
    # matching it would burn a spare on every expiring job, the same
    # trap the `died from signal {11,15}` exclusion avoids for PALS.
    r"|srun: error: \S+: tasks? [\d,-]+: Killed"
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

# Progress marker: evidence that training actually began. If two
# consecutive attempts show none, the run is broken before training
# starts (bad config, missing dataset, ...) and no amount of node
# swapping will help — bail out.
#
# This matches any of `History.update`'s counter names, not just
# `step`. That distinction is the whole bug: the counter bases are
# ("iter", "step", "epoch", "batch", "idx") -- see
# `ezpz.utils.format_compact_summary` -- and every ezpz example emits
# `iter=` (minimal.py:92, test.py:401), not `step=`. Matching `step=`
# alone meant a real ezpz job that hit a bad node twice was filed as
# "never started" and ABANDONED with spares still free, which is
# precisely the failure --auto-retry exists to survive.
#
# `torchtitan`-style `step: 1` (colon, spaces) is deliberately
# accepted too: the separator carries no information here, and being
# strict about it costs a real recovery.
#
# Erring toward accepting evidence of life is the right asymmetry. A
# false positive costs one extra swap attempt; a false negative
# abandons a recoverable job.
_PROGRESS_MARKER_RX = re.compile(
    r"\b(?:iter|step|epoch|batch|idx)\s*[=:]\s*\d+", re.MULTILINE
)

# Storage-layer errors: retryable, but the failure says nothing about
# any *host*.
#
# Sunspot job 12473704 (see docs/guides/fault-injection.md) died in a
# DCP checkpoint write with `OSError: [Errno 28] No space left on
# device`. On Aurora/Sunspot PALS one rank's death tears the whole job
# down and the teardown cascade prints a `<host>: rank N ...` line for
# some *other* host; the scraper picked that host up, the classifier
# said BAD_NODE_KNOWN, and a perfectly healthy node was retired into
# `bad_nodes.txt` while a spare was consumed for nothing.
#
# The fix is about ATTRIBUTION, not about retryability. That ENOSPC
# WAS retryable: `/lus/tegu` was 10% full, but 2 of its 4 OSTs were at
# 99-100% and the directory striped `stripe_count: 1`, so each shard
# file lands wholly on one round-robin-selected OST -- roughly half the
# writes failed and the rest succeeded. Reissuing the same write has a
# real chance of landing on a healthy OST. A terminal verdict matched
# on `Errno 28` would have converted a recoverable job into a dead one:
# the exact failure this guards against, with the sign flipped.
#
# So: retry, but do NOT blame a node, do NOT burn a spare, and do NOT
# write to `bad_nodes.txt`.
#
# Only the ENOSPC form below is attested by a captured log. The EDQUOT
# wording comes from the Linux strerror table (`errno 122`), not from a
# capture we hold -- it is included because a per-project quota is a
# storage fact with exactly the same non-attribution property (and can
# be freed by another job finishing), but if it ever proves to render
# differently in the wild, that is the line to fix.
#
# Deliberately NOT matched: `Permission denied`, `Read-only file
# system`, `No such file or directory`. Those really are usually
# terminal config errors, and a bounded no-blame retry is the wrong
# shape for them -- the normal path already surfaces them.
_UNATTRIBUTED_IO_RX = re.compile(
    r"No space left on device"
    r"|\[Errno 28\]"
    r"|Disk quota exceeded"
    r"|\[Errno 122\]"
)

# How many CONSECUTIVE attempts may end in RETRYABLE_UNATTRIBUTED
# before the loop stops giving the storage the benefit of the doubt
# and falls back to the normal bad-node path.
#
# The bound is the whole safety story here. Without it, a genuinely
# bad node that happens to emit a stale ENOSPC line would retry
# forever on the same broken host -- a false positive turning a real
# node fault into an unbounded loop. Three is enough to ride out the
# observed failure (round-robin OST selection, ~50/50 per write) and
# cheap enough that being wrong costs three attempts rather than a
# healthy node.
#
# Falling BACK to the normal path (rather than to a terminal verdict)
# is deliberate: after the budget the loop still makes forward
# progress by swapping, which is what it did before this existed.
_MAX_CONSECUTIVE_UNATTRIBUTED = 3


def _has_unattributed_io_failure(log_text: str) -> bool:
    """Return True iff the log shows a storage error that cannot be
    attributed to any host. See :data:`_UNATTRIBUTED_IO_RX`."""
    return _UNATTRIBUTED_IO_RX.search(log_text) is not None


class TerminationReason(Enum):
    """Outcomes that drive :func:`run_with_auto_retry`'s next step.

    Each value is the verb after ``FAILOVER STOP:`` in the postmortem
    log line, so grep'ing the log for ``FAILOVER STOP`` always lands
    on the final classifier verdict.

    SUCCESS / WALLTIME / STUCK_PRE_TRAINING / EXHAUSTED are terminal:
    :func:`run_with_auto_retry` returns immediately. The two BAD_NODE
    values trigger a swap-and-retry. RETRYABLE_UNATTRIBUTED retries
    WITHOUT a swap. INTERRUPTED is produced by the loop's SIGINT
    handler — :func:`classify_attempt` never returns it, since the
    classifier never sees the interrupt path (we re-raise
    KeyboardInterrupt before reaching the classifier).
    """

    SUCCESS = "success"
    WALLTIME = "walltime"
    BAD_NODE_KNOWN = "bad_node_known"
    BAD_NODE_BLIND = "bad_node_blind"
    RETRYABLE_UNATTRIBUTED = "retryable_unattributed"
    """A failure the log explains WITHOUT implicating a host.

    Today that means a storage error (ENOSPC/EDQUOT) raised inside the
    job -- see :data:`_UNATTRIBUTED_IO_RX`. Retry in place: no swap, no
    spare consumed, nothing appended to ``bad_nodes.txt``. Bounded by
    :data:`_MAX_CONSECUTIVE_UNATTRIBUTED` consecutive occurrences, after
    which the loop reverts to the normal bad-node path."""

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
    has_unattributed_io: bool = False
    """Whether the log showed an unattributed storage failure.

    Reported separately from ``reason`` because the loop's budget must
    count what the LOG said, not what the classifier decided. Once the
    budget is spent the verdict flips to a bad-node swap while the log
    still says ENOSPC -- if the counter tracked the verdict it would
    reset on that very swap, re-granting the budget forever and never
    actually bounding anything."""


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


# Provenance tags written to column 2 of ``bad_nodes.txt``. A
# ``scraped`` host was *named* by the log scraper from a failure
# signature; a ``blind`` host was evicted by :meth:`swap_one_blind`
# on a guess, with nothing in the log implicating it. Postmortems
# and any future "final attempt" logic need to tell those apart —
# blind-evicted hosts were never evidenced as faulty. See #233.
PROVENANCE_SCRAPED = "scraped"
PROVENANCE_BLIND = "blind"


@dataclass(frozen=True)
class BadNodeRecord:
    """One retired host plus *why* it was retired.

    The on-disk rendering (:meth:`to_line`) keeps the hostname in
    column 1 so bare-hostname consumers (``awk '{print $1}'``,
    ``cut -f1``) keep working against the richer file.
    """

    host: str
    provenance: str
    attempt: Optional[int] = None
    """1-based attempt that retired this host. ``None`` when the
    caller didn't supply one — the column is then omitted rather
    than filled with a guessed value."""

    def to_line(self) -> str:
        """Render as one whitespace-separated ``bad_nodes.txt`` line.

        Two spaces rather than one purely for legibility; every
        consumer splits on arbitrary whitespace.
        """
        parts = [self.host, self.provenance]
        if self.attempt is not None:
            parts.append(f"attempt={self.attempt}")
        return "  ".join(parts)


def parse_bad_nodes_file(path: Path) -> list[BadNodeRecord]:
    """Read a ``bad_nodes.txt`` back into records.

    Tolerates the legacy bare-hostname format (provenance is then
    reported as ``""``) so old artifacts stay readable.
    """
    records: list[BadNodeRecord] = []
    for raw in path.read_text().splitlines():
        fields = raw.split()
        if not fields:
            continue
        host = fields[0]
        provenance = fields[1] if len(fields) > 1 else ""
        attempt: Optional[int] = None
        for tok in fields[2:]:
            if tok.startswith("attempt="):
                try:
                    attempt = int(tok.split("=", 1)[1])
                except ValueError:
                    attempt = None
        records.append(BadNodeRecord(host, provenance, attempt))
    return records


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
    per swap for postmortem, each line carrying the hostname in
    column 1 and *why* it was retired in column 2 (see
    :class:`BadNodeRecord`).
    """

    active: list[str]
    spare: deque[str]
    hostfile_path: Path
    bad_nodes_path: Path
    bad_nodes: list[BadNodeRecord] = field(default_factory=list)
    """In-memory mirror of ``bad_nodes_path``, with provenance.

    Lets a caller ask "was anything actually *shown* to be bad?"
    without re-parsing the file — see :meth:`scraped_bad_hosts` and
    :meth:`blind_bad_hosts`."""

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

    def _append_bad(
        self,
        host: str,
        provenance: str,
        attempt: Optional[int] = None,
    ) -> None:
        """Record one retired host + why, on disk and in memory.

        Open + append + close per call (cheap; swaps are rare and
        a leaked fd outliving the process would be worse than the
        extra syscall).

        ``attempt`` is omitted from the line when the caller doesn't
        know it — an absent column beats a fabricated one."""
        record = BadNodeRecord(host, provenance, attempt)
        self.bad_nodes.append(record)
        with self.bad_nodes_path.open("a") as f:
            f.write(record.to_line() + "\n")

    def scraped_bad_hosts(self) -> list[str]:
        """Hosts a scraper *named* from a failure signature.

        The evidenced subset — the only entries a postmortem can
        call bad without qualification."""
        return [
            r.host
            for r in self.bad_nodes
            if r.provenance == PROVENANCE_SCRAPED
        ]

    def blind_bad_hosts(self) -> list[str]:
        """Hosts evicted on a guess, never implicated by a log.

        A future final attempt can reasonably reconstitute from
        these; nothing ever showed them to be faulty."""
        return [
            r.host
            for r in self.bad_nodes
            if r.provenance == PROVENANCE_BLIND
        ]

    @staticmethod
    def _node_key(host: str) -> str:
        """Canonical identity of a host, for comparison only.

        PBS hands out `x1921c7s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov`
        while the scraper's normalizer returns the plain
        `x1921c7s1b0n0.hsn.cm.sunspot.alcf.anl.gov`. Both name the same
        machine, and an exact `in` test says they do not.

        Sunspot job 12473750 is the cost of that: the scraper correctly
        identified the killed node, and the loop logged

            bad nodes: ['x1921c7s1b0n0.hsn...'] -- swapped 0

        then fell through to a blind rotation that retired the HEALTHY
        host and left the dead one running. Named attribution worked
        and was discarded on a string comparison.

        Reduce to the leading node token: strip any `-hsnN` interface
        suffix and the domain. That is the part that identifies the
        machine; everything after it describes how to reach it.
        """
        head = host.split(".", 1)[0]
        return head.split("-hsn", 1)[0]

    def _match_active(self, host: str) -> Optional[str]:
        """The active entry naming the same machine as *host*.

        Returns the string AS IT APPEARS in `self.active`, so callers
        keep writing hostfile-native names, not normalized ones.
        """
        if host in self.active:  # exact match: cheap and most common
            return host
        key = self._node_key(host)
        for a in self.active:
            if self._node_key(a) == key:
                return a
        return None

    def swap_in(
        self,
        bad_hosts: Sequence[str],
        attempt: Optional[int] = None,
    ) -> list[tuple[str, str]]:
        """Swap each known-bad host out for a spare.

        Skips hosts not currently in the active set (already
        replaced, or scraped from an older log). Returns the list
        of ``(bad, spare)`` pairs actually swapped — caller can
        log a summary.

        Every host retired here was *named* by the scraper, so it
        is recorded with :data:`PROVENANCE_SCRAPED`.

        Raises :class:`RuntimeError` if a swap is wanted but no
        spare is available. The caller catches that and ends the
        loop via :attr:`TerminationReason.EXHAUSTED`.
        """
        swaps: list[tuple[str, str]] = []
        for bad in bad_hosts:
            active_host = self._match_active(bad)
            if active_host is None:
                logger.debug("skip swap: %s not in active set", bad)
                continue
            bad = active_host
            if not self.spare:
                raise RuntimeError(
                    f"out of spare nodes — cannot replace {bad}"
                )
            spare = self.spare.popleft()
            idx = self.active.index(bad)
            self.active[idx] = spare
            self._append_bad(bad, PROVENANCE_SCRAPED, attempt)
            swaps.append((bad, spare))
        if swaps:
            self._write_active()
        return swaps

    def swap_one_blind(
        self, attempt: Optional[int] = None
    ) -> tuple[str, str]:
        """Rotate the first active host out for a spare.

        Used when the scraper can't pinpoint a culprit (silent hang,
        unknown crash pattern). Picks the first active host because
        that's also what the bash lib does — the choice is arbitrary
        but stable, so consecutive blind rotations cycle through the
        active set rather than thrashing on one host.

        Nothing implicated the evicted host, so it is recorded with
        :data:`PROVENANCE_BLIND` — a guess, not evidence.
        """
        if not self.spare:
            raise RuntimeError("out of spare nodes — cannot blind-rotate")
        if not self.active:
            raise RuntimeError("active set is empty — nothing to rotate")
        bad = self.active[0]
        spare = self.spare.popleft()
        self.active[0] = spare
        self._append_bad(bad, PROVENANCE_BLIND, attempt)
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
    consecutive_unattributed: int = 0,
) -> ClassificationResult:
    """Decide what the auto-retry loop should do after an attempt.

    Pure function — no side effects, single log read. Returns both
    the termination reason and the ``has_progress`` flag so the loop
    can thread the latter through to the next call without re-reading
    the log. The full termination matrix from PR #3's handoff doc:

    | rc       | log signals                                 | result                  |
    |----------|---------------------------------------------|-------------------------|
    | 0        | inner_rc=0 OR absent OR no crash pattern    | SUCCESS                 |
    | 0        | inner_rc != 0 (wrapper lied about success)  | classify by inner_rc    |
    | 0        | crash patterns present                      | bad-node retry          |
    | 143      | no crash patterns                           | WALLTIME                |
    | 143      | crash patterns present                      | bad-node retry          |
    | non-zero | ENOSPC/EDQUOT, under the budget             | RETRYABLE_UNATTRIBUTED  |
    | 124      | (idle-output watchdog tripped)              | BAD_NODE_BLIND          |
    | non-zero | scraper found named host(s)                 | BAD_NODE_KNOWN          |
    | non-zero | scraper empty                               | BAD_NODE_BLIND          |
    | -        | 0 progress twice AND scraper named nobody   | STUCK_PRE_TRAINING      |
    | -        | bad-node verdict but no spares left         | EXHAUSTED               |

    **Unattributed storage failures** (#231). A checkpoint write that
    dies of ENOSPC tears the whole job down, and on PALS the teardown
    cascade names some *other* host. Reading that host as bad retires a
    healthy node and burns a spare. So an I/O failure the log explains
    on its own short-circuits to :attr:`RETRYABLE_UNATTRIBUTED` — retry,
    but blame nothing. Bounded by ``consecutive_unattributed`` against
    :data:`_MAX_CONSECUTIVE_UNATTRIBUTED`: past the budget we stop
    believing the storage story and take the normal bad-node path, so a
    real node fault that happens to emit an I/O line cannot loop forever.
    Ordered AFTER the walltime guard (a clean walltime kill mid-write is
    still a walltime kill) and BEFORE the progress guard, since an
    ENOSPC during a long init is exactly the marker-free case #232 is
    about.

    **The progress guard** ("two consecutive attempts with no progress")
    is the user's preferred substitute for a numeric cap on blind
    rotations. It catches code bugs (broken config, missing dataset)
    that would otherwise burn the entire spare pool. The current
    attempt's progress status is checked against
    ``prior_attempt_had_progress``; the caller is responsible for
    tracking the prior value across iterations (use the ``has_progress``
    field of the returned :class:`ClassificationResult`).

    That guard now requires corroboration (#232). Missing progress
    markers are *absence* of evidence — equally consistent with a node
    that died during a long init, and with a trainer whose counter name
    :data:`_PROGRESS_MARKER_RX` does not know (which has already bitten
    us once: the regex was ``\\bstep=\\d+`` while every ezpz example
    emits ``iter=``). A scraper-named host is *presence* of evidence
    pointing the other way, and outranks the inference: given both, we
    fail over. Same shape as the ``and not crash`` clause that keeps
    :attr:`WALLTIME` from swallowing a bad-node death.

    INTERRUPTED is produced by the loop itself in the SIGINT handler,
    not here — the classifier never sees the interrupt path because
    KeyboardInterrupt is re-raised before we reach it.

    Args:
        shell_rc: Exit code of the attempt's child process.
        log_path: Path to that attempt's log.
        scraped_bad_nodes: Hosts the scraper named in that log.
        prior_attempt_had_progress: ``has_progress`` from the previous
            attempt, or ``None`` on the first attempt.
        has_spares: Whether a spare is available to swap in.
        consecutive_unattributed: How many attempts have ALREADY ended
            in :attr:`RETRYABLE_UNATTRIBUTED` in an unbroken run up to
            now. The loop tracks this and resets it to 0 on any other
            verdict.
    """
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    # Strip ANSI ONCE, here, so every matcher below sees plain text.
    #
    # ezpz's logger colorizes when attached to a tty, and the escapes
    # land INSIDE the tokens being matched: `\x1b[36miter\x1b[0m=12`
    # defeats `\biter=\d+`, and a colorized `rank 3 died from signal 15`
    # stops matching the innocent-cascade strip and is misread as a real
    # crash. `_extract_inner_rc` already stripped for exactly this
    # reason; doing it for the other two as well is the fix, rather
    # than teaching each pattern to tolerate escapes.
    log_text = _strip_ansi(log_text)

    inner_rc = _extract_inner_rc(log_text)
    crash = _has_crash_patterns(log_text)
    has_progress = _has_progress_markers(log_text)
    unattributed_io = _has_unattributed_io_failure(log_text)
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
        return ClassificationResult(
            reason=reason,
            has_progress=has_progress,
            has_unattributed_io=unattributed_io,
        )

    # Success: shell exit 0, no contrary inner_rc, no crash patterns.
    if effective_rc == 0:
        return _result(TerminationReason.SUCCESS)

    # Walltime guard. Real walltime: no point swapping nodes.
    # Walltime races: a true bad-node failure can land here when
    # mpiexec teardown races the wallclock kill. Use the crash
    # patterns to disambiguate.
    if effective_rc == _WALLTIME_RC and not crash:
        return _result(TerminationReason.WALLTIME)

    # Unattributed storage failure (#231). The log EXPLAINS this death
    # on its own -- an ENOSPC inside a checkpoint write -- so whatever
    # host the PALS teardown cascade happens to name is not evidence
    # about hardware. Retry in place: no swap, no spare, no entry in
    # bad_nodes.txt.
    #
    # Placed here on purpose:
    #   * AFTER walltime, so a clean wallclock kill that lands mid-write
    #     is still WALLTIME (swapping cannot buy more wallclock either).
    #   * BEFORE the progress guard, because a checkpoint write can be a
    #     resume-path `dcp.load` that dies before the first marker; two
    #     of those in a row must not read as "the job never started".
    #   * BEFORE the bad-node paths, which is the whole point.
    #
    # `consecutive_unattributed` is the bound. Past the budget we stop
    # believing the storage story and fall through to the normal
    # bad-node handling, so a real node fault that emits an I/O line
    # cannot pin the loop to one broken host forever.
    if (
        unattributed_io
        and consecutive_unattributed < _MAX_CONSECUTIVE_UNATTRIBUTED
    ):
        return _result(TerminationReason.RETRYABLE_UNATTRIBUTED)

    # The progress guard applies BEFORE we decide which swap path to
    # take — there's no point swapping nodes if the run is dying
    # before training starts. Note: we only check this on actual
    # failure paths; success already returned above.
    #
    # It needs corroboration (#232). "No progress marker" is an
    # INFERENCE from absence: it is equally true of a node that died
    # during a long init, and of a trainer whose counter name
    # `_PROGRESS_MARKER_RX` does not know. A scraper-named host is
    # positive evidence of a node fault, and positive evidence outranks
    # an inference from silence -- so when the scraper named someone we
    # fail over instead of concluding the job never started.
    #
    # This mirrors WALLTIME's `and not crash`: same structure, same
    # reason. The asymmetry is deliberate and matches the one
    # `_PROGRESS_MARKER_RX` already documents -- being wrong toward
    # failover costs one swap; being wrong toward STUCK abandons a
    # recoverable job with spares still free.
    #
    # A named host cannot make this loop forever: every failover
    # consumes a spare, so the pool bounds it and EXHAUSTED ends it.
    if (
        prior_attempt_had_progress is False
        and not has_progress
        and not scraped_bad_nodes
    ):
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
    #
    # `errors="replace"` is load-bearing: without it, ANY non-UTF-8
    # byte in the child's stdout (binary log fragment, terminal
    # control codes some libraries emit, a partial multibyte
    # sequence at a buffer boundary) raises UnicodeDecodeError
    # inside the `for line in proc.stdout` loop in `_drain`. That
    # crashes the drain thread, leaving the auto-retry monitor
    # "deaf" — training keeps going (tqdm writes via its own
    # handler) but the watchdog can no longer see crash
    # signatures, so a real failure later in the run would not
    # trigger a retry. Caught on Sunspot SFT job 12468338 — a
    # 32-min, 32-node run that lost its monitor thread to a
    # decode error mid-training.
    #
    # `replace` substitutes U+FFFD for bad bytes. We lose the
    # exact original byte in the log, which is fine: this is a
    # human-readable log + a crash-signature scraper, not a
    # binary protocol. The crash signatures we scrape for
    # (`shepherd died from signal 9`, `Connection closed by peer`,
    # etc.) are pure ASCII so substitution can't corrupt them.
    proc = subprocess.Popen(  # noqa: S603 — see comment above
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
        errors="replace",
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
            except Exception as exc:
                # Scraping is best-effort ATTRIBUTION: it upgrades a
                # blind rotation to a named one. Nothing it can fail at
                # is worth aborting a recoverable job for, so no
                # scraper failure is fatal here -- a blind rotation is
                # strictly better than a crash.
                #
                # This used to catch FileNotFoundError alone, so a
                # PermissionError from `getent` propagated out of the
                # retry loop entirely (#223). Logged rather than
                # swallowed silently: losing the bad node's NAME is
                # worth a line in the log.
                # Include the path and the traceback: this is the only
                # trace of WHY attribution was lost, and "scraper
                # failed" alone is not enough to debug from.
                logger.warning(
                    "[auto-retry] scraper failed on %s (%s: %s); falling "
                    "back to blind rotation",
                    p,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                return []

        scrape_fn = _default_scrape

    attempt = 0
    last_rc = 0
    prior_attempt_had_progress: Optional[bool] = None
    # Consecutive RETRYABLE_UNATTRIBUTED verdicts. Reset by ANY other
    # verdict: the budget exists to bound a *run* of storage failures,
    # not to cap them over the lifetime of the job. A run that hits
    # ENOSPC, recovers, trains for an hour and hits it again should get
    # the full budget the second time.
    consecutive_unattributed = 0

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
            consecutive_unattributed=consecutive_unattributed,
        )
        reason = result.reason
        # Thread the progress flag through to the next iteration —
        # classifier already parsed the log, no need to re-read.
        prior_attempt_had_progress = result.has_progress
        # Count what the LOG said, not what the classifier decided.
        # Keying off RETRYABLE_UNATTRIBUTED would be circular: spending
        # the budget flips the verdict to a bad-node swap, which would
        # then reset the counter and hand back a fresh budget on the
        # next identical failure -- an unbounded alternation instead of
        # a bound.
        if result.has_unattributed_io:
            consecutive_unattributed += 1
        else:
            consecutive_unattributed = 0

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
            # State plainly that this is an INFERENCE from absence
            # (#232). An operator whose trainer prints a counter this
            # regex does not know needs to be able to recognise the
            # misfire from the log line alone -- the previous wording
            # read as a finding rather than a guess.
            logger.error(
                "[auto-retry] FAILOVER STOP: stuck_pre_training "
                "(INFERRED, not observed: two consecutive attempts "
                "showed no iter=/step=/epoch=/batch=/idx= line, and the "
                "scraper named no host either, so the run is assumed to "
                "be dying before training starts. If your trainer prints "
                "a progress counter under some OTHER name, this verdict "
                "is wrong and a recoverable job was abandoned -- please "
                "report the counter you use. rc=%d)",
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

        # Unattributed storage failure: retry WITHOUT touching the
        # allocation (#231). No swap, no spare consumed, nothing
        # appended to bad_nodes.txt -- the failure said nothing about
        # any host, so neither do we.
        if reason is TerminationReason.RETRYABLE_UNATTRIBUTED:
            logger.warning(
                "[auto-retry] retryable_unattributed: the log shows an "
                "I/O failure (out of space / over quota) that is not "
                "evidence about any host, so no node is being retired "
                "and no spare is being consumed. Retrying on the SAME "
                "hosts (%d/%d of the consecutive budget, rc=%d). If this "
                "is a Lustre stripe landing on a full OST, the retry may "
                "well succeed; if it does not, check the filesystem "
                "before the budget runs out and normal node-swapping "
                "resumes.",
                consecutive_unattributed,
                _MAX_CONSECUTIVE_UNATTRIBUTED,
                last_rc,
            )
            continue

        # Bad-node paths fall through to a swap + continue.
        try:
            if reason is TerminationReason.BAD_NODE_KNOWN:
                swaps = allocation.swap_in(scraped, attempt=attempt)
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
                    bad, spare = allocation.swap_one_blind(
                        attempt=attempt
                    )
                    logger.warning(
                        "[auto-retry] blind rotation: %s -> %s",
                        bad,
                        spare,
                    )
            else:  # BAD_NODE_BLIND
                bad, spare = allocation.swap_one_blind(attempt=attempt)
                logger.warning(
                    "[auto-retry] blind rotation: %s -> %s", bad, spare
                )
        except RuntimeError as exc:
            logger.error(
                "[auto-retry] FAILOVER STOP: exhausted (%s)", exc
            )
            return last_rc
