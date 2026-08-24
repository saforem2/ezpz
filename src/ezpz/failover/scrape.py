"""Scrape training logs for bad-node hostnames.

The public entry point is :func:`scrape_bad_nodes`. It takes a path
to a training log and returns a deduplicated list of bad hostnames
in first-seen order. The patterns it matches are per-machine —
Aurora's PALS shepherd kills and gloo TCP failures are bundled by
default; other systems can register their own via the
:mod:`ezpz.failover.patterns` registry.

Typical usage from a failover wrapper::

    from ezpz.failover import scrape_bad_nodes

    hosts = scrape_bad_nodes("/path/to/attempt-1.log")
    if hosts:
        swap_in(hosts)
    else:
        swap_one_blind()  # nothing specific identified; rotate a spare

When no hostnames are matched the caller should treat that as
"failure with no actionable hostname" and apply whatever its
blind-rotation policy is — silence here does NOT mean the run
succeeded, only that the patterns couldn't pinpoint a culprit.
"""

from __future__ import annotations

import logging
import pathlib

from pathlib import Path

from ezpz.failover.patterns import (
    get_hostname_normalizer,
    get_patterns_for_machine,
)

# `ezpz.get_logger`, matching every other module in the package --
# a raw `logging.getLogger` here was invisible to pytest's caplog.
# Imported lazily inside the function that needs it so this module
# keeps its light import (the pattern helpers are used by unit tests
# that must not pull in the rest of ezpz).
def _get_logger():
    global _LOGGER
    if _LOGGER is None:
        try:
            import ezpz  # noqa: PLC0415

            _LOGGER = ezpz.get_logger(__name__)
        except Exception:
            _LOGGER = logging.getLogger(__name__)
    return _LOGGER


_LOGGER = None

# Machines already warned about, so the message appears once per
# process rather than once per retry attempt.
_WARNED_NO_PATTERNS: set = set()


def _registered_machines() -> "list[str]":
    """Machine keys that HAVE a pattern module available.

    Enumerates the package directory rather than the `_PATTERNS`
    registry: modules are imported on demand, so the registry only
    holds machines already looked up — asking it here reported
    "Registered: none" while aurora and sunspot both existed.

    Discovered rather than hardcoded, so the warning cannot drift out
    of date as modules are added.
    """
    try:
        from ezpz.failover import patterns as _pkg  # noqa: PLC0415

        d = pathlib.Path(_pkg.__file__).parent
        return sorted(
            f.stem
            for f in d.glob("*.py")
            if not f.stem.startswith("_")
        )
    except Exception:
        return []


def _detect_machine() -> str:
    """Best-effort detection of the current machine via ezpz's existing
    hostname → machine mapping. Imported lazily so the scrape module
    itself doesn't pull in the rest of ezpz when only the pattern
    helpers are needed (e.g. by unit tests)."""
    try:
        import ezpz  # noqa: PLC0415
    except ImportError:
        return ""
    try:
        return ezpz.get_machine().lower()
    except Exception:
        return ""


def _resolve_machine_key(machine: "str | None") -> str:
    """Pick the registry lookup key from an optional caller override.

    Registry keys are always lowercase (Aurora's module registers
    itself as ``"aurora"``). The auto-detect path already lowercases
    the result of ``ezpz.get_machine()``, but an explicit override
    arrives verbatim — so ``scrape_bad_nodes(log, machine="Aurora")``
    would silently miss every pattern unless we normalize here too.
    """
    if machine is None:
        return _detect_machine()
    return machine.lower()


def scrape_bad_nodes(
    log_path: "str | Path",
    *,
    machine: "str | None" = None,
) -> list[str]:
    """Return the deduplicated list of bad hostnames in *log_path*.

    Args:
        log_path: Path to the training log file. Read in full as
            text (errors='replace' so binary garbage in the middle
            of a log doesn't crash the scrape — these logs are
            sometimes >1GB and partially corrupted by mid-write
            crashes, and the patterns we care about always appear
            on lines that are clean ASCII).
        machine: Override machine detection. Defaults to the result
            of :func:`ezpz.get_machine`. Pass an explicit value
            (``"aurora"``, ``"polaris"``, ...) for tests or when
            scraping logs from a different cluster than the one
            running this code.

    Returns:
        Bad hostnames in first-seen order, deduplicated. Hostnames
        are canonicalized via the machine's hostname normalizer if
        one is registered (e.g. Aurora maps everything to the
        ``.hsn.cm.aurora.alcf.anl.gov`` form so downstream swap-in
        logic matches the PBS hostfile entries).

        Returns ``[]`` when no patterns matched OR when there is no
        pattern set registered for the detected machine.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"log file not found: {log_path}")

    machine = _resolve_machine_key(machine)
    patterns = get_patterns_for_machine(machine)
    if not patterns:
        # No registered patterns → nothing to do. Returning [] (rather
        # than raising) lets the caller's blind-rotation fallback fire,
        # which is the right behavior for unknown failure modes too.
        #
        # But SAY SO. An unsupported machine and a genuinely clean log
        # both produce [], and only the second means "no node to
        # blame" -- so every failover on a machine with no pattern set
        # is blind, silently and forever (#229). One line at WARNING
        # is the difference between a known limitation and a mystery.
        # Logged once per process: this runs on every attempt, and a
        # repeated warning in a retry loop trains people to ignore it.
        global _WARNED_NO_PATTERNS
        if machine not in _WARNED_NO_PATTERNS:
            _WARNED_NO_PATTERNS.add(machine)
            _get_logger().warning(
                "no bad-node patterns registered for machine %r — every "
                "failover on this machine will be a BLIND rotation "
                "(the sick node may stay in the allocation). Registered: "
                "%s",
                machine or "<undetected>",
                ", ".join(sorted(_registered_machines())) or "none",
            )
        return []

    normalize = get_hostname_normalizer(machine)
    # NOTE: this reads the entire log into memory. In practice
    # postmortem logs we've handled (multi-rank Aurora training,
    # hundreds of MB to a couple GB) fit fine on the head node where
    # failover runs. If you hit a 10+ GB log, pre-trim it with
    # `tail -c 1G` or similar before passing to this function.
    text = log_path.read_text(errors="replace")

    seen_order: list[str] = []
    seen_set: set[str] = set()

    for pattern in patterns:
        for raw_host in pattern.extractor(text):
            host = normalize(raw_host) if normalize else raw_host
            if host is None:
                continue
            if host in seen_set:
                continue
            seen_order.append(host)
            seen_set.add(host)

    return seen_order


def _collect_all_matches_for_debug(
    log_path: "str | Path",
    *,
    machine: "str | None" = None,
) -> dict[str, list[str]]:
    """Per-pattern breakdown of what the scraper matched. Not part of
    the public API — used by the CLI's ``--explain`` mode (when
    that's added) and by tests that want to verify a specific pattern
    fired, not just that *some* pattern fired.
    """
    log_path = Path(log_path)
    machine = _resolve_machine_key(machine)
    patterns = get_patterns_for_machine(machine)
    text = log_path.read_text(errors="replace")
    normalize = get_hostname_normalizer(machine)
    out: dict[str, list[str]] = {}
    for pattern in patterns:
        # Track seen separately so dedup is O(1) per check; otherwise
        # `host not in matches` is O(n) and the whole loop quadratic
        # on patterns that fire across many ranks (gloo peer-closed
        # routinely emits dozens of copies of the same host).
        matches: list[str] = []
        seen: set[str] = set()
        for raw_host in pattern.extractor(text):
            host = normalize(raw_host) if normalize else raw_host
            if host is None or host in seen:
                continue
            seen.add(host)
            matches.append(host)
        out[pattern.name] = matches
    return out
