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

from pathlib import Path
from typing import Iterable

from ezpz.failover.patterns import (
    get_hostname_normalizer,
    get_patterns_for_machine,
)


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

    if machine is None:
        machine = _detect_machine()
    patterns = get_patterns_for_machine(machine)
    if not patterns:
        # No registered patterns → nothing to do. Returning [] (rather
        # than raising) lets the caller's blind-rotation fallback fire,
        # which is the right behavior for unknown failure modes too.
        return []

    normalize = get_hostname_normalizer(machine)
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
    if machine is None:
        machine = _detect_machine()
    patterns = get_patterns_for_machine(machine)
    text = log_path.read_text(errors="replace")
    normalize = get_hostname_normalizer(machine)
    out: dict[str, list[str]] = {}
    for pattern in patterns:
        matches: list[str] = []
        for raw_host in pattern.extractor(text):
            host = normalize(raw_host) if normalize else raw_host
            if host is not None and host not in matches:
                matches.append(host)
        out[pattern.name] = matches
    return out
