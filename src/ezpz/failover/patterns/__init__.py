"""Per-machine bad-node detection patterns.

Each ALCF / NERSC / OLCF system has its own failure-mode signatures —
PALS shepherd messages on Aurora, NCCL timeouts on Polaris, etc. —
and its own hostname conventions for resolving an IP back to a node.
This package defines a small registry so :func:`ezpz.failover.scrape`
can dispatch to the right pattern set based on the detected machine.

To add support for a new machine:

  1. Create ``src/ezpz/failover/patterns/<machine>.py`` with one or
     more :class:`BadNodePattern` instances.
  2. Register them via :func:`register_patterns` at module import.
  3. The module gets imported automatically by
     :func:`get_patterns_for_machine` when needed.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable

# Pattern types --------------------------------------------------------------

# A pattern's ``match`` callable receives the FULL log text (not a stream of
# lines) and yields zero or more bad-host strings (possibly raw IPs that need
# resolution). Returning an iterable rather than a list keeps memory bounded
# on very large logs — the caller deduplicates downstream.
HostExtractor = Callable[[str], Iterable[str]]


@dataclass(frozen=True)
class BadNodePattern:
    """A single failure-mode signature for one machine.

    Args:
        name: Human-readable identifier (``"aurora.shepherd_signal_9"``).
            Used in log messages so postmortem reviewers know which
            pattern fired.
        extractor: Callable that takes the full log text and yields
            hostnames (and only hostnames). If the underlying log
            shape gives you IPs (e.g. the gloo "Connection closed by
            peer [IP]:port" line), the extractor is responsible for
            doing the IP→hostname conversion itself — the registry
            does NOT do a resolve step. Use :func:`reverse_resolve_ip`
            from this module to handle the lookup the same way the
            Aurora patterns do.
        description: One-line plain-English summary for help text +
            postmortem notes.
    """

    name: str
    extractor: HostExtractor
    description: str


# Machine registry -----------------------------------------------------------

_PATTERNS: dict[str, list[BadNodePattern]] = {}
_HOSTNAME_NORMALIZERS: dict[str, Callable[[str], "str | None"]] = {}


def register_patterns(
    machine: str,
    patterns: list[BadNodePattern],
    hostname_normalizer: "Callable[[str], str | None] | None" = None,
) -> None:
    """Register a set of patterns + an optional hostname normalizer for
    a machine. Idempotent — calling twice with the same machine
    replaces the previous registration (lets tests inject mocks).

    Re-registration FULLY replaces the previous state, including the
    normalizer slot: passing ``hostname_normalizer=None`` clears any
    previously-registered normalizer for this machine. Otherwise the
    old normalizer would silently apply to the new pattern set, which
    bit us in early development.
    """
    _PATTERNS[machine] = list(patterns)
    if hostname_normalizer is not None:
        _HOSTNAME_NORMALIZERS[machine] = hostname_normalizer
    else:
        # Clear any prior registration. This makes re-registration
        # truly idempotent regardless of whether the new caller
        # supplied a normalizer.
        _HOSTNAME_NORMALIZERS.pop(machine, None)


def get_patterns_for_machine(machine: str) -> list[BadNodePattern]:
    """Return the patterns registered for *machine*. Lazy-imports the
    per-machine module on first access so we don't pay the import cost
    for machines that aren't being used.

    Only "no such per-machine module" is silently treated as "unknown
    machine"; any OTHER ImportError (a missing dep, a syntax error,
    etc. INSIDE a known machine module) is re-raised so it surfaces
    instead of looking like an unsupported cluster.
    """
    if machine not in _PATTERNS:
        module_name = f"ezpz.failover.patterns.{machine}"
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            # `name` on ImportError tells us which module the import
            # machinery couldn't find. If it matches the one we asked
            # for, the machine genuinely doesn't have a pattern module
            # → return [] silently. If it matches some OTHER module
            # (e.g. the machine module exists but imports a missing
            # dep), re-raise so the failure isn't swallowed.
            if getattr(exc, "name", None) != module_name:
                raise
    return _PATTERNS.get(machine, [])


def get_hostname_normalizer(
    machine: str,
) -> "Callable[[str], str | None] | None":
    """Return the hostname normalizer for *machine* (or None)."""
    # Trigger registration if not yet loaded.
    get_patterns_for_machine(machine)
    return _HOSTNAME_NORMALIZERS.get(machine)


# Shared helpers (used by per-machine modules) -------------------------------

def reverse_resolve_ip(ip: str, timeout_s: float = 5.0) -> "str | None":
    """Reverse-resolve an IP to a hostname via ``getent hosts``.

    Returns the first hostname token in the lookup result, or None if
    the lookup fails (binary missing, timeout, empty result). The
    caller is responsible for further canonicalization (e.g. mapping
    to the cluster's HSN form).

    ``getent`` is preferred over ``socket.gethostbyaddr`` because it
    honors ``/etc/hosts`` + ``nsswitch.conf`` config — the IPs in
    gloo error messages are usually on the HSN fabric and only
    resolvable through the cluster's name service.
    """
    try:
        out = subprocess.check_output(
            ["getent", "hosts", ip],
            text=True,
            timeout=timeout_s,
        ).strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return None
    if not out:
        return None
    # `getent hosts <ip>` → `<ip>  <hostname1> <hostname2>...`
    parts = out.split()
    if len(parts) >= 2:
        return parts[1]
    return None


def compile_multiline(pattern: str) -> "re.Pattern[str]":
    """Compile *pattern* with ``re.MULTILINE`` set. Convenience for
    per-machine modules whose patterns all anchor on line starts."""
    return re.compile(pattern, re.MULTILINE)


__all__ = [
    "BadNodePattern",
    "register_patterns",
    "get_patterns_for_machine",
    "get_hostname_normalizer",
    "reverse_resolve_ip",
    "compile_multiline",
]
