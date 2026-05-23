"""Aurora-specific bad-node failure patterns.

These patterns were derived from postmortem analysis of real Aurora
training jobs. Each one is tied to a specific failure mode we've
seen take down a production training run:

  - **shepherd_signal_9** — PALS shepherd kill (node went bad
    mid-run). Aurora's most common hardware failure mode.
    Seen on jobs 8459818 / 8460301 / 8463659.

  - **gloo_connection_closed** — gloo TCP failure, usually one
    bad node taking down peer connections from many ranks. We
    reverse-resolve the IP to a hostname.
    Seen on jobs 8470102 / 8470103 / 8479581.

We DO NOT match ``rank N died from signal {11,15}`` because those
are almost always cascading deaths from a primary kill elsewhere on
a *different* node. Including them would falsely tag innocent nodes
(e.g. job 8466848 had rank 1304 die from signal 11 as a downstream
effect of a ``std::bad_alloc`` on rank 2413 — the kill happened on
the *first* node, but the signal-11 message named the *second*).
"""

from __future__ import annotations

import re
from typing import Iterable

from ezpz.failover.patterns import (
    BadNodePattern,
    compile_multiline,
    register_patterns,
    reverse_resolve_ip,
)


# ---------------------------------------------------------------------------
# Pattern 1: PALS shepherd kill
#
# Log line shape:
#     xNNNNcNsNbNnN.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9
#
# The hostname prefix is the literal compute node — PALS prepends it to
# every shepherd-level message. signal 9 = SIGKILL, sent by the runtime
# when the node-local daemon went non-responsive. Almost always a
# hardware fault (memory corruption, fabric flap, kernel hang).
# ---------------------------------------------------------------------------
_SHEPHERD_SIG9_RX = compile_multiline(
    r"^([a-zA-Z0-9.-]+\.hsn\.cm\.aurora\.alcf\.anl\.gov):\s+"
    r"shepherd\s+died\s+from\s+signal\s+9\b",
)


def _extract_shepherd_sig9(log_text: str) -> Iterable[str]:
    for m in _SHEPHERD_SIG9_RX.finditer(log_text):
        yield m.group(1)


# ---------------------------------------------------------------------------
# Pattern 2: gloo TCP peer-closed
#
# Log line shape (one of many — gloo wraps in RuntimeError):
#     RuntimeError: [..gloo..] Connection closed by peer [10.0.0.42]:12345
#
# Gloo errors typically point at a single peer IP across many
# "Connection closed" lines (every rank that was talking to the dead
# node logs its own copy), so deduplication usually collapses to one
# node. We reverse-resolve the IP to a hostname here (using the
# shared `reverse_resolve_ip` helper) and yield hostnames; the
# scraper's downstream normalizer canonicalizes the suffix.
# ---------------------------------------------------------------------------
_GLOO_PEER_RX = compile_multiline(
    r"Connection closed by peer\s+\[([0-9.]+)\]:\d+",
)


def _extract_gloo_peer(log_text: str) -> Iterable[str]:
    for m in _GLOO_PEER_RX.finditer(log_text):
        ip = m.group(1)
        host = reverse_resolve_ip(ip)
        if host is not None:
            yield host


# ---------------------------------------------------------------------------
# Hostname normalizer
#
# PBS hostfile entries are in the `.hsn.cm.aurora.alcf.anl.gov` form
# (the HSN fabric's name service). Other parts of Aurora occasionally
# emit the `.hostmgmtNNNN.cm.aurora...` management form. We normalize
# everything to the HSN form so swap_in (which greps the active
# hostfile) actually finds matches.
# ---------------------------------------------------------------------------
_AURORA_HSN_RX = re.compile(
    r"^x\d+c\d+s\d+b\d+n\d+\.hsn\.cm\.aurora\.alcf\.anl\.gov$"
)
# The .hostmgmtNNNN.cm.aurora.alcf.anl.gov form is the management
# interface; we know it maps 1:1 to the HSN interface and is safe to
# rewrite. Any OTHER suffix on a `x...n0.` host is something we
# haven't seen and shouldn't speculatively rewrite — better to drop
# (return None) than risk tagging a wrong node.
_AURORA_HOSTMGMT_RX = re.compile(
    r"^(x\d+c\d+s\d+b\d+n\d+)\.hostmgmt\d+\.cm\.aurora\.alcf\.anl\.gov$"
)


def normalize_aurora_hostname(host: str) -> "str | None":
    """Return the canonical ``.hsn.cm.aurora.alcf.anl.gov`` form, or
    None if *host* doesn't look like a valid Aurora compute hostname.

    Examples (in → out):
      ``x1234c0s0b0n0.hsn.cm.aurora.alcf.anl.gov``           → unchanged
      ``x1234c0s0b0n0.hostmgmt2042.cm.aurora.alcf.anl.gov``  → HSN form
      ``x1234c0s0b0n0.something-else.example.com``           → None
      ``some-other-host``                                    → None
    """
    if _AURORA_HSN_RX.match(host):
        return host
    m = _AURORA_HOSTMGMT_RX.match(host)
    if m:
        return f"{m.group(1)}.hsn.cm.aurora.alcf.anl.gov"
    return None


# ---------------------------------------------------------------------------
# Register at import time
# ---------------------------------------------------------------------------
AURORA_PATTERNS = [
    BadNodePattern(
        name="aurora.shepherd_signal_9",
        extractor=_extract_shepherd_sig9,
        description=(
            "PALS shepherd kill (signal 9). Node-local daemon went "
            "non-responsive; almost always a hardware fault."
        ),
    ),
    BadNodePattern(
        name="aurora.gloo_connection_closed",
        extractor=_extract_gloo_peer,
        description=(
            "gloo TCP peer-connection closed. IP reverse-resolved to "
            "an Aurora HSN hostname."
        ),
    ),
]

register_patterns(
    "aurora",
    AURORA_PATTERNS,
    hostname_normalizer=normalize_aurora_hostname,
)
# NOTE: Sunspot uses the same XPU fabric + PALS shepherd as Aurora,
# so the patterns SHOULD apply — but the full hostname suffix differs
# from Aurora's `.hsn.cm.aurora.alcf.anl.gov`. We haven't postmortemed
# a real Sunspot failure yet to confirm the exact suffix shape, so the
# Aurora pattern is NOT auto-registered for sunspot here. Add a
# dedicated `sunspot.py` module once a real failure provides the
# hostname format to match against.
