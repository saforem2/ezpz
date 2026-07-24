"""Sunspot-specific bad-node failure patterns.

Sunspot is the Aurora test-and-development system: same Intel PVC (XPU)
hardware, same PALS/cray-pals runtime, same failure modes. The signatures
are therefore identical to :mod:`ezpz.failover.patterns.aurora` — only the
hostname suffix differs (``.hsn.cm.sunspot.alcf.anl.gov`` vs Aurora's
``.hsn.cm.aurora.alcf.anl.gov``), plus Sunspot's HSN node token can carry a
``-hsnN`` suffix (e.g. ``x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov``)
that Aurora's does not.

  - **shepherd_signal_9** — PALS shepherd kill (node went bad mid-run).
    Same runtime as Aurora, so the same signature applies.

  - **gloo_connection_closed** — gloo TCP failure, IP reverse-resolved to
    a hostname. The extractor is IP-based and machine-agnostic.

We DO NOT match ``rank N died from signal {11,15}`` — those are almost
always cascading deaths downstream of a primary kill on a *different* node,
so including them would falsely tag innocent nodes. Same reasoning as
Aurora (see that module's docstring for the job-8466848 example).
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
# Log line shape (note the optional -hsnN on the node token):
#     x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: shepherd died from signal 9
#
# Same PALS runtime as Aurora; signal 9 = SIGKILL from the node-local
# daemon going non-responsive (almost always a hardware fault). The
# capture group is intentionally permissive ([A-Za-z0-9.-]+) so it accepts
# both the -hsnN and plain node-token forms; the anchoring `.hsn.cm.sunspot`
# suffix is what keeps it Sunspot-specific.
# ---------------------------------------------------------------------------
_SHEPHERD_SIG9_RX = compile_multiline(
    r"^([a-zA-Z0-9.-]+\.hsn\.cm\.sunspot\.alcf\.anl\.gov):\s+"
    r"shepherd\s+died\s+from\s+signal\s+9\b",
)


def _extract_shepherd_sig9(log_text: str) -> Iterable[str]:
    for m in _SHEPHERD_SIG9_RX.finditer(log_text):
        yield m.group(1)


# ---------------------------------------------------------------------------
# Pattern 2: gloo TCP peer-closed
#
# Log line shape:
#     RuntimeError: [..gloo..] Connection closed by peer [10.0.0.42]:12345
#
# IP-based and machine-agnostic — identical to the Aurora extractor. We
# reverse-resolve the IP to a hostname; the scraper's downstream normalizer
# canonicalizes the suffix.
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
# PBS hostfile entries and PALS shepherd lines use the
# `.hsn.cm.sunspot.alcf.anl.gov` form, with the HSN node token optionally
# carrying a `-hsnN` suffix. The `.hostmgmtNNNN.cm.sunspot...` management
# form maps 1:1 to the HSN interface and is safe to rewrite. Any OTHER
# suffix on an `x...n0.` host is something we haven't seen and shouldn't
# speculatively rewrite — return None (drop) rather than risk tagging a
# wrong node.
# ---------------------------------------------------------------------------
_SUNSPOT_HSN_RX = re.compile(
    r"^x\d+c\d+s\d+b\d+n\d+(?:-hsn\d+)?\.hsn\.cm\.sunspot\.alcf\.anl\.gov$"
)
_SUNSPOT_HOSTMGMT_RX = re.compile(
    r"^(x\d+c\d+s\d+b\d+n\d+)\.hostmgmt\d+\.cm\.sunspot\.alcf\.anl\.gov$"
)


def normalize_sunspot_hostname(host: str) -> "str | None":
    """Return the canonical ``.hsn.cm.sunspot.alcf.anl.gov`` form, or None
    if *host* doesn't look like a valid Sunspot compute hostname.

    Examples (in → out):
      ``x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov``      → unchanged
      ``x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov``           → unchanged
      ``x1922c7s6b0n0.hostmgmt2001.cm.sunspot.alcf.anl.gov``  → HSN form
      ``x1922c7s6b0n0.something-else.example.com``            → None
      ``some-other-host``                                     → None

    NOTE: the exact HSN suffix that a gloo reverse-DNS lookup returns on
    Sunspot (with vs without ``-hsn0``) is not yet confirmed against a real
    postmortem. The hostmgmt→HSN rewrite drops the ``-hsnN`` token (mirrors
    the observed PBS hostfile + shepherd-line form); revisit if a real
    failure shows a different mapping.
    """
    if _SUNSPOT_HSN_RX.match(host):
        return host
    m = _SUNSPOT_HOSTMGMT_RX.match(host)
    if m:
        return f"{m.group(1)}.hsn.cm.sunspot.alcf.anl.gov"
    return None


# ---------------------------------------------------------------------------
# Register at import time
# ---------------------------------------------------------------------------
SUNSPOT_PATTERNS = [
    BadNodePattern(
        name="sunspot.shepherd_signal_9",
        extractor=_extract_shepherd_sig9,
        description=(
            "PALS shepherd kill (signal 9). Node-local daemon went "
            "non-responsive; almost always a hardware fault. Same runtime "
            "as Aurora."
        ),
    ),
    BadNodePattern(
        name="sunspot.gloo_connection_closed",
        extractor=_extract_gloo_peer,
        description=(
            "gloo TCP peer-connection closed. IP reverse-resolved to a "
            "Sunspot HSN hostname."
        ),
    ),
]

register_patterns(
    "sunspot",
    SUNSPOT_PATTERNS,
    hostname_normalizer=normalize_sunspot_hostname,
)
