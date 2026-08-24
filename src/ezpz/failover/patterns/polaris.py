"""Polaris-specific bad-node failure patterns.

Polaris differs from Aurora/Sunspot in one way that shapes every
pattern here: its most common production failure is an NVIDIA
CUDA-runtime error raised inside a rank's Python process, NOT a
PALS shepherd kill. A shepherd kill arrives pre-labeled with the
node that died; a Python traceback does not -- mpiexec prints it
verbatim on stderr with no host prefix unless the launcher passes
``--label``.

Job 7550301 (2026-08-23, 130 nodes, 20B dolma chain) is the
motivating postmortem: two ranks raised

    torch.AcceleratorError: CUDA error: CUDA-capable device(s)
    is/are busy or unavailable

at ``torch.cuda.set_device()`` during ``setup_torch()``, on both
attempts. Because the tracebacks were unlabeled, the scraper found
no host, the caller fell back to BLIND rotation, and it rotated out
``active[0]`` -- a healthy node -- while the sick node stayed in the
allocation. Two spares burned, zero training steps, ~1 hour of 130
nodes charged against the allocation.

The fix has two halves and BOTH are required:

  1. The launcher passes ``--label`` to PALS mpiexec, so every line
     of a rank's stdout/stderr is prefixed
     ``<fqdn> <rank>: <text>``. Verified empirically on Polaris
     (job 7553963): the prefix is applied per-LINE to multi-line
     Python tracebacks on stderr, including the exact CUDA text
     above.
  2. These patterns read that prefix.

Without (1) the patterns below match nothing and the scraper
correctly falls back to blind rotation -- i.e. this module is a
no-op on unlabeled logs rather than a source of false positives.

We DO NOT match ``rank N died from signal {11,15}``. On Polaris the
overwhelmingly common source of signal 15 is the idle-output
watchdog's OWN SIGTERM (``launch_autoretry`` kills the process group
after ``--timeout`` seconds of silence), so the named rank is a
victim of our own teardown, not a culprit. In job 7550301 the only
host-attributed line in the entire log was exactly this -- naming
``x3007c0s13b1n0``, which was NOT the node that raised the CUDA
error. Matching it would have swapped yet another innocent node.
This mirrors the innocent-cascade exclusion in ``aurora.py`` and the
``_INNOCENT_RANK_CASCADE_RX`` strip in ``launch_autoretry``.
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

# The PALS ``--label`` prefix: ``<fqdn> <rank>: ``. Captured as a
# reusable fragment so every labeled pattern below anchors on the
# same shape. Group 1 is always the hostname.
#
# Confirmed on Polaris job 7553963:
#     x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov 1: RuntimeError: ...
_LABEL = r"^([a-zA-Z0-9.-]+\.hsn\.cm\.polaris\.alcf\.anl\.gov)\s+\d+:\s+"


# ---------------------------------------------------------------------------
# Pattern 1: CUDA device unavailable / busy
#
# Raised by cudaSetDevice when the GPU is held by a stale process, is
# in an error state, or has fallen off the bus. This is a node-local
# hardware/driver fault: the same node will fail again on retry, which
# is exactly why blind rotation could not clear job 7550301.
#
# Matches the torch wrapper class (``torch.AcceleratorError``), the
# older ``RuntimeError`` spelling, and the bare CUDA message, since
# the surfacing class has changed across torch versions and we care
# about the device state, not the wrapper.
# ---------------------------------------------------------------------------
_CUDA_UNAVAILABLE_RX = compile_multiline(
    _LABEL
    + r"(?:[\w.]*Error: )?"
    + r"CUDA error: CUDA-capable device\(s\) is/are busy or unavailable",
)


def _extract_cuda_unavailable(log_text: str) -> Iterable[str]:
    for m in _CUDA_UNAVAILABLE_RX.finditer(log_text):
        yield m.group(1)


# ---------------------------------------------------------------------------
# Pattern 2: CUDA initialization / no-device faults
#
# A GPU that has fallen off the PCIe bus surfaces as one of these
# rather than "busy or unavailable". Same remediation (swap the node),
# so they share a pattern.
#
# NOTE: `device-side assert triggered` is deliberately EXCLUDED. It is
# raised by a failing assertion inside a kernel -- an out-of-range index,
# a bad label, an invalid model input -- which is an APPLICATION defect,
# not a node fault. Matching it would retire a perfectly healthy node on
# every retry while the real bug persists, burning the spare pool and
# never converging. The rule for this pattern set: only match conditions
# where the same code would succeed on a different node.
# ---------------------------------------------------------------------------
_CUDA_INIT_RX = compile_multiline(
    _LABEL
    + r"(?:[\w.]*Error: )?"
    + r"CUDA error: (?:"
    + r"no CUDA-capable device is detected"
    + r"|initialization error"
    + r"|unknown error"
    + r"|system not yet initialized"
    + r")",
)


def _extract_cuda_init(log_text: str) -> Iterable[str]:
    for m in _CUDA_INIT_RX.finditer(log_text):
        yield m.group(1)


# ---------------------------------------------------------------------------
# Pattern 3: Xid / GPU hardware faults surfaced through NVML
#
# "GPU is lost" and ECC/uncorrectable errors mean the device is gone
# until the node is reset. Always node-fatal.
# ---------------------------------------------------------------------------
_GPU_LOST_RX = compile_multiline(
    _LABEL
    + r".*(?:"
    + r"GPU is lost"
    + r"|Uncorrectable ECC error encountered"
    + r"|has fallen off the bus"
    + r"|CUDA-capable device\(s\) is/are busy or unavailable"
    + r")",
)


def _extract_gpu_lost(log_text: str) -> Iterable[str]:
    for m in _GPU_LOST_RX.finditer(log_text):
        yield m.group(1)


# ---------------------------------------------------------------------------
# Pattern 4: PALS shepherd kill
#
# Same shape as Aurora's, with the Polaris suffix. This one is
# host-prefixed by PALS itself and so works WITHOUT ``--label``.
# ---------------------------------------------------------------------------
_SHEPHERD_SIG9_RX = compile_multiline(
    r"^([a-zA-Z0-9.-]+\.hsn\.cm\.polaris\.alcf\.anl\.gov):\s+"
    r"shepherd\s+died\s+from\s+signal\s+9\b",
)


def _extract_shepherd_sig9(log_text: str) -> Iterable[str]:
    for m in _SHEPHERD_SIG9_RX.finditer(log_text):
        yield m.group(1)


# ---------------------------------------------------------------------------
# Pattern 5: gloo TCP peer-closed
#
# Yields an IP that we reverse-resolve. Unlike the CUDA patterns this
# does not depend on ``--label`` -- the IP is inside the message. Note
# this names the node that was BEING talked to, which is the dead one.
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
# PBS hostfile entries on Polaris use the plain
# ``.hsn.cm.polaris.alcf.anl.gov`` form (verified against
# logs/failover-7550301/active.hostfile), but the node token
# sometimes carries an HSN interface suffix (``-hsn0``) in launcher
# output. Both name the same node; emit the suffix-less form so a
# single node cannot dedupe as two entries and so ``swap_in``'s
# hostfile lookup actually matches.
#
# The ``.hostmgmtNNNN.`` management form maps 1:1 to the HSN
# interface and is safe to rewrite. Any OTHER suffix is something we
# have not seen -- return None (drop) rather than risk tagging the
# wrong node.
# ---------------------------------------------------------------------------
_POLARIS_HSN_RX = re.compile(
    r"^(x\d+c\d+s\d+b\d+n\d+)(?:-hsn\d+)?\.hsn\.cm\.polaris\.alcf\.anl\.gov$"
)
_POLARIS_HOSTMGMT_RX = re.compile(
    r"^(x\d+c\d+s\d+b\d+n\d+)\.hostmgmt\d+\.cm\.polaris\.alcf\.anl\.gov$"
)


def normalize_polaris_hostname(host: str) -> "str | None":
    """Return the canonical ``.hsn.cm.polaris.alcf.anl.gov`` form, or
    None if *host* does not look like a valid Polaris compute hostname.

    Examples (in -> out):
      ``x3006c0s13b1n0.hsn.cm.polaris.alcf.anl.gov``          -> unchanged
      ``x3006c0s13b1n0-hsn0.hsn.cm.polaris.alcf.anl.gov``     -> suffix-less
      ``x3006c0s13b1n0.hostmgmt2042.cm.polaris.alcf.anl.gov`` -> HSN form
      ``x3006c0s13b1n0.something-else.example.com``           -> None
      ``some-other-host``                                     -> None
    """
    m = _POLARIS_HSN_RX.match(host)
    if m:
        return f"{m.group(1)}.hsn.cm.polaris.alcf.anl.gov"
    m = _POLARIS_HOSTMGMT_RX.match(host)
    if m:
        return f"{m.group(1)}.hsn.cm.polaris.alcf.anl.gov"
    return None


# ---------------------------------------------------------------------------
# Register at import time
# ---------------------------------------------------------------------------
POLARIS_PATTERNS = [
    BadNodePattern(
        name="polaris.cuda_device_unavailable",
        extractor=_extract_cuda_unavailable,
        description=(
            "cudaSetDevice reported the GPU busy/unavailable. Node-local "
            "driver or stale-process fault; recurs on retry. Requires the "
            "launcher to pass --label. Job 7550301."
        ),
    ),
    BadNodePattern(
        name="polaris.cuda_init_error",
        extractor=_extract_cuda_init,
        description=(
            "CUDA init failure (no device detected / initialization "
            "error / unknown error). Excludes device-side asserts, "
            "which are application defects, not node faults. "
            "Requires --label."
        ),
    ),
    BadNodePattern(
        name="polaris.gpu_lost",
        extractor=_extract_gpu_lost,
        description=(
            "GPU lost, uncorrectable ECC, or fell off the bus. Node-fatal "
            "until reset. Requires --label."
        ),
    ),
    BadNodePattern(
        name="polaris.shepherd_signal_9",
        extractor=_extract_shepherd_sig9,
        description=(
            "PALS shepherd kill (signal 9). Node-local daemon went "
            "non-responsive. Host-prefixed by PALS; no --label needed."
        ),
    ),
    BadNodePattern(
        name="polaris.gloo_connection_closed",
        extractor=_extract_gloo_peer,
        description=(
            "gloo TCP peer-connection closed. IP reverse-resolved to a "
            "Polaris HSN hostname. No --label needed."
        ),
    ),
]

register_patterns(
    "polaris",
    POLARIS_PATTERNS,
    hostname_normalizer=normalize_polaris_hostname,
)
