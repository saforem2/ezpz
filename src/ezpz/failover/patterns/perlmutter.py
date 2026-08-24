"""Perlmutter (NERSC) bad-node failure patterns.

Perlmutter differs from Aurora/Sunspot in the way that matters here:
it runs **SLURM/srun**, not PBS/PALS, so none of the PALS signatures
appear. A killed node emits no ``shepherd died from signal 9`` and no
``rank N exited with code K`` — the scraper found nothing, and
``srun``'s rc=143 additionally collapsed to ``WALLTIME`` so the loop
did not even retry (see ``_CRASH_PATTERNS_RX`` and ezpz #238).

What ``srun`` DOES emit names the host directly::

    srun: error: nid001321: tasks 4-7: Killed       <- the victim
    srun: error: nid001320: tasks 0-3: Terminated   <- the cascade

Captured on Perlmutter job 57540936, which SIGKILLed the ranks on one
node of a 2-active/2-spare allocation.

We match ``Killed`` and NOT ``Terminated``. ``Killed`` is SIGKILL —
something outside the job stopped that rank. ``Terminated`` is the
SIGTERM every rank receives when a step is torn down normally, so
matching it would retire a node on every clean walltime expiry. Same
reasoning as the ``rank N died from signal {11,15}`` exclusion the
PALS modules document.
"""

from __future__ import annotations

import re
from typing import Iterable

from ezpz.failover.patterns import (
    BadNodePattern,
    compile_multiline,
    register_patterns,
)

# ---------------------------------------------------------------------------
# Pattern: srun task-kill
#
#     srun: error: nid001321: tasks 4-7: Killed
#     srun: error: nid001321: task 4: Killed
#
# Host-first and `srun:`-prefixed, unlike the PALS forms. The task list
# may be a range, a comma list, or a single index.
# ---------------------------------------------------------------------------
_SRUN_KILLED_RX = compile_multiline(
    r"^srun:\s+error:\s+(nid\d+):\s+tasks?\s+[\d,\-]+:\s+Killed\b",
)


def _extract_srun_killed(log_text: str) -> Iterable[str]:
    for m in _SRUN_KILLED_RX.finditer(log_text):
        yield m.group(1)


# Perlmutter node names are bare `nidNNNNNN` — no domain, no `-hsnN`
# interface suffix — and that is exactly the form SLURM puts in the
# hostfile, so there is nothing to canonicalize.
_NID_RX = re.compile(r"^nid\d+$")


def normalize_perlmutter_hostname(host: str) -> "str | None":
    """Accept a bare ``nidNNNNNN``; reject anything else.

    Deliberately strict: a name that is not in SLURM's own form would
    never match the active hostfile, so admitting it could only put a
    host in ``bad_nodes.txt`` that no swap can ever act on.
    """
    h = host.strip().split(".", 1)[0]
    return h if _NID_RX.match(h) else None


PERLMUTTER_PATTERNS = [
    BadNodePattern(
        name="perlmutter.srun_task_killed",
        extractor=_extract_srun_killed,
        description=(
            "srun reported a rank as Killed (SIGKILL) on this node. "
            "Distinct from Terminated, which is the normal step "
            "teardown. Confirmed on job 57540936."
        ),
    ),
]

register_patterns(
    "perlmutter",
    PERLMUTTER_PATTERNS,
    hostname_normalizer=normalize_perlmutter_hostname,
)
