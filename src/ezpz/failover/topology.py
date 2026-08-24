"""Recover the rank-to-host mapping a training log already records.

``setup_torch`` logs one line per host at startup, from ``local_rank=0``::

    ['x1921c1s0b0n0'][device='xpu'][node=0/1][local_rank=00/11][rank=00/23]
    ['x1921c1s1b0n0'][device='xpu'][node=1/1][local_rank=00/11][rank=12/23]

That is a complete topology description -- host 0 owns ranks 0-11, host 1
owns 12-23 -- and until now nothing read it back. The failover scraper
looks only for *failure* signatures, so a log with no such signature
yields no host, even when the log plainly says which hosts were running.

This module is deliberately NOT a bad-node detector. It answers "which
host owned rank N", which is only useful when something else names a
rank. Read :func:`hosts_for_ranks`'s docstring for what that rules out.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

__all__ = ["RankHostMap", "parse_rank_host_map", "hosts_for_ranks"]

# ezpz colorizes when attached to a tty, and the escapes land INSIDE
# the tokens: `[^[[94mnode^[[0m=^[[1;36m1^[[0m/...`, so `node=(\d+)`
# cannot match a real log. Strip once before matching -- the same fix
# `classify_attempt` needed for its progress and crash matchers, and
# the same trap that made the documented `grep "iter="` recipe return
# nothing.
_ANSI_RX = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# Anchored on the bracketed-quoted host + the node=/rank= pair, so an
# arbitrary log line mentioning a hostname cannot be mistaken for a
# topology line. `local_rank` is not captured: it is always 0 here (only
# local_rank 0 logs) and says nothing about the span.
_SETUP_RX = re.compile(
    r"\['(?P<host>[^']+)'\]"
    r".*?\[node=(?P<node>\d+)/(?P<nnodes>\d+)\]"
    r".*?\[rank=(?P<rank>\d+)/(?P<last_rank>\d+)\]"
)


class RankHostMap:
    """Which host owned which ranks, as recovered from a log."""

    def __init__(self, first_rank_by_host: dict[str, int], world_size: int):
        self.world_size = world_size
        # Sort by first rank so adjacent hosts bound each other's span.
        self._hosts = sorted(first_rank_by_host.items(), key=lambda kv: kv[1])

    @property
    def hosts(self) -> list[str]:
        return [h for h, _ in self._hosts]

    def host_for_rank(self, rank: int) -> Optional[str]:
        """The host that owned *rank*, or ``None`` if unknown.

        ``None`` rather than a guess: a rank outside the recovered
        world, or a log whose topology lines were truncated, must not
        silently resolve to whichever host happens to sort last.
        """
        if rank < 0 or rank >= self.world_size:
            return None
        owner = None
        for host, first in self._hosts:
            if first <= rank:
                owner = host
            else:
                break
        return owner

    def ranks_for_host(self, host: str) -> range:
        """The contiguous rank span *host* owned (empty if unknown).

        Contiguity is an assumption, and it is the assumption most
        likely to be wrong: it holds for the block distribution PALS
        and srun use by default, and would break under a round-robin
        or custom rank-reorder. The map is only as good as that.
        """
        for i, (h, first) in enumerate(self._hosts):
            if h != host:
                continue
            end = (
                self._hosts[i + 1][1]
                if i + 1 < len(self._hosts)
                else self.world_size
            )
            return range(first, end)
        return range(0)


def parse_rank_host_map(log_text: str) -> Optional[RankHostMap]:
    """Recover the topology from *log_text*, or ``None`` if absent.

    ``None`` means "this log does not say", which callers must treat
    as unknown rather than as an empty mapping.
    """
    first_by_host: dict[str, int] = {}
    world = 0
    for m in _SETUP_RX.finditer(_ANSI_RX.sub("", log_text)):
        host = m.group("host")
        rank = int(m.group("rank"))
        # `rank=NN/LAST` is inclusive of the last rank, so world = last+1.
        world = max(world, int(m.group("last_rank")) + 1)
        # Keep the LOWEST rank seen per host: with EZPZ_LOG_ALL_RANKS=1
        # every rank logs, and the span start is the smallest.
        if host not in first_by_host or rank < first_by_host[host]:
            first_by_host[host] = rank
    if not first_by_host or world <= 0:
        return None
    return RankHostMap(first_by_host, world)


def hosts_for_ranks(log_text: str, ranks: "list[int]") -> list[str]:
    """Map *ranks* to the hosts that owned them, deduplicated.

    The intended use is turning a rank-tagged failure (``[rank23]:
    OSError``) into the host that produced it, for a human reading a
    postmortem.

    **This is not evidence that the host is faulty**, and must not be
    wired into bad-node retirement. Two reasons, both observed:

    * A rank-tagged error is frequently not local to that rank. DCP's
      collective save re-raises the originating failure on every rank,
      so a full filesystem surfaces as ``[rank23]`` on a host that did
      nothing wrong (ezpz #231).
    * A hard ``kill -9`` produces no rank-tagged line at all, so this
      returns nothing exactly when attribution is most wanted
      (ezpz #234).

    It closes the gap where a log names a rank and the reader cannot
    tell which machine that was. Nothing more.
    """
    tmap = parse_rank_host_map(log_text)
    if tmap is None:
        return []
    out: list[str] = []
    for r in ranks:
        h = tmap.host_for_rank(r)
        if h is not None and h not in out:
            out.append(h)
    return out


def parse_rank_host_map_file(path: "str | Path") -> Optional[RankHostMap]:
    """:func:`parse_rank_host_map` for a log on disk."""
    p = Path(path)
    if not p.exists():
        return None
    return parse_rank_host_map(p.read_text(errors="replace"))
