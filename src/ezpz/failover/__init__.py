"""Bad-node failover support for distributed training on HPC clusters.

This package provides the primitives the (forthcoming) failover
wrapper builds on:

  - :func:`scrape_bad_nodes` — parse a training log for known
    failure-mode signatures and return the bad hostnames.
  - :mod:`ezpz.failover.patterns` — per-machine pattern registry
    (Aurora bundled, others addable).

The shell-level orchestration (split nodefile, yeet env, swap in
spares, retry) lives in ``src/ezpz/bin/failover.sh`` and is
introduced in a separate PR.
"""

from __future__ import annotations

from ezpz.failover.scrape import scrape_bad_nodes

__all__ = ["scrape_bad_nodes"]
