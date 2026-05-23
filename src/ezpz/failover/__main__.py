"""CLI entry point: ``python -m ezpz.failover <log-file>``.

Drop-in replacement for the old ``scrape_bad_nodes.py`` script that
lived in torchtitan. Reads one log file, prints one bad hostname per
line on stdout (deduplicated, first-seen order). Exits non-zero on
usage errors; exits 0 even when nothing was matched (silence here
means "no actionable hostname", not "log was clean").

Usage:
    python -m ezpz.failover <log-file>
    python -m ezpz.failover --machine aurora <log-file>
    python -m ezpz.failover --explain <log-file>   # per-pattern breakdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ezpz.failover.scrape import (
    _collect_all_matches_for_debug,
    scrape_bad_nodes,
)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ezpz.failover",
        description=(
            "Scrape a training log for bad-node hostnames. Prints one "
            "hostname per line. Exit 0 even when nothing matched."
        ),
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the training log to scrape.",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help=(
            "Override machine detection (e.g. 'aurora'). Defaults to "
            "the result of ezpz.get_machine() lower-cased."
        ),
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Print a per-pattern breakdown to stderr in addition to "
            "the deduplicated hostnames on stdout. Useful when "
            "postmorteming an unfamiliar failure mode."
        ),
    )
    args = parser.parse_args(argv)

    if not args.log_file.is_file():
        print(f"log file not found: {args.log_file}", file=sys.stderr)
        return 1

    if args.explain:
        per_pattern = _collect_all_matches_for_debug(
            args.log_file, machine=args.machine
        )
        print("[explain] per-pattern matches:", file=sys.stderr)
        for name, hosts in per_pattern.items():
            print(f"  {name}: {hosts or '<none>'}", file=sys.stderr)

    hosts = scrape_bad_nodes(args.log_file, machine=args.machine)
    for host in hosts:
        print(host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
