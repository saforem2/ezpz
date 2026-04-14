#!/usr/bin/env python3
"""Thin wrapper — delegates to ``ezpz.examples.report``.

Usage::

    python3 scripts/generate_report.py --outdir outputs/benchmarks/2026-03-16-103000
"""
from ezpz.examples.report import main

if __name__ == "__main__":
    main()
