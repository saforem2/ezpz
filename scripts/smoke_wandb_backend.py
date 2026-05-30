#!/usr/bin/env python3
"""Manual smoke test for the WandbBackend → setup_wandb delegation.

Run from any machine that has wandb logged in (`wandb login` once).
The test creates a single wandb run, asserts that all 18 expected
keys land in run.config, and then finishes the run cleanly.

Usage:

    .venv/bin/python scripts/smoke_wandb_backend.py

Or directly in any wandb-authenticated environment:

    python -m scripts.smoke_wandb_backend

What it checks (this is the manual test plan item from PR #147 that
the mocked unit tests can't cover):

  1. WandbBackend actually delegates to setup_wandb in production
  2. verify_wandb() passes when there's a real API key
  3. wandb.init() succeeds (network reachable, project name valid)
  4. All 18 env-tracking keys end up readable on the live run
  5. WandbBackend(config=...) keys land at TOP level (not nested)

Exits 0 on success, 1 on failure with the missing-key list.
"""

from __future__ import annotations

import os
import sys


# Required keys after the PR #147 env-field expansion. Mirrors
# tests/test_tracker.py::test_env_fields_from_setup_wandb_applied.
REQUIRED_KEYS = [
    # Original set
    "hostname",
    "pytorch_backend",
    "world_size",
    "machine",
    "ezpz_version",
    "working_directory",
    "torch_version",
    # Timestamp fields
    "year",
    "month",
    "day",
    "tstamp",
    # Filtering / grouping dimensions
    "jobid",
    "scheduler",
    "num_nodes",
    "ranks_per_node",
    "device_type",
    # Debugging / postmortems
    "python_version",
    "ezpz_git_sha",
]


def main() -> int:
    print("=" * 70)
    print("WandbBackend → setup_wandb delegation smoke test")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"CWD:    {os.getcwd()}")
    print()

    # Pre-flight: bail before we touch the network if there's no key.
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            import netrc

            netrc_path = os.path.expanduser("~/.netrc")
            if os.path.isfile(netrc_path):
                auth = netrc.netrc(netrc_path).authenticators(
                    "api.wandb.ai"
                )
                if not auth:
                    raise RuntimeError("no entry for api.wandb.ai")
            else:
                raise RuntimeError("no ~/.netrc")
        except Exception as exc:
            print(f"✗ No wandb credentials found ({exc}).")
            print("  Run `wandb login` first, or set WANDB_API_KEY.")
            return 1

    from ezpz.tracker import WandbBackend

    # User config kwargs land at the TOP level (not run.config.config.*)
    # — that's the WandbBackend contract we're verifying.
    user_config = {"lr": 0.01, "batch_size": 32, "smoke_test": True}

    print("Creating wandb run via WandbBackend(...)")
    backend = WandbBackend(
        project_name="ezpz-smoke-tests",
        config=user_config,
        rank=0,
    )

    if backend.run is None:
        print("✗ WandbBackend.run is None — init failed silently.")
        print("  Check the wandb logs above for the underlying error.")
        return 1

    run = backend.run
    print(f"✓ Run created: {run.name} ({run.url})")
    print()

    # Pull the live config back. wandb.Config supports dict-like access.
    live_config = dict(run.config)

    # Check 1: top-level user keys
    print("Checking WandbBackend(config=...) lands at TOP level:")
    failures: list[str] = []
    for key, expected in user_config.items():
        actual = live_config.get(key)
        marker = "✓" if actual == expected else "✗"
        print(f"  {marker} run.config.{key} = {actual!r} (expected {expected!r})")
        if actual != expected:
            failures.append(
                f"top-level user key '{key}': "
                f"expected {expected!r}, got {actual!r}"
            )
    print()

    # Check 2: all 18 env-tracking keys present
    print(f"Checking {len(REQUIRED_KEYS)} env-tracking keys from setup_wandb:")
    missing = [k for k in REQUIRED_KEYS if k not in live_config]
    for key in REQUIRED_KEYS:
        present = key in live_config
        marker = "✓" if present else "✗"
        val = live_config.get(key)
        # jobid/ezpz_git_sha are allowed to be None outside a PBS job /
        # in pip installs — still need the KEY to be present though.
        val_repr = repr(val) if val is not None else "None (OK)"
        print(f"  {marker} run.config.{key:<20s} = {val_repr}")
        if not present:
            failures.append(f"missing env key: {key}")
    print()

    backend.finish()
    print(f"Run URL: {run.url}")
    print()

    if failures:
        print(f"✗ {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("✓ All checks passed.")
    if missing:
        # Shouldn't happen given the loop above, but be paranoid.
        print(f"  ({len(missing)} keys missing: {missing})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
