#!/usr/bin/env python3
"""A trainer that fails on demand, for exercising ``--auto-retry``.

Run as a subprocess by ``tests/test_autoretry_faultinject.py``. It prints
a few progress lines and then dies in whichever way the test asked for.

Deliberately imports **nothing** from ezpz or torch: it must start in
milliseconds (the harness spawns it dozens of times) and must not trip
`conftest.py`'s `_no_rendezvous_leak` guard, which fails any test that
leaks `MASTER_ADDR`/`MASTER_PORT`.

Configured by environment rather than argv so the harness can vary one
knob without rebuilding a command line, and so ``FI_COUNTER`` can make
attempt N behave differently from attempt N+1 -- the idiom already used
by ``tests/test_failover_lib.sh`` and ``test_launch_watchdog.py``.

Every failure signature below is copied from a real captured log (see
``ezpz.failover.patterns`` and ``_CRASH_PATTERNS_RX`` in
``launch_autoretry.py``), not invented. A made-up signature would prove
only that the harness can match its own strings.

| env var       | meaning                                              |
| ------------- | ---------------------------------------------------- |
| `FI_COUNTER`  | file holding the attempt number (incremented here)   |
| `FI_MODE`     | which failure to inject (see `_MODES`)               |
| `FI_FAIL_ON`  | comma-separated attempts that fail; default all      |
| `FI_STEPS`    | how many progress lines to print first (default 2)   |
| `FI_MARKER`   | counter name in those lines: `step` (default) or `iter` |
| `FI_TRAILER`  | if set, print ezpz launch's `Execution finished with N.` |
| `FI_HOST`     | hostname to blame in `shepherd` mode                 |
| `FI_CKPT`     | checkpoint file; enables save/resume + restart timing |
| `FI_TOTAL`    | total steps to reach across all attempts (with FI_CKPT) |
| `FI_CKPT_EVERY` | save interval in steps (default 10)                |
| `FI_FAIL_AT`  | step to fail at, instead of after FI_STEPS lines      |
| `FI_STEP_MS`  | simulated cost of one step, milliseconds (default 0) |
"""

from __future__ import annotations

import json
import os
import sys
import time

# Process entry. `restart_seconds` is measured from HERE, matching
# fsdp_tp's `train/restart_seconds`, which is timed before setup_torch
# so it captures the full cold path rather than just the ckpt read.
_T_ENTRY = time.monotonic()

# A real Sunspot node name. The scraper's normalizer rejects anything
# that is not `x<d>c<d>s<d>b<d>n<d>`, so `h1`-style test names would be
# silently dropped and the named-host path would never be reached.
DEFAULT_HOST = "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"

# Exit code PBS walltime kills surface as (128 + SIGTERM).
WALLTIME_RC = 143


def _bump_counter(path: str) -> int:
    """Return this attempt's 1-based number, persisting across runs."""
    try:
        n = int(open(path).read().strip() or 0)
    except (OSError, ValueError):
        n = 0
    n += 1
    with open(path, "w") as fh:
        fh.write(str(n))
    return n


def _emit(line: str) -> None:
    # Column 0 matters: the shepherd pattern is `^`-anchored under
    # MULTILINE, so any prefix (timestamp, `[rank0]`) breaks the match.
    sys.stdout.write(line + "\n")
    sys.stdout.flush()



def _die(mode: str, host: str) -> int:
    """Emit the chosen failure signature and return its rc."""
    if mode == "shepherd":
        # PALS shepherd kill -- the one named-host signature reachable
        # off-ALCF (the gloo pattern yields an IP and needs `getent`).
        _emit(f"{host}: shepherd died from signal 9")
        rc = 1
    elif mode == "gloo_peer":
        _emit(
            "RuntimeError: [enforce fail at gloo/transport/tcp/pair.cc:598] "
            "Connection closed by peer [10.0.0.42]:53121"
        )
        rc = 1
    elif mode == "ur_oom":
        _emit(
            "[rank7]: RuntimeError: level_zero backend failed with "
            "error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)"
        )
        rc = 1
    elif mode == "rank_exit":
        # One rank's `exit 1` tears the job down: PALS reports the
        # aggregate as 143, indistinguishable from a walltime kill
        # except for this line.
        _emit(f"{host}: rank 7 exited with code 1")
        rc = WALLTIME_RC
    elif mode == "innocent_cascade":
        # A clean walltime kill SIGTERMs every rank. These lines look
        # alarming but must NOT be read as a bad node.
        for r in range(3):
            _emit(f"rank {r} died from signal 15")
        rc = WALLTIME_RC
    elif mode == "clean_walltime":
        _emit("normal training output, then the wallclock ran out")
        rc = WALLTIME_RC
    elif mode == "hang":
        # Go silent. The idle watchdog should SIGTERM us; the sleep is
        # long enough that a broken watchdog shows up as a timeout
        # rather than a pass.
        time.sleep(float(os.environ.get("FI_HANG_S", "120")))
        rc = 0
    elif mode == "sigkill":
        sys.stdout.flush()
        os.kill(os.getpid(), 9)  # -> Popen.poll() returns -9
        rc = 0  # unreachable
    elif mode == "silent_fail":
        # Fails with nothing a scraper can name: forces blind rotation.
        rc = 1
    else:  # pragma: no cover - guards a typo in a test
        raise SystemExit(f"_faultinject: unknown FI_MODE={mode!r}")
    return rc


def _run_checkpointed(attempt: int, mode: str, host: str) -> int:
    """A training loop that resumes from a checkpoint after a fault.

    Enabled by ``FI_CKPT``. The point is to measure what a restart
    actually costs -- ``restart_seconds``, timed from process entry to
    the first productive step, the same definition
    ``fsdp_tp``'s ``train/restart_seconds`` uses -- so the numbers on
    the docs page are measured rather than asserted.

    The "model" is a single integer. That is deliberate: this measures
    the loop's restart overhead, not tensor I/O, and a real model would
    bury a ~30ms signal under seconds of framework startup.
    """
    ckpt_path = os.environ["FI_CKPT"]
    total = int(os.environ.get("FI_TOTAL", "100"))
    every = int(os.environ.get("FI_CKPT_EVERY", "10"))
    step_ms = float(os.environ.get("FI_STEP_MS", "0"))
    # FI_FAIL_AT may be a comma-separated list: the Nth entry is used on
    # attempt N, so a single run can fail at different points in
    # training rather than repeating one scenario.
    _fa = [x.strip() for x in os.environ.get("FI_FAIL_AT", "").split(",") if x.strip()]
    fail_at = _fa[min(attempt - 1, len(_fa) - 1)] if _fa else None
    marker = os.environ.get("FI_MARKER", "step")

    start = 0
    resumed_from = None
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            start = json.load(fh)["step"]
        resumed_from = start
        _emit(f"resumed from checkpoint at {marker}={start}")

    restart_s = time.monotonic() - _T_ENTRY
    _emit(
        f"attempt={attempt} resumed_from={resumed_from} "
        f"restart_seconds={restart_s:.4f}"
    )

    for step in range(start, total):
        if step_ms:
            time.sleep(step_ms / 1000.0)
        _emit(f"{marker}={step} loss={1.0 / (step + 1):.4f}")
        if (step + 1) % every == 0:
            with open(ckpt_path, "w") as fh:
                json.dump({"step": step + 1}, fh)
        if fail_at is not None and step == int(fail_at) and _should_fail(
            attempt
        ):
            return _die(mode, host)

    _emit(f"training complete at {marker}={total}")
    return 0


def _should_fail(attempt: int) -> bool:
    spec = os.environ.get("FI_FAIL_ON", "").strip()
    if not spec:
        return True
    return attempt in {int(x) for x in spec.split(",") if x.strip()}


def main() -> int:
    counter = os.environ.get("FI_COUNTER")
    attempt = _bump_counter(counter) if counter else 1

    fail_on = os.environ.get("FI_FAIL_ON", "").strip()
    should_fail = (
        True
        if not fail_on
        else attempt in {int(x) for x in fail_on.split(",") if x.strip()}
    )

    host = os.environ.get("FI_HOST", DEFAULT_HOST)
    mode = os.environ.get("FI_MODE", "shepherd")
    n_steps = int(os.environ.get("FI_STEPS", "2"))

    # Progress first. `_PROGRESS_MARKER_RX` is `\bstep=\d+` and nothing
    # else, so the counter's NAME decides whether the loop believes
    # training started. `FI_MARKER=iter` reproduces what every real ezpz
    # example actually prints (minimal.py:92, test.py:401 both use
    # "iter"), which does NOT match -- that is the point of the option.
    if os.environ.get("FI_CKPT"):
        return _run_checkpointed(attempt, mode, host)

    marker = os.environ.get("FI_MARKER", "step")
    for i in range(n_steps):
        _emit(f"{marker}={attempt * 100 + i} loss={1.0 / (i + 1):.4f}")

    if not should_fail:
        _emit("training complete")
        if os.environ.get("FI_TRAILER"):
            _emit("Execution finished with 0.")
        return 0

    rc = _die(mode, host)
    if os.environ.get("FI_TRAILER"):
        _emit(f"Execution finished with {rc}.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
