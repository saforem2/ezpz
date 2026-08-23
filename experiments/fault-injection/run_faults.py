#!/usr/bin/env python3
"""Drive `--auto-retry` through injected faults and record what happened.

Runs the real `run_with_auto_retry` loop against `tests/_faultinject.py`,
a trainer that checkpoints and then dies on cue. Emits one JSONL record
per attempt so `plot_faults.py` can chart progress against wall-clock and
annotate what each fault cost.

This is the local, allocation-free counterpart to
`experiments/checkpoint-restart/`, which measures the same thing on two
Sunspot nodes with a real 23 GB model. The numbers here are milliseconds
where those are seconds -- the point is the *shape* of the recovery and
the loop's decisions, not the I/O cost.

    python3 experiments/fault-injection/run_faults.py --out run.jsonl

Every fault signature comes from `ezpz.failover.patterns`; none is
invented.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FAULT_SCRIPT = REPO / "tests" / "_faultinject.py"

# A real Sunspot node name: the scraper's normalizer rejects anything
# that is not x<d>c<d>s<d>b<d>n<d>, so a placeholder would never be
# matched and the named-host path would never be exercised.
BAD_HOST = "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"

_STEP_RX = re.compile(r"\bstep=(\d+)")
_RESTART_RX = re.compile(
    r"attempt=(\d+) resumed_from=(\S+) restart_seconds=([0-9.]+)"
)


def _parse_attempt(log: Path, wall_start: float) -> dict:
    """Pull one attempt's shape out of its log."""
    text = log.read_text(errors="replace")
    steps = [int(m) for m in _STEP_RX.findall(text)]
    rec: dict = {
        "log": log.name,
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "n_steps": len(steps),
        "resumed_from": None,
        "restart_seconds": None,
        "fault": None,
    }
    m = _RESTART_RX.search(text)
    if m:
        rec["attempt"] = int(m.group(1))
        rec["resumed_from"] = (
            None if m.group(2) == "None" else int(m.group(2))
        )
        rec["restart_seconds"] = float(m.group(3))
    # Name the fault by its signature, the way an operator would read it.
    for needle, label in (
        ("shepherd died from signal 9", "shepherd_sig9"),
        ("Connection closed by peer", "gloo_peer"),
        ("UR_RESULT_ERROR_OUT_OF_RESOURCES", "ur_oom"),
        ("exited with code 1", "rank_exit"),
    ):
        if needle in text:
            rec["fault"] = label
            break
    rec["mtime_offset_s"] = log.stat().st_mtime - wall_start
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fault_run.jsonl", type=Path)
    ap.add_argument("--total-steps", type=int, default=120)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument(
        "--fail-at",
        default="35",
        help="step to fail at; comma-separated for a different point "
        "per attempt (e.g. 60,145,240)",
    )
    ap.add_argument(
        "--fail-on",
        default="1,2",
        help="attempts that should fail (comma-separated)",
    )
    ap.add_argument(
        "--step-ms",
        type=float,
        default=8.0,
        help="simulated per-step cost, so the chart has a time axis",
    )
    ap.add_argument("--mode", default="shepherd")
    ap.add_argument("--workdir", type=Path, default=None)
    args = ap.parse_args()

    from ezpz.launch_autoretry import (
        AutoRetryConfig,
        NodeAllocation,
        run_with_auto_retry,
    )

    work = args.workdir or Path(
        os.environ.get("TMPDIR", "/tmp")
    ) / f"fault-inject-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    counter = work / "attempts.txt"
    counter.write_text("0")
    ckpt = work / "ckpt.json"
    if ckpt.exists():
        ckpt.unlink()

    for k, v in {
        "FI_COUNTER": str(counter),
        "FI_CKPT": str(ckpt),
        "FI_TOTAL": str(args.total_steps),
        "FI_CKPT_EVERY": str(args.ckpt_every),
        "FI_FAIL_AT": str(args.fail_at),
        "FI_FAIL_ON": args.fail_on,
        "FI_MODE": args.mode,
        "FI_STEP_MS": str(args.step_ms),
        "FI_HOST": BAD_HOST,
    }.items():
        os.environ[k] = v

    cfg = AutoRetryConfig(
        cmd=[sys.executable, str(FAULT_SCRIPT)],
        log_dir=work / "logs",
        # A live watchdog also shortens the poll interval to
        # min(1.0, max(0.1, ...)), so attempt wall-time reflects the
        # workload instead of being quantized to the 1.0s idle poll.
        idle_timeout_s=30,
        max_failover_retries=None,
        machine="sunspot",
    )
    alloc = NodeAllocation.from_full_nodelist(
        [BAD_HOST, "spare-1", "spare-2", "spare-3"],
        1,
        work / "active.hostfile",
        work / "bad_nodes.txt",
    )

    # Real backoff would put 5s and 10s of dead air into the chart and
    # say nothing about recovery cost. Zero the RETRY backoff only --
    # not time.sleep, which is also the watchdog's poll.
    import ezpz.launch_autoretry as M

    M._backoff_for_attempt = lambda _n: 0.0

    # Wrap the runner to time each attempt from the OUTSIDE. The
    # in-process `restart_seconds` omits interpreter startup and process
    # spawn -- on a real job that is most of the restart, so reporting
    # only the in-process figure would understate recovery by orders of
    # magnitude.
    orig_runner = M._run_attempt_with_tee
    spawn_times: list[float] = []

    def _timed(cmd, log_path, idle_timeout_s):
        t = time.monotonic()
        try:
            return orig_runner(cmd, log_path, idle_timeout_s)
        finally:
            spawn_times.append(time.monotonic() - t)

    M._run_attempt_with_tee = _timed

    t0 = time.monotonic()
    wall_start = time.time()
    rc = run_with_auto_retry(cfg, alloc)
    elapsed = time.monotonic() - t0
    M._run_attempt_with_tee = orig_runner

    logs = sorted((work / "logs").glob("attempt-*.log"))
    attempts = [_parse_attempt(p, wall_start) for p in logs]
    for i, a in enumerate(attempts, 1):
        a.setdefault("attempt", i)
        if i <= len(spawn_times):
            a["attempt_wall_s"] = round(spawn_times[i - 1], 4)

    summary = {
        "kind": "summary",
        "rc": rc,
        "elapsed_s": round(elapsed, 4),
        "n_attempts": len(attempts),
        "total_steps": args.total_steps,
        "ckpt_every": args.ckpt_every,
        "step_ms": args.step_ms,
        "mode": args.mode,
        "bad_nodes": (work / "bad_nodes.txt").read_text().split(),
        "final_hostfile": (work / "active.hostfile").read_text().split(),
    }

    with args.out.open("w") as fh:
        for a in attempts:
            fh.write(json.dumps({"kind": "attempt", **a}) + "\n")
        fh.write(json.dumps(summary) + "\n")

    print(f"rc={rc}  attempts={len(attempts)}  elapsed={elapsed:.2f}s")
    for a in attempts:
        print(
            f"  attempt {a['attempt']}: steps {a['first_step']}..."
            f"{a['last_step']}  resumed_from={a['resumed_from']}  "
            f"restart={a['restart_seconds']}  "
            f"wall={a.get('attempt_wall_s')}  fault={a['fault']}"
        )
    print(f"  bad nodes retired: {summary['bad_nodes']}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
