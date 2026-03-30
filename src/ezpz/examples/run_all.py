"""run_all.py — Run all ezpz examples sequentially and generate a report.

This is the Python replacement for ``scripts/run_benchmarks.sh``.  Each example
runs in its own subprocess via ``ezpz launch`` so that distributed state
(``setup_torch`` / ``cleanup``) is fully isolated between runs.

Usage::

    # Run all examples:
    python -m ezpz.examples.run_all

    # Run specific examples:
    python -m ezpz.examples.run_all --examples test fsdp vit

    # Override model size:
    python -m ezpz.examples.run_all --model small

    # Custom output directory:
    python -m ezpz.examples.run_all --outdir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Example definitions ──────────────────────────────────────────────────────
# Each entry mirrors a ``run_example`` call from the original bash script.
# Args containing ``{model}``, ``{bench_dir}``, or ``{timestamp}`` are
# formatted at runtime by ``build_command()``.

EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "test",
        "module": "ezpz.examples.test",
        "args": ["--model", "{model}"],
    },
    {
        "name": "fsdp",
        "module": "ezpz.examples.fsdp",
        "args": ["--model", "{model}"],
    },
    {
        "name": "vit",
        "module": "ezpz.examples.vit",
        "args": ["--model", "{model}", "--warmup", "0", "--fsdp"],
    },
    {
        "name": "fsdp_tp",
        "module": "ezpz.examples.fsdp_tp",
        "args": ["--model", "{model}", "--dataset", "stanfordnlp/imdb"],
    },
    {
        "name": "diffusion",
        "module": "ezpz.examples.diffusion",
        "args": ["--model", "{model}", "--dataset", "stanfordnlp/imdb"],
    },
    {
        "name": "hf",
        "module": "ezpz.examples.hf",
        "args": [
            "--dataset_name=eliplutchok/fineweb-small-sample",
            "--streaming",
            "--model_name_or_path", "meta-llama/Llama-3.2-1B",
            "--bf16=true",
            "--do_train=true",
            "--do_eval=true",
            "--report-to=wandb",
            "--logging-steps=1",
            "--max-steps=100",
            "--optim=adamw_torch",
            "--logging-first-step",
            "--include-for-metrics=inputs,loss",
            "--max-eval-samples=100",
            "--per_device_train_batch_size=1",
            "--per_device_eval_batch_size=1",
            "--block_size=2048",
            "--fsdp=auto_wrap",
            "--output_dir={bench_dir}/outputs/ezpz.hf/{timestamp}",
        ],
    },
    {
        "name": "hf_trainer",
        "module": "ezpz.examples.hf_trainer",
        "args": [
            "--dataset_name=eliplutchok/fineweb-small-sample",
            "--streaming",
            "--model_name_or_path", "meta-llama/Llama-3.2-1B",
            "--bf16=true",
            "--do_train=true",
            "--do_eval=true",
            "--report-to=wandb",
            "--logging-steps=1",
            "--max-steps=100",
            "--optim=adamw_torch",
            "--logging-first-step",
            "--include-for-metrics=inputs,loss",
            "--max-eval-samples=100",
            "--per_device_train_batch_size=1",
            "--per_device_eval_batch_size=1",
            "--block_size=2048",
            "--fsdp=auto_wrap",
            "--output_dir={bench_dir}/outputs/ezpz.hf_trainer/{timestamp}",
        ],
    },
]

ALL_EXAMPLE_NAMES = [e["name"] for e in EXAMPLES]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _git(cmd: list[str]) -> str:
    """Run a git command and return stripped stdout, or ``'unknown'``."""
    try:
        result = subprocess.run(
            ["git", *cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _python_eval(expr: str) -> str:
    """Evaluate a one-liner in the current Python and return stdout."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", expr],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "N/A"


def capture_env(bench_dir: Path) -> dict[str, Any]:
    """Gather environment metadata and write ``env.json``."""
    scheduler = "local"
    job_id = str(os.getpid())
    num_nodes = 1

    if os.environ.get("PBS_JOBID"):
        scheduler = "PBS"
        job_id = os.environ["PBS_JOBID"]
        nodefile = os.environ.get("PBS_NODEFILE", "")
        if nodefile and Path(nodefile).exists():
            num_nodes = len(Path(nodefile).read_text().splitlines())
        else:
            num_nodes = int(os.environ.get("PBS_NUM_NODES", 1))
    elif os.environ.get("SLURM_JOB_ID"):
        scheduler = "SLURM"
        job_id = os.environ["SLURM_JOB_ID"]
        num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    gpus_per_node = int(
        os.environ.get(
            "NGPU_PER_HOST",
            _python_eval(
                "import ezpz; print(ezpz.distributed.get_gpus_per_node())"
            ),
        )
        or 0
    )

    env_info: dict[str, Any] = {
        "git_commit": _git(["rev-parse", "--short", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "job_id": job_id,
        "scheduler": scheduler,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "total_gpus": num_nodes * gpus_per_node,
        "hostname": socket.gethostname(),
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "python": _python_eval(
            "import sys; print("
            "f'{sys.version_info.major}.{sys.version_info.minor}"
            ".{sys.version_info.micro}')"
        ),
        "torch": _python_eval("import torch; print(torch.__version__)"),
        "ezpz_version": _python_eval(
            "from ezpz.__about__ import __version__; print(__version__)"
        ),
    }

    env_path = bench_dir / "env.json"
    env_path.write_text(json.dumps(env_info, indent=2) + "\n", encoding="utf-8")
    print(f"Environment info written to {env_path}")
    return env_info


def build_command(
    example: dict[str, Any],
    model: str,
    bench_dir: Path,
    timestamp: str,
) -> list[str]:
    """Build the full subprocess command for an example."""
    formatted_args = [
        a.format(model=model, bench_dir=bench_dir, timestamp=timestamp)
        for a in example["args"]
    ]
    return [
        "ezpz", "launch",
        "python3", "-m", example["module"],
        *formatted_args,
    ]


def run_example(
    name: str,
    cmd: list[str],
    bench_dir: Path,
) -> dict[str, Any]:
    """Run an example as a subprocess, capturing output to a log file."""
    logfile = bench_dir / f"{name}.log"
    print()
    print("\u2550" * 64)
    print(f"  Running: {name}")
    print("\u2550" * 64)

    t0 = time.perf_counter()
    with logfile.open("w") as log_fh:
        proc = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = int(time.perf_counter() - t0)

    if proc.returncode == 0:
        print(f"  \u2713 {name} completed in {elapsed}s")
    else:
        print(f"  \u2717 {name} FAILED (exit {proc.returncode}) after {elapsed}s")

    return {
        "name": name,
        "exit_code": proc.returncode,
        "wall_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run all ezpz examples sequentially and generate a report.",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=ALL_EXAMPLE_NAMES,
        default=ALL_EXAMPLE_NAMES,
        metavar="NAME",
        help=f"Examples to run (default: all). Choices: {', '.join(ALL_EXAMPLE_NAMES)}",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Model size preset passed to examples (default: small).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/benchmarks/{timestamp}).",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = args.outdir or Path("outputs/benchmarks") / timestamp
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark output directory: {bench_dir}")

    # ── Capture environment ──────────────────────────────────────────────
    capture_env(bench_dir)

    # ── Run examples ─────────────────────────────────────────────────────
    timings_path = bench_dir / "timings.csv"
    timings_path.write_text("name,exit_code,wall_seconds\n", encoding="utf-8")

    selected = [e for e in EXAMPLES if e["name"] in args.examples]
    results: list[dict[str, Any]] = []

    for example in selected:
        cmd = build_command(
            example,
            model=args.model,
            bench_dir=bench_dir,
            timestamp=timestamp,
        )
        result = run_example(example["name"], cmd, bench_dir)
        results.append(result)

        with timings_path.open("a") as f:
            f.write(
                f"{result['name']},{result['exit_code']},{result['wall_seconds']}\n"
            )

    # ── Generate report ──────────────────────────────────────────────────
    print()
    print("\u2550" * 64)
    print("  Generating report")
    print("\u2550" * 64)

    from ezpz.examples.report import generate_report

    report = generate_report(bench_dir)
    report_path = bench_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {report_path}")
    print(f"\nDone. Results in: {bench_dir}")


if __name__ == "__main__":
    main()
