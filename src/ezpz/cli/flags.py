"""Shared argparse builders for ezpz CLI entrypoints."""

from __future__ import annotations

import argparse


class _RawDescAndDefaultsFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Combine raw multi-line descriptions with auto-appended defaults."""


def _non_negative_int(value: str) -> int:
    """argparse type validator that rejects negative integers.

    Used by `--timeout` and `--retries` on `ezpz launch` — both treat
    `0` as a meaningful value (off / no-retry) but negative numbers
    have no sensible semantics. Reject them at parse time with a
    clear message instead of silently coercing them away.
    """
    try:
        as_int = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from exc
    if as_int < 0:
        raise argparse.ArgumentTypeError(
            f"must be >= 0 (got {as_int})"
        )
    return as_int


def _spare_nodes_value(value: str) -> "int | str":
    """argparse type for ``--spare-nodes``.

    Accepts either the literal string ``"auto"`` (derive from
    ``total_pbs_nodes - ceil($nproc / $ppn)``) or a non-negative
    integer. Returning
    a union here keeps the auto/explicit distinction visible to the
    downstream resolver — `launch.py` checks for the string before
    treating the value as a count.
    """
    if value == "auto":
        return "auto"
    return _non_negative_int(value)


def build_test_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``ezpz test`` (ezpz.examples.test)."""
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            """
            ezpz test: A simple PyTorch distributed smoke test

            Trains a simple MLP on MNIST dataset using DDP.

            NOTE: `ezpz test` is a lightweight wrapper around:

            ```bash
            ezpz launch python3 -m ezpz.examples.test
            ```
            """
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline length",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="deepspeed_config.json",
        help="Deepspeed config file",
    )
    parser.add_argument(
        "--cp",
        type=int,
        default=1,
        help="Context parallel size",
    )
    parser.add_argument(
        "--backend",
        required=False,
        type=str,
        default="DDP",
        help="Backend (DDP, DeepSpeed, etc.)",
    )
    parser.add_argument(
        "--pyinstrument-profiler",
        action="store_true",
        help="Profile the training loop",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=False,
        dest="pytorch_profiler",
        required=False,
        action="store_true",
        help="Use PyTorch profiler",
    )
    parser.add_argument(
        "--rank-zero-only",
        action="store_true",
        help="Run profiler only on rank 0",
    )
    parser.add_argument(
        "--pytorch-profiler-wait",
        type=int,
        default=1,
        help="Wait time before starting the PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-warmup",
        type=int,
        default=2,
        help="Warmup iterations for the PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-active",
        type=int,
        default=3,
        help="Active iterations for the PyTorch profiler",
    )
    parser.add_argument(
        "--pytorch-profiler-repeat",
        type=int,
        default=5,
        help="Repeat iterations for the PyTorch profiler",
    )
    # The next five flags default to True and were declared as
    # `action="store_true"`, which made them unreachable as False —
    # passing the flag and not passing it both produced True.
    # BooleanOptionalAction generates the matching --no-* form so users
    # can actually disable each one (e.g. `--no-with-stack`).
    parser.add_argument(
        "--profile-memory",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Profile memory usage",
    )
    parser.add_argument(
        "--record-shapes",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Record shapes in the profiler",
    )
    parser.add_argument(
        "--save-datasets",
        action="store_true",
        default=False,
        help="Save datasets",
    )
    parser.add_argument(
        "--with-stack",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include stack traces in the profiler",
    )
    parser.add_argument(
        "--with-flops",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include FLOPs in the profiler",
    )
    parser.add_argument(
        "--with-modules",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include module information in the profiler",
    )
    parser.add_argument(
        "--acc-events",
        default=False,
        action="store_true",
        help="Accumulate events in the profiler",
    )
    parser.add_argument(
        "--train-iters",
        "--train_iters",
        type=int,
        default=200,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        # Choices kept in sync with `ezpz.examples.test.MODEL_PRESETS`
        # and `MODEL_ALIASES`. If you add a size to test.py, add it
        # here too (this parser is in cli/flags.py, separate module
        # from test.py — couldn't import MODEL_PRESETS without
        # creating a circular dep).
        choices=[
            "debug", "small", "medium", "large",
            "xl", "xlarge", "extra-large",
            "xxl", "xxlarge", "extra-extra-large",
            "xxxl", "xxxlarge", "extra-extra-extra-large",
        ],
        help=(
            "Model size preset for the smoke test. "
            "xl/xxl/xxxl accept long-form aliases too: "
            "`xlarge`/`extra-large`, `xxlarge`/`extra-extra-large`, etc."
        ),
    )
    parser.add_argument(
        "--log-freq",
        "--log_freq",
        type=int,
        default=1,
        help="Logging frequency",
    )
    parser.add_argument(
        "--print-freq",
        "--print_freq",
        type=int,
        default=10,
        help="Printing frequency",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=28 * 28,
        help="Input size",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=10,
        help="Output size",
    )
    parser.add_argument(
        "--layer-sizes",
        help="Comma-separated list of layer sizes",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[512, 256, 128],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Data type (fp16, float16, bfloat16, bf16, float32, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to use for training (e.g., mnist).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Directory to cache dataset downloads.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers to use.",
    )
    parser.add_argument(
        "--no-distributed-history",
        action="store_true",
        help="Disable distributed history aggregation",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile after FSDP/DDP wrap.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode (only used when --compile is set). "
            "`default` is safest. `reduce-overhead` enables cudagraphs "
            "for small models / large batches. `max-autotune` does "
            "extensive kernel search - slow startup, fastest steady state."
        ),
    )
    return parser


def build_launch_parser(
    *, prog: str | None = None, include_command: bool = True
) -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``ezpz launch``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Launch a command on the current PBS/SLURM job.\n"
        "\n"
        "Additional `<launcher flags>` can be passed through directly\n"
        "to the launcher by including '--' as a separator before\n"
        "the command.\n"
        "\n"
        "Examples:\n"
        "\n"
        "    ezpz launch <launcher flags> -- <command> <args>\n"
        "\n"
        "    ezpz launch -n 8 -ppn 4 --verbose --tag-output -- python3 -m ezpz.examples.fsdp_tp\n"
        "\n"
        "    ezpz launch --nproc 8 -x EZPZ_LOG_LEVEL=DEBUG -- python3 my_script.py --my-arg val\n"
        if include_command
        else "Launch a command on the current PBS/SLURM job.",
        # ),
        # description="\n".join(
        #     [
        #         "Launch a command on the current PBS/SLURM job.",
        #         "",
        #         "Additional `<launcher flags>` can be passed through directly",
        #         "to the launcher by including '--' as a separator before ",
        #         "the command.",
        #         "Examples:",
        #         "\t$ # ezpz launch <launcher flags> -- <command> <args>"
        #         "\t$ ezpz launch --nproc 8 -x EZPZ_LOG_LEVEL=DEBUG -- python3 my_script.py --my-arg val",
        #     ]
        # )
        formatter_class=_RawDescAndDefaultsFormatter,
    )
    parser.add_argument(
        "--print-source",
        action="store_true",
        help="Print the location of the launch CLI source and exit.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        help="Deprecated: output filtering has been removed. This flag is ignored.",
    )
    parser.add_argument(
        "-n",
        "-np",
        "--n",
        "--np",
        "--nproc",
        "--world_size",
        "--nprocs",
        type=int,
        dest="nproc",
        default=-1,
        help="Number of processes.",
    )
    parser.add_argument(
        "-ppn",
        "--ppn",
        "--nproc_per_node",
        type=int,
        default=-1,
        dest="nproc_per_node",
        help="Processes per node.",
    )
    parser.add_argument(
        "-nh",
        "--nh",
        "--nhost",
        "--nnode",
        "--nnodes",
        "--nhosts",
        "--nhosts",
        type=int,
        default=-1,
        dest="nhosts",
        help="Number of nodes to use.",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        dest="hostfile",
        help="Hostfile to use for launching.",
    )
    parser.add_argument(
        "--cpu-bind",
        type=str,
        default=None,
        dest="cpu_bind",
        help=(
            "CPU binding value to pass to the launcher. "
            "Takes precedence over CPU_BIND when both are specified."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=_non_negative_int,
        default=None,
        dest="idle_timeout_s",
        help=(
            "Idle-output watchdog timeout in seconds. If the launched "
            "process produces no output (on stdout OR stderr — they "
            "are merged) for this many seconds, send SIGTERM (then "
            "SIGKILL after a 10s grace period) and exit with code 124. "
            "NOT a total walltime — the process can run indefinitely "
            "as long as it keeps emitting output on either stream. "
            "Off by default, BUT defaults to 1800 (30 min) when "
            "--auto-retry is set (matches FAILOVER_IDLE_TIMEOUT in "
            "src/ezpz/bin/failover.sh — silent xccl hangs otherwise "
            "burn the full PBS walltime). Pass 0 explicitly to disable "
            "even under --auto-retry. Useful for catching collective "
            "hangs (e.g. xccl silent deadlock on XPU)."
        ),
    )
    parser.add_argument(
        "--retries",
        type=_non_negative_int,
        default=0,
        dest="retries",
        help=(
            "Re-execute the launched command up to N times on non-zero "
            "exit (including a watchdog kill, exit 124). Applies "
            "exponential backoff between attempts (5s, 10s, 20s, ..., "
            "capped at 60s). Default: 0 (no retry)."
        ),
    )
    parser.add_argument(
        "--auto-retry",
        action="store_true",
        dest="auto_retry",
        help=(
            "Run with automatic bad-node failover. On every non-zero "
            "exit (including watchdog kill, exit 124, and walltime-"
            "racing crashes that surface as 143), scrape the log for "
            "known bad-node signatures (Aurora PALS shepherd-9, gloo "
            "peer-closed) and either swap the named hosts for spares "
            "or rotate one spare blindly. Loops until success, a real "
            "walltime hit, spare exhaustion, two consecutive attempts "
            "with zero training progress, or SIGINT. "
            "Requires --nproc to be set explicitly. "
            "Mutually exclusive with --retries."
        ),
    )
    parser.add_argument(
        "--spare-nodes",
        type=_spare_nodes_value,
        default=None,
        dest="spare_nodes",
        help=(
            "Number of spare nodes to reserve for --auto-retry "
            "swap-ins. Pass 'auto' (the default when --auto-retry is "
            "set) to derive from total_pbs_nodes - ceil($nproc / $ppn) "
            "(ranks → hosts, since --nproc counts ranks). Pass an "
            "integer for an explicit count. Ignored when --auto-retry "
            "is not set."
        ),
    )
    parser.add_argument(
        "--max-failover-retries",
        type=_non_negative_int,
        default=None,
        dest="max_failover_retries",
        help=(
            "Upper bound on --auto-retry attempts. Default: unbounded "
            "— termination is governed by the matrix in "
            "docs/cli/launch/index.md (success, walltime, exhaustion, "
            "stuck-pre-training, or SIGINT)."
        ),
    )
    if include_command:
        parser.add_argument(
            "command",
            nargs=argparse.REMAINDER,
            help="Command (and arguments) to execute. Use '--' to separate options when needed.",
        )
    return parser


def build_doctor_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``ezpz doctor``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inspect the current environment for ezpz launch readiness.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-friendly text.",
    )
    return parser


def build_generate_parser(
    *, prog: str | None = None
) -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``ezpz generate``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate text using a model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type to use for the model.",
    )
    return parser
