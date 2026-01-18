"""Shared argparse builders for ezpz CLI entrypoints."""

from __future__ import annotations

import argparse


def build_test_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``ezpz test`` (ezpz.examples.test)."""
    parser = argparse.ArgumentParser(
        prog=prog,
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
    parser.add_argument(
        "--profile-memory",
        default=True,
        action="store_true",
        help="Profile memory usage",
    )
    parser.add_argument(
        "--record-shapes",
        default=True,
        action="store_true",
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
        action="store_true",
        help="Include stack traces in the profiler",
    )
    parser.add_argument(
        "--with-flops",
        default=True,
        action="store_true",
        help="Include FLOPs in the profiler",
    )
    parser.add_argument(
        "--with-modules",
        default=True,
        action="store_true",
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Filter output lines by these strings.",
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
