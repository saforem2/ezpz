"""Lightweight argparse builder for ezpz.test_dist."""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``ezpz.test_dist``."""
    parser = argparse.ArgumentParser(
        description="Training configuration parameters"
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
