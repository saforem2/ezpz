"""Minimal PyTorch-profiler example: trace a distributed step on any device.

The smallest thing that produces a real ``torch.profiler`` trace from a
distributed ezpz run. Launch it with:

    ezpz launch python3 -m ezpz.examples.profiler --profile

which writes a Chrome trace per profiled step and logs a key-averages
table. Load the JSON in ``chrome://tracing``, `Perfetto
<https://ui.perfetto.dev>`_, or TensorBoard.

Why this module exists
----------------------
``fsdp``, ``vit``, ``diffusion``, ``fsdp_tp``, ``hf`` and ``hf_trainer``
all already accept ``--profile`` — but each is a full training script, so
"how do I profile on Aurora" is answered by a ~900-line file. This is the
same wiring with nothing else in it: a synthetic MLP, one loop, and the
three lines that matter.

    with profiling_context_from_args(args, outdir) as prof:
        for step in range(steps):
            ...                       # forward / backward / step
            if prof is not None:
                prof.step()           # advance the profiler schedule

Device coverage is automatic. :func:`ezpz.profile.get_torch_profiler`
picks the activity set from what is available — ``ProfilerActivity.XPU``
on Aurora and Sunspot, ``CUDA`` on Polaris and Perlmutter, ``CPU``
everywhere — so the same command profiles on every system.

``prof.step()`` is not optional: the schedule
(``wait``/``warmup``/``active``/``repeat``) only advances when it is
called. Omit it and the profiler stays in ``wait`` forever and writes no
trace at all — a silent no-op rather than an error.

**Every rank profiles by default** (``rank_zero_only`` defaults to
``False``); pass ``--rank-zero-only`` to restrict to rank 0. On a
many-rank job that matters: an 8-rank Polaris run wrote 24 traces, and
Aurora runs 12 ranks per node. See
:func:`ezpz.profile.get_profiling_context`.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

import ezpz
from ezpz.cli.flags import add_profiling_args
from ezpz.examples import get_example_outdir
from ezpz.profile import profiling_context_from_args

logger = ezpz.get_logger(__name__)

MODULE_NAME = "ezpz.examples.profiler"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the profiler example.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments, including the shared profiling flag-set from
        :func:`ezpz.cli.flags.add_profiling_args`.
    """
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Minimal distributed training loop with torch profiling.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Training steps to run (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-rank batch size (default: 64).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="Width of each hidden layer (default: 1024).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of hidden layers (default: 4).",
    )
    # --profile, --pyinstrument-profiler, --pytorch-profiler-{wait,warmup,
    # active,repeat}, --rank-zero-only, --record-shapes, --with-stack, ...
    add_profiling_args(parser)
    return parser.parse_args(argv)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build a small MLP on the local accelerator.

    Args:
        args: Parsed arguments supplying ``hidden_size`` and ``layers``.

    Returns:
        The model, moved to the current device and DDP-wrapped when the
        world size is greater than one.
    """
    device_type = ezpz.get_torch_device_type()
    sizes = [args.hidden_size] * args.layers
    from ezpz.models.minimal import SequentialLinearNet

    model = SequentialLinearNet(
        input_dim=args.hidden_size,
        output_dim=args.hidden_size,
        sizes=sizes,
    )
    model.to(device_type)
    if ezpz.get_world_size() > 1:
        model = ezpz.distributed.wrap_model_for_ddp(model)
    return model


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    outdir: os.PathLike | str,
) -> None:
    """Run the training loop inside a profiling context.

    The profiler is a ``nullcontext`` (so ``prof`` is ``None``) unless
    ``--profile`` or ``--pyinstrument-profiler`` was passed, which keeps
    the unprofiled path free of overhead.

    Args:
        model: Model to train.
        optimizer: Optimizer for ``model``.
        args: Parsed arguments, forwarded to the profiling context.
        outdir: Directory that receives traces and metrics.
    """
    device_type = ezpz.get_torch_device_type()
    bsize, isize = args.batch_size, args.hidden_size
    model.train()

    with profiling_context_from_args(args, outdir) as prof:
        for step in range(args.steps):
            t0 = time.perf_counter()
            x = torch.rand((bsize, isize)).to(device_type)
            y = model(x)
            loss = ((y - x) ** 2).sum()
            dtf = (t1 := time.perf_counter()) - t0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dtb = time.perf_counter() - t1

            # Advance the profiler schedule. Without this the schedule
            # never leaves `wait` and no trace is ever emitted.
            #
            # The `is not None` check is load-bearing beyond the
            # unprofiled case: PyInstrumentProfiler.__enter__ returns
            # None (it has no step()), so --pyinstrument-profiler lands
            # here too and must skip the call rather than AttributeError.
            if prof is not None:
                prof.step()

            # One line per host, not per rank: a 96-rank job logging
            # from every rank is unreadable (see AGENTS.md).
            if step % 5 == 0 and ezpz.get_local_rank() == 0:
                logger.info(
                    "iter=%d loss=%.6f dtf=%.6f dtb=%.6f",
                    step,
                    loss.item(),
                    dtf,
                    dtb,
                )


def main(argv: list[str] | None = None) -> None:
    """Entrypoint for the profiler example."""
    args = parse_args(argv)
    ezpz.setup_torch(seed=int(os.environ.get("SEED", 0)))
    outdir = get_example_outdir(MODULE_NAME)
    logger.info("Outputs will be saved to %s", outdir)

    model = build_model(args)
    optimizer = torch.optim.Adam(model.parameters())
    train(model, optimizer, args, outdir)

    if args.pytorch_profiler and ezpz.get_rank() == 0:
        traces = sorted(Path(outdir).glob("torch-profiler-*.json"))
        if traces:
            logger.info(
                "Wrote %d trace(s) to %s -- load in chrome://tracing, "
                "https://ui.perfetto.dev, or TensorBoard",
                len(traces),
                outdir,
            )
        else:
            # Reachable when the schedule never hit an `active` step.
            # Compute the threshold from the ACTUAL flags: quoting the
            # defaults misdiagnoses a customized schedule (e.g.
            # --pytorch-profiler-wait 50 --steps 20 needs 55, not 6).
            needed = (
                args.pytorch_profiler_wait
                + args.pytorch_profiler_warmup
                + args.pytorch_profiler_active
            )
            logger.warning(
                "--profile was passed but no trace was written. This "
                "schedule needs wait+warmup+active = %d+%d+%d = %d "
                "steps before the first trace; --steps is %d.",
                args.pytorch_profiler_wait,
                args.pytorch_profiler_warmup,
                args.pytorch_profiler_active,
                needed,
                args.steps,
            )


if __name__ == "__main__":
    main()
