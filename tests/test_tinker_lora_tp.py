"""LoRA under REAL tensor parallelism, on a real 2-rank mesh.

Why this file exists at all: every other LoRA test runs single-rank, and
single-rank cannot see these bugs. A ``world_size=1`` mesh makes every
placement trivially satisfiable, so a plan that mislabels a sharded
activation as replicated passes locally and fails at ``tp=2``. Worse,
``ColwiseParallel``'s input hook calls ``DTensor.from_local(...,
run_check=False)``, so torch *believes* a mislabeled tensor rather than
raising -- a single-rank probe reports "OK" for a config that is wrong.

Two bugs were found this way, both invisible below world_size=2:

1. ``A``/``B`` styles were hardcoded to ``ColwiseParallel``. Under a
   ``RowwiseParallel`` base (``attention.wo``) the base's input arrives
   sharded on the feature dim, while a Colwise ``A`` declares it
   replicated and keeps a full-width weight::

       Sharding propagation failed for aten.mm.default(
           Spec(f32[16, 64](R)), Spec(f32[128, 8](S(1))))

2. ``lora_tp_plan`` retargeted ``output`` unconditionally, but
   ``--lora-target`` defaults to ``attn,mlp``, which leaves the
   unembedding unwrapped. ``parallelize_module`` ignores plan keys that
   match no module **without raising**, so ``output`` silently stayed an
   unsharded ``nn.Linear`` and died on its first DTensor input.

The gate is numerical, not "no exception": tp=2 output must match tp=1,
and every trainable parameter must receive a gradient. A layout bug can
produce a perfectly runnable wrong answer.

Uses ``mp.spawn`` rather than ``torchrun``: it needs no rendezvous
socket, which keeps this runnable on a laptop (torchrun's IPv6
rendezvous hangs on macOS) in about 30s.
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="mp.spawn + gloo fixture is POSIX-only"
)

RANK = 8
ALPHA = 16
PORT = "29873"


def _build(train_attn: bool, train_mlp: bool, train_unembed: bool):
    from ezpz.models.llama import ModelArgs, Transformer
    from ezpz.tinker.lora import LoraConfig, apply_lora

    torch.manual_seed(1234)
    cfg = ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=8,
        n_kv_heads=8,
        vocab_size=256,
        multiple_of=16,
        hidden_dim=256,
        max_seq_len=64,
        depth_init=True,
    )
    model = Transformer.from_model_args(cfg)
    model.init_weights()
    model = apply_lora(
        model,
        LoraConfig(
            rank=RANK,
            alpha=ALPHA,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        ),
        verbose=False,
    )
    # B is zero-init by design, which would make the adapter branch a
    # no-op and hide every layout bug in it. Give it real values.
    torch.manual_seed(7)
    for mod in model.modules():
        if type(mod).__name__ == "LoRALinear":
            nn.init.normal_(mod.B.weight, std=0.02)
    return model


def _plans():
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PrepareModuleInput,
        RowwiseParallel,
        SequenceParallel,
    )

    root = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1), output_layouts=Replicate()
        ),
    }
    block = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }
    return root, block


def _worker(rank, ws, flags, outdir):
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=PORT,
        RANK=str(rank),
        WORLD_SIZE=str(ws),
    )
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import parallelize_module

        from ezpz.tinker.lora import lora_tp_plan

        model = _build(*flags)
        tokens = torch.arange(16).reshape(1, 16) % 256

        if ws > 1:
            mesh = init_device_mesh("cpu", (ws,))
            root, block = _plans()
            parallelize_module(model, mesh, lora_tp_plan(root, model))
            for blk in model.layers:
                blk.attention.n_heads //= ws
                blk.attention.n_kv_heads //= ws
                parallelize_module(blk, mesh, lora_tp_plan(block, blk))

        out = model(tokens)
        if hasattr(out, "full_tensor"):
            out = out.full_tensor()
        loss = out.float().pow(2).mean()
        loss.backward()

        if rank == 0:
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.save(
                {
                    "out": out.detach().flatten()[:64].clone(),
                    "loss": float(loss),
                    "with_grad": sum(
                        1 for p in trainable if p.grad is not None
                    ),
                    "trainable": len(trainable),
                },
                os.path.join(outdir, f"res_{ws}.pt"),
            )
    finally:
        dist.destroy_process_group()


def _run(ws, flags, outdir):
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(ws, flags, outdir), nprocs=ws, join=True)
    return torch.load(os.path.join(outdir, f"res_{ws}.pt"), weights_only=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "train_attn,train_mlp,train_unembed",
    [
        (True, True, False),  # the default target set
        (True, False, False),  # attn only
        (False, True, False),  # mlp only -- no Rowwise `wo` adapter
        (True, True, True),  # `output` wrapped, so it IS retargeted
    ],
    ids=["attn+mlp", "attn", "mlp", "attn+mlp+unembed"],
)
def test_tp2_matches_tp1(train_attn, train_mlp, train_unembed, tmp_path):
    """tp=2 must produce tp=1's answer, not merely run without raising."""
    pytest.importorskip("torch.distributed")
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo unavailable")

    flags = (train_attn, train_mlp, train_unembed)
    one = _run(1, flags, str(tmp_path))
    two = _run(2, flags, str(tmp_path))

    assert two["trainable"] > 0, "no trainable params -- test proves nothing"
    assert two["with_grad"] == two["trainable"], (
        f"{two['trainable'] - two['with_grad']} adapter params got no "
        "gradient; some adapter is detached from the graph"
    )
    delta = (one["out"] - two["out"]).abs().max().item()
    assert delta < 1e-4, (
        f"tp=2 output differs from tp=1 by {delta:.3e} -- a layout bug "
        "can run cleanly and still be wrong"
    )


@pytest.mark.slow
def test_unwrapped_output_is_not_retargeted():
    """`output` must keep its own plan entry when it is not wrapped.

    Retargeting it to `output.base` would match no module, and
    parallelize_module would ignore that silently.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel

    from ezpz.tinker.lora import lora_tp_plan

    model = _build(True, True, False)  # unembed NOT wrapped
    style = ColwiseParallel()
    plan = lora_tp_plan({"output": style}, model)
    assert plan == {"output": style}, (
        "an unwrapped `output` was retargeted; the resulting keys match "
        "no module and torch drops them without raising"
    )


@pytest.mark.slow
def test_wrapped_output_is_retargeted():
    """The converse: when `output` IS wrapped it must be retargeted."""
    from torch.distributed.tensor.parallel import ColwiseParallel

    from ezpz.tinker.lora import lora_tp_plan

    model = _build(True, True, True)
    plan = lora_tp_plan({"output": ColwiseParallel()}, model)
    assert set(plan) == {"output.base", "output.A", "output.B"}


def test_adapter_styles_follow_the_base():
    """A mirrors what the base consumes; B what it produces.

    Pure plan-shape check, so it needs no distributed setup.
    """
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
    )

    from ezpz.tinker.lora import lora_tp_plan

    row = RowwiseParallel(output_layouts=Shard(1))
    plan = lora_tp_plan({"attention.wo": row})

    a = plan["attention.wo.A"]
    assert isinstance(a, RowwiseParallel), (
        "A must mirror the base's class -- a Colwise A under a Rowwise "
        "base declares a sharded activation replicated and the "
        "contraction dims disagree"
    )
    assert a.output_layouts == (Replicate(),)
    assert a.use_local_output is False, (
        "use_local_output=True unwraps to this rank's shard, so B would "
        "see r/tp instead of r"
    )

    b = plan["attention.wo.B"]
    assert isinstance(b, RowwiseParallel)
    assert b.input_layouts == (Replicate(),), "B is fed by A, not by x"
    assert b.output_layouts == row.output_layouts, (
        "B's output is summed with the base's, so it must land in the "
        "same layout"
    )

    col = ColwiseParallel()
    cplan = lora_tp_plan({"attention.wq": col})
    assert isinstance(cplan["attention.wq.A"], ColwiseParallel)
    assert isinstance(cplan["attention.wq.B"], ColwiseParallel)
