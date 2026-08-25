"""FSDP2 collective symmetry when LoRA leaves whole units frozen (#239).

Under ``--lora-target attn,mlp`` every adapter lands inside a transformer
block, so ``apply_lora``'s freeze-everything-first pass (``tinker/lora.py``)
leaves ``tok_embeddings`` and ``[norm, output]`` with no trainable parameter
at all. ``parallelize`` still makes each of those its own FSDP2 unit.

A fully-frozen unit is asymmetric in backward: ``reshard_after_forward``
discarded its parameters after forward, so backward re-gathers them, but
``post_backward`` returns before the reduce-scatter when a group has no
gradients. The unit emits an all-gather with no matching reduce-scatter.

On agpt-2b (12 layers) that is 14 backward all-gathers against 12
reduce-scatters. This file pins that the asymmetry is gone.

**What this does NOT claim.** The asymmetry is present in *both* hanging
(r8/r16) and working (r32/r64) #239 configurations, so it is a
precondition, not a proven cause. These tests assert the invariant the
fix establishes -- not that #239 is fixed.

Real ``fully_shard`` on 2 gloo ranks via ``mp.spawn``, following
``test_tinker_lora_tp.py``: no rendezvous socket, runs on a laptop.
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

PORT = "29874"
DIM = 64
LAYERS = 12
VOCAB = 512


class _Blk(nn.Module):
    """Stand-in for a TransformerBlock: frozen base + trainable adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.A = nn.Linear(DIM, 8, bias=False)
        self.B = nn.Linear(8, DIM, bias=False)
        self.wq.weight.requires_grad_(False)
        self.wo.weight.requires_grad_(False)

    def forward(self, x):
        return x + self.wo(self.wq(x)) + self.B(self.A(x))


class _M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList([_Blk() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        h = self.tok_embeddings(ids)
        for b in self.layers:
            h = b(h)
        return self.output(self.norm(h))


def _worker(rank: int, ws: int, keep_frozen_gathered: bool, q) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=PORT,
        RANK=str(rank),
        WORLD_SIZE=str(ws),
    )
    torch.set_default_device("cpu")
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    try:
        torch.manual_seed(1234)
        m = _M()
        # What apply_lora does: freeze the base everywhere. Adapters live
        # only inside blocks, so embeddings and norm/output end up with no
        # trainable parameter at all.
        m.tok_embeddings.weight.requires_grad_(False)
        m.output.weight.requires_grad_(False)
        for p in m.norm.parameters():
            p.requires_grad_(False)

        trace: list[tuple[str, str]] = []
        phase = ["fwd"]
        real_ag = dist.all_gather_into_tensor
        real_rs = dist.reduce_scatter_tensor

        def _ag(*a, **k):
            trace.append((phase[0], "AG"))
            return real_ag(*a, **k)

        def _rs(*a, **k):
            trace.append((phase[0], "RS"))
            return real_rs(*a, **k)

        dist.all_gather_into_tensor = _ag
        dist.reduce_scatter_tensor = _rs
        try:
            # The REAL shipped helper -- reverting the fix in fsdp_tp.py
            # must break these tests, so do not reimplement it here.
            from ezpz.examples.fsdp_tp import frozen_unit_kwargs

            os.environ["EZPZ_FSDP_FROZEN_RESHARD"] = (
                "0" if keep_frozen_gathered else "1"
            )
            mesh = init_device_mesh("cpu", (ws,))
            kw = {"mesh": mesh, "reshard_after_forward": True}
            fully_shard(
                m.tok_embeddings, **frozen_unit_kwargs(m.tok_embeddings, kw)
            )
            for b in m.layers:
                fully_shard(b, **kw)
            fully_shard(
                [m.norm, m.output],
                **frozen_unit_kwargs([m.norm, m.output], kw),
            )
            fully_shard(m, **kw)

            ids = torch.randint(0, VOCAB, (2, 8))
            loss = m(ids).float().pow(2).mean()
            phase[0] = "bwd"
            loss.backward()
        finally:
            dist.all_gather_into_tensor = real_ag
            dist.reduce_scatter_tensor = real_rs

        if rank == 0:
            bwd = [t for t in trace if t[0] == "bwd"]
            q.put(
                (
                    sum(1 for t in bwd if t[1] == "AG"),
                    sum(1 for t in bwd if t[1] == "RS"),
                )
            )
    finally:
        dist.destroy_process_group()


def _count(keep_frozen_gathered: bool) -> tuple[int, int]:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.spawn(_worker, args=(2, keep_frozen_gathered, q), nprocs=2, join=True)
    return q.get()


@pytest.mark.slow
def test_frozen_units_resharded_are_asymmetric() -> None:
    """The bug's precondition: 14 all-gathers, 12 reduce-scatters.

    This is the *old* behaviour. Pinned so the fix below is measured
    against a real, reproduced asymmetry rather than an assumed one --
    if torch ever stops emitting the unmatched all-gather, this fails
    and the fix becomes unnecessary rather than silently inert.
    """
    ag, rs = _count(keep_frozen_gathered=False)
    assert rs == LAYERS, f"expected {LAYERS} reduce-scatters, got {rs}"
    assert ag == LAYERS + 2, (
        f"expected {LAYERS + 2} all-gathers (12 blocks + 2 frozen units), "
        f"got {ag}"
    )
    assert ag != rs, "asymmetry should be present without the fix"


@pytest.mark.slow
def test_frozen_units_kept_gathered_are_symmetric() -> None:
    """The fix: every backward all-gather has a matching reduce-scatter."""
    ag, rs = _count(keep_frozen_gathered=True)
    assert rs == LAYERS, f"expected {LAYERS} reduce-scatters, got {rs}"
    assert ag == rs, (
        f"asymmetry survived the fix: {ag} all-gathers vs {rs} reduce-scatters"
    )


def test_frozen_unit_kwargs_predicate(monkeypatch) -> None:
    """Single-rank guard on the predicate in the shipped helper."""
    from ezpz.examples.fsdp_tp import frozen_unit_kwargs

    monkeypatch.delenv("EZPZ_FSDP_FROZEN_RESHARD", raising=False)
    base = {"mesh": object(), "reshard_after_forward": True}

    frozen = nn.Linear(4, 4, bias=False)
    frozen.weight.requires_grad_(False)
    trainable = nn.Linear(4, 4, bias=False)

    assert frozen_unit_kwargs(frozen, base)["reshard_after_forward"] is False
    assert frozen_unit_kwargs(trainable, base)["reshard_after_forward"] is True
    assert (
        frozen_unit_kwargs([frozen, trainable], base)["reshard_after_forward"]
        is True
    )
    # LayerNorm carries trainable affine params, so the pair is NOT frozen.
    assert (
        frozen_unit_kwargs([frozen, nn.LayerNorm(4)], base)[
            "reshard_after_forward"
        ]
        is True
    )
    # The caller's dict must not be mutated.
    assert base["reshard_after_forward"] is True

    monkeypatch.setenv("EZPZ_FSDP_FROZEN_RESHARD", "1")
    assert frozen_unit_kwargs(frozen, base)["reshard_after_forward"] is True


def test_apply_lora_really_leaves_units_frozen(monkeypatch) -> None:
    """Bridge from the stand-in model above to the real one.

    The spawn tests use a hand-built module. This asserts the *real*
    ``Transformer`` + ``apply_lora`` produces the same shape, so those
    tests are not measuring an artifact of the stand-in:

    * ``--lora-target attn,mlp`` leaves ``tok_embeddings`` and
      ``[norm, output]`` with zero trainable params -> fix engages.
    * ``unembed`` puts an adapter in ``[norm, output]`` -> fix correctly
      declines, since that unit is no longer frozen.
    """
    from ezpz.examples.fsdp_tp import frozen_unit_kwargs
    from ezpz.models.llama import ModelArgs, Transformer
    from ezpz.tinker.lora import LoraConfig, apply_lora

    monkeypatch.delenv("EZPZ_FSDP_FROZEN_RESHARD", raising=False)
    base = {"reshard_after_forward": True}
    cfg = ModelArgs(
        dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=128
    )

    def trainable(mods) -> int:
        ms = mods if isinstance(mods, list) else [mods]
        return sum(p.requires_grad for m in ms for p in m.parameters())

    m = apply_lora(
        Transformer(cfg),
        LoraConfig(
            rank=8, train_attn=True, train_mlp=True, train_unembed=False
        ),
        verbose=False,
    )
    assert trainable(m.tok_embeddings) == 0
    assert trainable([m.norm, m.output]) == 0
    assert (
        frozen_unit_kwargs(m.tok_embeddings, base)["reshard_after_forward"]
        is False
    )
    assert (
        frozen_unit_kwargs([m.norm, m.output], base)["reshard_after_forward"]
        is False
    )
    # Blocks keep the sharded path -- 14 trainable tensors is the count the
    # #239 collective-size arithmetic is built on.
    assert trainable(m.layers[0]) == 14

    m2 = apply_lora(
        Transformer(cfg),
        LoraConfig(
            rank=8, train_attn=True, train_mlp=True, train_unembed=True
        ),
        verbose=False,
    )
    assert trainable([m2.norm, m2.output]) > 0, (
        "unembed adapter should land here"
    )
    assert (
        frozen_unit_kwargs([m2.norm, m2.output], base)["reshard_after_forward"]
        is True
    ), "a unit holding a trainable adapter must keep the default path"
