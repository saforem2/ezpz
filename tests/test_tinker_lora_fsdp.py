"""LoRA under REAL multi-rank FSDP2, sharded the way ``fsdp_tp`` shards.

Why this file exists: ``tests/test_tinker_lora.py::TestLoraUnderFSDP2``
runs ``fully_shard`` at ``world_size=1`` and groups only
``[each layer, root]``. Both simplifications hide the shape that
production actually builds:

* At ws=1 FSDP2 issues **no collectives at all** (verified: the
  all-gather/reduce-scatter trace is empty), so no ordering property is
  under test. Every collective-ordering bug is invisible.
* ``fsdp_tp.parallelize`` (fsdp_tp.py:2124-2137) shards
  ``tok_embeddings``, each block, ``[norm, output]``, then the root.
  Under ``--lora-target attn,mlp`` the adapters land only inside the
  blocks, so ``tok_embeddings`` and ``[norm, output]`` become
  **fully-frozen FSDP units** -- units with zero trainable params. The
  existing test's grouping never builds one (its only all-frozen unit is
  the root, which holds no sharded params).

Issue #239: at ws=8 on Perlmutter (torch 2.13.0), ``--lora-rank 8``/``16``
with ``--lora-target attn,mlp`` deadlocked in ``.backward()``; r=32/64 and
attn-only did not. The NCCL watchdog put all 8 ranks on
``_REDUCE_SCATTER_BASE`` SeqNum=18 while "last enqueued work" was 39 --
an *ordering* deadlock, not a shape mismatch.

These tests assert the ORDERING INVARIANT, not the absence of a hang:
every rank must issue byte-identical collective sequences, and the
schedule must not depend on the adapter rank. A rank-dependent schedule
is the precondition for exactly the deadlock #239 hit; pinning it here
turns a silent 3600s hang on 8 GPUs into a red CI line on a laptop.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="mp.spawn + gloo fixture is POSIX-only"
)


def _free_port() -> str:
    """Pick a probably-unused port in the parent.

    A hard-coded MASTER_PORT fails *deterministically* when two
    spawn-based test files run concurrently -- they collide every time.
    Binding :0 and releasing turns that into a small race: the port is
    free when we read it, and something else could claim it before the
    workers bind. That window is narrow and the failure is loud
    (rendezvous refuses rather than silently misbehaving), so this is a
    real improvement, but it is NOT a guarantee -- do not read this
    helper as providing exclusion.

    A true fix would keep the socket open and pass the fd to the
    workers, which torch's env:// rendezvous cannot consume, or use a
    filesystem rendezvous. Both are more machinery than these two test
    files justify.
    """
    import socket

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return str(sk.getsockname()[1])


WS = 2
# agpt-2b's depth. The layer count sets where the first reduce-scatter
# lands in the sequence; #239's watchdog named SeqNum=18, which is the
# first RS of a 12-block model. Keep 12 so the index is comparable.
N_LAYERS = 12


def _worker(rank, ws, lora_rank, targets, outdir, port):
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=port,
        RANK=str(rank),
        WORLD_SIZE=str(ws),
    )
    dist.init_process_group("gloo", rank=rank, world_size=ws)

    trace: list[tuple[str, int, int]] = []
    # torch 2.13 renamed these: *_into_tensor/_tensor -> *_single. Patch
    # whichever pair exists, else the trace is silently empty on 2.13.
    _AG_NAMES = ("all_gather_into_tensor", "all_gather_single")
    _RS_NAMES = ("reduce_scatter_tensor", "reduce_scatter_single")
    _saved = {
        n: getattr(dist, n)
        for n in _AG_NAMES + _RS_NAMES
        if getattr(dist, n, None) is not None
    }
    _ag = next(_saved[n] for n in _AG_NAMES if n in _saved)
    _rs = next(_saved[n] for n in _RS_NAMES if n in _saved)

    def _pair(a, k, out_names, in_names):
        o = a[0] if a else next(k[n] for n in out_names if n in k)
        i = a[1] if len(a) > 1 else next(k[n] for n in in_names if n in k)
        return int(i.numel()), int(o.numel())

    def ag(*a, **k):
        trace.append(
            (
                "AG",
                *_pair(
                    a,
                    k,
                    ("output_tensor", "output"),
                    ("input_tensor", "input"),
                ),
            )
        )
        return _ag(*a, **k)

    def rs(*a, **k):
        trace.append(
            (
                "RS",
                *_pair(
                    a,
                    k,
                    ("output", "output_tensor"),
                    ("input", "input_tensor"),
                ),
            )
        )
        return _rs(*a, **k)

    for _n in _AG_NAMES:
        if _n in _saved:
            setattr(dist, _n, ag)
    for _n in _RS_NAMES:
        if _n in _saved:
            setattr(dist, _n, rs)
    err = None
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        from ezpz.models.llama import ModelArgs, Transformer
        from ezpz.tinker.lora import LoraConfig, apply_lora

        torch.manual_seed(0)
        model = Transformer.from_model_args(
            ModelArgs(
                dim=64,
                n_layers=N_LAYERS,
                n_heads=4,
                n_kv_heads=2,
                vocab_size=128,
                multiple_of=16,
                hidden_dim=128,
                max_seq_len=64,
                depth_init=True,
            )
        )
        model.init_weights()
        if lora_rank:
            model = apply_lora(
                model,
                LoraConfig(
                    rank=lora_rank,
                    train_attn=("attn" in targets),
                    train_mlp=("mlp" in targets),
                ),
                verbose=False,
            )

        mesh = init_device_mesh("cpu", (ws,))
        kw = {"mesh": mesh, "reshard_after_forward": True}
        # EXACT fsdp_tp.parallelize grouping (fsdp_tp.py:2124-2137).
        fully_shard(model.tok_embeddings, **kw)
        for block in model.layers:
            fully_shard(block, **kw)
        fully_shard([model.norm, model.output], **kw)
        fully_shard(model, **kw)

        model(torch.randint(0, 128, (2, 16))).float().pow(2).mean().backward()

        trainable = [p for p in model.parameters() if p.requires_grad]
        payload = {
            "trace": trace,
            "trainable": len(trainable),
            "with_grad": sum(1 for p in trainable if p.grad is not None),
            "frozen_with_grad": [
                n
                for n, p in model.named_parameters()
                if not p.requires_grad and p.grad is not None
            ],
        }
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        payload = {"trace": trace}
    finally:
        for _n, _f in _saved.items():
            setattr(dist, _n, _f)
        payload["err"] = err
        with open(os.path.join(outdir, f"fsdp_{rank}.json"), "w") as f:
            json.dump(payload, f)
        dist.destroy_process_group()


def _run(lora_rank, targets, outdir, ws=WS):
    import torch.multiprocessing as mp

    os.makedirs(outdir, exist_ok=True)
    mp.spawn(
        _worker,
        args=(ws, lora_rank, targets, outdir, _free_port()),
        nprocs=ws,
        join=True,
    )
    out = []
    for r in range(ws):
        with open(os.path.join(outdir, f"fsdp_{r}.json")) as f:
            d = json.load(f)
        d["trace"] = [tuple(e) for e in d["trace"]]
        out.append(d)
    return out


def _gloo_or_skip():
    pytest.importorskip("torch.distributed")
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo unavailable")


@pytest.mark.slow
@pytest.mark.parametrize("lora_rank", [8, 16, 32])
def test_collective_schedule_is_identical_across_ranks(lora_rank, tmp_path):
    """Every rank must issue the SAME collectives in the SAME order.

    A rank whose schedule diverges is a deadlock the moment the backend
    is a real one: peers block in a collective the diverging rank never
    posts. gloo will not necessarily hang here, but the divergence is
    observable, and it is the thing that hangs.
    """
    _gloo_or_skip()
    res = _run(lora_rank, "attn,mlp", str(tmp_path))
    assert {r["err"] for r in res} == {None}, [r["err"] for r in res]

    ref = res[0]["trace"]
    assert ref, (
        "no collectives recorded -- ws must be > 1 or this proves nothing"
    )
    for i, r in enumerate(res[1:], start=1):
        assert r["trace"] == ref, (
            f"rank {i} issued a different collective sequence than rank 0.\n"
            f"  rank0: {len(ref)} ops, {''.join(o[0] for o, _, _ in ref)}\n"
            f"  rank{i}: {len(r['trace'])} ops, "
            f"{''.join(o[0] for o, _, _ in r['trace'])}"
        )


@pytest.mark.slow
def test_schedule_does_not_depend_on_adapter_rank(tmp_path):
    """r=8 must issue the SAME op sequence as r=32, only wider payloads.

    #239's signature: r=8/16 deadlocked while r=32/64 trained. If the
    LoRA rank changes WHICH collectives fire or in WHAT order -- rather
    than only how big they are -- that is the bug's structural
    precondition. Compare the op-kind sequence only; numels legitimately
    scale with r.
    """
    _gloo_or_skip()
    shape = {}
    for r in (8, 32):
        res = _run(r, "attn,mlp", str(tmp_path / str(r)))
        assert {x["err"] for x in res} == {None}
        shape[r] = [op for op, _, _ in res[0]["trace"]]
    assert shape[8] == shape[32], (
        "the collective SEQUENCE changes with --lora-rank:\n"
        f"  r=8:  {''.join(o[0] for o in shape[8])}\n"
        f"  r=32: {''.join(o[0] for o in shape[32])}\n"
        "A rank-dependent schedule is how #239 deadlocked (r=8/16 hung, "
        "r=32/64 did not)."
    )


@pytest.mark.slow
def test_schedule_does_not_depend_on_lora_target(tmp_path):
    """attn-only vs attn+mlp must not change the op sequence.

    #239: r=16 attn+mlp hung; r=16 attn-only trained. Same reasoning as
    the rank sweep -- the target set decides which blocks contain
    trainable params, and must not decide the collective ORDER.
    """
    _gloo_or_skip()
    shape = {}
    for tgt in ("attn", "attn,mlp"):
        res = _run(16, tgt, str(tmp_path / tgt.replace(",", "_")))
        assert {x["err"] for x in res} == {None}
        shape[tgt] = [op for op, _, _ in res[0]["trace"]]
    assert shape["attn"] == shape["attn,mlp"], (
        "the collective SEQUENCE changes with --lora-target:\n"
        f"  attn:     {''.join(shape['attn'])}\n"
        f"  attn,mlp: {''.join(shape['attn,mlp'])}"
    )


@pytest.mark.slow
def test_fully_frozen_units_do_not_break_grad_routing(tmp_path):
    """Under ``--lora-target attn,mlp``, tok_embeddings and [norm, output]
    are FSDP units with ZERO trainable params.

    The existing single-rank test never builds one of these: its grouping
    is [each layer, root], and the root holds no sharded params. A frozen
    unit that still posts a reduce-scatter -- or that skips one its peers
    post -- is an ordering hazard.
    """
    _gloo_or_skip()
    res = _run(8, "attn,mlp", str(tmp_path))
    assert {r["err"] for r in res} == {None}
    for i, r in enumerate(res):
        assert r["trainable"] > 0, f"rank {i}: nothing trainable"
        assert r["with_grad"] == r["trainable"], (
            f"rank {i}: {r['trainable'] - r['with_grad']} adapter params got "
            "no gradient -- an adapter is detached under FSDP2"
        )
        assert r["frozen_with_grad"] == [], (
            f"rank {i}: frozen base params accumulated grads: "
            f"{r['frozen_with_grad'][:5]}"
        )


@pytest.mark.slow
def test_frozen_units_actually_exist_in_this_grouping():
    """Guard the guard: prove the sharding above really does create
    zero-trainable FSDP units, so the tests are not vacuous.

    If a future change to `apply_lora` starts training the embedding or
    the unembedding, the frozen-unit hazard disappears and the tests
    above quietly stop testing it. This fails loudly instead.
    """
    from ezpz.models.llama import ModelArgs, Transformer
    from ezpz.tinker.lora import LoraConfig, apply_lora

    torch.manual_seed(0)
    model = Transformer.from_model_args(
        ModelArgs(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=128,
            multiple_of=16,
            hidden_dim=128,
            max_seq_len=64,
            depth_init=True,
        )
    )
    model.init_weights()
    model = apply_lora(model, LoraConfig(rank=8), verbose=False)

    units = {
        "tok_embeddings": list(model.tok_embeddings.parameters()),
        "[norm,output]": list(model.norm.parameters())
        + list(model.output.parameters()),
    }
    for name, params in units.items():
        assert params, f"{name} has no params -- grouping assumption changed"
        assert not any(p.requires_grad for p in params), (
            f"{name} now has trainable params, so it is no longer a "
            "fully-frozen FSDP unit; the #239 hazard shape changed and "
            "these tests need revisiting"
        )
