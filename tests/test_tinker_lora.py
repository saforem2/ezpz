"""Tests for the LoRA layer in :mod:`ezpz.tinker.lora`.

Three of these pin constraints that were measured against torch before
the code was written, and that are easy to regress silently:

* ``.weight`` must proxy to the base, or ``Transformer.init_weights``
  raises ``AttributeError`` and no native model can be built.
* ``B`` must be zero-initialized, or the adapted model does not start
  equal to the base and every "LoRA doesn't change step 0" assumption
  breaks.
* A TP plan keyed on ``attention.wq`` must be retargeted once ``wq`` is a
  wrapper, or ``tp>1`` raises ``NotImplementedError``.

CPU-only. The FSDP2 and TP cases build a single-rank gloo process group;
they skip cleanly where that is unavailable.
"""

from __future__ import annotations

import contextlib
import os

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from ezpz.tinker.lora import (  # noqa: E402
    ATTN_TARGETS,
    MLP_TARGETS,
    LoraConfig,
    LoRALinear,
    adapter_state_dict,
    apply_lora,
    iter_lora_modules,
    lora_tp_plan,
    merge_adapters,
    reset_lora_after_materialize,
)


def _tiny_model_args():
    from ezpz.models.llama import ModelArgs

    return ModelArgs(
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


def _tiny_transformer():
    from ezpz.models.llama import Transformer

    torch.manual_seed(0)
    return Transformer.from_model_args(_tiny_model_args())


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


class TestLoRALinear:
    def test_zero_init_is_identity(self):
        """B starts at zero, so the adapter is a no-op at step 0."""
        torch.manual_seed(0)
        base = nn.Linear(16, 32, bias=False)
        x = torch.randn(4, 16)
        want = base(x).clone()
        got = LoRALinear(base, rank=4)(x)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_weight_proxy_supports_llama_init(self):
        """REGRESSION: llama.py:359 does nn.init.trunc_normal_(l.weight).

        Without the ``.weight`` property this raises AttributeError and
        every native model build fails.
        """
        lora = LoRALinear(nn.Linear(8, 8, bias=False), rank=2)
        nn.init.trunc_normal_(
            lora.weight, mean=0.0, std=0.02
        )  # must not raise
        assert lora.weight is lora.base.weight

    def test_base_is_frozen_adapters_are_not(self):
        lora = LoRALinear(nn.Linear(8, 8, bias=False), rank=2)
        assert not lora.base.weight.requires_grad
        assert lora.A.weight.requires_grad
        assert lora.B.weight.requires_grad

    def test_only_adapters_receive_grads(self):
        lora = LoRALinear(nn.Linear(8, 8, bias=False), rank=2)
        lora(torch.randn(2, 8)).sum().backward()
        assert lora.base.weight.grad is None
        # A's grad is zero at step 0 (B == 0) but must still be populated.
        assert lora.A.weight.grad is not None
        assert lora.B.weight.grad is not None

    def test_scaling(self):
        assert LoRALinear(nn.Linear(4, 4), rank=8).scaling == 1.0
        assert LoRALinear(nn.Linear(4, 4), rank=8, alpha=16).scaling == 2.0

    def test_rejects_non_linear(self):
        with pytest.raises(TypeError, match="expects nn.Linear"):
            LoRALinear(nn.Conv1d(4, 4, 1), rank=2)  # type: ignore[arg-type]

    def test_shape_preserved(self):
        lora = LoRALinear(nn.Linear(16, 32, bias=False), rank=4)
        assert lora(torch.randn(2, 5, 16)).shape == (2, 5, 32)

    def test_init_weights_resets_b_to_zero(self):
        """After to_empty() adapter storage is garbage; init must re-zero B."""
        lora = LoRALinear(nn.Linear(8, 8, bias=False), rank=2)
        with torch.no_grad():
            lora.B.weight.fill_(0.5)
        lora.init_weights()
        assert torch.count_nonzero(lora.B.weight) == 0


# ---------------------------------------------------------------------------
# LoraConfig
# ---------------------------------------------------------------------------


class TestLoraConfig:
    def test_target_names_by_role(self):
        assert set(LoraConfig(train_mlp=False).target_names()) == set(
            ATTN_TARGETS
        )
        assert set(LoraConfig(train_attn=False).target_names()) == set(
            MLP_TARGETS
        )

    def test_rejects_bad_rank(self):
        with pytest.raises(ValueError, match="rank must be > 0"):
            LoraConfig(rank=0)

    def test_rejects_bad_dropout(self):
        with pytest.raises(ValueError, match="dropout must be"):
            LoraConfig(dropout=1.0)

    def test_rejects_empty_target_set(self):
        with pytest.raises(ValueError, match="adapts nothing"):
            LoraConfig(train_attn=False, train_mlp=False)

    def test_extra_targets_dedupe(self):
        names = LoraConfig(extra_targets=("wq", "custom")).target_names()
        assert names.count("wq") == 1
        assert "custom" in names


# ---------------------------------------------------------------------------
# apply_lora over a real Transformer
# ---------------------------------------------------------------------------


class TestApplyLora:
    def test_wraps_expected_modules(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        names = {n for n, _ in iter_lora_modules(model)}
        assert "layers.0.attention.wq" in names
        assert "layers.0.feed_forward.w1" in names
        assert len(names) == 2 * (len(ATTN_TARGETS) + len(MLP_TARGETS))

    def test_output_untouched_by_default(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        assert not isinstance(model.output, LoRALinear)

    def test_train_unembed_wraps_output(self):
        model = apply_lora(
            _tiny_transformer(),
            LoraConfig(rank=4, train_unembed=True),
            verbose=False,
        )
        assert isinstance(model.output, LoRALinear)

    def test_trainable_fraction_is_small(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        assert 0 < n_train < n_all
        assert n_train / n_all < 0.5

    def test_every_trainable_param_is_an_adapter(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert ".A." in name or ".B." in name, (
                    f"{name} unexpectedly trainable"
                )

    def test_forward_unchanged_at_step_zero(self):
        """The whole model -- not just one layer -- starts as the base."""
        base = _tiny_transformer().eval()
        tokens = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            want = base(tokens).clone()
        adapted = apply_lora(base, LoraConfig(rank=4), verbose=False).eval()
        with torch.no_grad():
            got = adapted(tokens)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_backward_leaves_base_grads_none(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        model(torch.randint(0, 128, (2, 16))).sum().backward()
        frozen_with_grad = [
            n
            for n, p in model.named_parameters()
            if not p.requires_grad and p.grad is not None
        ]
        assert frozen_with_grad == []

    def test_init_weights_still_works_after_wrapping(self):
        """The .weight proxy must survive a full re-init of the wrapped model."""
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        model.init_weights()  # must not raise
        for _, mod in iter_lora_modules(model):
            assert torch.count_nonzero(mod.B.weight) == 0

    def test_raises_when_nothing_matches(self):
        with pytest.raises(RuntimeError, match="matched no modules"):
            apply_lora(
                nn.Sequential(nn.ReLU()), LoraConfig(rank=4), verbose=False
            )


# ---------------------------------------------------------------------------
# TP plan retargeting -- the constraint that breaks tp>1
# ---------------------------------------------------------------------------


class TestLoraTpPlan:
    def test_retargets_wrapped_leaves(self):
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
        )

        col, row = ColwiseParallel(), RowwiseParallel()
        plan = lora_tp_plan({"attention.wq": col, "attention.wo": row})
        # base keeps the original style object; A and B are derived.
        assert plan["attention.wq.base"] is col
        assert plan["attention.wo.base"] is row
        assert set(plan) == {
            "attention.wq.base",
            "attention.wq.A",
            "attention.wq.B",
            "attention.wo.base",
            "attention.wo.A",
            "attention.wo.B",
        }

    def test_A_follows_the_base_style(self):
        """A mirrors the base's class -- it consumes the same activation.

        This test previously asserted the OPPOSITE (A is always
        Colwise). That belief came from a toy single-rank probe, and it
        is wrong: under a Rowwise base the activation arrives sharded on
        the feature dim, so a Colwise A -- which declares that input
        replicated and keeps a full-width weight -- disagrees on the
        contraction dim::

            Sharding propagation failed for aten.mm.default(
                Spec(f32[16, 64](R)), Spec(f32[128, 8](S(1))))

        Verified numerically against tp=1 on a real 2-rank mesh in
        tests/test_tinker_lora_tp.py.
        """
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
        )

        for base_style in (ColwiseParallel(), RowwiseParallel()):
            a = lora_tp_plan({"attention.wq": base_style})["attention.wq.A"]
            assert isinstance(a, type(base_style)), (
                f"A must mirror {type(base_style).__name__}, got "
                f"{type(a).__name__}"
            )

    def test_B_lands_in_the_bases_output_layout(self):
        """B's output is summed with the base's, so layouts must agree.

        B's *input* comes from A, not from the base's input, so the
        base's `input_layouts` must NOT be copied onto it.
        """
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import RowwiseParallel

        row = RowwiseParallel(output_layouts=Shard(1))
        b = lora_tp_plan({"attention.wo": row})["attention.wo.B"]
        assert isinstance(b, RowwiseParallel)
        assert b.output_layouts == row.output_layouts
        assert b.input_layouts == (Replicate(),)

    def test_A_never_unwraps_to_a_local_tensor(self):
        """use_local_output must be False so B receives a DTensor.

        The default (True) hands back this rank's local shard, so B sees
        r/tp instead of r.
        """
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
        )

        for base_style in (ColwiseParallel(), RowwiseParallel()):
            a = lora_tp_plan({"attention.wq": base_style})["attention.wq.A"]
            assert a.use_local_output is False, (
                f"A under {type(base_style).__name__} must keep its DTensor"
            )

    def test_A_is_parallelized_with_replicated_output(self):
        """REGRESSION (Sunspot tp=2): leaving A unparallelized makes it
        return a plain tensor while base returns a DTensor, so their sum
        raises `mixed torch.Tensor and DTensor`. A must be a DTensor op
        whose output is Replicate, because B (Colwise) needs a
        replicated input."""
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor.parallel import ColwiseParallel

        a_style = lora_tp_plan({"attention.wq": ColwiseParallel()})[
            "attention.wq.A"
        ]
        assert isinstance(a_style, ColwiseParallel)
        layouts = a_style.output_layouts
        layouts = layouts if isinstance(layouts, tuple) else (layouts,)
        assert all(isinstance(x, Replicate) for x in layouts), (
            f"A must output Replicate for B to consume it, got {layouts}"
        )

    def test_passes_through_non_targets(self):
        plan = lora_tp_plan({"attention_norm": "SP", "attention": "PREP"})
        assert plan == {"attention_norm": "SP", "attention": "PREP"}

    def test_covers_every_role_target(self):
        from torch.distributed.tensor.parallel import ColwiseParallel

        base = {f"attention.{n}": ColwiseParallel() for n in ATTN_TARGETS}
        base |= {f"feed_forward.{n}": ColwiseParallel() for n in MLP_TARGETS}
        plan = lora_tp_plan(base)
        assert len(plan) == 3 * len(base)  # base + A + B
        assert all(k.endswith((".base", ".A", ".B")) for k in plan)


# A single-rank gloo PG needs a rendezvous address. Set it EXPLICITLY
# rather than with `os.environ.setdefault`: setdefault *defers* to
# whatever is already in the environment, so a value leaked by an
# earlier test wins. That is how a fabricated `MASTER_ADDR` from a pure
# unit test reached this rendezvous and blocked it for the 30-minute
# default PG timeout. Restoring the previous values on the way out
# keeps this test from becoming the next leaker (tests/conftest.py's
# `_no_rendezvous_leak` enforces that).
@contextlib.contextmanager
def _rendezvous_env(port: str):
    prev = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.skipif(os.name != "posix", reason="needs a gloo PG")
class TestLoraTpPlanAgainstTorch:
    """The retarget must actually satisfy ``parallelize_module``."""

    @staticmethod
    def _pg():
        import torch.distributed as dist

        if dist.is_initialized():
            return True
        with _rendezvous_env("29733"):
            try:
                dist.init_process_group(
                    "gloo", rank=0, world_size=1
                )
            except Exception:
                return False
            return True

    def test_unretargeted_plan_raises_but_retargeted_works(self):
        import torch.distributed as dist

        if not self._pg():
            pytest.skip("could not init gloo PG")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            parallelize_module,
        )

        mesh = init_device_mesh("cpu", (1,))

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return self.wq(x)

        # Unretargeted: the plan names a LoRALinear -> torch refuses.
        bad = Attn()
        bad.wq = LoRALinear(bad.wq, rank=4)
        with pytest.raises(
            NotImplementedError, match="only support nn.Linear"
        ):
            parallelize_module(bad, mesh, {"wq": ColwiseParallel()})

        # Retargeted: shards and runs.
        good = Attn()
        good.wq = LoRALinear(good.wq, rank=4)
        parallelize_module(good, mesh, lora_tp_plan({"wq": ColwiseParallel()}))
        good(torch.randn(2, 4, 16)).sum().backward()

        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP2 composition
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="needs a gloo PG")
class TestLoraUnderFSDP2:
    def test_fully_shard_with_mixed_requires_grad(self):
        import torch.distributed as dist

        if dist.is_initialized():
            pytest.skip("PG already initialized by another test")
        with _rendezvous_env("29734"):
            try:
                dist.init_process_group(
                    "gloo", rank=0, world_size=1
                )
            except Exception:
                pytest.skip("could not init gloo PG")
        try:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh("cpu", (1,))
            model = apply_lora(
                _tiny_transformer(), LoraConfig(rank=4), verbose=False
            )
            for layer in model.layers:
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)

            model(torch.randint(0, 128, (2, 16))).sum().backward()
            frozen_with_grad = [
                n
                for n, p in model.named_parameters()
                if not p.requires_grad and p.grad is not None
            ]
            assert frozen_with_grad == []
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


# ---------------------------------------------------------------------------
# export helpers
# ---------------------------------------------------------------------------


class TestExportHelpers:
    def test_adapter_state_dict_is_adapters_only(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        sd = adapter_state_dict(model)
        assert sd
        assert all(".A." in k or ".B." in k for k in sd)
        full = sum(p.numel() for p in model.parameters())
        assert sum(t.numel() for t in sd.values()) < full / 4

    def test_merge_is_a_noop_at_step_zero(self):
        """B == 0, so merging changes nothing and restores a plain model."""
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        model.eval()
        tokens = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            before = model(tokens).clone()
        merged = merge_adapters(model).eval()
        with torch.no_grad():
            after = merged(tokens)
        torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-6)
        assert not list(iter_lora_modules(merged))

    def test_merge_applies_a_trained_delta(self):
        model = apply_lora(
            _tiny_transformer(), LoraConfig(rank=4), verbose=False
        )
        model.eval()
        tokens = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            before = model(tokens).clone()
            # Simulate training having moved B off zero.
            for _, mod in iter_lora_modules(model):
                mod.B.weight.normal_(0.0, 0.02)
            adapted = model(tokens).clone()
        assert not torch.allclose(adapted, before)
        merged = merge_adapters(model).eval()
        with torch.no_grad():
            after = merged(tokens)
        torch.testing.assert_close(after, adapted, rtol=1e-5, atol=1e-5)


class TestAdapterDeviceAndDtype:
    """Adapters must inherit the base's device and dtype.

    Built with the process defaults instead, a bf16 base gets fp32
    adapters and dies on its first adapter matmul, and a base on `meta`
    gets adapters with REAL cpu storage -- silently defeating meta-init
    on exactly the large models it exists for. (Codex P1 on #207.)
    """

    def test_dtype_follows_the_base(self):
        base = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        w = LoRALinear(base, rank=4)
        assert w.A.weight.dtype == torch.bfloat16
        assert w.B.weight.dtype == torch.bfloat16

    def test_bf16_forward_does_not_raise(self):
        # Pre-fix: RuntimeError: expected m1 and m2 to have the same
        # dtype, but got: c10::BFloat16 != float
        base = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        w = LoRALinear(base, rank=4)
        out = w(torch.randn(2, 16, dtype=torch.bfloat16))
        assert out.shape == (2, 16)

    def test_meta_base_keeps_adapters_on_meta(self):
        with torch.device("meta"):
            base = nn.Linear(16, 16, bias=False)
        w = LoRALinear(base, rank=4)
        assert w.A.weight.is_meta and w.B.weight.is_meta, (
            "adapters allocated real storage for a meta base; meta-init "
            "exists to avoid exactly this"
        )


class TestResetAfterMaterialize:
    """`to_empty()` leaves adapter storage uninitialized.

    `Transformer.init_weights` reaches through LoRALinear's `.weight`
    proxy to the BASE weight only -- nothing calls
    `LoRALinear.init_weights`. So on the meta-init path `B` keeps
    whatever was in memory, breaking "adapter is an exact no-op at step
    0" and starting training from garbage. (Codex P1 on #207.)
    """

    @staticmethod
    def _meta_model():
        from ezpz.models.llama import ModelArgs, Transformer

        cfg = ModelArgs(
            dim=64, n_layers=1, n_heads=4, n_kv_heads=4, vocab_size=128,
            multiple_of=8, hidden_dim=128, max_seq_len=32, depth_init=True,
        )
        with torch.device("meta"):
            model = Transformer.from_model_args(cfg)
        return apply_lora(model, LoraConfig(rank=4), verbose=False)

    def test_init_weights_alone_leaves_B_dirty(self):
        """Pins WHY the extra call is needed -- not a redundant step."""
        model = self._meta_model()
        model.to_empty(device="cpu")
        for _, mod in iter_lora_modules(model):
            mod.B.weight.data.fill_(7.0)
        model.init_weights()
        dirty = [
            n for n, mod in iter_lora_modules(model)
            if float(mod.B.weight.detach().abs().max()) != 0.0
        ]
        assert dirty, (
            "Transformer.init_weights now resets adapters on its own; "
            "reset_lora_after_materialize may be redundant -- re-check "
            "before deleting it"
        )

    def test_reset_zeroes_every_B(self):
        model = self._meta_model()
        model.to_empty(device="cpu")
        for _, mod in iter_lora_modules(model):
            mod.B.weight.data.fill_(7.0)
            mod.A.weight.data.fill_(7.0)
        model.init_weights()
        n = reset_lora_after_materialize(model)
        assert n == 7, f"expected 7 adapters, reset {n}"
        for name, mod in iter_lora_modules(model):
            assert float(mod.B.weight.detach().abs().max()) == 0.0, name
            # A must be re-randomized, not left at the poison value
            assert float(mod.A.weight.detach().abs().max()) != 7.0, name

    def test_no_adapters_is_a_free_noop(self):
        from ezpz.models.llama import ModelArgs, Transformer

        cfg = ModelArgs(
            dim=64, n_layers=1, n_heads=4, n_kv_heads=4, vocab_size=128,
            multiple_of=8, hidden_dim=128, max_seq_len=32, depth_init=True,
        )
        assert reset_lora_after_materialize(
            Transformer.from_model_args(cfg)
        ) == 0

    def test_fsdp_tp_calls_it_on_the_meta_path(self):
        import inspect

        from ezpz.examples import fsdp_tp

        src = inspect.getsource(fsdp_tp.parallelize)
        assert "reset_lora_after_materialize" in src, (
            "the meta-init path no longer resets adapters; B would keep "
            "whatever to_empty left in its storage"
        )
