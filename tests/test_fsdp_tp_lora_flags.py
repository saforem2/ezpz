"""Tests for the ``--lora-*`` flags on ``ezpz.examples.fsdp_tp``.

Covers the CLI surface and the two integration points where LoRA meets
the existing pipeline:

* ``_lora_is_applied`` gates the TP-plan retarget, so it must be exactly
  right -- a false negative means ``tp>1`` raises
  ``NotImplementedError`` deep in ``parallelize_module``; a false
  positive would rewrite plan keys that have no ``.base``.
* ``--lora-rank 0`` (the default) must leave everything untouched, so
  every existing run is unaffected.

CPU-only; arg parsing and module inspection need no accelerator.
"""

from __future__ import annotations

import importlib

import pytest


def _fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except ModuleNotFoundError as exc:
        pytest.skip(f"missing optional dependency for fsdp_tp: {exc}")


class TestLoraFlags:
    def test_defaults_off(self):
        m = _fsdp_tp()
        a = m.parse_args(["--model", "debug", "--tp", "1"])
        assert a.lora_rank == 0
        assert a.lora_alpha is None
        assert a.lora_dropout == 0.0
        assert a.lora_target == "attn,mlp"

    def test_rank_and_alpha(self):
        m = _fsdp_tp()
        a = m.parse_args(
            [
                "--model",
                "debug",
                "--tp",
                "1",
                "--lora-rank",
                "16",
                "--lora-alpha",
                "32",
            ]
        )
        assert a.lora_rank == 16 and a.lora_alpha == 32.0

    @pytest.mark.parametrize(
        "target", ["attn", "mlp", "unembed", "attn,mlp", "attn,mlp,unembed"]
    )
    def test_target_spellings_parse(self, target):
        m = _fsdp_tp()
        a = m.parse_args(
            ["--model", "debug", "--tp", "1", "--lora-target", target]
        )
        assert a.lora_target == target


class TestLoraIsApplied:
    """The predicate that gates the TP-plan retarget."""

    def test_false_on_a_plain_model(self):
        torch = pytest.importorskip("torch")
        m = _fsdp_tp()
        assert m._lora_is_applied(torch.nn.Linear(4, 4)) is False

    def test_true_after_apply_lora(self):
        torch = pytest.importorskip("torch")
        nn = torch.nn
        m = _fsdp_tp()
        from ezpz.tinker.lora import LoraConfig, apply_lora

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(8, 8, bias=False)

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attn()

        model = nn.Module()
        model.layers = nn.ModuleList([Blk()])
        apply_lora(model, LoraConfig(rank=4, train_mlp=False), verbose=False)
        assert m._lora_is_applied(model) is True


class TestTpSupported:
    """LoRA at tp>1 works; it must NOT be refused at setup.

    It used to be, because the plan hardcoded A/B styles instead of
    deriving them from the base's, and because it retargeted `output`
    even when --lora-target left it unwrapped. Both are fixed and
    verified numerically against tp=1 (tests/test_tinker_lora_tp.py).
    """

    def test_no_tp_guard_remains(self):
        m = _fsdp_tp()
        src = __import__("inspect").getsource(m.train)
        assert "--lora-rank with --tp > 1 is not supported" not in src

    def test_per_block_plan_is_not_reassigned(self):
        """Retargeting must not accumulate across layers.

        `layer_tp_plan = lora_tp_plan(layer_tp_plan)` inside the loop
        would feed layer 2 the plan already rewritten for layer 1.
        """
        m = _fsdp_tp()
        src = __import__("inspect").getsource(m.parallelize)
        assert "layer_tp_plan = _lora.lora_tp_plan(" not in src

    def test_plan_calls_pass_the_module(self):
        """Every lora_tp_plan call must pass a module.

        Without it, `output` is retargeted even when --lora-target left
        it unwrapped, and parallelize_module ignores the unmatched keys
        silently -- leaving `output` an unsharded nn.Linear that dies on
        its first DTensor input.
        """
        import re

        m = _fsdp_tp()
        src = __import__("inspect").getsource(m.parallelize)
        calls = re.findall(r"lora_tp_plan\((.*?)\)", src, re.S)
        assert calls, "no lora_tp_plan call found in parallelize()"
        for args in calls:
            assert "," in args, (
                f"lora_tp_plan({args.strip()}) omits the module"
            )


class TestHfPathReachesLora:
    """`--lora-rank` must not be silently ignored for HF models.

    The LoRA block used to live inside the native model's ``else:``
    branch, so an HF run accepted the flag, logged nothing unusual, and
    full fine-tuned. Source-level checks because instantiating an HF
    model here would need weights.
    """

    @staticmethod
    def _train_src():
        return __import__("inspect").getsource(_fsdp_tp().train)

    def test_lora_block_is_not_nested_in_the_native_branch(self):
        """`if args.lora_rank` must sit at function level.

        Its indentation is the whole bug: one level deeper and only
        native models reach it.
        """
        import re

        src = self._train_src()
        m = re.search(r'^(\s*)if getattr\(args, "lora_rank"', src, re.M)
        assert m, "the --lora-rank block moved or was renamed"
        assert len(m.group(1)) == 4, (
            f"the LoRA block is indented {len(m.group(1))} spaces, so it "
            "is nested inside a branch; at function level it must be 4, "
            "or HF models silently skip LoRA again"
        )

    def test_partial_application_is_refused(self):
        """A requested role that adapts nothing must fail loudly."""
        src = self._train_src()
        assert "iter_lora_modules(model)" in src, (
            "nothing inspects the finished tree, so a partially-applied "
            "LoRA request would pass silently"
        )

    def test_every_role_is_validated_not_just_unembed(self):
        """The guard must cover attn and mlp too, not only unembed.

        An earlier version checked `unembed` alone, so a model that
        matched `attn` but had no recognizable MLP passed with fewer
        adapters than requested -- the exact silent-partial-application
        this guard exists to prevent.
        """
        src = self._train_src()
        for role in ("ATTN_TARGETS", "MLP_TARGETS", "UNEMBED_TARGETS"):
            assert role in src, f"the guard ignores {role}"
        assert "HF_ATTN_TARGETS" in src and "HF_MLP_TARGETS" in src, (
            "the guard checks only native spellings, so every HF model "
            "would look like an empty role"
        )

    def test_partial_role_match_is_detected(self):
        """Behavioural check of the guard's predicate, not its text.

        Builds a model with attention projections but no MLP, asks for
        both roles, and confirms the empty role is identified.
        """
        torch = pytest.importorskip("torch")
        nn = torch.nn
        from ezpz.tinker import lora as _lora

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)

        model = nn.Module()
        model.layers = nn.ModuleList([Attn()])
        model = _lora.apply_lora(
            model,
            _lora.LoraConfig(rank=2, train_attn=True, train_mlp=True),
            verbose=False,
        )
        leaves = {
            n.rsplit(".", 1)[-1] for n, _ in _lora.iter_lora_modules(model)
        }
        roles = {
            "attn": set(_lora.ATTN_TARGETS) | set(_lora.HF_ATTN_TARGETS),
            "mlp": set(_lora.MLP_TARGETS) | set(_lora.HF_MLP_TARGETS),
        }
        empty = sorted(r for r in roles if not (leaves & roles[r]))
        assert empty == ["mlp"], (
            f"expected 'mlp' to be flagged as unmatched, got {empty}; "
            "apply_lora does NOT raise here because attn matched"
        )

    def test_help_text_no_longer_claims_native_only(self):
        m = _fsdp_tp()
        p = m.build_parser() if hasattr(m, "build_parser") else None
        help_txt = p.format_help() if p else __import__("inspect").getsource(m)
        assert "Native models only" not in help_txt, (
            "--lora-rank still advertises itself as native-only"
        )


class TestTargetValidation:
    """An unknown --lora-target must fail loudly at setup, not silently
    adapt nothing (which would look like LoRA 'not working')."""

    def test_unknown_target_rejected(self):
        from ezpz.tinker.lora import LoraConfig

        # The SystemExit is raised in train(); here we pin the underlying
        # contract that an empty target set is refused.
        with pytest.raises(ValueError, match="adapts nothing"):
            LoraConfig(rank=4, train_attn=False, train_mlp=False)


class TestTpPlanRetargetWiring:
    """The retarget must fire only when LoRA is present."""

    def test_plan_unchanged_without_lora(self):
        from torch.distributed.tensor.parallel import ColwiseParallel

        from ezpz.tinker.lora import lora_tp_plan

        plan = {"attention.wq": ColwiseParallel(), "attention_norm": "SP"}
        # lora_tp_plan is only CALLED when _lora_is_applied is True; this
        # pins that calling it needlessly would change keys, which is why
        # the call site is gated.
        assert lora_tp_plan(plan) != plan

    def test_norm_entries_never_retargeted(self):
        from ezpz.tinker.lora import lora_tp_plan

        out = lora_tp_plan({"attention_norm": "SP", "ffn_norm": "SP"})
        assert out == {"attention_norm": "SP", "ffn_norm": "SP"}
