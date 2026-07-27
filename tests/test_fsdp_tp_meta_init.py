"""Tests for meta-device model init in ezpz.examples.fsdp_tp.

Two layers:
  * Pure tier-selection logic (_estimate_param_count / _resolve_meta_init) —
    no torch.distributed, runs anywhere.
  * A real meta build -> fully_shard -> to_empty -> init_weights -> forward
    round-trip on a tiny model at single rank, proving the mechanism works
    and produces finite, initialized params (not meta/garbage).
"""
from __future__ import annotations

import argparse
import importlib
import os

import pytest


def _import():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not import fsdp_tp: {exc}")


def _torch_or_skip():
    try:
        import torch  # noqa: F401

        return torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")


def _args(meta_init="auto", model="agpt-2b"):
    return argparse.Namespace(meta_init=meta_init, model=model)


class TestEstimateParamCount:
    """The closed-form estimate must separate small from large presets."""

    def _cfg(self, m, **kw):
        ModelArgs = m.ModelArgs
        base = dict(
            dim=2048, n_layers=12, n_heads=16, n_kv_heads=4,
            vocab_size=256128, multiple_of=256, hidden_dim=11008,
            rope_theta=50000.0, max_seq_len=8192,
        )
        base.update(kw)
        return ModelArgs(**base)

    def test_agpt_2b_below_threshold(self):
        m = _import()
        est = m._estimate_param_count(self._cfg(m))  # agpt-2b dims
        assert est < 6e9, f"agpt-2b estimate {est/1e9:.1f}B should be < 6B"

    def test_agpt_20b_above_threshold(self):
        m = _import()
        est = m._estimate_param_count(
            self._cfg(m, dim=5120, n_layers=64, n_heads=40, n_kv_heads=8,
                      hidden_dim=14336, multiple_of=1024, rope_theta=500000.0,
                      max_seq_len=2048)
        )
        assert est >= 6e9, f"agpt-20b estimate {est/1e9:.1f}B should be >= 6B"

    def test_debug_tiny_below_threshold(self):
        m = _import()
        est = m._estimate_param_count(
            self._cfg(m, dim=256, n_layers=4, n_heads=8, n_kv_heads=8,
                      vocab_size=1024, hidden_dim=1024)
        )
        assert est < 6e9


class TestResolveMetaInit:
    """Mode + tier resolution. Uses a real single-rank PG so ezpz.get_rank()
    (called inside the resolver for logging) is well-defined."""

    def _cfg(self, m, **kw):
        base = dict(
            dim=2048, n_layers=12, n_heads=16, n_kv_heads=4,
            vocab_size=256128, multiple_of=256, hidden_dim=11008,
            rope_theta=50000.0, max_seq_len=8192,
        )
        base.update(kw)
        return m.ModelArgs(**base)

    def test_off_is_false(self):
        m = _import()
        big = self._cfg(m, dim=5120, n_layers=64, vocab_size=256128)
        assert m._resolve_meta_init(_args("off"), big, is_hf_model=False) is False

    def test_on_is_true_for_native(self):
        m = _import()
        small = self._cfg(m, dim=256, n_layers=2, vocab_size=1024)
        assert m._resolve_meta_init(_args("on"), small, is_hf_model=False) is True

    def test_hf_always_false(self):
        m = _import()
        big = self._cfg(m, dim=5120, n_layers=64)
        # even with mode=on, HF forces False
        assert m._resolve_meta_init(
            _args("on", model="meta-llama/Llama-3.2-1B"), big, is_hf_model=True
        ) is False

    def test_auto_small_is_dense(self):
        m = _import()
        small = self._cfg(m)  # agpt-2b: ~2B < 6B
        assert m._resolve_meta_init(_args("auto"), small, is_hf_model=False) is False

    def test_auto_large_is_meta(self):
        m = _import()
        big = self._cfg(m, dim=5120, n_layers=64, n_heads=40, n_kv_heads=8,
                        hidden_dim=14336, multiple_of=1024)
        assert m._resolve_meta_init(_args("auto"), big, is_hf_model=False) is True

    def test_auto_threshold_env_override(self, monkeypatch):
        m = _import()
        small = self._cfg(m)  # ~2B
        # Lower the threshold so the 2b preset now counts as "large".
        monkeypatch.setenv("EZPZ_META_INIT_MIN_PARAMS", "1e9")
        assert m._resolve_meta_init(_args("auto"), small, is_hf_model=False) is True


class TestMetaBuildRoundTrip:
    """Build a tiny Transformer on meta, materialize with to_empty +
    init_weights, and confirm params are real + finite (the core mechanism)."""

    def test_meta_build_then_to_empty_init(self):
        torch = _torch_or_skip()
        m = _import()
        cfg = m.ModelArgs(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=128,
            multiple_of=32, hidden_dim=128, rope_theta=10000.0, max_seq_len=64,
        )
        with torch.device("meta"):
            model = m.Transformer.from_model_args(cfg)
        # On meta: params have no storage.
        assert all(p.is_meta for p in model.parameters())
        # Materialize on CPU (no sharding here — this isolates to_empty+init).
        dev = torch.device("cpu")
        model.to_empty(device=dev)
        model.init_weights(buffer_device=dev)
        # Now every param is real + finite (initialized, not garbage/meta).
        for name, p in model.named_parameters():
            assert not p.is_meta, f"{name} still on meta"
            assert torch.isfinite(p).all(), f"{name} not finite after init"
        # freqs_cis buffer recomputed on the real device + finite.
        assert not model.freqs_cis.is_meta
        assert torch.isfinite(torch.view_as_real(model.freqs_cis)).all()

    def test_init_weights_buffer_device_default_backcompat(self):
        """No-arg init_weights still works (backward-compat for dense path)."""
        torch = _torch_or_skip()
        m = _import()
        cfg = m.ModelArgs(
            dim=64, n_layers=1, n_heads=4, n_kv_heads=4, vocab_size=128,
            multiple_of=32, hidden_dim=128, rope_theta=10000.0, max_seq_len=64,
        )
        model = m.Transformer.from_model_args(cfg)  # dense, on CPU
        model.init_weights()  # no buffer_device — must not raise
        assert torch.isfinite(model.tok_embeddings.weight).all()
