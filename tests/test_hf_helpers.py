"""Tests for pure-Python helpers in ``ezpz.examples.hf``.

The full module pulls in transformers/datasets/accelerate which are
heavy and may not be importable in every environment.  We import the
helper lazily and skip if hf can't load.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def hf_module():
    try:
        from ezpz.examples import hf
    except Exception as exc:
        pytest.skip(f"ezpz.examples.hf unavailable: {exc}")
    return hf


class TestStripMetricPrefix:
    """Tests for ``_strip_metric_prefix``."""

    def test_simple(self, hf_module):
        result = hf_module._strip_metric_prefix(
            "train/loss=0.5 train/dt=0.1", "train/",
        )
        assert result == "loss=0.5 dt=0.1"

    def test_only_strips_at_token_start(self, hf_module):
        """A metric whose name happens to contain the prefix as a
        substring (e.g. cosine_train/foo) should not be mangled."""
        result = hf_module._strip_metric_prefix(
            "train/loss=0.5 cosine_train/foo=0.1", "train/",
        )
        assert result == "loss=0.5 cosine_train/foo=0.1"

    def test_no_match_passthrough(self, hf_module):
        result = hf_module._strip_metric_prefix(
            "loss=0.5 dt=0.1", "train/",
        )
        assert result == "loss=0.5 dt=0.1"

    def test_empty(self, hf_module):
        assert hf_module._strip_metric_prefix("", "train/") == ""

    def test_eval_prefix(self, hf_module):
        result = hf_module._strip_metric_prefix(
            "eval/loss=0.5 eval/perplexity=2.5", "eval/",
        )
        assert result == "loss=0.5 perplexity=2.5"


class TestSafetensorsSaveErrors:
    """Tests for ``_safetensors_save_errors``."""

    def test_includes_oserror_runtimeerror_valueerror(self, hf_module):
        errors = hf_module._safetensors_save_errors()
        assert OSError in errors
        assert RuntimeError in errors
        assert ValueError in errors

    def test_includes_safetensor_error_when_installed(self, hf_module):
        """When safetensors is importable, its native error type is added.

        Skip if safetensors is not installed in this environment.
        """
        try:
            from safetensors import SafetensorError
        except ImportError:
            pytest.skip("safetensors not installed")
        errors = hf_module._safetensors_save_errors()
        assert SafetensorError in errors

    def test_module_constant_matches_function(self, hf_module):
        """The cached _SAFETENSORS_SAVE_ERRORS matches a fresh call."""
        assert hf_module._SAFETENSORS_SAVE_ERRORS == hf_module._safetensors_save_errors()
