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


class TestSavePretrainedWithFallback:
    """Tests for ``_save_pretrained_with_fallback``.

    Uses a fake model whose ``save_pretrained`` we control, so we can
    drive the success / retry / no-catch paths without touching disk
    or pulling in real HF model weights.
    """

    class _FakeModel:
        """Minimal stand-in for an HF model with save_pretrained."""

        def __init__(self, raises_on=None):
            # Tuple of exception instances to raise on successive calls.
            self.raises_on = list(raises_on or [])
            self.calls: list[dict] = []

        def save_pretrained(self, output_dir, **kwargs):
            self.calls.append({"output_dir": output_dir, **kwargs})
            if self.raises_on:
                exc = self.raises_on.pop(0)
                if exc is not None:
                    raise exc

    def test_success_no_retry(self, hf_module, tmp_path):
        """First call succeeds → no retry, no safe_serialization=False."""
        model = self._FakeModel()
        save_fn = lambda *a, **kw: None
        out = str(tmp_path / "out")
        hf_module._save_pretrained_with_fallback(
            model, out,
            is_main_process=True,
            save_function=save_fn,
        )
        assert len(model.calls) == 1
        # First (and only) call should NOT pass safe_serialization
        assert "safe_serialization" not in model.calls[0]
        # All other kwargs are forwarded verbatim
        assert model.calls[0]["output_dir"] == out
        assert model.calls[0]["is_main_process"] is True
        assert model.calls[0]["save_function"] is save_fn

    def test_retries_on_oserror_with_safe_serialization_false(
        self, hf_module, tmp_path, caplog,
    ):
        """OSError 'Argument list too long' triggers retry with
        safe_serialization=False, and a warning is logged.
        """
        import logging

        # Simulate Lustre E2BIG failure
        e2big = OSError(7, "Argument list too long")
        save_fn = lambda *a, **kw: None
        out = str(tmp_path / "out")
        model = self._FakeModel(raises_on=[e2big, None])
        with caplog.at_level(logging.WARNING, logger="ezpz.examples.hf"):
            hf_module._save_pretrained_with_fallback(
                model, out,
                is_main_process=True,
                save_function=save_fn,
            )
        assert len(model.calls) == 2
        # First attempt — no safe_serialization arg
        assert "safe_serialization" not in model.calls[0]
        # Retry — safe_serialization=False, all other kwargs forwarded
        assert model.calls[1]["safe_serialization"] is False
        assert model.calls[1]["output_dir"] == out
        assert model.calls[1]["is_main_process"] is True
        assert model.calls[1]["save_function"] is save_fn
        # Warning identifies the exception type and mentions retrying
        retry_warnings = [
            r for r in caplog.records
            if "safetensors" in r.message and "retrying" in r.message
        ]
        assert len(retry_warnings) == 1
        assert "OSError" in retry_warnings[0].message

    def test_retries_on_safetensor_error_when_available(
        self, hf_module, tmp_path,
    ):
        """SafetensorError from the rust core triggers retry."""
        try:
            from safetensors import SafetensorError
        except ImportError:
            pytest.skip("safetensors not installed")
        model = self._FakeModel(raises_on=[SafetensorError("rust boom"), None])
        hf_module._save_pretrained_with_fallback(
            model, str(tmp_path / "out"),
            is_main_process=True,
            save_function=lambda *a, **kw: None,
        )
        assert len(model.calls) == 2
        assert model.calls[1]["safe_serialization"] is False

    def test_does_not_catch_genuine_bugs(self, hf_module, tmp_path):
        """TypeError / AttributeError must propagate, not be silently retried."""
        model = self._FakeModel(raises_on=[TypeError("genuine bug")])
        with pytest.raises(TypeError, match="genuine bug"):
            hf_module._save_pretrained_with_fallback(
                model, str(tmp_path / "out"),
                is_main_process=True,
                save_function=lambda *a, **kw: None,
            )
        # Only the first call happened — we did not retry under
        # safe_serialization=False after a non-Lustre failure.
        assert len(model.calls) == 1

    def test_does_not_catch_keyboard_interrupt(self, hf_module, tmp_path):
        """KeyboardInterrupt must propagate even though the previous
        BaseException-typed signature could have caught it."""
        model = self._FakeModel(raises_on=[KeyboardInterrupt()])
        with pytest.raises(KeyboardInterrupt):
            hf_module._save_pretrained_with_fallback(
                model, str(tmp_path / "out"),
                is_main_process=True,
                save_function=lambda *a, **kw: None,
            )
        assert len(model.calls) == 1
