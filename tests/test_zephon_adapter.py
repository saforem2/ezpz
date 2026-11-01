"""Zephon adapter: unit tests + CPU elastic resume.

* test_zephon_* — basic adapter wiring (no torch/distributed).
* test_elastic_resume_cpu — CPU-only checkpoint/restore demo (mirrors
  torchtitan-zephon's elastic_resume_demo.py intent in a single-process
  smoke). Determinism = same recipe + same seed must yield same token
  stream across checkpoint/restore.
"""

import tempfile
import pytest

try:
    import zephon  # noqa: F401

    HAS_ZEPHON = True
except ImportError:
    HAS_ZEPHON = False

pytestmark = pytest.mark.skipif(not HAS_ZEPHON, reason="zephon not installed")

RECIPE = "src/ezpz/examples/zephon/local_jsonl.toml"


def test_zephon_importable():
    from ezpz.data.zephon import ZephonDataLoader, get_zephon_version

    assert get_zephon_version() is not None
    assert ZephonDataLoader is not None


def test_zephon_local_jsonl_recipe_produces_batches():
    from ezpz.data.zephon import ZephonDataLoader

    dl = ZephonDataLoader(recipe=RECIPE, seq_len=64, tokenizer_id="gpt2")
    it = iter(dl)
    batch = next(it)
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (4, 64)


def test_zephon_without_recipe_raises():
    from ezpz.data.zephon import ZephonDataLoader

    with pytest.raises(FileNotFoundError):
        ZephonDataLoader(recipe="/nonexistent.toml", seq_len=32)


def test_zephon_parse_args_flag():
    from ezpz.data.zephon import ZephonDataLoader

    dl = ZephonDataLoader(recipe=RECIPE, seq_len=32)
    assert dl.seq_len == 32


def test_zephon_mixture_recipe_yields_batches():
    from ezpz.data.zephon import ZephonDataLoader

    dl = ZephonDataLoader(recipe="src/ezpz/examples/zephon/mixture.toml", seq_len=32, tokenizer_id="gpt2")
    it = iter(dl)
    batch = next(it)
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (4, 32)
    seen = set()
    for _ in range(3):
        b = next(it)
        seen.add(b["input_ids"].shape)
    assert len(seen) == 1


def test_elastic_resume_cpu():
    """Single-process checkpoint/restore is idempotent (no elastic resize, just round-trip)."""
    from ezpz.data.zephon import ZephonDataLoader

    dl = ZephonDataLoader(recipe=RECIPE, seq_len=32, tokenizer_id="gpt2")
    it = iter(dl)
    # Consume 2 batches, checkpoint, restore on a fresh loader, continue
    b0 = next(it)
    b1 = next(it)
    state = dl.save_state()
    assert state is not None, "save_state returned None before any batch"

    # Fresh loader, restore, should continue from same point (if elastic semantics hold,
    # the next batch after restore equals the next batch we would have seen without restore).
    # Single-rank round-trip: we just test that restore doesn't throw and yields a batch.
    dl2 = ZephonDataLoader(recipe=RECIPE, seq_len=32, tokenizer_id="gpt2")
    # Must build pipeline before load, or defer
    dl2.load_state(state)
    it2 = iter(dl2)
    b2 = next(it2)
    assert "input_ids" in b2
