"""Regression: `--hf-limit 0` must mean "no limit", not "no rows".

`diffusion.py` selected rows with `dataset.select(range(limit))`. The flag
is documented as "0 (default) = no limit (use the full dataset)", but
`range(0)` is empty -- so the DEFAULT invocation selected nothing and
raised `ValueError: No text rows found from HF dataset.` on a dataset that
loaded perfectly.

Measured on Aurora during `ezpz benchmark --model large`: stanfordnlp/imdb
loads 25000 rows with a `text` column, and the diffusion example still died
in 14 s. Every default-invocation run with an HF dataset was affected.

`ezpz.data.hf.get_hf_text_dataset` already treated `limit <= 0` as "all
rows"; these tests pin the two to the same convention.
"""

from __future__ import annotations

import pytest

from ezpz.examples.diffusion import load_hf_texts


class _FakeDataset:
    """Minimal stand-in for a `datasets.Dataset`.

    Only what `load_hf_texts` touches: `column_names`, `len`, and a
    `select(indices)` that honours the requested range.
    """

    def __init__(self, rows: int, column: str = "text") -> None:
        self._rows = [{column: f"row-{i}"} for i in range(rows)]
        self.column_names = [column]

    def __len__(self) -> int:
        return len(self._rows)

    def select(self, indices):
        return [self._rows[i] for i in indices]


@pytest.fixture
def patched_load(monkeypatch):
    """Patch `datasets.load_dataset` to return a fake of a given size."""

    def _install(rows: int, column: str = "text"):
        import datasets

        monkeypatch.setattr(
            datasets, "load_dataset", lambda *a, **k: _FakeDataset(rows, column)
        )

    return _install


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_uses_every_row(patched_load, limit):
    """The regression: 0 (and any <= 0) means the whole dataset."""
    pytest.importorskip("datasets")
    patched_load(25)
    texts = load_hf_texts("x/y", "train", "text", limit)
    assert len(texts) == 25, (
        f"limit={limit} is documented as 'no limit' but selected {len(texts)} rows"
    )


def test_positive_limit_truncates(patched_load):
    pytest.importorskip("datasets")
    patched_load(25)
    assert len(load_hf_texts("x/y", "train", "text", 10)) == 10


def test_limit_larger_than_dataset_is_clamped(patched_load):
    """`select(range(99999))` on 25 rows would raise; clamp instead."""
    pytest.importorskip("datasets")
    patched_load(25)
    assert len(load_hf_texts("x/y", "train", "text", 99999)) == 25


def test_missing_text_column_still_raises(patched_load):
    """The guard that SHOULD fire keeps firing."""
    pytest.importorskip("datasets")
    patched_load(25, column="content")
    with pytest.raises(ValueError, match="text_column"):
        load_hf_texts("x/y", "train", "text", 0)


def test_genuinely_empty_dataset_still_raises(patched_load):
    """An empty dataset is a real error -- do not mask it with the fix."""
    pytest.importorskip("datasets")
    patched_load(0)
    with pytest.raises(ValueError, match="No text rows found"):
        load_hf_texts("x/y", "train", "text", 0)
