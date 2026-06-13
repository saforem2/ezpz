"""Tests for ``ezpz.data.hf`` rank-0-first cache plumbing.

Regression coverage for the HF ``datasets`` filesystem-cache race
fixed by ``_main_process_first``. Pre-fix every rank called
``load_dataset`` / ``Dataset.map`` in parallel against the same
fingerprint-named Arrow cache file on a shared filesystem; ranks
raced for ``os.chmod`` and partial Arrow stream reads
(``FileNotFoundError`` / ``pyarrow.lib.ArrowInvalid``). Observed
on a 48-rank fsdp_tp run.

These tests inspect the order of ``torch.distributed.barrier``
calls vs the wrapped work, without actually initialising a real
process group.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import ezpz.data.hf as hf


class TestMainProcessFirst:
    """``_main_process_first`` must:

    1. No-op when ``torch.distributed`` isn't initialised (single-
       process callers, notebooks, unit tests).
    2. Issue rank-0 → work → barrier on rank 0 (releases everyone).
    3. Issue barrier (wait) → work → no second barrier on non-rank-0.
    4. Still release the barrier even if the wrapped work raises
       (otherwise rank 0's crash leaves all other ranks hung).
    """

    def test_noop_when_distributed_unavailable(self, monkeypatch):
        """No barriers when ``torch.distributed.is_available()`` is False."""
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: False
        )
        # If is_available is False, is_initialized must not even be
        # reached — patch it to blow up so we'd notice if it were.
        monkeypatch.setattr(
            torch.distributed,
            "is_initialized",
            MagicMock(side_effect=AssertionError("should not be called")),
        )
        mock_barrier = MagicMock()
        monkeypatch.setattr(torch.distributed, "barrier", mock_barrier)

        with hf._main_process_first():
            pass

        mock_barrier.assert_not_called()

    def test_noop_when_distributed_not_initialised(self, monkeypatch):
        """No barriers when distributed is available but not initialised."""
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: False
        )
        mock_barrier = MagicMock()
        monkeypatch.setattr(torch.distributed, "barrier", mock_barrier)

        with hf._main_process_first():
            pass

        mock_barrier.assert_not_called()

    def test_rank_zero_runs_work_first_then_releases(self, monkeypatch):
        """On rank 0: work runs first, ONE barrier fires after (release)."""
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        events: list[str] = []
        monkeypatch.setattr(
            torch.distributed,
            "barrier",
            lambda *a, **kw: events.append("barrier"),
        )

        with hf._main_process_first():
            events.append("work")

        assert events == ["work", "barrier"], (
            f"Rank 0 must run work BEFORE the release barrier; got {events}. "
            "Pre-fix, all ranks ran work concurrently and raced for the "
            "Arrow cache file."
        )

    def test_non_rank_zero_waits_then_runs(self, monkeypatch):
        """On non-rank-0: ONE barrier fires before work (wait for rank 0)."""
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

        events: list[str] = []
        monkeypatch.setattr(
            torch.distributed,
            "barrier",
            lambda *a, **kw: events.append("barrier"),
        )

        with hf._main_process_first():
            events.append("work")

        assert events == ["barrier", "work"], (
            f"Non-rank-0 must wait at a barrier BEFORE running work; "
            f"got {events}. Without the wait, all ranks would race "
            "for the Arrow cache file rank 0 is still writing."
        )

    def test_rank_zero_releases_on_exception(self, monkeypatch):
        """If rank 0's work raises, the release barrier MUST still fire.

        Otherwise every other rank is stuck at their first barrier
        forever (or until torch's collective timeout).
        """
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        mock_barrier = MagicMock()
        monkeypatch.setattr(torch.distributed, "barrier", mock_barrier)

        with pytest.raises(RuntimeError, match="boom"):
            with hf._main_process_first():
                raise RuntimeError("boom")

        mock_barrier.assert_called_once()

    def test_non_rank_zero_does_not_call_extra_barrier_on_exception(
        self, monkeypatch
    ):
        """Non-rank-0: exactly one barrier (the upfront wait), even on raise.

        The release barrier is rank-0's job; a non-rank-0 failure
        should not double-barrier the group.
        """
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True
        )
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

        mock_barrier = MagicMock()
        monkeypatch.setattr(torch.distributed, "barrier", mock_barrier)

        with pytest.raises(RuntimeError, match="boom"):
            with hf._main_process_first():
                raise RuntimeError("boom")

        mock_barrier.assert_called_once()


class TestGetHfTextDatasetRankGating:
    """``get_hf_text_dataset`` must wrap both cache-writing calls
    (``datasets.load_dataset`` and ``Dataset.map``) in
    ``_main_process_first``.

    Pre-fix both calls fired from every rank concurrently — the
    exact failure mode in the original traceback.
    """

    def _make_fake_dataset(self):
        """Build a MagicMock that quacks like a ``datasets.Dataset``."""
        ds = MagicMock()
        ds.column_names = ["text"]
        ds.__len__ = MagicMock(return_value=4)
        # `dataset.map(...)` returns a tokenized dataset; that
        # tokenized dataset gets `.set_format` and attribute
        # assignment, so it needs to be a MagicMock too.
        tokenized = MagicMock()
        ds.map = MagicMock(return_value=tokenized)
        return ds, tokenized

    def test_load_dataset_wrapped_in_main_process_first(self, monkeypatch):
        """``load_dataset`` must be inside the context manager."""
        ds, _tokenized = self._make_fake_dataset()
        events: list[str] = []

        def _fake_load(*_a, **_kw):
            events.append("load_dataset")
            return ds

        # Patch the context manager to record entry/exit; if
        # load_dataset fires outside it, ordering will reveal that.
        from contextlib import contextmanager

        @contextmanager
        def _recording_cm():
            events.append("enter_main_first")
            try:
                yield
            finally:
                events.append("exit_main_first")

        monkeypatch.setattr(hf, "_main_process_first", _recording_cm)
        monkeypatch.setattr(hf.datasets, "load_dataset", _fake_load)
        monkeypatch.setattr(
            hf.AutoTokenizer,
            "from_pretrained",
            lambda *_a, **_kw: MagicMock(pad_token="x", pad_token_id=0, vocab_size=32),
        )

        hf.get_hf_text_dataset(
            dataset_name="dummy",
            split="train",
            text_column="text",
            tokenizer_name="dummy",
            seq_len=16,
            limit=0,
            seed=1,
        )

        # load_dataset must appear between an enter/exit pair.
        assert "load_dataset" in events
        ld_idx = events.index("load_dataset")
        # Find the surrounding enter/exit pair.
        enters = [i for i, e in enumerate(events) if e == "enter_main_first" and i < ld_idx]
        exits = [i for i, e in enumerate(events) if e == "exit_main_first" and i > ld_idx]
        assert enters and exits, (
            f"load_dataset is not wrapped in _main_process_first; "
            f"events: {events}. The cache-populating call MUST be "
            "rank-0-gated or every rank races for the same file."
        )

    def test_map_wrapped_in_main_process_first(self, monkeypatch):
        """``dataset.map(...)`` must be inside the context manager."""
        ds, _tokenized = self._make_fake_dataset()
        events: list[str] = []

        def _fake_load(*_a, **_kw):
            return ds

        def _record_map(*_a, **_kw):
            events.append("map")
            t = MagicMock()
            return t

        ds.map = _record_map

        from contextlib import contextmanager

        @contextmanager
        def _recording_cm():
            events.append("enter_main_first")
            try:
                yield
            finally:
                events.append("exit_main_first")

        monkeypatch.setattr(hf, "_main_process_first", _recording_cm)
        monkeypatch.setattr(hf.datasets, "load_dataset", _fake_load)
        monkeypatch.setattr(
            hf.AutoTokenizer,
            "from_pretrained",
            lambda *_a, **_kw: MagicMock(pad_token="x", pad_token_id=0, vocab_size=32),
        )

        hf.get_hf_text_dataset(
            dataset_name="dummy",
            split="train",
            text_column="text",
            tokenizer_name="dummy",
            seq_len=16,
            limit=0,
            seed=1,
        )

        assert "map" in events
        m_idx = events.index("map")
        enters = [i for i, e in enumerate(events) if e == "enter_main_first" and i < m_idx]
        exits = [i for i, e in enumerate(events) if e == "exit_main_first" and i > m_idx]
        assert enters and exits, (
            f"dataset.map is not wrapped in _main_process_first; "
            f"events: {events}. Tokenization writes to a shared "
            "Arrow cache file and MUST be rank-0-gated."
        )


class TestLoadHfTextsLimitContract:
    """``load_hf_texts(limit=...)`` contract pinning.

    Pre-PR-#166, ``limit=0`` raised ``ValueError``. PR #166 changed
    the semantics: ``limit <= 0`` now means "use the full dataset",
    matching ``get_hf_text_dataset``'s convention. These tests pin
    the new contract so a future "validate limit > 0" refactor
    can't silently regress it.
    """

    def _make_fake_load_dataset(self, n_rows: int):
        """Build a fake ``datasets.load_dataset`` returning ``n_rows`` rows.

        The returned object only needs to support the surface used by
        ``load_hf_texts``: ``column_names``, ``len()``, ``shuffle()``,
        and ``select()``.
        """
        rows = [{"text": f"row {i} text payload"} for i in range(n_rows)]

        class _FakeDataset:
            def __init__(self, _rows):
                self._rows = _rows
                self.column_names = ["text"]

            def __len__(self):
                return len(self._rows)

            def shuffle(self, seed=None):
                # Deterministic "no-op shuffle" — we just want to
                # verify the limit-handling branch, not RNG behavior.
                return self

            def select(self, indices):
                return _FakeDataset([self._rows[i] for i in indices])

            def __iter__(self):
                return iter(self._rows)

        ds = _FakeDataset(rows)

        def _fake_load_dataset(*_a, **_kw):
            return ds

        return _fake_load_dataset

    def test_limit_zero_returns_all_rows(self, monkeypatch):
        """``limit=0`` means "no limit" — must return every row, not
        raise ValueError (pre-PR-#166 behavior) and not return [].
        """
        fake_load = self._make_fake_load_dataset(n_rows=10)
        # ``load_hf_texts`` does ``from datasets import load_dataset``
        # at call time, so patch the source module's symbol.
        import datasets as _datasets
        monkeypatch.setattr(_datasets, "load_dataset", fake_load)

        texts = hf.load_hf_texts(
            dataset_name="dummy",
            split="train",
            text_column="text",
            limit=0,
        )
        assert len(texts) == 10, (
            f"limit=0 must return all 10 rows; got {len(texts)}. "
            "PR #166 redefined limit<=0 as 'no limit' — a regression "
            "to the pre-#166 'raise ValueError' contract would break "
            "callers that pass limit=0 to mean 'use the full dataset'."
        )

    def test_limit_negative_returns_all_rows(self, monkeypatch):
        """``limit=-1`` also means "no limit" per the implementation
        (``if limit <= 0 or limit >= total``)."""
        fake_load = self._make_fake_load_dataset(n_rows=10)
        import datasets as _datasets
        monkeypatch.setattr(_datasets, "load_dataset", fake_load)

        texts = hf.load_hf_texts(
            dataset_name="dummy",
            split="train",
            text_column="text",
            limit=-1,
        )
        assert len(texts) == 10, (
            f"limit=-1 must return all 10 rows; got {len(texts)}. "
            "Any limit <= 0 is the 'no limit' sentinel."
        )

    def test_limit_positive_truncates(self, monkeypatch):
        """A positive ``limit`` below the dataset size must subsample
        exactly that many rows."""
        fake_load = self._make_fake_load_dataset(n_rows=10)
        import datasets as _datasets
        monkeypatch.setattr(_datasets, "load_dataset", fake_load)

        texts = hf.load_hf_texts(
            dataset_name="dummy",
            split="train",
            text_column="text",
            limit=5,
        )
        assert len(texts) == 5, (
            f"limit=5 must return exactly 5 rows; got {len(texts)}."
        )
