"""Tests for globusfs streaming integration (storage_options forwarding).

Covers HfDataTrainingArguments storage_options parsing and the
forwarding of storage_options / streaming / data_files through
split_dataset and get_hf_datasets without hitting the network.
Runs without torch — stubs it so ezpz.data.hf imports.
"""

from __future__ import annotations

import itertools
import json
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub torch so ezpz.data.hf imports without it (macOS .venv has no torch)
# ---------------------------------------------------------------------------

def _ensure_torch_stubs():
    if "torch" in sys.modules:
        return

    torch_mod = types.ModuleType("torch")
    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.is_available = lambda: False
    dist_mod.is_initialized = lambda: False
    dist_mod.get_rank = lambda: 0
    dist_mod.barrier = MagicMock()
    torch_mod.distributed = dist_mod
    torch_mod.Generator = MagicMock()
    torch_mod.tensor = MagicMock()
    torch_mod.Tensor = MagicMock
    torch_mod.long = "long"
    torch_mod.float = "float"
    torch_mod.get_num_threads = lambda: 1
    torch_mod.set_num_threads = lambda n: None
    torch_mod.get_num_interop_threads = lambda: 1

    utils_data_mod = types.ModuleType("torch.utils.data")
    utils_data_mod.IterableDataset = object
    utils_data_mod.DistributedSampler = MagicMock
    utils_data_mod.DataLoader = MagicMock
    utils_data_mod.Dataset = object
    utils_data_mod.get_worker_info = MagicMock(return_value=None)

    torch_utils_mod = types.ModuleType("torch.utils")

    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = dist_mod
    sys.modules["torch.utils"] = torch_utils_mod
    sys.modules["torch.utils.data"] = utils_data_mod

    # datasets stub if missing
    if "datasets" not in sys.modules:
        ds_mod = types.ModuleType("datasets")
        ds_mod.load_dataset = MagicMock()
        ds_mod.Dataset = MagicMock
        ds_mod.IterableDataset = MagicMock
        ds_mod.DatasetDict = MagicMock
        ds_mod.IterableDatasetDict = MagicMock
        sys.modules["datasets"] = ds_mod

    # transformers stub
    if "transformers" not in sys.modules:
        tf_mod = types.ModuleType("transformers")
        tf_mod.AutoTokenizer = MagicMock()
        tf_mod.utils = types.ModuleType("transformers.utils")
        tf_mod.utils.versions = types.ModuleType("transformers.utils.versions")
        tf_mod.utils.versions.require_version = lambda *a, **kw: None
        tf_mod.utils.logging = types.ModuleType("transformers.utils.logging")
        sys.modules["transformers"] = tf_mod
        sys.modules["transformers.utils"] = tf_mod.utils
        sys.modules["transformers.utils.logging"] = tf_mod.utils.logging
        sys.modules["transformers.utils.versions"] = tf_mod.utils.versions


_ensure_torch_stubs()

from ezpz.configs import HfDataTrainingArguments
import ezpz.data.hf as hf_module
from ezpz.data.hf import get_hf_datasets, split_dataset


# ---------------------------------------------------------------------------
# 1-3: HfDataTrainingArguments storage_options parsing
# ---------------------------------------------------------------------------

def test_storage_options_default_is_none():
    args = HfDataTrainingArguments(dataset_name="test/dummy")
    assert args.storage_options is None


def test_storage_options_json_string_parsed():
    args = HfDataTrainingArguments(
        dataset_name="test/dummy",
        storage_options='{"globus": {"anon": true}}',
    )
    assert args.storage_options == {"globus": {"anon": True}}


def test_storage_options_dict_passthrough():
    so = {"globus": {"https_url": "https://example.com"}}
    args = HfDataTrainingArguments(
        dataset_name="test/dummy",
        storage_options=so,
    )
    assert args.storage_options == so
    assert args.storage_options is so


# ---------------------------------------------------------------------------
# 4: split_dataset forwards storage_options (streaming=True)
# ---------------------------------------------------------------------------

def test_split_dataset_forwards_storage_options():
    storage_options = {"globus": {"https_url": "https://example.com"}}
    args = HfDataTrainingArguments(
        dataset_name="test/dummy",
        streaming=True,
        storage_options=storage_options,
    )

    with patch("ezpz.data.hf.datasets.load_dataset") as mock_load, \
         patch("ezpz.data.hf.datasets.IterableDatasetDict", side_effect=lambda x: x), \
         patch("ezpz.data.hf.datasets.DatasetDict", side_effect=lambda x: x):
        mock_load.return_value = MagicMock()

        split_dataset(
            args,
            train_split_name="train",
            validation_split_name="validation",
        )

        assert mock_load.call_count >= 2, f"expected >=2 calls, got {mock_load.call_count}"
        for c in mock_load.call_args_list:
            assert c.kwargs.get("storage_options") == storage_options, (
                f"storage_options not forwarded in call {c}"
            )
            assert c.kwargs.get("streaming") is True


# ---------------------------------------------------------------------------
# 5: split_dataset without storage_options -> None
# ---------------------------------------------------------------------------

def test_split_dataset_storage_options_none_by_default():
    args = HfDataTrainingArguments(
        dataset_name="test/dummy",
        streaming=True,
    )
    assert args.storage_options is None

    with patch("ezpz.data.hf.datasets.load_dataset") as mock_load, \
         patch("ezpz.data.hf.datasets.IterableDatasetDict", side_effect=lambda x: x), \
         patch("ezpz.data.hf.datasets.DatasetDict", side_effect=lambda x: x):
        mock_load.return_value = MagicMock()

        split_dataset(
            args,
            train_split_name="train",
            validation_split_name="validation",
        )

        assert mock_load.call_count >= 2
        for c in mock_load.call_args_list:
            assert c.kwargs.get("storage_options") is None
            assert c.kwargs.get("streaming") is True


# ---------------------------------------------------------------------------
# Helpers for get_hf_datasets data_files branch tests (6,7,8)
# ---------------------------------------------------------------------------

@contextmanager
def _dummy_main_process_first(*_a, **_kw):
    yield


def _make_mock_raw_datasets():
    mock_ds = MagicMock()
    mock_ds.features = ["text"]
    mock_tokenized = MagicMock()
    mock_tokenized.features = ["text"]
    mock_lm = MagicMock()
    mock_lm.__getitem__ = MagicMock(return_value=MagicMock())
    mock_tokenized.map = MagicMock(return_value=mock_lm)

    raw = MagicMock()
    raw.keys.return_value = ["train", "validation"]
    raw.__contains__ = lambda self, k: k in ["train", "validation"]
    raw.__getitem__.return_value = mock_ds
    raw.map = MagicMock(return_value=mock_tokenized)
    return raw


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.model_max_length = 1024
    tok.pad_token = "[PAD]"
    tok.pad_token_id = 0
    tok.vocab_size = 100
    return tok


def _patch_missing_globals():
    import itertools as _it
    patches = []
    mock_training_args = MagicMock()
    mock_training_args.main_process_first = _dummy_main_process_first
    mock_training_args.do_train = False
    mock_training_args.do_eval = False
    patches.append(patch.object(hf_module, "training_args", mock_training_args, create=True))
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 1024
    patches.append(patch.object(hf_module, "config", mock_config, create=True))
    patches.append(patch.object(hf_module, "chain", _it.chain, create=True))
    patches.append(patch.object(hf_module, "evaluate", MagicMock(), create=True))
    return patches


# ---------------------------------------------------------------------------
# 6: get_hf_datasets data_files branch forwards streaming + storage_options
# ---------------------------------------------------------------------------

def test_get_hf_datasets_data_files_forwards_streaming_and_storage_options():
    storage_options = {"globus": {"anon": True}}
    args = HfDataTrainingArguments(
        train_file="globus://my-bucket/data.json",
        streaming=True,
        storage_options=storage_options,
    )
    assert args.dataset_name is None

    mock_raw = _make_mock_raw_datasets()
    patches = _patch_missing_globals()
    for p in patches:
        p.start()

    try:
        with patch("ezpz.data.hf.datasets.load_dataset", return_value=mock_raw) as mock_load, \
             patch("ezpz.data.hf.AutoTokenizer.from_pretrained", return_value=_make_mock_tokenizer()):
            try:
                get_hf_datasets(
                    args,
                    tokenizer_name="dummy-tokenizer",
                    model_name_or_path=None,
                )
            except Exception:
                pass

            assert mock_load.call_count >= 1, "expected at least one load_dataset call for data_files branch"
            first_kwargs = mock_load.call_args_list[0].kwargs
            assert first_kwargs.get("streaming") is True, f"streaming not forwarded: {first_kwargs}"
            assert first_kwargs.get("storage_options") == storage_options, f"storage_options not forwarded: {first_kwargs}"
            for c in mock_load.call_args_list:
                assert c.kwargs.get("streaming") is True
                assert c.kwargs.get("storage_options") == storage_options
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# 7: globus:// URLs are passed through unchanged in data_files
# ---------------------------------------------------------------------------

def test_globus_url_passthrough_in_data_files():
    globus_url = "globus://my-bucket/path/to/train.json"
    args = HfDataTrainingArguments(
        train_file=globus_url,
        streaming=True,
        storage_options={"globus": {"anon": True}},
    )

    mock_raw = _make_mock_raw_datasets()
    patches = _patch_missing_globals()
    for p in patches:
        p.start()

    try:
        with patch("ezpz.data.hf.datasets.load_dataset", return_value=mock_raw) as mock_load, \
             patch("ezpz.data.hf.AutoTokenizer.from_pretrained", return_value=_make_mock_tokenizer()):
            try:
                get_hf_datasets(
                    args,
                    tokenizer_name="dummy-tokenizer",
                    model_name_or_path=None,
                )
            except Exception:
                pass

            assert mock_load.call_count >= 1
            first_kwargs = mock_load.call_args_list[0].kwargs
            data_files = first_kwargs.get("data_files")
            if data_files is None:
                assert False, f"data_files not in kwargs: {first_kwargs}, args: {mock_load.call_args_list[0].args}"
            assert isinstance(data_files, dict), f"data_files should be dict, got {data_files}"
            assert data_files.get("train") == globus_url, f"globus URL not passed through: {data_files}"
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# 8: storage_options forwarded even when streaming=False
# ---------------------------------------------------------------------------

def test_storage_options_with_streaming_false_still_forwarded():
    storage_options = {"globus": {"https_url": "https://example.com"}}
    args = HfDataTrainingArguments(
        dataset_name="test/dummy",
        streaming=False,
        storage_options=storage_options,
    )
    assert args.streaming is False

    with patch("ezpz.data.hf.datasets.load_dataset") as mock_load, \
         patch("ezpz.data.hf.datasets.IterableDatasetDict", side_effect=lambda x: x), \
         patch("ezpz.data.hf.datasets.DatasetDict", side_effect=lambda x: x):
        mock_load.return_value = MagicMock()

        split_dataset(
            args,
            train_split_name="train",
            validation_split_name="validation",
        )

        assert mock_load.call_count >= 2
        for c in mock_load.call_args_list:
            assert c.kwargs.get("storage_options") == storage_options, (
                f"storage_options must be forwarded even when streaming=False, got {c.kwargs.get('storage_options')}"
            )
            assert c.kwargs.get("streaming") is False
