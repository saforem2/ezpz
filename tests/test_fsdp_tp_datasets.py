import pytest
import torch
import torch.nn.functional as F

from ezpz.data.hf import ToyTextDataset, build_vocab, load_hf_texts
from ezpz.data.distributed import get_random_dataset_fsdp_tp
from ezpz.models.llama import ModelArgs, Transformer


def _build_tiny_model(vocab_size: int, seq_len: int) -> Transformer:
    config = ModelArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=8,
        batch_size=2,
        max_seq_len=seq_len,
        depth_init=True,
    )
    return Transformer.from_model_args(config)


def _loss_is_finite(model: Transformer, batch: torch.Tensor) -> bool:
    tokens = batch.to(torch.long)
    inp = tokens[:, :-1]
    labels = tokens[:, 1:]
    pred = model(inp)
    loss = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return bool(torch.isfinite(loss).all().item())


def test_fsdp_tp_dataset_random_loss_is_finite():
    torch.manual_seed(0)
    seq_len = 64
    data = get_random_dataset_fsdp_tp(
        batch_size=2,
        vocab_size=128,
        seq_length=seq_len,
        broadcast_within_tp=False,
        drop_last=False,
    )
    batch = next(iter(data["dataloader"]))
    model = _build_tiny_model(vocab_size=128, seq_len=seq_len)
    assert _loss_is_finite(model, batch)


def test_fsdp_tp_dataset_hf_loss_is_finite():
    pytest.importorskip("datasets")
    torch.manual_seed(0)
    seq_len = 64
    texts = load_hf_texts(
        dataset_name="stanfordnlp/imdb",
        split="train",
        text_column="text",
        limit=8,
    )
    vocab, _ = build_vocab(texts)
    dataset = ToyTextDataset(texts, vocab, seq_len=seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, drop_last=False
    )
    batch = next(iter(dataloader))
    model = _build_tiny_model(vocab_size=len(vocab), seq_len=seq_len)
    assert _loss_is_finite(model, batch)
