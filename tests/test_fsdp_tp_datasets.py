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


def test_hf_dataloader_drop_last_yields_static_batch_shape():
    """Every batch must have a static batch dim under ``drop_last=True``.

    Regression for the ``--loss-impl=compiled`` step-53 OOM: a ragged
    final batch (batch_size 2 -> 1) makes torch.compile mark the batch
    dim dynamic and recompile at the epoch boundary; the recompiled
    backward drops the fused CE and materializes the full (B*T, vocab)
    logits grad (~15.6 GiB for agpt-2b) -> XPU OOM. The fsdp_tp HF loader
    now sets drop_last=True, so no ragged batch reaches the compiled step.

    Dataset size (5) is deliberately NOT a multiple of batch_size (2):
    with drop_last=False the last batch would be size 1 (the recompile
    trigger); with drop_last=True it must be dropped.
    """

    class _TinyDS(torch.utils.data.Dataset):
        def __init__(self, n, seq_len):
            self.data = torch.arange(n * seq_len).reshape(n, seq_len)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    batch_size = 2
    seq_len = 8
    dataset = _TinyDS(5, seq_len)  # 5 % 2 == 1 -> ragged tail without drop_last

    # drop_last=True (the fix): only full-size batches, none ragged.
    dl_fixed = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    batches = list(dl_fixed)
    assert len(batches) == 2  # floor(5 / 2)
    assert all(b.shape[0] == batch_size for b in batches), (
        "drop_last=True must yield only full-size batches (static batch dim)"
    )

    # Sanity: the pre-fix config (drop_last=False) DID produce the ragged
    # size-1 tail batch that triggered the recompile. Locks the root cause.
    dl_ragged = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    ragged = list(dl_ragged)
    assert len(ragged) == 3 and ragged[-1].shape[0] == 1
