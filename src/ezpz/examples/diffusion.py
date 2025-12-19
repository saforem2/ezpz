"""
Toy diffusion example for short text generation.

This script trains a tiny denoising diffusion model on a handful of toy
sentences and then samples new sentences by running the reverse process.
The goal is to keep the code minimal while showcasing the full flow:

    1. Build a small vocabulary from a list of prompts.
    2. Train a denoising network to predict noise on token embeddings.
    3. Sample text by iterating the reverse diffusion process.

Typical usage (customize with args as needed):

    ezpz-launch -m ezpz.examples.diffusion --timesteps 64 --train-steps 500 --batch-size 16
    # with FSDP and a HF dataset slice:
    WORLD_SIZE=2 ezpz-launch -m ezpz.examples.diffusion --hf-dataset ag_news --fsdp
"""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple
from contextlib import nullcontext
import ezpz

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import MixedPrecision

import wandb

logger = ezpz.get_logger(__name__)

fp = Path(__file__)
WBPROJ_NAME = f"ezpz.{fp.parent.stem}.{fp.stem}"
WBRUN_NAME = f"{ezpz.get_timestamp()}"


def build_vocab(texts: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create a tiny vocabulary from a list of strings."""
    specials = ["<pad>", "<unk>"]
    words = sorted({word for text in texts for word in text.lower().split()})
    vocab = {tok: idx for idx, tok in enumerate(specials + words)}
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    return vocab, inv_vocab


class ToyTextDataset(Dataset):
    """Pads or truncates sentences to a fixed length."""

    def __init__(
        self, texts: List[str], vocab: Dict[str, int], seq_len: int = 12
    ):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, text: str) -> torch.Tensor:
        tokens = [
            self.vocab.get(tok, self.unk_id) for tok in text.lower().split()
        ]
        tokens = tokens[: self.seq_len]
        tokens += [self.pad_id] * (self.seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type:ignore
        return self._encode(self.texts[idx])


@dataclass
class DiffusionSchedule:
    """Precompute alpha/beta schedule values for DDPM style updates."""

    timesteps: int = 64
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.timesteps
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


class DiffusionTextModel(nn.Module):
    """Simple transformer that predicts noise on token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int,
        timesteps: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size  # type:ignore
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.time_emb = nn.Embedding(timesteps, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=4 * hidden_size,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.proj = nn.Linear(hidden_size, hidden_size)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # Clone avoids autograd complaints about views when using sharded params.
        # return self.token_emb(tokens).clone() * math.sqrt(self.hidden_size)
        return self.token_emb(tokens).clone() * math.sqrt(self.hidden_size)

    def forward(
        self, noisy_embs: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        _, seq_len, _ = noisy_embs.shape
        pos = self.pos_emb(torch.arange(seq_len, device=noisy_embs.device))
        temb = self.time_emb(t).unsqueeze(1)
        h = noisy_embs + pos.unsqueeze(0) + temb
        h = self.encoder(h)
        return self.proj(h)

    def decode_tokens(self, embs: torch.Tensor) -> torch.Tensor:
        weights = self.token_emb.weight  # (vocab, hidden)
        logits = torch.einsum("bld,vd->blv", embs, weights)
        return logits.argmax(dim=-1)


def sample_timesteps(
    batch_size: int, schedule: DiffusionSchedule, device: torch.device
) -> torch.Tensor:
    return torch.randint(0, schedule.timesteps, (batch_size,), device=device)


def add_noise(
    x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    alpha_bar = schedule.alpha_bars.to(x0.device)[t].view(-1, 1, 1)
    noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    return noisy, noise


def p_sample(
    model: DiffusionTextModel,
    xt: torch.Tensor,
    t: int,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    t_batch = torch.full((xt.size(0),), t, device=xt.device, dtype=torch.long)
    beta = schedule.betas.to(xt.device)[t]
    alpha = schedule.alphas.to(xt.device)[t]
    alpha_bar = schedule.alpha_bars.to(xt.device)[t]
    eps = model(xt, t_batch)
    mean = (xt - (beta / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
    if t == 0:
        return mean
    noise = torch.randn_like(xt)
    return mean + torch.sqrt(beta) * noise


def generate_text(
    model: DiffusionTextModel,
    schedule: DiffusionSchedule,
    inv_vocab: Dict[int, str],
    seq_len: int,
    num_samples: int,
    skip_tokens: Tuple[str, ...] = ("<pad>", "<unk>"),
) -> List[str]:
    model.eval()
    samples: List[str] = []
    is_fsdp = isinstance(model, FSDP)
    # base_model = model.module if is_fsdp else model
    full_param_ctx = (
        FSDP.summon_full_params(model)  # , recursive=True)
        if is_fsdp
        else nullcontext()
    )

    with torch.no_grad():
        with full_param_ctx:
            token_emb_weight = model.token_emb.weight
            for _ in range(num_samples):
                xt = torch.randn(
                    (1, seq_len, model.hidden_size),
                    device=token_emb_weight.device,
                )
                for t in reversed(range(schedule.timesteps)):
                    xt = p_sample(model, xt, t, schedule)
                logits = torch.einsum("bld,vd->blv", xt, token_emb_weight)
                token_ids = logits.argmax(dim=-1)[0].tolist()
                words = [
                    inv_vocab.get(tok_id, "<unk>") for tok_id in token_ids
                ]
                words = [w for w in words if w not in skip_tokens]
                samples.append(" ".join(words))
    return samples


@ezpz.timeitlogit(rank=ezpz.get_rank())
def test(model, test_loader):
    DEVICE = ezpz.get_torch_device()
    DEVICE_ID = f"{DEVICE}:{ezpz.get_local_rank()}"
    model.eval()
    # correct = 0
    ddp_loss = torch.zeros(3).to(DEVICE_ID)
    with torch.no_grad():
        for batch, target in test_loader:
            batch, target = batch.to(DEVICE_ID), target.to(DEVICE_ID)
            output = model(batch)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)  # type:ignore

    test_loss = ddp_loss[0] / ddp_loss[2]

    return {
        "test_loss": test_loss,
        "test_acc": 100.0 * ddp_loss[1] / ddp_loss[2],
    }


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    model: DiffusionTextModel,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    args: argparse.Namespace,
    steps: int,
    lr: float = 1e-3,
    outdir: Optional[str | Path | os.PathLike] = None,
) -> ezpz.History:
    device = ezpz.get_torch_device(as_torch_device=True)
    # if not isinstance(model, (DistributeFSDP):
    model.to(device)
    model.train()
    wrapped_model = ezpz.dist.wrap_model(
        model, use_fsdp=args.fsdp, dtype=args.dtype
    )
    optim = torch.optim.AdamW(wrapped_model.parameters(), lr=lr)
    mstr = ezpz.models.summarize_model(
        wrapped_model,
        verbose=False,
        depth=2,
        # input_size=(
        #     torch.tensor((int(args.batch_size), int(args.seq_length))).to(
        #         torch.long
        #     )
        # ).shape,
    )
    logger.info("Model summary:\n%s", mstr)

    outdir_parent = Path(os.getcwd()) if outdir is None else outdir
    outdir = Path(outdir_parent).joinpath(ezpz.history.get_timestamp())
    metrics_path = outdir.joinpath("metrics.jsonl")
    outdir.mkdir(parents=True, exist_ok=True)
    history = ezpz.history.History(
        report_dir=outdir,
        report_enabled=True,
        jsonl_path=metrics_path,
        jsonl_overwrite=True,
        distributed_history=(
            1 < ezpz.get_world_size() <= 384  # and not config.pytorch_profiler
        ),
    )

    # log_freq = max(1, steps // 100)
    assert isinstance(
        wrapped_model, (nn.Module, FSDP, DistributedDataParallel)
    ), "Model should be wrapped for training."
    # assert hasattr(wrapped_model, "module"), (
    #     "Model should be wrapped for training."
    # )
    # assert callable(getattr(wrapped_model.module, "embed_tokens", None)), (
    #     "Model should have embed_tokens method."
    # )
    loader_iter = iter(loader)
    for step in range(steps):
        t0 = time.perf_counter()
        try:
            tokens = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            tokens = next(loader_iter)
        tokens = tokens.to(device)
        t1 = time.perf_counter()
        ezpz.dist.synchronize()
        x0 = model.embed_tokens(tokens)
        t = sample_timesteps(tokens.size(0), schedule, device=device)
        xt, noise = add_noise(x0, t, schedule)
        pred_noise = model(xt, t)
        loss = torch.mean((pred_noise - noise) ** 2)
        t2 = time.perf_counter()
        ezpz.dist.synchronize()

        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        t3 = time.perf_counter()
        ezpz.dist.synchronize()

        if step % args.log_freq == 0 or step == steps - 1:
            logger.info(
                history.update(
                    {
                        "train/step": step,
                        "train/loss": loss.item(),
                        "train/dt": t3 - t0,
                        "train/dtd": t1 - t0,
                        "train/dtf": t2 - t1,
                        "train/dtb": t3 - t2,
                    }
                ).replace("train/", "")
            )

    # loader_iter = iter(loader)
    # for step in range(steps):
    #     try:
    #         tokens = next(loader_iter)
    #     except StopIteration:
    #         loader_iter = iter(loader)
    #         tokens = next(loader_iter)
    #     tokens = tokens.to(device)
    #     x0 = model.embed_tokens(tokens)
    #     t = sample_timesteps(tokens.size(0), schedule, device=device)
    #     xt, noise = add_noise(x0, t, schedule)
    #     pred_noise = model(xt, t)
    #     loss = torch.mean((pred_noise - noise) ** 2)
    #
    #     loss.backward()
    #     optim.step()
    #     optim.zero_grad(set_to_none=True)
    #
    #     if step % log_freq == 0 or step == steps - 1:
    #         summary = history.update({"step": step, "loss": loss.item()})
    #         logger.info(summary)
    return history


def get_default_texts() -> List[str]:
    return [
        "the product team ships updates every week",
        "customers ask for faster onboarding",
        "the service autoscaling keeps latency steady",
        "data pipelines need reliable monitoring",
        "large language models assist with code reviews",
        "cloud costs drop when workloads are right sized",
        "edge devices sync logs during quiet hours",
        "dashboards show live metrics for incidents",
    ]


def load_hf_texts(
    dataset_name: str,
    split: str,
    text_column: str,
    limit: int,
) -> List[str]:
    """
    Pull a small slice of text from a Hugging Face dataset for quick experiments.

    This uses only a limited number of rows (`limit`) to keep the example light.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - best-effort import
        raise RuntimeError(
            "datasets package is required for --hf-dataset usage"
        ) from exc

    logger.info(
        "Loading HF dataset %s split=%s column=%s limit=%s",
        dataset_name,
        split,
        text_column,
        limit,
    )
    dataset = load_dataset(dataset_name, split=split)
    if text_column not in list(dataset.column_names):
        raise ValueError(
            f"text_column '{text_column}' not in dataset columns {dataset.column_names}"
        )
    texts = [str(row[text_column]) for row in dataset.select(range(limit))]
    if not texts:
        raise ValueError("No text rows found from HF dataset.")
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny diffusion example for text generation."
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 8))
    )
    parser.add_argument(
        "--dtype", type=str, default=os.environ.get("DTYPE", "float32")
    )
    parser.add_argument(
        "--extra-text",
        type=str,
        nargs="*",
        default=None,
        help="Additional sentences to add to the tiny corpus.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP wrapping (requires WORLD_SIZE>1 and torch.distributed init).",
    )
    parser.add_argument(
        "--fsdp-mixed-precision",
        action="store_true",
        help="Use bfloat16 parameters with FSDP for speed (defaults to float32).",
    )
    parser.add_argument(
        "--hidden", type=int, default=int(os.environ.get("HIDDEN", 128))
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset name (e.g., 'ag_news'). When set, replaces the toy corpus.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--hf-text-column",
        type=str,
        default="text",
        help="Column containing raw text in the dataset.",
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=512,
        help="Number of rows to sample from the HF dataset for quick experiments.",
    )
    parser.add_argument(
        "--log_freq", type=int, default=int(os.environ.get("LOG_FREQ", 1))
    )
    parser.add_argument(
        "--samples", type=int, default=int(os.environ.get("SAMPLES", 3))
    )
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("SEED", 0))
    )
    parser.add_argument(
        "--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", 12))
    )
    parser.add_argument(
        "--timesteps", type=int, default=int(os.environ.get("TIMESTEPS", 64))
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=int(os.environ.get("TRAIN_STEPS", 400)),
    )
    parser.add_argument(
        "--lr", type=float, default=float(os.environ.get("LR", 3e-3))
    )
    # parser.add_argument(
    #     "--ddp",
    #     action="store_true",
    #     help="Enable DDP wrapping (requires WORLD_SIZE>1 and torch.distributed init).",
    # )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = ezpz.setup_torch(seed=args.seed)
    if ezpz.get_rank() == 0:
        run = ezpz.dist.setup_wandb(project_name=WBPROJ_NAME)
        assert run is not None and run is wandb.run
        wandb.config.update({**vars(args)})
        wandb.config.update(ezpz.get_dist_info())

    base_texts: List[str]
    if args.hf_dataset:
        base_texts = load_hf_texts(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            text_column=args.hf_text_column,
            limit=args.hf_limit,
        )
    else:
        base_texts = get_default_texts()
        if args.extra_text:
            base_texts = base_texts + args.extra_text

    vocab, inv_vocab = build_vocab(base_texts)
    dataset = ToyTextDataset(base_texts, vocab, seq_len=args.seq_len)
    sampler = (
        DistributedSampler(dataset) if ezpz.get_world_size() > 1 else None
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        drop_last=False,
    )

    schedule = DiffusionSchedule(timesteps=args.timesteps)
    model = DiffusionTextModel(
        vocab_size=len(vocab),
        hidden_size=args.hidden,
        max_seq_len=args.seq_len,
        timesteps=args.timesteps,
    )
    device = ezpz.get_torch_device(as_torch_device=True)
    model.to(device)

    history = train(
        model=model,
        loader=loader,
        schedule=schedule,
        args=args,
        steps=args.train_steps,
        lr=args.lr,
    )

    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            run_name=WBRUN_NAME,
            dataset_fname="train",
            warmup=0.1,
        )
        samples = generate_text(
            model,
            schedule,
            inv_vocab,
            seq_len=args.seq_len,
            num_samples=args.samples,
        )
        for idx, text in enumerate(samples):
            logger.info("sample %s: %s", idx, text)


if __name__ == "__main__":
    main()
