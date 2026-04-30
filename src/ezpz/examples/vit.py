"""Train a lightweight Vision Transformer on fake or MNIST data.

Launch with:

    ezpz launch -m ezpz.examples.vit --dataset mnist --batch_size 256

Quick smoke test on a laptop:

    python -m ezpz.examples.vit --dataset fake --max_iters 1 \
        --batch_size 4 --img_size 64 --patch_size 8 \
        --num_heads 2 --head_dim 16 --depth 2 --num_classes 10

Model presets:

    --model debug|small|medium|med|large

Help output (``python3 -m ezpz.examples.vit --help``):

    usage: ezpz.examples.vit [-h] [--img_size IMG_SIZE] [--batch_size BATCH_SIZE]
                             [--num_heads NUM_HEADS] [--head_dim HEAD_DIM]
                             [--hidden-dim HIDDEN_DIM] [--mlp-dim MLP_DIM]
                             [--dropout DROPOUT]
                             [--attention-dropout ATTENTION_DROPOUT]
                             [--num_classes NUM_CLASSES] [--dataset {fake,mnist}]
                             [--depth DEPTH] [--patch_size PATCH_SIZE]
                             [--dtype DTYPE] [--compile]
                             [--num_workers NUM_WORKERS] [--max_iters MAX_ITERS]
                             [--warmup WARMUP] [--attn_type {native,sdpa}]
                             [--cuda_sdpa_backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}]
                             [--fsdp]

    Train a simple ViT

    options:
      -h, --help            show this help message and exit
      --img_size IMG_SIZE, --img-size IMG_SIZE
                            Image size
      --batch_size BATCH_SIZE, --batch-size BATCH_SIZE
                            Batch size
      --num_heads NUM_HEADS, --num-heads NUM_HEADS
                            Number of heads
      --head_dim HEAD_DIM, --head-dim HEAD_DIM
                            Hidden Dimension
      --hidden-dim HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                            Hidden Dimension
      --mlp-dim MLP_DIM, --mlp_dim MLP_DIM
                            MLP Dimension
      --dropout DROPOUT     Dropout rate
      --attention-dropout ATTENTION_DROPOUT, --attention_dropout ATTENTION_DROPOUT
                            Attention Dropout rate
      --num_classes NUM_CLASSES, --num-classes NUM_CLASSES
                            Number of classes
      --dataset {fake,mnist}
                            Dataset to use
      --depth DEPTH         Depth
      --patch_size PATCH_SIZE, --patch-size PATCH_SIZE
                            Patch size
      --dtype DTYPE         Data type
      --compile             Compile model
      --num_workers NUM_WORKERS, --num-workers NUM_WORKERS
                            Number of workers
      --max_iters MAX_ITERS, --max-iters MAX_ITERS
                            Maximum iterations
      --warmup WARMUP       Warmup iterations (or fraction) before starting to collect metrics.
      --attn_type {native,sdpa}, --attn-type {native,sdpa}
                            Attention function to use.
      --cuda_sdpa_backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}, --cuda-sdpa-backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}
                            CUDA SDPA backend to use.
      --fsdp                Use FSDP
"""

import argparse
import functools
import math
from pathlib import Path
import sys
import time
from typing import Any, Optional

import torch

import ezpz
import ezpz.distributed

# from TORCH_DTYPES_MAP
from ezpz.data.vision import get_fake_data, get_mnist
from ezpz.examples import get_example_outdir
from ezpz.flops import compute_mfu, try_estimate
from ezpz.models import summarize_model
from ezpz.models.vit.attention import AttentionBlock

logger = ezpz.get_logger(__name__)

fp = Path(__file__)
WBPROJ_NAME = f"ezpz.{fp.parent.stem}.{fp.stem}"

MODEL_PRESETS = {
    "debug": {
        "batch_size": 4,
        "num_heads": 2,
        "head_dim": 16,
        "depth": 2,
    },
    "small": {
        "batch_size": 128,
        "num_heads": 16,
        "head_dim": 64,
        "depth": 24,
    },
    "medium": {
        "batch_size": 64,
        "num_heads": 12,
        "head_dim": 64,
        "depth": 16,
    },
    "large": {
        "batch_size": 32,
        "num_heads": 16,
        "head_dim": 64,
        "depth": 32,
    },
}
MODEL_ALIASES = {"med": "medium"}
MODEL_PRESET_FLAGS = {
    "batch_size": ["--batch_size", "--batch-size"],
    "num_heads": ["--num_heads", "--num-heads"],
    "head_dim": ["--head_dim", "--head-dim"],
    "depth": ["--depth"],
}
MNIST_DEFAULTS = {
    # 28×28 single-channel images, 10 classes — patch=4 gives 49 tokens.
    "img_size": 28,
    "num_classes": 10,
    "patch_size": 4,
    # MNIST is tiny.  The ViT-Large-ish defaults (depth=24, heads=16,
    # head_dim=64 → embed_dim=1024, ~73M params) overfit before they
    # learn anything useful.  Use a smaller, faster model by default
    # so a single run actually trains.
    "num_heads": 4,
    "head_dim": 32,
    "depth": 4,
    # 100 iters × bs=128 = ~0.2 epochs of MNIST.  Bump so a default
    # run gets through enough data to see the loss move.
    "max_iters": 2000,
}
MNIST_DEFAULT_FLAGS = {
    "img_size": ["--img_size", "--img-size"],
    "num_classes": ["--num_classes", "--num-classes"],
    "patch_size": ["--patch_size", "--patch-size"],
    "num_heads": ["--num_heads", "--num-heads"],
    "head_dim": ["--head_dim", "--head-dim"],
    "depth": ["--depth"],
    "max_iters": ["--max_iters", "--max-iters"],
}


class PatchEmbed(torch.nn.Module):
    """Convert images into patch embeddings."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = torch.nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleVisionTransformer(torch.nn.Module):
    """Minimal Vision Transformer implementation without timm."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_classes: int,
        block_fn: Any,
        class_token: bool = False,
        global_pool: str = "avg",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.class_token = class_token
        self.global_pool = global_pool
        if class_token:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches += 1
        else:
            self.cls_token = None
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.pos_drop = torch.nn.Dropout(p=dropout)
        self.blocks = torch.nn.ModuleList(
            [
                block_fn(dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ]
        )
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = (
            torch.nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else torch.nn.Identity()
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Standard ViT init recipe (per the original An Image is Worth 16x16
        # Words paper and timm's vision_transformer.py):
        # - positional embedding and class token: trunc_normal(std=0.02)
        # - all Linear weights: trunc_normal(std=0.02), bias zeroed
        # - LayerNorm: weight=1, bias=0 (PyTorch default already, kept
        #   explicit so the recipe is self-contained)
        # - patch-embed Conv2d: xavier_uniform (treats it as a Linear over
        #   the patch, which is what it functionally is)
        # PyTorch's default Linear init (kaiming_uniform) is calibrated for
        # ReLU-MLPs and produces visibly worse loss curves on ViTs.
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Patch-embed Conv2d: a Linear over flattened patches in disguise.
        torch.nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            torch.nn.init.zeros_(self.patch_embed.proj.bias)
        # Walk every Linear / LayerNorm in the transformer stack
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.global_pool == "avg":
            if self.cls_token is not None:
                x = x[:, 1:]
            x = x.mean(dim=1)
        elif self.cls_token is not None:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        return self.head(x)


def _arg_provided(argv: list[str], flags: list[str]) -> bool:
    return any(flag in argv for flag in flags)


def apply_model_preset(args: argparse.Namespace, argv: list[str]) -> None:
    if args.model is None:
        return
    model_name = args.model
    model_key = MODEL_ALIASES.get(model_name)
    if model_key is None:
        model_key = model_name
    preset = MODEL_PRESETS[model_key]
    for field_name, value in preset.items():
        flags = MODEL_PRESET_FLAGS.get(field_name, [])
        if not _arg_provided(argv, flags):
            setattr(args, field_name, value)


def apply_dataset_overrides(args: argparse.Namespace, argv: list[str]) -> None:
    if args.dataset != "mnist":
        return
    for field_name, value in MNIST_DEFAULTS.items():
        flags = MNIST_DEFAULT_FLAGS.get(field_name, [])
        if not _arg_provided(argv, flags):
            setattr(args, field_name, value)


def validate_dataset_args(args: argparse.Namespace) -> None:
    if args.dataset != "mnist":
        return
    if args.img_size % args.patch_size != 0:
        raise ValueError(
            "MNIST img_size must be divisible by patch_size; "
            f"got img_size={args.img_size} and patch_size={args.patch_size}."
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for ViT training."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="ezpz.examples.vit",
        description="Train a simple ViT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img_size",
        "--img-size",
        type=int,
        default=224,
        help="Image size",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--num_heads",
        "--num-heads",
        type=int,
        default=16,
        help="Number of heads",
    )
    parser.add_argument(
        "--head_dim",
        "--head-dim",
        type=int,
        default=64,
        help="Hidden Dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden Dimension",
    )
    parser.add_argument(
        "--mlp-dim", "--mlp_dim", type=int, default=2048, help="MLP Dimension"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )
    parser.add_argument(
        "--attention-dropout",
        "--attention_dropout",
        type=float,
        default=0.0,
        help="Attention Dropout rate",
    )
    parser.add_argument(
        "--num_classes",
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["fake", "mnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted([*MODEL_PRESETS.keys(), *MODEL_ALIASES.keys()]),
        help="Model size preset (overrides defaults)",
    )
    parser.add_argument("--depth", type=int, default=24, help="Depth")
    parser.add_argument(
        "--patch_size",
        "--patch-size",
        type=int,
        default=16,
        help="Patch size",
    )
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type")
    parser.add_argument("--compile", action="store_true", help="Compile model")
    parser.add_argument(
        "--num_workers",
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers",
    )
    parser.add_argument(
        "--max_iters",
        "--max-iters",
        type=int,
        default=100,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.1,
        help="Warmup iterations (or fraction) before starting to collect metrics.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help=(
            "Peak learning rate.  AdamW's stdlib default of 1e-3 is too "
            "aggressive for from-scratch ViTs and tends to either diverge "
            "or stall on the trivial constant prediction."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        type=float,
        default=0.05,
        help="AdamW weight decay (matches the standard ViT recipe).",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        "--lr_warmup_iters",
        type=float,
        default=0.05,
        help=(
            "Linear LR warmup duration. Integer = iterations; float in "
            "(0, 1) = fraction of --max_iters.  Distinct from --warmup, "
            "which only gates metric collection."
        ),
    )
    parser.add_argument(
        "--min-lr-ratio",
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help=(
            "End-of-cosine LR as a fraction of --lr (e.g. 0.1 = decay "
            "to 10%% of peak by --max_iters)."
        ),
    )
    parser.add_argument(
        "--attn_type",
        "--attn-type",
        default="native",
        choices=["native", "sdpa"],
        help="Attention function to use.",
    )
    parser.add_argument(
        "--cuda_sdpa_backend",
        "--cuda-sdpa-backend",
        default="all",
        choices=[
            "flash_sdp",
            "mem_efficient_sdp",
            "math_sdp",
            "cudnn_sdp",
            "all",
        ],
        help="CUDA SDPA backend to use.",
    )
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP")
    parser.add_argument(
        "--fsdp-sharding-strategy",
        type=str,
        default="full-shard",
        choices=list(ezpz.distributed.FSDP_SHARDING_STRATEGIES),
        help="FSDP sharding strategy",
    )
    args = parser.parse_args(argv)
    apply_model_preset(args, argv)
    apply_dataset_overrides(args, argv)
    validate_dataset_args(args)
    return args


# def get_device_type():
#     """Resolve the torch device type, falling back if MPS lacks collectives."""
#     import os
#
#     device_override = os.environ.get("TORCH_DEVICE")
#     device_type = device_override or ezpz.get_torch_device()
#     if isinstance(device_type, str) and device_type.startswith("mps"):
#         logger.warning(
#             "MPS does not support torch.distributed collectives; falling back to CPU"
#         )
#         return "cpu"
#     return ezpz.get_torch_device_type()


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train_fn(
    block_fn: Any,
    args: argparse.Namespace,
    dataset: Optional[str] = "fake",
) -> ezpz.History:
    """Train the Vision Transformer on fake or MNIST data.

    Args:
        block_fn: Attention block constructor with attn_fn injected.
        args: Training hyperparameters.
        dataset: Dataset choice, either ``fake`` or ``mnist``.

    Returns:
        History of training metrics.
    """
    # seed = int(os.environ.get('SEED', '0'))
    # rank = ezpz.setup(backend='DDP', seed=seed)
    world_size = ezpz.distributed.get_world_size()

    local_rank = ezpz.distributed.get_local_rank()
    # device_type = str(ezpz.get_torch_device(as_torch_device=False))
    device_type = ezpz.distributed.get_torch_device_type()
    device = torch.device(f"{device_type}:{local_rank}")
    # torch.set_default_device(device)
    logger.info("train_args=%s", vars(args))

    if dataset == "fake":
        dataset_dict = get_fake_data(
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif dataset == "mnist":
        dataset_dict = get_mnist(
            train_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            download=(ezpz.distributed.get_rank() == 0),
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Expected 'fake' or 'mnist'."
        )

    # data = get

    # train_set = FakeImageDataset(config.img_size)
    # logger.info(f'{len(train_set)=}')
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=config.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    in_chans = 1 if dataset == "mnist" else 3
    model = SimpleVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=in_chans,
        embed_dim=(args.num_heads * args.head_dim),
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        class_token=False,
        global_pool="avg",
        block_fn=block_fn,
        dropout=args.dropout,
    )

    mstr = summarize_model(
        model,
        verbose=False,
        depth=1,
        input_size=(
            args.batch_size,
            in_chans,
            args.img_size,
            args.img_size,
        ),
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    model_size_in_billions = num_params / 1e9
    logger.info(f"\n{mstr}")
    logger.info(f"Model size: nparams={model_size_in_billions:.2f} B")

    # model = ezpz.distributed.wrap_model(
    #     model=model,
    #     use_fsdp=args.fsdp,
    #     dtype=args.dtype,
    #     # device_id=int(ezpz.get_local_rank())
    # )
    _model_flops = try_estimate(
        model, (args.batch_size, in_chans, args.img_size, args.img_size),
    )

    if world_size > 1:
        reshard = ezpz.distributed.resolve_fsdp_strategy(
            args.fsdp_sharding_strategy
        )
        use_fsdp = args.fsdp and reshard is not None
        model = ezpz.distributed.wrap_model(
            model=model,
            use_fsdp=use_fsdp,
            dtype=args.dtype,
            **({"reshard_after_forward": reshard} if reshard is not None else {}),
        )
        # if args.fsdp:
        #     logger.info("Using FSDP for distributed training")
        #     if args.dtype in {"fp16", "bf16", "fp32"}:
        #         try:
        #             model = FSDP(
        #                 model,
        #                 mixed_precision=MixedPrecision(
        #                     param_dtype=TORCH_DTYPES_MAP[args.dtype],
        #                     reduce_dtype=torch.float32,
        #                     cast_forward_inputs=True,
        #                 ),
        #             )
        #         except Exception as exc:
        #             logger.warning(f"Encountered exception: {exc}")
        #             logger.warning(
        #                 "Unable to wrap model with FSDP. Falling back to DDP..."
        #             )
        #             model = ezpz.distributed.wrap_model(model=model, f)
        #     else:
        #         try:
        #             model = FSDP(model)
        #         except Exception:
        #             model = ezpz.distributed.wrap_model(args=args, model=model)
        # else:
        #     logger.info("Using DDP for distributed training")
        #     model = ezpz.distributed.prepare_model_for_ddp(model)

    if args.compile:
        logger.info("Compiling model")
        model = torch.compile(model)

    torch_dtype = ezpz.distributed.TORCH_DTYPES_MAP[args.dtype]
    criterion = torch.nn.CrossEntropyLoss()
    # Use the standard ViT recipe explicitly: AdamW(lr, weight_decay,
    # betas=(0.9, 0.999)) — defaults gave us lr=1e-3 which routinely
    # diverges or stalls a from-scratch ViT.
    optimizer = torch.optim.AdamW(  # type:ignore[arg-type]
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    # Linear warmup → cosine decay to args.lr * args.min_lr_ratio over
    # max_iters.  Lets us crank --lr without blowing up the early steps.
    if args.max_iters is not None and args.max_iters > 0:
        total_iters = int(args.max_iters)
        if 0.0 < args.lr_warmup_iters < 1.0:
            lr_warmup_iters = max(1, int(args.lr_warmup_iters * total_iters))
        else:
            lr_warmup_iters = max(1, int(args.lr_warmup_iters))
        min_lr_ratio = max(0.0, min(1.0, args.min_lr_ratio))

        def _lr_lambda(step: int) -> float:
            if step < lr_warmup_iters:
                # Linear warmup from 0 → 1 over the first lr_warmup_iters
                return float(step + 1) / float(lr_warmup_iters)
            # Cosine decay from 1 → min_lr_ratio over the remainder
            decay_steps = max(1, total_iters - lr_warmup_iters)
            progress = (step - lr_warmup_iters) / decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        )
        logger.info(
            "LR schedule: linear warmup %d steps → cosine decay to %.2g "
            "(peak lr=%.2g, weight_decay=%.2g)",
            lr_warmup_iters,
            args.lr * min_lr_ratio,
            args.lr,
            args.weight_decay,
        )
    else:
        # No max_iters → no schedule.  Constant LR for streaming runs.
        lr_scheduler = None
        logger.info(
            "Constant LR (no --max_iters cap): peak lr=%.2g, weight_decay=%.2g",
            args.lr, args.weight_decay,
        )

    model.train()  # type:ignore

    outdir = get_example_outdir(WBPROJ_NAME)
    logger.info("Outputs will be saved to %s", outdir)
    metrics_path = outdir.joinpath("metrics.jsonl")
    history = ezpz.history.History(
        report_dir=outdir,
        report_enabled=True,
        jsonl_path=metrics_path,
        jsonl_overwrite=True,
        distributed_history=(1 < world_size <= 384),
        project_name=WBPROJ_NAME,
        config={"args": vars(args), **ezpz.get_dist_info()},
    )
    try:
        if args.compile:
            logger.info("Skipping tracker watch while compiling")
        else:
            history.tracker.watch(model, log="all")
    except Exception as e:
        logger.exception(e)
        logger.warning("Failed to watch model with tracker; continuing...")
    logger.info(
        f"Training with {world_size} x {device_type} (s), using {torch_dtype=}"
    )
    warmup_iters = (
        int(args.warmup)
        if args.warmup >= 1.0
        else int(
            args.warmup
            * (
                args.max_iters
                if args.max_iters is not None
                else len(dataset_dict["train"]["loader"])
            )
        )
    )
    # data["train"].to(ezpz.distributed.get_torch_device_type())
    last_step = -1
    for step, batch in enumerate(dataset_dict["train"]["loader"]):
        last_step = step
        if args.max_iters is not None and step > int(args.max_iters):
            break
        if step < warmup_iters:
            logger.info("warmup step %d / %d", step, warmup_iters)
        t0 = time.perf_counter()
        inputs = batch[0].to(device=device, non_blocking=True)
        label = batch[1].to(device=device, non_blocking=True)
        ezpz.distributed.synchronize()
        with torch.autocast(device_type=device_type, dtype=torch_dtype):
            t1 = time.perf_counter()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            acc = (outputs.argmax(dim=-1) == label).float().mean()
            t2 = time.perf_counter()
        ezpz.distributed.synchronize()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ezpz.distributed.synchronize()
        t3 = time.perf_counter()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        ezpz.distributed.synchronize()
        t4 = time.perf_counter()
        if step >= warmup_iters:
            loss_value = float(loss.detach().item())
            acc_value = float(acc.detach().item())
            if not math.isfinite(loss_value) or not math.isfinite(acc_value):
                logger.warning(
                    "Skipping non-finite train metrics at step=%s", step
                )
                continue
            train_metrics = {
                    "train/iter": step,
                    "train/loss": loss_value,
                    "train/acc": acc_value,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/dt": t4 - t0,
                    "train/dtd": t1 - t0,
                    "train/dtf": t2 - t1,
                    "train/dto": t3 - t2,
                    "train/dtb": t4 - t3,
            }
            if _model_flops > 0 and (t4 - t0) > 0:
                # Full step: data load + forward + optimizer + backward.
                # MFU here is the most "honest" number — the step time
                # the user feels — but is not directly comparable with
                # examples that exclude data loading (fsdp_tp, minimal).
                train_metrics["train/tflops"] = _model_flops / (t4 - t0) / 1e12
                train_metrics["train/mfu"] = compute_mfu(_model_flops, t4 - t0)
            train_msg = history.update(train_metrics).replace("train/", "")
            logger.info("[train] %s", train_msg)

    if "test" in dataset_dict:
        model.eval()  # type:ignore
        eval_loss = 0.0
        eval_acc = 0.0
        eval_count = 0
        eval_step = 0
        with torch.no_grad():
            for batch in dataset_dict["test"]["loader"]:
                inputs = batch[0].to(device=device, non_blocking=True)
                labels = batch[1].to(device=device, non_blocking=True)
                with torch.autocast(device_type=device_type, dtype=torch_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    correct = (outputs.argmax(dim=-1) == labels).sum()
                batch_size = labels.numel()
                eval_loss += loss.item() * batch_size
                eval_acc += correct.item()
                eval_count += batch_size
                batch_loss = float(loss.detach().item())
                batch_acc = float(correct.item() / batch_size)
                if math.isfinite(batch_loss) and math.isfinite(batch_acc):
                    eval_msg = history.update(
                        {
                            "eval/iter": eval_step,
                            "eval/loss": batch_loss,
                            "eval/acc": batch_acc,
                        }
                    ).replace("eval/", "")
                    logger.info("[eval] %s", eval_msg)
                eval_step += 1
        if eval_count:
            total_loss = torch.tensor(eval_loss, device=device)
            total_correct = torch.tensor(eval_acc, device=device)
            total_count = torch.tensor(eval_count, device=device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(total_loss)
                torch.distributed.all_reduce(total_correct)
                torch.distributed.all_reduce(total_count)
            eval_loss_value = float(total_loss.item() / total_count.item())
            eval_acc_value = float(total_correct.item() / total_count.item())
            if not math.isfinite(eval_loss_value) or not math.isfinite(
                eval_acc_value
            ):
                logger.warning("Skipping non-finite eval metrics")
                model.train()  # type:ignore
                return history
            summary_rows = [
                ("loss", f"{eval_loss_value:.6f}"),
                ("acc", f"{eval_acc_value:.6f}"),
                ("samples", f"{int(total_count.item())}"),
            ]
            header = ("eval metric", "value")
            col1 = max(len(header[0]), *(len(row[0]) for row in summary_rows))
            col2 = max(len(header[1]), *(len(row[1]) for row in summary_rows))
            summary_table = [
                "Eval summary:",
                f"| {header[0]:<{col1}} | {header[1]:>{col2}} |",
                f"|:{'-' * (col1 - 1)} | {'-' * (col2 - 1)}:|",
            ]
            summary_table.extend(
                f"| {name:<{col1}} | {value:>{col2}} |"
                for name, value in summary_rows
            )
            logger.info("\n".join(f"[eval] {line}" for line in summary_table))
        model.train()  # type:ignore

    return history, outdir


@ezpz.timeitlogit(rank=ezpz.get_rank())
def main(args: argparse.Namespace):
    """CLI entrypoint to configure logging and launch ViT training."""
    t0 = time.perf_counter()
    _ = ezpz.distributed.setup_torch()
    t_setup = time.perf_counter()

    def attn_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Scaled dot-product attention with configurable backend."""
        scale = args.head_dim ** (-0.5)
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        if args.attention_dropout > 0.0:
            attn = torch.nn.functional.dropout(
                attn,
                p=args.attention_dropout,
                training=torch.is_grad_enabled(),
            )
        x = attn @ v
        return x

    logger.info(f"Using {args.attn_type} for SDPA backend")
    if args.attn_type == "native":
        block_fn = functools.partial(AttentionBlock, attn_fn=attn_fn)
    # if args.sdpa_backend == 'by_hand':
    elif args.attn_type == "sdpa":
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)

            if args.cuda_sdpa_backend in ["flash_sdp", "all"]:
                torch.backends.cuda.enable_flash_sdp(True)
            if args.cuda_sdpa_backend in ["mem_efficient_sdp", "all"]:
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if args.cuda_sdpa_backend in ["math_sdp", "all"]:
                torch.backends.cuda.enable_math_sdp(True)
            if args.cuda_sdpa_backend in ["cudnn_sdp", "all"]:
                torch.backends.cuda.enable_cudnn_sdp(True)

        block_fn = functools.partial(
            AttentionBlock,
            attn_fn=lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=(
                    args.attention_dropout if torch.is_grad_enabled() else 0.0
                ),
            ),
        )
    else:
        raise ValueError(f"Unknown attention type: {args.attn_type}")
    logger.info(f"Using AttentionBlock Attention with {args.compile=}")
    train_start = time.perf_counter()
    history, outdir = train_fn(block_fn, args=args, dataset=args.dataset)
    train_end = time.perf_counter()
    t1 = time.perf_counter()
    timings = {
        "main/setup_torch": t_setup - t0,
        "main/train": train_end - train_start,
        "main/total": t1 - t0,
    }
    logger.info("Timings: %s", timings)
    history.tracker.log(
        {
            (f"timings/{k}" if not k.startswith("timings/") else k): v
            for k, v in timings.items()
        }
    )
    if ezpz.distributed.get_rank() == 0:
        if history.history and any(len(v) for v in history.history.values()):
            dataset = history.finalize(
                outdir=outdir,
                run_name=WBPROJ_NAME,
                verbose=False,
            )
            del dataset  # logged by finalize()
        else:
            logger.warning("No metrics recorded; skipping dataset save")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    ezpz.distributed.cleanup()
