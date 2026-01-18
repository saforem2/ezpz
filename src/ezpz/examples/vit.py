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
from dataclasses import asdict
from dataclasses import dataclass, field
import functools
from pathlib import Path
import sys
import time
from typing import Any, Optional

import torch

import ezpz
import ezpz.dist

# from TORCH_DTYPES_MAP
from ezpz.configs import timmViTConfig
from ezpz.data.vision import get_fake_data, get_mnist
from ezpz.models import summarize_model
from ezpz.models.vit.attention import AttentionBlock

logger = ezpz.get_logger(__name__)

fp = Path(__file__)
WBPROJ_NAME = f"ezpz.{fp.parent.stem}.{fp.stem}"
WBRUN_NAME = f"{ezpz.get_timestamp()}"

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


try:
    import wandb
except Exception:
    wandb = None  # type:ignore
    logger.warning("Failed to import wandb")


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
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for ViT training."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="ezpz.examples.vit",
        description="Train a simple ViT",
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
        default="fake",
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
    args = parser.parse_args(argv)
    apply_model_preset(args, argv)
    return args


@dataclass
class VitTrainArgs:
    """Structured configuration for Vision Transformer training."""

    img_size: int = 224
    batch_size: int = 128
    num_heads: int = 16
    compile: bool = False
    depth: int = 8
    dtype: str = "bf16"
    head_dim: int = 64
    hidden_dim: int = 1024
    mlp_dim: int = 2048
    max_iters: int = 1000
    dropout: float = 0.1
    attention_dropout: float = 0.0
    num_classes: int = 1000
    dataset: str = "fake"
    depth: int = 24
    patch_size: int = 16
    num_workers: int = 0
    warmup: float = 0.1
    attn_type: str = "native"
    fsdp: Optional[bool] = None
    format: Optional[str] = field(default_factory=str)
    cuda_sdpa_backend: Optional[str] = "all"


def get_device_type():
    """Resolve the torch device type, falling back if MPS lacks collectives."""
    import os

    device_override = os.environ.get("TORCH_DEVICE")
    device_type = device_override or ezpz.get_torch_device()
    if isinstance(device_type, str) and device_type.startswith("mps"):
        logger.warning(
            "MPS does not support torch.distributed collectives; falling back to CPU"
        )
        return "cpu"
    return ezpz.get_torch_device_type()


def train_fn(
    block_fn: Any,
    args: VitTrainArgs,
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
    world_size = ezpz.dist.get_world_size()

    local_rank = ezpz.dist.get_local_rank()
    # device_type = str(ezpz.get_torch_device(as_torch_device=False))
    device_type = ezpz.dist.get_torch_device_type()
    device = torch.device(f"{device_type}:{local_rank}")
    # torch.set_default_device(device)
    config = timmViTConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        depth=args.depth,
        patch_size=args.patch_size,
    )

    logger.info(f"{asdict(config)=}")

    if dataset == "fake":
        data = get_fake_data(
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
    elif dataset == "mnist":
        data = get_mnist(
            train_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            download=(ezpz.dist.get_rank() == 0),
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
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=in_chans,
        embed_dim=(config.num_heads * config.head_dim),
        depth=config.depth,
        num_heads=config.num_heads,
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
            config.batch_size,
            in_chans,
            config.img_size,
            config.img_size,
        ),
    )
    model.to(device)
    num_params = sum(
        [
            sum(
                [
                    getattr(p, "ds_numel", 0)
                    if hasattr(p, "ds_id")
                    else p.nelement()
                    for p in model_module.parameters()
                ]
            )
            for model_module in model.modules()
        ]
    )
    model_size_in_billions = num_params / 1e9
    logger.info(f"\n{mstr}")
    logger.info(f"Model size: nparams={model_size_in_billions:.2f} B")
    if wandb is not None and ezpz.verify_wandb():
        if (run := getattr(wandb, "run")) is not None and run is wandb.run:
            try:
                wandb.run.watch(model, log="all")  # type:ignore
            except Exception as e:
                logger.exception(e)
                logger.warning(
                    "Failed to watch model with wandb; continuing..."
                )

    # model = ezpz.dist.wrap_model(
    #     model=model,
    #     use_fsdp=args.fsdp,
    #     dtype=args.dtype,
    #     # device_id=int(ezpz.get_local_rank())
    # )
    if world_size > 1:
        model = ezpz.dist.wrap_model(
            model=model,
            use_fsdp=args.fsdp,
            dtype=args.dtype,
            device_id=ezpz.get_torch_device(as_torch_device=True),
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
        #             model = ezpz.dist.wrap_model(model=model, f)
        #     else:
        #         try:
        #             model = FSDP(model)
        #         except Exception:
        #             model = ezpz.dist.wrap_model(args=args, model=model)
        # else:
        #     logger.info("Using DDP for distributed training")
        #     model = ezpz.dist.prepare_model_for_ddp(model)

    if args.compile:
        logger.info("Compiling model")
        model = torch.compile(model)

    torch_dtype = ezpz.dist.TORCH_DTYPES_MAP[args.dtype]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())  # type:ignore
    model.train()  # type:ignore

    history = ezpz.history.History()
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
                else len(data["train"]["loader"])
            )
        )
    )
    # data["train"].to(ezpz.dist.get_torch_device_type())
    for step, data in enumerate(data["train"]["loader"]):
        if args.max_iters is not None and step > int(args.max_iters):
            break
        t0 = time.perf_counter()
        inputs = data[0].to(device=device, non_blocking=True)
        label = data[1].to(device=device, non_blocking=True)
        ezpz.dist.synchronize()
        with torch.autocast(device_type=device_type, dtype=torch_dtype):
            t1 = time.perf_counter()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            t2 = time.perf_counter()
        ezpz.dist.synchronize()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ezpz.dist.synchronize()
        t3 = time.perf_counter()
        optimizer.step()
        ezpz.dist.synchronize()
        t4 = time.perf_counter()
        if step >= warmup_iters:
            logger.info(
                history.update(
                    {
                        "train/iter": step,
                        "train/loss": loss,
                        "train/dt": t4 - t0,
                        "train/dtd": t1 - t0,
                        "train/dtf": t2 - t1,
                        "train/dto": t3 - t2,
                        "train/dtb": t4 - t3,
                    }
                ).replace("train/", "")
            )

    if ezpz.dist.get_rank() == 0:
        dataset = history.finalize(
            run_name=WBRUN_NAME, dataset_fname="train", verbose=False
        )
        logger.info(f"{dataset=}")

    return history


def main(args: argparse.Namespace):
    """CLI entrypoint to configure logging and launch ViT training."""
    rank = ezpz.dist.setup_torch()
    if rank == 0 and ezpz.verify_wandb():
        try:
            fp = Path(__file__).resolve()
            run = ezpz.setup_wandb(
                project_name=f"ezpz.{fp.parent.name}.{fp.stem}"
            )
            if wandb is not None and run is not None and run is wandb.run:
                # assert run is not None and run is wandb.run
                wandb.config.update(ezpz.get_dist_info())
                wandb.config.update({**vars(args)})
        except Exception:
            logger.warning("Failed to setup wandb, continuing without!")

    targs = dict(**vars(args))
    targs.pop("dataset", None)
    targs.pop("use_timm", None)
    targs.pop("model", None)
    train_args = VitTrainArgs(**targs)
    # train_args:  = (**targs)
    config = timmViTConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        depth=args.depth,
        patch_size=int(args.patch_size),
    )

    def attn_fn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Scaled dot-product attention with configurable backend."""
        scale = config.head_dim ** (-0.5)
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
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
            attn_fn=torch.nn.functional.scaled_dot_product_attention,
        )
    else:
        raise ValueError(f"Unknown attention type: {args.attn_type}")
    logger.info(f"Using AttentionBlock Attention with {args.compile=}")
    train_fn(block_fn, args=train_args, dataset=args.dataset)


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter()
    main(args)
    ezpz.dist.cleanup()
    logger.info(f"Took {time.perf_counter() - t0:.2f} seconds")
