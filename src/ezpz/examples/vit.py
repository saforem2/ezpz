"""
ezpz/examples/vit.py
"""

import argparse
from dataclasses import asdict
from dataclasses import dataclass, field
import functools
from pathlib import Path
import time
from typing import Any, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

import ezpz
from ezpz import TORCH_DTYPES_MAP
from ezpz.configs import timmViTConfig
from ezpz.data.vision import get_fake_data, get_mnist
from ezpz.models import summarize_model
from ezpz.models.vit.attention import AttentionBlock

logger = ezpz.get_logger(__name__)

try:
    import wandb
except Exception:
    wandb = None  # type:ignore
    logger.warning("Failed to import wandb")

try:
    from timm.models.vision_transformer import VisionTransformer  # type:ignore
except (ImportError, ModuleNotFoundError) as e:
    logger.exception(
        "Please install timm to use VisionTransformer: uv pip install timm (`--no-deps` on Aurora)"
    )
    raise e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ezpz.examples.vit",
        description="Train a simple ViT",
    )
    parser.add_argument("--img_size", default=224, help="Image size")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of heads"
    )
    parser.add_argument(
        "--head_dim",
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
        "--num_classes", type=int, default=1000, help="Number of classes"
    )
    parser.add_argument(
        "--dataset",
        default="fake",
        choices=["fake", "mnist"],
        help="Dataset to use",
    )
    parser.add_argument("--depth", type=int, default=24, help="Depth")
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Patch size"
    )
    parser.add_argument("--dtype", default="bf16", help="Data type")
    parser.add_argument("--compile", action="store_true", help="Compile model")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers"
    )
    parser.add_argument("--max_iters", default=None, help="Maximum iterations")
    parser.add_argument(
        "--warmup",
        default=0.1,
        help="Warmup iterations (or fraction) before starting to collect metrics.",
    )
    parser.add_argument(
        "--attn_type",
        default="native",
        choices=["native", "sdpa"],
        help="Attention function to use.",
    )
    parser.add_argument(
        "--cuda_sdpa_backend",
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
    # return TrainArgs(**parser.parse_args())
    # return TrainArgs(**vars(parser.parse_args()))
    return parser.parse_args()


@dataclass
class VitTrainArgs:
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

    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=(config.num_heads * config.head_dim),
        depth=config.depth,
        num_heads=config.num_heads,
        class_token=False,
        global_pool="avg",
        block_fn=block_fn,
    )

    mstr = summarize_model(
        model,
        verbose=False,
        depth=1,
        input_size=(
            config.batch_size,
            3,
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
    if wandb is not None:
        if wandb.run is not None:
            wandb.run.watch(model, log="all")

    model = ezpz.dist.wrap_model(
        model=model,
        use_fsdp=args.fsdp,
        dtype=args.dtype,
    )
    if world_size > 1:
        if args.fsdp:
            logger.info("Using FSDP for distributed training")
            if args.dtype in {"fp16", "bf16", "fp32"}:
                try:
                    model = FSDP(
                        model,
                        mixed_precision=MixedPrecision(
                            param_dtype=TORCH_DTYPES_MAP[args.dtype],
                            reduce_dtype=torch.float32,
                            cast_forward_inputs=True,
                        ),
                    )
                except Exception as exc:
                    logger.warning(f"Encountered exception: {exc}")
                    logger.warning(
                        "Unable to wrap model with FSDP. Falling back to DDP..."
                    )
                    model = ezpz.dist.wrap_model(model=model, f)
            else:
                try:
                    model = FSDP(model)
                except Exception:
                    model = ezpz.dist.wrap_model(args=args, model=model)
        else:
            logger.info("Using DDP for distributed training")
            model = ezpz.dist.prepare_model_for_ddp(model)

    if args.compile:
        logger.info("Compiling model")
        model = torch.compile(model)

    torch_dtype = TORCH_DTYPES_MAP[args.dtype]
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
                        "train/loss": loss.item(),
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
            run_name="ezpz-vit", dataset_fname="train", verbose=False
        )
        logger.info(f"{dataset=}")

    return history


def main():
    rank = ezpz.dist.setup_torch()
    args = parse_args()
    if rank == 0:
        try:
            fp = Path(__file__).resolve()
            run = ezpz.setup_wandb(
                project_name=f"ezpz.{fp.parent.name}.{fp.stem}"
            )
            if wandb is not None:
                assert run is not None and run is wandb.run
                wandb.config.update(ezpz.get_dist_info())
                wandb.config.update({**vars(args)})  # type:ignore
        except Exception:
            logger.warning("Failed to setup wandb, continuing without!")

    targs = dict(**vars(args))
    targs.pop("dataset", None)
    targs.pop("use_timm", None)
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
    t0 = time.perf_counter()
    main()
    logger.info(f"Took {time.perf_counter() - t0:.2f} seconds")
