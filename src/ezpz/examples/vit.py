"""
ezpz/examples/vit.py
"""

import argparse
import functools
import os
from pathlib import Path
import time
from typing import Any, Optional

from dataclasses import asdict

import ezpz
from timm.models.vision_transformer import VisionTransformer
import torch
# import torch._dynamo
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

from ezpz import TORCH_DTYPES_MAP
from ezpz.configs import TrainArgs, ViTConfig, timmViTConfig
from ezpz.models import summarize_model

from ezpz.data.vision import get_fake_data, get_mnist
from ezpz.models.vit.attention import AttentionBlock

# from mmm.data.vision import get_fake_data  # , get_mnist
# from mmm.models.vit.attention import AttentionBlock

logger = ezpz.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='ezpz.examples.vit',
        description='Train a simple ViT',
    )
    parser.add_argument('--img_size', default=224, help='Image size')
    parser.add_argument('--batch_size', default=128, help='Batch size')
    parser.add_argument('--num_heads', default=16, help='Number of heads')
    parser.add_argument('--head_dim', default=64, help='Hidden Dimension')
    parser.add_argument(
        "--dataset",
        default="fake",
        choices=["fake", "mnist"],
        help="Dataset to use",
    )
    parser.add_argument('--depth', default=24, help='Depth')
    parser.add_argument('--patch_size', default=16, help='Patch size')
    parser.add_argument('--dtype', default='bf16', help='Data type')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    parser.add_argument('--num_workers', default=0, help='Number of workers')
    parser.add_argument('--max_iters', default=None, help='Maximum iterations')
    parser.add_argument(
        '--attn_type',
        default='native',
        choices=['native', 'sdpa'],
        help='Attention function to use.',
    )
    parser.add_argument(
        '--cuda_sdpa_backend',
        default='all',
        choices=[
            'flash_sdp',
            'mem_efficient_sdp',
            'math_sdp',
            'cudnn_sdp',
            'all',
        ],
        help='CUDA SDPA backend to use.',
    )
    # return TrainArgs(**parser.parse_args())
    # return TrainArgs(**vars(parser.parse_args()))
    return parser.parse_args()


def train_fn(
        block_fn: Any,
        args: TrainArgs,
        dataset: Optional[str]='fake',
) -> ezpz.History:
    # seed = int(os.environ.get('SEED', '0'))
    # rank = ezpz.setup(backend='DDP', seed=seed)
    world_size = ezpz.get_world_size()

    local_rank = ezpz.get_local_rank()
    device_type = str(ezpz.get_torch_device(as_torch_device=False))
    device = torch.device(f'{device_type}:{local_rank}')
    config = timmViTConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        depth=args.depth,
        patch_size=args.patch_size,
    )

    logger.info(f'{asdict(config)=}')


    if dataset == "fake":
        data = get_fake_data(
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
    elif dataset == "mnist":
        data = get_mnist(
            train_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            download=(ezpz.get_rank() == 0),
            shuffle=True,
        )
        # data = get_mnist(
        #     img_size=args.img_size,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=True,
        #     drop_last=True,
        #     shuffle=True,
        # )
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
        global_pool='avg',
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
    if ezpz.get_rank() == 0:
        try:
            wandb.watch(model, log='all')  # type:ignore
        except Exception:
            pass
    logger.info(f'\n{mstr}')
    model.to(device)
    num_params = sum(
        [
            sum(
                [
                    getattr(p, 'ds_numel', 0)
                    if hasattr(p, 'ds_id')
                    else p.nelement()
                    for p in model_module.parameters()
                ]
            )
            for model_module in model.modules()
        ]
    )
    model_size_in_billions = num_params / 1e9
    logger.info(f'Model size: nparams={model_size_in_billions:.2f} B')

    if world_size > 1:
        if args.dtype in {'fp16', 'bf16', 'fp32'}:
            model = FSDP(
                model,
                mixed_precision=MixedPrecision(
                    param_dtype=TORCH_DTYPES_MAP[args.dtype],
                    reduce_dtype=torch.float32,
                    cast_forward_inputs=True,
                ),
            )
        else:
            model = FSDP(model)

    if args.compile:
        logger.info('Compiling model')
        model = torch.compile(model)

    torch_dtype = TORCH_DTYPES_MAP[args.dtype]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())  # type:ignore
    model.train()  # type:ignore

    history = ezpz.History()
    logger.info(
        f'Training with {world_size} x {device_type} (s), using {torch_dtype=}'
    )
    for step, data in enumerate(data['train']['loader']):
        if args.max_iters is not None and step > int(args.max_iters):
            break
        t0 = time.perf_counter()
        inputs = data[0].to(device=device, non_blocking=True)
        label = data[1].to(device=device, non_blocking=True)
        t1 = time.perf_counter()
        with torch.autocast(device_type=device_type, dtype=torch_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, label)
        t2 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        t3 = time.perf_counter()
        logger.info(
            history.update(
                {
                    'train/iter': step,
                    'train/loss': loss.item(),
                    'train/dt': t3 - t0,
                    'train/dtf': t2 - t1,
                    'train/dtb': t3 - t2,
                }
            ).replace('train/', '')
        )

    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            run_name='ezpz-vit', dataset_fname='train', verbose=False
        )
        logger.info(f'{dataset=}')

    return history


def main():
    # torch._dynamo.config.suppress_errors = True  # type:ignore
    rank = ezpz.setup_torch()
    #     backend=os.environ.get('BACKEND', 'DDP'),
    # )
    # return TrainArgs(**vars(parser.parse_args()))
    args = parse_args()
    if ezpz.get_rank() == 0 and not os.environ.get('WANDB_DISABLED', False):
        try:
            import wandb
        except Exception as e:
            logger.exception('Failed to import wandb')
            raise e
        fp = Path(__file__).resolve()
        run = ezpz.setup_wandb(project_name=f'ezpz.{fp.parent.name}.{fp.stem}')
        assert run is not None and run is wandb.run
        wandb.run.config.update({**vars(args)})  # type:ignore

    targs = dict(**vars(args))
    targs.pop("dataset", None)
    targs.pop("use_timm", None)
    train_args: TrainArgs = TrainArgs(**targs)
    config = timmViTConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        depth=args.depth,
        patch_size=args.patch_size,
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

    logger.info(f'Using {args.attn_type} for SDPA backend')
    if args.attn_type == 'native':
        block_fn = functools.partial(AttentionBlock, attn_fn=attn_fn)
    # if args.sdpa_backend == 'by_hand':
    elif args.attn_type == 'sdpa':
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)

            if args.cuda_sdpa_backend in ['flash_sdp', 'all']:
                torch.backends.cuda.enable_flash_sdp(True)
            if args.cuda_sdpa_backend in ['mem_efficient_sdp', 'all']:
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if args.cuda_sdpa_backend in ['math_sdp', 'all']:
                torch.backends.cuda.enable_math_sdp(True)
            if args.cuda_sdpa_backend in ['cudnn_sdp', 'all']:
                torch.backends.cuda.enable_cudnn_sdp(True)

        block_fn = functools.partial(
            AttentionBlock,
            attn_fn=torch.nn.functional.scaled_dot_product_attention,
        )
    else:
        raise ValueError(f'Unknown attention type: {args.attn_type}')
    logger.info(f'Using AttentionBlock Attention with {args.compile=}')
    train_fn(block_fn, args=train_args, dataset=args.dataset)


if __name__ == '__main__':
    t0 = time.perf_counter()
    main()
    logger.info(f'Took {time.perf_counter() - t0:.2f} seconds')
