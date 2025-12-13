"""
ezpz/data/vision.py

Sam Foreman
2024-12-27
"""

import os

from pathlib import Path
from typing import Optional

import ezpz
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from ezpz.dist import TORCH_DTYPES_MAP
from torch.utils.data import Dataset

OUTPUT_DIR = Path(os.getcwd()).joinpath(".cache", "ezpz", "data", "vision")


class FakeImageDataset(Dataset):
    def __init__(
        self,
        size: int,
        dtype: Optional[str | torch.dtype] = None,
    ):
        super().__init__()
        self.size = size
        self.dtype = (
            torch.float32
            if dtype is None
            else (TORCH_DTYPES_MAP[dtype] if isinstance(dtype, str) else dtype)
        )

    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        rand_image = torch.randn(
            [3, self.size, self.size],
            dtype=(torch.float32 if self.dtype is None else self.dtype),
        )
        label = torch.tensor(data=(index % 1000), dtype=torch.int64)
        return rand_image, label


def get_mnist(
    train_batch_size: int = 128,
    test_batch_size: int = 128,
    outdir: Optional[str | Path] = None,
    num_workers: int = 1,
    download: bool = True,
    pin_memory: bool = True,
) -> dict:
    outdir = OUTPUT_DIR if outdir is None else outdir
    datadir = Path(outdir).joinpath("data", "mnist")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()

    if rank == 0:
        _ = datasets.MNIST(
            datadir.as_posix(),
            train=True,
            download=download,
            transform=transform,
        )
    if world_size > 1:
        ezpz.barrier()

    dset_train = datasets.MNIST(
        datadir.as_posix(),
        train=True,
        download=False,
        transform=transform,
    )
    dset_test = datasets.MNIST(datadir, train=False, transform=transform)
    sampler_train = (
        DistributedSampler(dset_train, rank=rank, num_replicas=world_size)
        if world_size > 1
        else None
    )
    sampler_test = (
        DistributedSampler(dset_test, rank=rank, num_replicas=world_size)
        if world_size > 1
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dset_train,
        batch_size=train_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=sampler_train,
        shuffle=(sampler_train is None),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dset_test,
        batch_size=test_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=sampler_test,
        shuffle=False,
    )

    return {
        "train": {
            "data": dset_train,
            "loader": train_loader,
            "sampler": sampler_train,
        },
        "test": {
            "data": dset_test,
            "loader": test_loader,
            "sampler": sampler_test,
        },
    }


def get_imagenet(
    train_batch_size: int = 128,
    test_batch_size: int = 128,
    outdir: Optional[str | Path] = None,
    num_workers: int = 1,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> dict:
    """Return train/test ImageNet datasets, loaders, and (optional) samplers.

    Expects directory layout:

        <outdir>/data/imagenet/
            train/
                class1/
                class2/
                ...
            val/
                class1/
                class2/
                ...

    where `train/` and `val/` are standard ImageNet-style folders.
    """
    outdir = OUTPUT_DIR if outdir is None else Path(outdir)
    datadir = Path(outdir).joinpath("data", "imagenet")

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Basic sanity check only on rank 0 (no auto-download for ImageNet)
    if ezpz.get_rank() == 0:
        train_dir = datadir / "train"
        val_dir = datadir / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"Expected ImageNet data under:\n"
                f"  {train_dir}\n"
                f"  {val_dir}\n"
                "with standard ImageFolder layout."
            )

    if ezpz.get_world_size() > 1:
        ezpz.barrier()

    dataset1 = datasets.ImageFolder(
        root=datadir / "train", transform=train_transform
    )
    dataset2 = datasets.ImageFolder(
        root=datadir / "val", transform=test_transform
    )

    train_kwargs: dict = {
        "batch_size": train_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    test_kwargs: dict = {
        "batch_size": test_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }

    sampler1, sampler2 = None, None
    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()

    if world_size > 1:
        sampler1 = DistributedSampler(
            dataset1,
            rank=rank,
            num_replicas=world_size,
            shuffle=True,
        )
        sampler2 = DistributedSampler(
            dataset2,
            rank=rank,
            num_replicas=world_size,
            shuffle=False,
        )
        train_kwargs["sampler"] = sampler1
        test_kwargs["sampler"] = sampler2
    else:
        train_kwargs["shuffle"] = shuffle

    loader_train = torch.utils.data.DataLoader(
        dataset=dataset1, **train_kwargs
    )
    loader_test = torch.utils.data.DataLoader(dataset=dataset2, **test_kwargs)

    return {
        "train": {
            "data": dataset1,
            "loader": loader_train,
            "sampler": sampler1,
        },
        "test": {
            "data": dataset2,
            "loader": loader_test,
            "sampler": sampler2,
        },
    }


class HFImageNet1K(Dataset):
    """Thin wrapper to use HF imagenet-1k with torchvision transforms."""

    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        example = self.ds[int(idx)]
        img = example["image"]  # PIL.Image or array
        label = int(example["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_imagenet1k(
    train_batch_size: int = 128,
    test_batch_size: int = 128,
    outdir: Optional[str | Path] = None,
    num_workers: int = 1,
    download: bool = True,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> dict:
    """ILSVRC/imagenet-1k via Hugging Face, mirroring get_mnist API/behavior."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for get_imagenet_hf.\n"
            "Install via `pip install datasets`."
        ) from exc

    outdir = OUTPUT_DIR if outdir is None else Path(outdir)
    datadir = Path(outdir).joinpath("data", "imagenet_hf")
    datadir.mkdir(parents=True, exist_ok=True)

    # Optional "don't download" behavior
    if not download and not any(datadir.iterdir()):
        raise FileNotFoundError(
            f"No cached imagenet-1k dataset found in {datadir} and download=False."
        )

    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()

    # Only rank 0 triggers the initial download into cache_dir
    if rank == 0 and download:
        _ = load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            cache_dir=datadir.as_posix(),
        )
        _ = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            cache_dir=datadir.as_posix(),
        )

    if world_size > 1:
        ezpz.barrier()

    # Now every rank loads from the shared cache_dir
    hf_train = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        cache_dir=datadir.as_posix(),
    )
    hf_val = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        cache_dir=datadir.as_posix(),
    )

    # ImageNet-style normalization and transforms
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset1 = HFImageNet1K(hf_train, transform=train_transform)
    dataset2 = HFImageNet1K(hf_val, transform=test_transform)

    train_kwargs: dict = {
        "batch_size": train_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    test_kwargs: dict = {
        "batch_size": test_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }

    sampler1, sampler2 = None, None
    if world_size > 1:
        sampler1 = DistributedSampler(
            dataset1,
            rank=rank,
            num_replicas=world_size,
            shuffle=True,
        )
        sampler2 = DistributedSampler(
            dataset2,
            rank=rank,
            num_replicas=world_size,
            shuffle=False,
        )
        train_kwargs["sampler"] = sampler1
        test_kwargs["sampler"] = sampler2
    else:
        train_kwargs["shuffle"] = shuffle

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset1,
        **train_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset2,
        **test_kwargs,
    )

    return {
        "train": {
            "data": dataset1,
            "loader": train_loader,
            "sampler": sampler1,
        },
        "test": {
            "data": dataset2,
            "loader": test_loader,
            "sampler": sampler2,
        },
    }


def get_openimages(
    train_batch_size: int = 128,
    test_batch_size: int = 128,
    outdir: Optional[str | Path] = None,
    num_workers: int = 1,
    download: bool = False,  # kept for API parity; not used
    shuffle: bool = False,
    pin_memory: bool = True,
) -> dict:
    """Return train/test OpenImages datasets, loaders, and samplers.

    Expects an ImageFolder-style layout:

        <outdir>/data/openimages/
            train/
                class_000/
                class_001/
                ...
            val/
                class_000/
                class_001/
                ...

    `download` is a no-op here; you need to stage the data yourself.
    """
    outdir = OUTPUT_DIR if outdir is None else Path(outdir)
    datadir = Path(outdir).joinpath("data", "openimages")

    train_dir = datadir / "train"
    val_dir = datadir / "val"

    # Sanity check (only on rank 0)
    if ezpz.get_rank() == 0:
        if not train_dir.is_dir() or not val_dir.is_dir():
            from ezpz.data.utils import download_openimages_subset

            download_openimages_subset(
                outdir=datadir,
                split="train",
                max_classes=50,
                num_workers=num_workers,
            )
            # raise FileNotFoundError(
            #     f"Expected OpenImages data under:\n"
            #     f"  {train_dir}\n"
            #     f"  {val_dir}\n"
            #     "with standard ImageFolder layout."
            # )
    if ezpz.get_world_size() > 1:
        ezpz.barrier()

    # Use standard ImageNet/OpenImages-like normalization
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Datasets
    dataset1 = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
    )
    dataset2 = datasets.ImageFolder(
        root=val_dir,
        transform=test_transform,
    )

    train_kwargs: dict = {
        "batch_size": train_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    test_kwargs: dict = {
        "batch_size": test_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }

    sampler1, sampler2 = None, None
    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()

    if world_size > 1:
        sampler1 = DistributedSampler(
            dataset1,
            rank=rank,
            num_replicas=world_size,
            shuffle=True,
        )
        sampler2 = DistributedSampler(
            dataset2,
            rank=rank,
            num_replicas=world_size,
            shuffle=False,
        )
        train_kwargs["sampler"] = sampler1
        test_kwargs["sampler"] = sampler2
    else:
        train_kwargs["shuffle"] = shuffle

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset1,
        **train_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset2,
        **test_kwargs,
    )

    return {
        "train": {
            "data": dataset1,
            "loader": train_loader,
            "sampler": sampler1,
        },
        "test": {
            "data": dataset2,
            "loader": test_loader,
            "sampler": sampler2,
        },
    }


def get_fake_data(
    img_size: int,
    batch_size: int,
    num_workers: int = 1,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    dtype: Optional[str | torch.dtype] = None,
) -> dict:
    dataset = FakeImageDataset(size=img_size, dtype=dtype)
    # loader = torch.utils.data.DataLoader(  # type:ignore
    #     dataset, batch_size=batch_size, shuffle=shuffle
    # )
    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()
    kwargs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "drop_last": drop_last,
    }
    sampler = None
    if world_size > 1:  # use DistributedSampler when > 1 device
        sampler = DistributedSampler(
            dataset, rank=rank, num_replicas=world_size, shuffle=shuffle
        )
        kwargs |= {"sampler": sampler}
    train_loader = torch.utils.data.DataLoader(  # type:ignore
        dataset, **kwargs
    )
    return {
        "train": {
            "data": dataset,
            "loader": train_loader,
            "sampler": sampler,
        },
    }
