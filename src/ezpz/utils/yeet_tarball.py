#!/usr/bin/env python
"""
ezpz/utils/yeet_tarball.py

Utility to transfer and extract a tarball across distributed nodes using ezpz.
"""

import os
import time
import ezpz
import argparse

from pathlib import Path

logger = ezpz.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str, required=False)
    parser.add_argument(
        "--d", default=True, action="store_true", help="decompress"
    )
    parser.add_argument("--flags", type=str, default="xf")
    parser.add_argument("--chunk_size", type=int, default=1024 * 1024 * 128)
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing tarball"
    )

    # src = Path(args.src)
    # if args.dst is None:
    #     args.dst = f"/tmp/{Path(args.src).stem}"
    #     logger.warning(f"Destination not provided, using {args.dst}")
    # Auto decompress if:
    #   - `src` ends with `.tar, .tar.gz, .tgz, .tar.bz2, .tbz` _and_
    #   - `dst` ends with `.tar, .tar.gz, .tgz, .tar.bz2, .tbz`
    # if args.d is False and src.suffix in [
    #     ".tar",
    #     ".gz",
    #     ".tgz",
    #     ".bz2",
    #     ".tbz",
    # ]:
    #     args.d = True
    return parser.parse_args()


def bcast_chunk(A, chunk_size: int = 1024 * 1024 * 128) -> bytearray:
    if ezpz.get_rank() == 0:
        size = len(A)
        logger.info(f"size of data {size}")
    else:
        size = 0
    size = ezpz.dist.broadcast(size, root=0)
    nc = size // chunk_size + 1
    B = bytearray(size)
    import tqdm

    for i in tqdm.trange(nc, disable=(ezpz.get_rank() != 0)):
        if i * chunk_size < size:
            end = min(i * chunk_size + chunk_size, size)
            data = A[i * chunk_size : end] if ezpz.get_rank() == 0 else None
            B[i * chunk_size : end] = ezpz.dist.broadcast(data, root=0)
    return B


CHUNK_SIZE = 1024 * 1024 * 128


def transfer(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    decompress: bool = True,
    chunk_size: int = CHUNK_SIZE,
    flags: str = "xf",
):
    logger.info(f"Transfer started at {ezpz.get_timestamp()}")
    start_time = time.perf_counter()
    if ezpz.get_rank() == 0:
        start = time.perf_counter()
        with open(src, "rb") as f:
            data = f.read()
        end = time.perf_counter()
        logger.info("\n")
        logger.info("==================")
        logger.info(f"Rank-0 loading library {src} took {end - start} seconds")
    else:
        data = None
    start = time.perf_counter()
    data = bcast_chunk(data, chunk_size)
    end = time.perf_counter()
    with open(dst, "wb") as f:
        f.write(data)
    if ezpz.get_rank() == 0:
        logger.info(f"Broadcast took {end - start} seconds")
        logger.info(
            f"Writing to the disk {dst} took {time.perf_counter() - end}"
        )
    start = time.perf_counter()
    dirname = os.path.dirname(dst)
    assert os.path.isfile(dst)
    if decompress:
        t0d = time.perf_counter()
        os.system(f"tar -p -{flags} {dst} -C {dirname}")
        logger.info(f"untar took {time.perf_counter() - t0d:.2f} seconds")
    logger.info(f"Total time: {time.perf_counter() - start_time} seconds")
    logger.info("==================\n")


def _create_tarball_if_needed(
    src: str | os.PathLike, overwrite: bool = False
) -> Path:
    # Create a tarball
    # src_fp = src_fp.parent / f"{src_fp.name}.tar.gz"
    src = Path(src).absolute().resolve()
    tarball_name = f"{src.name}.tar.gz"
    tarball_fp = Path(tarball_name).absolute().resolve()
    # tarball_fp = fp.parent / tarball_name
    logger.warning(
        f"{src} is a directory! creating a tarball at {src.as_posix()}"
    )
    if tarball_fp.exists():
        logger.info(f"Tarball {tarball_fp} already exists")
        if overwrite and ezpz.get_rank() == 0:
            fp_bak = src.with_suffix(
                src.suffix + f"{ezpz.get_timestamp()}.bak"
            )
            logger.info(f"Backing up existing tarball to {fp_bak}")
            os.rename(src, fp_bak)
        else:
            logger.info("Not overwriting existing tarball, exiting.")
            raise FileExistsError(
                f"Tarball {tarball_fp} already exists. Use --overwrite to overwrite."
            )
    from ezpz.utils import create_tarball

    t0 = time.perf_counter()

    src_tarball = create_tarball(src)
    assert src_tarball.exists(), f"{src} does not exist"
    logger.info(
        " ".join(
            [
                f"Created tarball at {src_tarball.as_posix()}",
                f"in {time.perf_counter() - t0:.2f} seconds",
            ]
        )
    )
    return src_tarball


def main(
    src: str | os.PathLike,
    dst: str | os.PathLike | None = None,
    overwrite: bool = False,
    decompress: bool = True,
    chunk_size: int = 1024 * 1024 * 128,
    flags: str = "xf",
):
    src_fp = Path(src).absolute().resolve()
    assert src_fp.exists(), f"{src_fp} does not exist"
    if src_fp.is_dir():
        src_tarball = _create_tarball_if_needed(src_fp, overwrite=overwrite)
    elif src_fp.is_file():
        src_tarball = src_fp
    else:
        raise ValueError(f"{src_fp} is neither a file nor a directory")
    dst_name = f"{src_tarball.name}".replace(".tar", "").replace(".gz", "")
    dst_fp = Path("/tmp") / f"{dst_name}.tar.gz" if dst is None else Path(dst)
    logger.info(f"Copying {src_tarball} to {dst_fp}")
    return transfer(
        src=src_tarball.as_posix(),
        dst=dst_fp.as_posix(),
        decompress=decompress,
        chunk_size=chunk_size,
        flags=flags,
    )


if __name__ == "__main__":
    import ezpz

    _ = ezpz.setup_torch()
    args = parse_args()
    main(
        src=args.src,
        dst=args.dst,
        decompress=args.d,
        chunk_size=args.chunk_size,
        flags=args.flags,
        overwrite=args.overwrite,
    )
