#!/usr/bin/env python
# -----------------------------------------------------------------------
# This script is to transfer the package to the local drives.
# Rank 0 will load the data and then do the broadcast and write them back
# -----------------------------------------------------------------------
import ezpz
import argparse
import os
import time


logger = ezpz.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--d", action="store_true")
    parser.add_argument("--flags", type=str, default="xf")
    parser.add_argument("--chunk_size", type=int, default=1024 * 1024 * 128)
    args = parser.parse_args()
    return args


def bcast_chunk(A, chunk_size: int = 1024 * 1024 * 128) -> bytearray:
    if ezpz.get_rank() == 0:
        size = len(A)
        logger.info(f"size of data {size}")
    else:
        size = 0
    size = ezpz.dist.broadcast(size, root=0)
    nc = size // chunk_size + 1
    B = bytearray(size)
    for i in range(nc):
        if i * chunk_size < size:
            end = min(i * chunk_size + chunk_size, size)
            if ezpz.get_rank() == 0:
                data = A[i * chunk_size : end]
            else:
                data = None
            B[i * chunk_size : end] = ezpz.dist.broadcast(data, root=0)
    return B


def Transfer(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    decompress=True,
    chunk_size: int = 1024 * 1024 * 128,
    flags: str = "xf",
):
    start_time = time.time()
    if ezpz.get_rank() == 0:
        start = time.time()
        with open(src, "rb") as f:
            data = f.read()
        end = time.time()
        logger.info("\n")
        logger.info("==================")
        logger.info(f"Rank-0 loading library {src} took {end - start} seconds")
    else:
        data = None
    start = time.time()
    data = bcast_chunk(data, chunk_size)
    end = time.time()
    with open(dst, "wb") as f:
        f.write(data)
    if ezpz.get_rank() == 0:
        logger.info(f"Broadcast took {end - start} seconds")
        logger.info(f"Writing to the disk {dst} took {time.time() - end}")
    start = time.time()
    dirname = os.path.dirname(dst)
    assert os.path.isfile(dst)
    if decompress:
        os.system(f"tar -p -{flags} {dst} -C {dirname}")
    if ezpz.get_rank() == 0:
        if decompress:
            logger.info(f"untar took {time.time() - start} seconds")
        logger.info(f"Total time: {time.time() - start_time} seconds")
        logger.info("==================\n")


def main():
    args = parse_args()
    Transfer(
        src=args.src,
        dst=args.dst,
        decompress=args.d,
        chunk_size=args.chunk_size,
        flags=args.flags,
    )


if __name__ == "__main__":
    _ = ezpz.setup_torch()
    main()
