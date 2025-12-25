import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import ezpz

logger = ezpz.get_logger(__name__)

CHUNK_SIZE = 1024 * 1024 * 128


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument(
        "--decompress",
        dest="decompress",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--flags", type=str, default="xf")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing tarball"
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal flag; when set, run transfer instead of launching.",
    )
    return parser.parse_args(argv)


def bcast_chunk(data: bytes | bytearray | None, chunk_size: int) -> bytearray:
    if ezpz.get_rank() == 0:
        size = len(data) if data is not None else 0
        logger.info(f"size of data {size}")
    else:
        size = 0
    size = ezpz.dist.broadcast(size, root=0)
    nc = size // chunk_size + 1
    buffer = bytearray(size)
    import tqdm

    for i in tqdm.trange(nc, disable=(ezpz.get_rank() != 0)):
        if i * chunk_size < size:
            end = min(i * chunk_size + chunk_size, size)
            payload = data[i * chunk_size : end] if ezpz.get_rank() == 0 else None
            buffer[i * chunk_size : end] = ezpz.dist.broadcast(payload, root=0)
    return buffer


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
        logger.info(f"Writing to the disk {dst} took {time.perf_counter() - end}")
    dirname = os.path.dirname(dst)
    assert os.path.isfile(dst)
    if decompress:
        t0d = time.perf_counter()
        os.system(f"tar -p -{flags} {dst} -C {dirname}")
        logger.info(f"untar took {time.perf_counter() - t0d:.2f} seconds")
    logger.info(f"Total time: {time.perf_counter() - start_time} seconds")
    logger.info("==================\n")


def _create_tarball_if_needed(src: str | os.PathLike, overwrite: bool = False) -> Path:
    src_path = Path(src).absolute().resolve()
    tarball_name = f"{src_path.name}.tar.gz"
    tarball_fp = Path(tarball_name).absolute().resolve()
    logger.warning(f"{src_path} is a directory! creating a tarball at {tarball_fp}")
    if tarball_fp.exists():
        logger.info(f"Tarball {tarball_fp} already exists")
        if overwrite and ezpz.get_rank() == 0:
            backup = src_path.with_suffix(src_path.suffix + f"{ezpz.get_timestamp()}.bak")
            logger.info(f"Backing up existing tarball to {backup}")
            os.rename(src_path, backup)
        else:
            logger.info("Not overwriting existing tarball, exiting.")
            raise FileExistsError(
                f"Tarball {tarball_fp} already exists. Use --overwrite to overwrite."
            )
    from ezpz.utils import create_tarball

    t0 = time.perf_counter()
    src_tarball = create_tarball(src_path)
    assert src_tarball.exists(), f"{src_path} does not exist"
    logger.info(
        " ".join(
            [
                f"Created tarball at {src_tarball.as_posix()}",
                f"in {time.perf_counter() - t0:.2f} seconds",
            ]
        )
    )
    return src_tarball


def execute_transfer(args: argparse.Namespace) -> int:
    src_fp = Path(args.src).absolute().resolve()
    assert src_fp.exists(), f"{src_fp} does not exist"
    if src_fp.is_dir():
        src_tarball = _create_tarball_if_needed(src_fp, overwrite=args.overwrite)
    elif src_fp.is_file():
        src_tarball = src_fp
    else:
        raise ValueError(f"{src_fp} is neither a file nor a directory")
    dst_name = f"{src_tarball.name}".replace(".tar", "").replace(".gz", "")
    dst_fp = Path("/tmp") / f"{dst_name}.tar.gz" if args.dst is None else Path(args.dst)
    logger.info(f"Copying {src_tarball} to {dst_fp}")
    transfer(
        src=src_tarball.as_posix(),
        dst=dst_fp.as_posix(),
        decompress=args.decompress,
        chunk_size=args.chunk_size,
        flags=args.flags,
    )
    return 0


def launch_workers(argv: Sequence[str]) -> int:
    import ezpz.launch
    import ezpz.pbs

    worker_flags = " ".join(argv)
    return ezpz.launch.launch(
        launch_cmd=ezpz.pbs.get_pbs_launch_cmd(ngpu_per_host=1),
        include_python=False,
        cmd_to_launch=(f"{sys.executable} -m ezpz.utils.yeet_env --worker {worker_flags}"),
    )


def run(argv: Optional[Sequence[str]]) -> int:
    if os.environ.get("MAKE_TARBALL") is not None:
        from ezpz.utils import check_for_tarball

        tarball_fp = check_for_tarball()
        argv = ["--src", tarball_fp.as_posix(), "--decompress"]
    if argv is None:
        raise ValueError(f"Received {argv=}")
    return launch_workers(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    if args.worker:
        _ = ezpz.setup_torch()
        return execute_transfer(args)
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
