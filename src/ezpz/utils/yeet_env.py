import os
import sys

from pathlib import Path


def main():
    import ezpz.pbs
    import ezpz.launch

    # if overwrite and ezpz.get_rank() == 0:
    #     fp_bak = src.with_suffix(
    #         src.suffix + f"{ezpz.get_timestamp()}.bak"
    #     )
    #     logger.info(f"Backing up existing tarball to {fp_bak}")
    #     os.rename(src, fp_bak)
    # else:
    #     logger.info("Not overwriting existing tarball, exiting.")
    #     raise FileExistsError(
    #         f"Tarball {tarball_fp} already exists. Use --overwrite to overwrite."
    #     )
    if os.environ.get("MAKE_TARBALL") is not None:
        from ezpz.utils import check_for_tarball

        tarball_fp = check_for_tarball()
        flags = f"--src {tarball_fp} --decompress"
    else:
        flags = " ".join(sys.argv[1:])
    # single process on each node
    ezpz.launch.launch(
        launch_cmd=ezpz.pbs.get_pbs_launch_cmd(ngpu_per_host=1),
        include_python=False,
        cmd_to_launch=(f"{sys.executable} -m ezpz.utils.yeet_tarball {flags}"),
    )


if __name__ == "__main__":
    main()
