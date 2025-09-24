import os
import sys


def main():
    import ezpz.launch
    import ezpz.pbs

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
