import os
import sys
from typing import Optional, Sequence


def run(argv: Optional[Sequence[str]]) -> int:
    import ezpz.launch
    import ezpz.pbs

    if os.environ.get("MAKE_TARBALL") is not None:
        from ezpz.utils import check_for_tarball

        tarball_fp = check_for_tarball()
        flags = f"--src {tarball_fp} --decompress"
    elif argv is not None:
        flags = " ".join(argv)
    else:
        raise ValueError(f"Received {argv=}")
    # single process on each node
    return ezpz.launch.launch(
        launch_cmd=ezpz.pbs.get_pbs_launch_cmd(ngpu_per_host=1),
        include_python=False,
        cmd_to_launch=(f"{sys.executable} -m ezpz.utils.yeet_tarball {flags}"),
    )


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    main()
