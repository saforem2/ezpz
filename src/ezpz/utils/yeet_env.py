from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def run(argv: Sequence[str] | None = None) -> int:
    """Distribute an environment tarball across cluster nodes."""
    import ezpz.launch
    import ezpz.pbs

    args = [] if argv is None else list(argv)
    if os.environ.get("MAKE_TARBALL") is not None:
        from ezpz.utils import check_for_tarball

        tarball_fp = check_for_tarball()
        flags = f"--src {tarball_fp} --decompress"
    else:
        flags = " ".join(args)
    # single process on each node
    return ezpz.launch.launch(
        launch_cmd=ezpz.pbs.get_pbs_launch_cmd(ngpu_per_host=1),
        include_python=False,
        cmd_to_launch=(f"{sys.executable} -m ezpz.utils.yeet_tarball {flags}"),
    )


def main() -> int:
    """Backward-compatible console script entry point."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
