"""Create or reuse an environment tarball and report its path."""

from __future__ import annotations

import sys
from collections.abc import Sequence
import ezpz

logger = ezpz.get_logger(__name__)


def run(argv: Sequence[str] | None = None) -> int:
    """Inspect/create the environment tarball and print its location."""
    del argv  # tar-env does not accept CLI options yet
    from ezpz.utils import check_for_tarball

    tarball_fp = check_for_tarball()
    logger.info(tarball_fp.as_posix())
    return 0


def main() -> int:
    """Backward-compatible console script entry point."""
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
