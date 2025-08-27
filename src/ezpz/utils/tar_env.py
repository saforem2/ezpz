"""
utils/tar_env.py
"""

import os
import sys
import ezpz

from pathlib import Path

logger = ezpz.get_logger(__name__)


def main():
    from ezpz.utils import check_for_tarball
    # _ = ezpz.setup_torch()
    tarball_fp = check_for_tarball()
    logger.info(tarball_fp.as_posix())


if __name__ == "__main__":
    main()
