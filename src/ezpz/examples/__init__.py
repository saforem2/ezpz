"""Collection of runnable `ezpz` training and inference examples.

Launch any script with:

    ezpz launch -m ezpz.examples.<module_name> [args...]

Package initializers are not direct CLIs, so ``python3 -m ezpz.examples`` only
exposes the package and does not define additional ``--help`` output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import ezpz
from ezpz.configs import PathLike


def get_example_outdir(
    module_name: str, base_dir: Optional[PathLike] = None
) -> Path:
    """Return the shared output directory for example scripts.

    Args:
        module_name: Fully-qualified module name for the example.
        base_dir: Optional base output directory override.

    Returns:
        The output directory for this run.
    """
    created_at = os.environ.get("EZPZ_LOG_TIMESTAMP")
    if created_at is None:
        created_at = ezpz.get_timestamp() if ezpz.get_rank() == 0 else None
    created_at = ezpz.dist.broadcast(created_at, root=0)
    if created_at is not None:
        os.environ["EZPZ_LOG_TIMESTAMP"] = created_at
    base_path = (
        Path(os.getcwd()).joinpath("outputs")
        if base_dir is None
        else Path(base_dir).expanduser()
    )
    outdir = base_path.joinpath(module_name, f"{created_at}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir
