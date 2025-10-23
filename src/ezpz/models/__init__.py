# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
mmm/data/__init__.py
"""

from typing import Optional, Sequence

import ezpz

logger = ezpz.get_logger(__name__)


def summarize_model(
    model,
    verbose: bool = False,
    depth: int = 1,
    input_size: Optional[Sequence[int]] = None,
):
    try:
        from torchinfo import summary

        summary_str = summary(
            model,
            input_size=input_size,
            depth=depth,
            verbose=verbose,
        )
        # logger.info(f'\n{summary_str}')
        return summary_str

    except (ImportError, ModuleNotFoundError):
        logger.warning("torchinfo not installed, unable to print model summary!")
