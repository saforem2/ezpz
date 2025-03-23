# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
ezpz/tp/__init__.py

modified from:
https://github.com/facebookresearch/fairscale/blob/5f484b3545f27eddb19d970fbe1d361b9c5f2b07/fairscale/nn/tensor_parallel/initialize.py
"""

import logging

from typing import List, Optional

import socket

# NOTE: Need to swap import order on Polaris (hostname: [x3...])
if socket.gethostname().startswith('x3'):
    from mpi4py import MPI
    import torch
else:
    import torch
    from mpi4py import MPI  # type:ignore

import torch.distributed as tdist
from datetime import timedelta
from ezpz.tp.utils import (
    ensure_divisibility,
    divide_and_check_no_remainder,
    split_tensor_along_last_dim,
)

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


__all__ = [
    'initialize_tensor_parallel',
    'tensor_parallel_is_initialized',
    'get_tensor_parallel_group',
    'get_data_parallel_group',
    'get_pipeline_parallel_group',
    'get_tensor_parallel_world_size',
    'get_tensor_parallel_rank',
    'get_tensor_parallel_src_rank',
    'get_data_parallel_world_size',
    'get_data_parallel_rank',
    'destroy_tensor_parallel',
    'ensure_divisibility',
    'divide_and_check_no_remainder',
    'split_tensor_along_last_dim',
    'get_context_parallel_group',
    'get_context_parallel_ranks',
    'get_context_parallel_world_size',
    'get_context_parallel_rank',
    'get_pipeline_parallel_ranks',
    'get_context_parallel_rank',
]

# tensor parallel group that the current rank belongs to.
_TENSOR_PARALLEL_GROUP = None
_TENSOR_PARALLEL_RANKS = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_RANKS = None
# Pipeline parallel group that the current rank belongs to.
_PIPELINE_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_RANKS = None

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP_RANKS = None


def initialize_tensor_parallel(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
    tensor_parallel_backend: Optional[str] = None,
    pipeline_parallel_backend: Optional[str] = None,
    context_parallel_backend: Optional[str] = None,
    data_parallel_backend: Optional[str] = None,
    timeout: Optional[timedelta] = None,
) -> None:
    """
    Initialize tensor data parallel groups.

    Arguments:
        tensor_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 tensor parallel groups and 2 data parallel groups as:
        4 tensor parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    process groups initialized in the order of TP, CP, PP, DP.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the tensor tensor, 2 GPUs to parallelize context(seq len), and 2 GPUs to parallelize
    the tensor pipeline. The present function will
    create 8 tensor model-parallel groups, 8 context-parallel group, 8 pipeline model-parallel groups
    and 8 data-parallel groups as:
    when alternate_pp_config = False,
        8 data_parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7], [g8, g12], [g9, g13], [g10, g14], [g11, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 context-parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 pipeline model-parallel groups:
            [g0, g8], [g1, g9], [g2, g10], [g3, g11], [g4, g12], [g5, g13], [g6, g16], [g7, g15]
    """
    # Get world size and rank. Ensure some consistencies.
    assert tdist.is_initialized()
    world_size = tdist.get_world_size()
    tensor_parallel_size = int(min(tensor_parallel_size, world_size))
    ensure_divisibility(world_size, tensor_parallel_size)
    ensure_divisibility(world_size, context_parallel_size)
    ensure_divisibility(
        world_size,
        tensor_parallel_size * pipeline_parallel_size * context_parallel_size,
    )
    rank = tdist.get_rank()

    dpsize = int(
        world_size
        / (tensor_parallel_size * pipeline_parallel_size * context_parallel_size)
    )

    if tdist.get_rank() == 0:
        logger.info(
            '> initializing tensor parallel with size {}'.format(
                tensor_parallel_size
            )
        )
        logger.info(
            '> initializing context parallel with size {}'.format(
                context_parallel_size
            )
        )
        logger.info(
            '> initializing pipeline with size {}'.format(pipeline_parallel_size)
        )
        logger.info(
            '> initializing ddp with size {}'.format(dpsize)
        )

    groups = torch.LongTensor(range(world_size)).reshape(
        dpsize,
        pipeline_parallel_size,
        context_parallel_size,
        tensor_parallel_size,
    )

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_RANKS
    assert (
        _DATA_PARALLEL_GROUP is None
    ), 'data parallel group is already initialized'
    assert (
        _DATA_PARALLEL_RANKS is None
    ), 'data parallel ranks are already initialized'
    for i in range(pipeline_parallel_size):
        for j in range(context_parallel_size):
            for k in range(tensor_parallel_size):
                ranks = groups[:, i, j, k].tolist()
                group = tdist.new_group(
                    groups[:, i, j, k].tolist(),
                    backend=data_parallel_backend,
                    timeout=timeout,
                )
                if i == found[1] and j == found[2] and k == found[3]:
                    _DATA_PARALLEL_GROUP = group
                    _DATA_PARALLEL_RANKS = ranks

    # Build the tensor parallel groups.
    global _TENSOR_PARALLEL_GROUP
    global _TENSOR_PARALLEL_RANKS
    assert (
        _TENSOR_PARALLEL_GROUP is None
    ), 'tensor parallel group is already initialized'
    assert (
        _TENSOR_PARALLEL_RANKS is None
    ), 'tensor parallel ranks are already initialized'
    for i in range(dpsize):
        for j in range(pipeline_parallel_size):
            for k in range(context_parallel_size):
                ranks = groups[i, j, k, :].tolist()
                group = tdist.new_group(
                    groups[i, j, k, :].tolist(),
                    backend=tensor_parallel_backend,
                    timeout=timeout,
                )
                if i == found[0] and j == found[1] and k == found[2]:
                    _TENSOR_PARALLEL_GROUP = group
                    _TENSOR_PARALLEL_RANKS = ranks

    # Build the pipeline parallel groups.
    global _PIPELINE_PARALLEL_GROUP
    global _PIPELINE_PARALLEL_RANKS
    assert (
        _PIPELINE_PARALLEL_GROUP is None
    ), 'Pipeline parallel group is already initialized'
    for i in range(dpsize):
        for j in range(context_parallel_size):
            for k in range(tensor_parallel_size):
                ranks = groups[i, :, j, k].tolist()
                group = tdist.new_group(
                    ranks, backend=pipeline_parallel_backend, timeout=timeout
                )
                if i == found[0] and j == found[2] and k == found[3]:
                    _PIPELINE_PARALLEL_GROUP = group
                    _PIPELINE_PARALLEL_RANKS = ranks

    # Build the context parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GROUP_RANKS

    assert (
        _CONTEXT_PARALLEL_GROUP is None
    ), 'Context parallelism is already initialized.'
    for i in range(dpsize):
        for j in range(pipeline_parallel_size):
            for k in range(tensor_parallel_size):
                ranks = groups[i, j, :, k].tolist()
                group = tdist.new_group(
                    ranks, backend=context_parallel_backend, timeout=timeout
                )
                if i == found[0] and j == found[1] and k == found[3]:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GROUP_RANKS = ranks


def tensor_parallel_is_initialized() -> bool:
    """Check if tensor and data parallel groups are initialized."""
    if (
        _TENSOR_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
        or _PIPELINE_PARALLEL_GROUP is None
        or _CONTEXT_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_tensor_parallel_group() -> tdist.ProcessGroup:
    """Get the tensor parallel group the caller rank belongs to."""
    assert (
        _TENSOR_PARALLEL_GROUP is not None
    ), 'tensor parallel group is not initialized'
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_ranks() -> List[int]:
    """Get the tensor parallel group the caller rank belongs to."""
    assert (
        _TENSOR_PARALLEL_RANKS is not None
    ), 'tensor parallel group is not initialized'
    return _TENSOR_PARALLEL_RANKS


def get_tensor_parallel_world_size() -> int:
    """Return world size for the tensor parallel group."""
    return tdist.get_world_size(group=get_tensor_parallel_group())


def get_tensor_parallel_rank() -> int:
    """Return my rank for the tensor parallel group."""
    return tdist.get_rank(group=get_tensor_parallel_group())


def get_tensor_parallel_src_rank() -> int:
    """
    Calculate the global rank corresponding to local rank 0 in the TP group.
    """
    global_rank = tdist.get_rank()
    local_world_size = get_tensor_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_context_parallel_group() -> tdist.ProcessGroup:
    """Get the context parallel group the caller rank belongs to."""
    assert (
        _CONTEXT_PARALLEL_GROUP is not None
    ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_ranks() -> List[int]:
    """Return context parallel ranks for the context parallel group."""
    assert (
        _CONTEXT_PARALLEL_GROUP_RANKS is not None
    ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP_RANKS


def get_context_parallel_world_size() -> int:
    """Return world size for the context parallel group."""
    return tdist.get_world_size(group=get_context_parallel_group())


def get_context_parallel_rank() -> int:
    """Return my rank for the context parallel group."""
    return tdist.get_rank(group=get_context_parallel_group())


def get_pipeline_parallel_group() -> tdist.ProcessGroup:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_PARALLEL_GROUP is not None
    ), 'pipeline parallel group is not initialized'
    return _PIPELINE_PARALLEL_GROUP


def get_pipeline_parallel_ranks() -> List[int]:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_PARALLEL_RANKS is not None
    ), 'pipeline parallel group is not initialized'
    return _PIPELINE_PARALLEL_RANKS


def get_pipeline_parallel_world_size() -> int:
    """Return world size for the context parallel group."""
    return tdist.get_world_size(group=get_pipeline_parallel_group())


def get_pipeline_parallel_rank() -> int:
    """Return my rank for the pipeline parallel group."""
    return tdist.get_rank(group=get_pipeline_parallel_group())


def get_data_parallel_group() -> tdist.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert (
        _DATA_PARALLEL_GROUP is not None
    ), 'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_ranks() -> List[int]:
    """Get the data parallel group the caller rank belongs to."""
    assert (
        _DATA_PARALLEL_RANKS is not None
    ), 'data parallel group is not initialized'
    return _DATA_PARALLEL_RANKS


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    return tdist.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return tdist.get_rank(group=get_data_parallel_group())


def destroy_tensor_parallel() -> None:
    """Set the groups to none."""
    global _TENSOR_PARALLEL_GROUP
    _TENSOR_PARALLEL_GROUP = None
    global _TENSOR_PARALLEL_RANKS
    _TENSOR_PARALLEL_RANKS = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_RANKS
    _DATA_PARALLEL_RANKS = None

    global _PIPELINE_PARALLEL_GROUP
    _PIPELINE_PARALLEL_GROUP = None
    global _PIPELINE_PARALLEL_RANKS
    _PIPELINE_PARALLEL_RANKS = None

    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GROUP_RANKS
    _CONTEXT_PARALLEL_GROUP_RANKS = None
