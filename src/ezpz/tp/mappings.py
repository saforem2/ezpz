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
"""
ezpz/tp/mappings.py

Modified from:
<https://github.com/facebookresearch/fairscale/blob/5f484b3545f27eddb19d970fbe1d361b9c5f2b07/fairscale/nn/model_parallel/mappings.py>
"""

from typing import Any

import ezpz

import torch
import torch.distributed as tdist
from ezpz.tp import get_tensor_parallel_group
from ezpz.tp.utils import split_tensor_along_last_dim
# from ezpz.mp.utils import split_tensor_along_last_dim
# from ezpz.utils import split_tensor_along_last_dim
# from .initialize import get_tensor_parallel_group


logger = ezpz.get_logger(__name__)


def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_tensor_parallel_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if tdist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    tdist.all_reduce(input_, group=group)

    return input_


def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_tensor_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if tdist.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = tdist.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = tdist.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = get_tensor_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if tdist.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = tdist.get_rank(group=group)
    world_size = tdist.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    tdist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_) -> torch.Tensor:  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_) -> torch.Tensor:  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_) -> torch.Tensor:  # type: ignore
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _gather(grad_output)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_) -> torch.Tensor:  # type: ignore
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(_CopyToTensorParallelRegion.apply(input_))


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(_ReduceFromTensorParallelRegion.apply(input_))


def scatter_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(_ScatterToTensorParallelRegion.apply(input_))


def gather_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(_GatherFromTensorParallelRegion.apply(input_))
