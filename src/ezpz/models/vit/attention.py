"""Attention blocks for Vision Transformer experiments.

Provides :class:`AttentionBlock` (using ``torchvision.ops.MLP``) and
:class:`timmAttentionBlock` (using ``timm.models.layers.Mlp``), both of
which accept a pluggable attention function for benchmarking different
attention backends.

Based on `Increasing Transformer Model Efficiency Through Attention Layer
Optimization <https://towardsdatascience.com/increasing-transformer-model-efficiency-through-attention-layer-optimization-fefa6f87b1d6>`_.
"""

import functools

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """ViT attention block using ``torchvision.ops.MLP`` for the FFN.

    Args:
        attn_fn: Callable implementing ``(q, k, v) -> out`` attention.
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        format: QKV permutation format.  Use ``"bshd"`` for batch-first
            layouts; defaults to the standard ``(heads, seq, dim)`` order.
        **kwargs: Ignored (accepted for API compatibility).
    """

    def __init__(
        self,
        attn_fn,
        dim: int = 768,
        num_heads: int = 12,
        format: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_fn = attn_fn
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        from torchvision.ops import MLP

        self.mlp = MLP(
            in_channels=dim,
            hidden_channels=4 * [dim],
        )
        permute = (2, 0, 3, 1, 4)
        self.permute_attn = functools.partial(torch.transpose, dim0=1, dim1=2)

        if format == "bshd":
            permute = (2, 0, 1, 3, 4)
            self.permute_attn = nn.Identity()
        self.permute_qkv = functools.partial(torch.permute, dims=permute)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x_in)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # permute tensor based on the specified format
        qkv = self.permute_qkv(qkv)
        q, k, v = qkv.unbind(0)
        # use the attention function specified by the user
        x = self.attn_fn(q, k, v)
        # permute output according to the specified format
        x = self.permute_attn(x).reshape(B, N, C)
        x = self.proj(x)
        x = x + x_in
        x = x + self.mlp(self.norm2(x))
        return x


class timmAttentionBlock(nn.Module):
    """ViT attention block using ``timm.models.layers.Mlp`` for the FFN.

    Identical to :class:`AttentionBlock` except the MLP is sourced from
    the ``timm`` library instead of ``torchvision``.

    Args:
        attn_fn: Callable implementing ``(q, k, v) -> out`` attention.
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        format: QKV permutation format (see :class:`AttentionBlock`).
        **kwargs: Ignored (accepted for API compatibility).
    """

    def __init__(
        self,
        attn_fn,
        dim: int = 768,
        num_heads: int = 12,
        format: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_fn = attn_fn
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        from timm.models.layers import Mlp

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim * 4,
        )
        permute = (2, 0, 3, 1, 4)
        self.permute_attn = functools.partial(torch.transpose, dim0=1, dim1=2)

        if format == "bshd":
            permute = (2, 0, 1, 3, 4)
            self.permute_attn = nn.Identity()
        self.permute_qkv = functools.partial(torch.permute, dims=permute)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x_in)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # permute tensor based on the specified format
        qkv = self.permute_qkv(qkv)
        q, k, v = qkv.unbind(0)
        # use the attention function specified by the user
        x = self.attn_fn(q, k, v)
        # permute output according to the specified format
        x = self.permute_attn(x).reshape(B, N, C)
        x = self.proj(x)
        x = x + x_in
        x = x + self.mlp(self.norm2(x))
        return x
