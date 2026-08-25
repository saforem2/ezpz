# Attention from Scratch

**Goal**:
Compare a Query ($Q$) against a set of keys ($K$) to determine how much
"attention" to pay to corresponding values ($V$)

Recall:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left\(\frac{Q @
                                                      K^{T}}{\sqrt{d_{k}}\right\)
                                                                                V,$$

Where $d_{k}$ is the dimensionality of the keys and is used for scaling to
normalize the gradients.

## Implementation

### Single Head Attention

```python
import math
import torch

class SingleHeadAttention(nn.Module):
    def __init__(self, dim: int, use_bias: False) -> None:
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def forward(x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = q @ k.transpose(-2, -1) / self.scale
        weights = scores.softmax(-1)
        return weights @ V
```


### Scaled Dot Product Att

```python
import torch
import math


def my_scaled_dot_product_attention(query, key=None, value=None):
    key = key if key is not None else query
    value = value if value is not None else query
    # query and key must have same embedding dimension
    assert query.size(-1) == key.size(-1)
    dk = key.size(-1)  # embed dimension of key
    # query, key, value = (bs, seq_len, embed_dim)
    # compute dot-product to obtain pairwise "similarity" and scale it
    qk = query @ key.transpose(-1, -2) / dk**0.5
    # apply softmax
    # attn_weights = (bs, seq_len, seq_len)
    attn_weights = torch.softmax(qk, dim=-1)
    # compute weighted sum of value vectors
    # attn = (bs, seq_len, embed_dim)
    attn = attn_weights @ value
    return attn  # , attn_weights


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Scaled Dot Product Attention

    The attention value from element i to element j is based on its similarity
    of the query (q_i) and key (k_j).

    This is done using the similarity metric:

    Attention(q, k, v) = softmax(q @ k.T / k.dim(-1)) @ v
    """
    assert q.size(-1) == k.size(-1)
    qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
    attn_weights = qk.softmax(-1)
    attn = attn_weights @ v
    return attn  # , attn_weights


def _sdpa(q, k, v):
    return (q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))).softmax(-1) @ v


def sdpa_(
    q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None
) -> torch.Tensor:
    k = q if k is None else k
    v = q if v is None else v
    dk = k.size(-1)
    # q, k, v = (batch_size, seq_len, embed_dim) = (b, s, d)
    assert q.size(-1) == dk
    # compute dot-product to get pairwise similarity and scale it:
    qk = q @ k.transpose(-2, -1) / math.sqrt(dk)
    # apply softmax
    attn_weights = qk.softmax(-1)
    attn = attn_weights @ v
    return attn  # , attn_weights


def sdpa_torch(q, k=None, v=None):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def sse(x, y):
    return ((x - y) ** 2).sum()
```

#### Test

```python
def main():
    x = torch.normal(0, 1, (2, 3, 6))
    print(sse(sdpa_torch(x, x, x), my_scaled_dot_product_attention(x, x, x)))
    print(sse(sdpa(x, x, x), my_scaled_dot_product_attention(x, x, x)))
    print(sse(_sdpa(x, x, x), my_scaled_dot_product_attention(x, x, x)))
    print(sse(sdpa_(x, x, x), my_scaled_dot_product_attention(x, x, x)))
    print(sse(sdpa_torch(x, x, x), _sdpa(x, x, x)))
    print(sse(sdpa(x, x, x), _sdpa(x, x, x)))
    print(sse(_sdpa(x, x, x), _sdpa(x, x, x)))
    print(sse(sdpa_(x, x, x), _sdpa(x, x, x)))
    print(sse(sdpa_torch(x, x, x), sdpa(x, x, x)))
    print(sse(sdpa(x, x, x), sdpa(x, x, x)))
    print(sse(_sdpa(x, x, x), sdpa(x, x, x)))
    print(sse(sdpa_(x, x, x), sdpa(x, x, x)))
    print(sse(sdpa_torch(x, x, x), sdpa_(x, x, x)))
    print(sse(sdpa(x, x, x), sdpa_(x, x, x)))
    print(sse(_sdpa(x, x, x), sdpa_(x, x, x)))
    print(sse(sdpa_(x, x, x), sdpa_(x, x, x)))
```

### Multi Head Attention

```python
import math
import torch

def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    assert q.size(-1) == k.size(-1)
    qk = (q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))).softmax(-1)
    return qk @ v


class AttentionBlock(torch.nn.Module):
    def __init__(self, d_in, d_out, bias: bool = False):
        super().__init__()
        self.wq = torch.nn.Linear(d_in, d_out, bias=bias)
        self.wk = torch.nn.Linear(d_in, d_out, bias=bias)
        self.wv = torch.nn.Linear(d_in, d_out, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return sdpa(self.wq(q), self.wk(k), self.wv(v))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim: int, nheads: int, bias: bool = False):
        super().__init__()
        assert dim % nheads == 0, "Number of heads must divide embedding dimension"
        self.nheads = nheads
        self.dh = dim // nheads  # head dim
        self.head_blocks = torch.nn.ModuleList([
            AttentionBlock(dim, self.dh, bias=bias) for _ in range(self.nheads)
        ])
        self.projection = torch.nn.Linear(dim, dim)

    def forward(self, q, k, v):
        attns = [head(q, k, v) for head in self.head_blocks]
        attns = torch.cat(attns, dim=-1)
        return self.projection(attns)
```



