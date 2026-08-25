# Attention From Scratch

```python
import math
import torch

def sdpa(q: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None):
    assert q.size(-1) == k.size(-1), "Queries and Keys must have same embedding dimension"
    scores = (q @ k.transpose(-2, -1) / math.sqrt(k.size(-1)))
    weights = scores.softmax(-1)
    return weights @ v


class AttentionBlock(torch.nn.Module):
    def __init__(self, di: int, do: int, bias: bool = False):
        self.wq = torch.nn.Linear(di, do, bias=bias)
        self.wk = torch.nn.Linear(di, do, bias=bias)
        self.wv = torch.nn.Linear(di, do, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return sdpa(self.wq(q), self.wk(k), self.wv(v))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d: int, nh: int, bias: bool = False):
        assert d % nh == 0, "Number of heads must evenly divide embedding dimension."
        dh = d // nh
        self.heads = torch.nn.ModuleList([
            AttentionBlock(d, dh, bias=bias) for _ in range(nh)
        ])
        self.projection = torch.nn.Linear(d, d, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.projection(torch.cat([head(q, k, v) for head in self.heads]))
```

