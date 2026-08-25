```python
import math
import torch

def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    assert q.size(-1) == k.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
    attn = scores.softmax(-1)
    return attn @ v

class AttentionBlock(torch.nn.Module):
    def __init__(self, d_in, d_out, bias: bool = False):
        self.wq = torch.nn.Linear(d_in, d_out, bias=bias)
        self.wk = torch.nn.Linear(d_in, d_out, bias=bias)
        self.wv = torch.nn.Linear(d_in, d_out, bias=bias)

    def forward(self, q, k, v):
        return sdpa(self.wq(q), self.wk(k), self.wv(v))

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, nheads, bias):
        super().__init__()
        assert dim % nheads == 0
        dh = dim // nheads
        self.heads = torch.nn.ModuleList([
            AttentionBlock(dim, dh, bias=bias) for _ in range(nheads)
        ])
        self.projection = torch.nn.Linear(dim, dim)

    def forward(self, q, k, v):
        attns = torch.cat([
            head(q, k, v) for head in self.heads
        ], dim=-1)
        return self.projection(attns)
```
