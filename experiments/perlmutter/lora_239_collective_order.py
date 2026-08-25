"""Does the backward collective ORDER change with LoRA rank?

The #239 trace shows work #18 (a reduce-scatter) skipped while #19 (an
all-gather) started. If the frozen-unit all-gather lands at a different
position relative to the block reduce-scatter chain depending on r, that
would be the rank-dependence the trace otherwise lacks.
"""
import os, torch, torch.nn as nn, torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

D, L, V = 64, 6, 512


def mk(r):
    class Blk(nn.Module):
        def __init__(s):
            super().__init__()
            s.wq = nn.Linear(D, D, bias=False); s.wo = nn.Linear(D, D, bias=False)
            s.A = nn.Linear(D, r, bias=False);  s.B = nn.Linear(r, D, bias=False)
            for p in (s.wq.weight, s.wo.weight):
                p.requires_grad_(False)

        def forward(s, x):
            return x + s.wo(s.wq(x)) + s.B(s.A(x))

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok_embeddings = nn.Embedding(V, D)
            s.layers = nn.ModuleList([Blk() for _ in range(L)])
            s.norm = nn.LayerNorm(D); s.output = nn.Linear(D, V, bias=False)

        def forward(s, i):
            h = s.tok_embeddings(i)
            for b in s.layers:
                h = b(h)
            return s.output(s.norm(h))

    return M()


def run(rank, ws, r, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29601",
                      RANK=str(rank), WORLD_SIZE=str(ws))
    torch.set_default_device("cpu")
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    m = mk(r)
    m.tok_embeddings.weight.requires_grad_(False)
    m.output.weight.requires_grad_(False)
    for p in m.norm.parameters():
        p.requires_grad_(False)
    seq = []; ph = ["fwd"]
    oag, ors = dist.all_gather_into_tensor, dist.reduce_scatter_tensor

    def ag(*a, **k):
        seq.append((ph[0], "A")); return oag(*a, **k)

    def rs(*a, **k):
        i = k.get("input", a[1] if len(a) > 1 else None)
        seq.append((ph[0], "R", i.numel() if i is not None else -1)); return ors(*a, **k)

    dist.all_gather_into_tensor = ag; dist.reduce_scatter_tensor = rs
    mesh = init_device_mesh("cpu", (ws,)); kw = dict(mesh=mesh, reshard_after_forward=True)
    fully_shard(m.tok_embeddings, **kw)
    for b in m.layers:
        fully_shard(b, **kw)
    fully_shard([m.norm, m.output], **kw); fully_shard(m, **kw)
    ph[0] = "fwd"
    out = m(torch.randint(0, V, (2, 8)))
    ph[0] = "bwd"
    out.float().pow(2).mean().backward()
    dist.all_gather_into_tensor = oag; dist.reduce_scatter_tensor = ors
    if rank == 0:
        allseq = "".join(x[1] for x in seq)
        bwd = "".join(x[1] for x in seq if x[0] == "bwd")
        q.put((allseq, bwd))
    dist.destroy_process_group()


if __name__ == "__main__":
    ref = None
    for r in (8, 16, 32, 64):
        q = mp.get_context("spawn").SimpleQueue()
        mp.spawn(run, args=(2, r, q), nprocs=2, join=True)
        allseq, bwd = q.get()
        tag = ""
        if ref is None:
            ref = (allseq, bwd)
        else:
            tag = "  <-- DIFFERS" if (allseq, bwd) != ref else "  (identical to r=8)"
        print(f"r={r:<3} full={allseq}")
        print(f"      bwd ={bwd}{tag}")
