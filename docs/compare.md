# Comparison vs. alternatives

Why pick `ezpz` over the obvious alternatives — raw `torchrun`, HuggingFace
`accelerate`, Microsoft DeepSpeed? Honest take below; none of these are
strictly better than another, they're optimized for different things.

## TL;DR

| Tool | Best at | Pick it when |
|---|---|---|
| **raw `torchrun`** | Single-cluster setups; maximum control | You only target one cluster type and want zero abstraction |
| **`accelerate`** | HF-ecosystem ergonomics | You're already in HF Trainer / `transformers` |
| **DeepSpeed** | ZeRO-Infinity, NVMe offload, MoE | You need offload, very-large models, or DeepSpeed's MoE |
| **`ezpz`** | Cross-cluster portability; HPC-scheduler integration | You launch on PBS/SLURM; you target NVIDIA + Intel + AMD; you want one launch incantation everywhere |

If your training stays on a single-cluster, single-GPU-vendor setup,
`torchrun` directly is fine — `ezpz launch` is a thin wrapper around it
in that case anyway. The value of `ezpz` shows up when any of the
following apply:

- You move workloads between Aurora (Intel XPU), Polaris (NVIDIA),
  Frontier (AMD), Perlmutter, or your laptop without rewriting launch
  scripts.
- You submit to PBS or SLURM and want the same script to work in both.
- You want one `ezpz launch` command that auto-detects the scheduler,
  hostfile, GPU count, NIC, and CPU bindings — no per-cluster
  bookkeeping.
- You want node-local environment broadcast ([`ezpz yeet`](./cli/yeet.md))
  for fast startup at scale.

## vs. raw `torchrun` / `mpirun`

What `torchrun` gives you: a battle-tested launcher, clean integration
with `torch.distributed`, no extra dependencies.

What you have to do yourself:

```bash
# Polaris (PBS, NVIDIA, MPI)
mpiexec -n $((NNODES * 4)) --hostfile $PBS_NODEFILE \
    --cpu-bind depth -d 16 \
    python -m torch.distributed.run --nnodes=$NNODES \
        --nproc-per-node=4 --rdzv-backend=c10d \
        --rdzv-endpoint=$HEAD_NODE:29500 train.py

# Aurora (PBS, Intel XPU, also MPI but different bindings + ranks-per-node)
mpiexec -n $((NNODES * 12)) --ppn 12 --hostfile $PBS_NODEFILE \
    --cpu-bind verbose,list:1-7:8-15:16-23:24-31:32-39:40-47:53-59:60-67:68-75:76-83:84-91:92-99 \
    gpu_tile_compact.sh \
    python train.py

# Perlmutter (SLURM, NVIDIA)
srun -N $NNODES -n $((NNODES * 4)) --gpus-per-node=4 --cpu-bind=cores \
    python train.py
```

vs.

```bash
ezpz launch python train.py    # any of the above
```

`ezpz launch` reads `PBS_NODEFILE` / `SLURM_NODELIST`, detects the GPU
type (`nvidia-smi` / `xpu-smi` / `rocm-smi`), picks the right launcher
(`mpiexec` / `srun`), and applies known-good CPU bindings per cluster.
The same script runs on your laptop with `world_size=1`. See [Why
ezpz?](./index.md) for the diff version.

What you give up: complete control over launcher flags. You can pass
launcher-specific args through with `ezpz launch -- <launcher flags> --
python train.py`, but if you need exotic options not covered by the
common path, raw `torchrun` is more direct.

## vs. HuggingFace `accelerate`

What `accelerate` gives you: rich integration with HF Trainer / PEFT /
TRL, FSDP/DeepSpeed wrappers via config files, automatic mixed
precision, the broader HF ecosystem.

Where they differ:

- **Scope.** `accelerate` is HF's distributed-training abstraction —
  it owns model + optimizer wrapping. `ezpz` doesn't try to own any
  of that; it stays focused on launch + scheduler discovery + a thin
  `wrap_model()` helper. You can use both together (most of `ezpz`'s
  HF examples do).
- **HPC scheduler integration.** `accelerate launch` doesn't know
  about PBS or SLURM hostfiles directly — you write a YAML config per
  cluster. `ezpz launch` auto-detects.
- **Cross-vendor.** `accelerate` works on NVIDIA and (via PRs) Intel
  XPU, but the cluster-level "where are my nodes, which CPU bindings
  do I want" piece is up to you. `ezpz` ships those defaults.

When to pick `accelerate`: your project lives in the HF ecosystem
already. Use `ezpz launch` to wrap the `accelerate launch` invocation
if you also want HPC-scheduler integration.

When to pick `ezpz`: you want minimal abstraction, run outside HF,
or move between clusters often.

## vs. DeepSpeed

DeepSpeed is the answer for **ZeRO-Infinity, NVMe parameter offload,
MoE expert parallelism, and very-large-model training** (10B+ where
optimizer state alone is too big for HBM). PyTorch FSDP2 — what
`ezpz.wrap_model` uses by default — is now competitive with DeepSpeed
ZeRO-3 for the more common case (1B-70B models, no offload, no
MoE-specific routing).

Where they overlap: ZeRO-2 / ZeRO-3 sharding strategies. `ezpz` exposes
both via `reshard_after_forward={False,True}` (see
[Distributed Training Guide](./guides/distributed-training.md#fsdp-sharding-strategies)).

Where DeepSpeed wins:

- **Offload.** CPU offload (ZeRO-Offload) and NVMe offload
  (ZeRO-Infinity) for fitting models that exceed your aggregate HBM.
- **MoE.** First-class expert parallelism, expert-data hybrid
  parallelism, gating-loss reductions.
- **Pipeline parallelism.** DeepSpeed's pipeline engine has been
  battle-tested at trillion-parameter scale.
- **Curriculum learning** and other training-loop helpers.

Where `ezpz` wins:

- **Simpler API.** `wrap_model(model)` returns a wrapped model. No
  config JSON, no engine tuple to unpack.
- **Cross-vendor.** DeepSpeed officially supports NVIDIA + (recently)
  Intel XPU; `ezpz` adds AMD ROCm and Apple MPS as first-class
  targets for the launch / discovery layer.

`ezpz` already has DeepSpeed plumbing for users who need it
(`get_deepspeed_config_json`, `_init_deepspeed`, example scripts under
`src/ezpz/examples/deepspeed/`). A first-class `wrap_model_for_deepspeed`
helper is on the roadmap (see [TODO §16](https://github.com/saforem2/ezpz/blob/main/TODO.md#16-explicit-deepspeed-wrapper-low)) —
for now, build a config dict with `get_deepspeed_config_json(stage=...)`
and call `deepspeed.initialize()` directly.

## See also

- [Why ezpz? (landing page diff)](./index.md) — side-by-side bash
  comparison of with vs. without
- [Distributed Training Guide](./guides/distributed-training.md) —
  including ZeRO/FSDP sharding strategy mapping
- [`ezpz launch`](./cli/launch/index.md) — the launcher this page
  compares against
