# FSDP + LoRA + **TP** validation matrix

## What has NOT been validated

Everything run for #239 used **`--tp 1`** — that is FSDP + LoRA with
tensor parallelism *disabled*. The XPU runs were also single-node.

| | nodes | `tp` | FSDP+LoRA | FSDP+LoRA+**TP** |
|---|---|---|---|---|
| Perlmutter | 2 | 1 | hangs (#239) | **not tested** |
| Polaris | 2 | 1 | ✅ | **not tested** |
| Sunspot | 1 | 1 | ✅ single-node | **not tested** |
| Aurora | 1 | 1 | ✅ single-node | **not tested** |

So "LoRA works on Polaris/Sunspot/Aurora" is true only for the
`tp=1`, and only multi-node on Polaris.

## Why `tp>1` is a genuinely different path, not a bigger version

1. **A 2D mesh.** `tp=1` gives one flat sharded dp dim; `tp>1` builds
   `(dp_shard, tp)` with `dp_replicate * dp_shard * tp == world_size`.
   Different collective groups, different reduce-scatter topology.
2. **The TP plan targets modules by NAME, and LoRA breaks it.**
   `fsdp_tp.parallelize` maps `"attention.wq" -> ColwiseParallel()`.
   Swapping `wq` for a LoRA wrapper raises
   `NotImplementedError: ColwiseParallel currently only support
   nn.Linear and nn.Embedding!`. `lora_tp_plan` rewrites those keys to
   point at the inner `nn.Linear`. **Without it, `tp>1` dies** — and
   that rewrite has only ever been checked at `world_size=1`, where a
   mesh satisfies every placement trivially.

That second point is the reason this matters: the one piece of code
specifically written to make LoRA and TP coexist has never run on a
real multi-rank mesh.

## RESULTS

**Sunspot (PVC/xccl, torch 2.13, 2 nodes, `world_size=24`)** — job
`12473890`, 3/3 pass:

| `tp` | mesh | result |
|---|---|---|
| 1 | `dp_shard=24, tp=1` | trains, 106 s |
| 2 | `dp_shard=12, tp=2` | trains, 86 s |
| 4 | `dp_shard=6,  tp=4` | trains, 82 s |

**Polaris (A100/NCCL, torch 2.13, 2 nodes, `world_size=8`)** — job
`7563719`:

| `tp` | mesh | result |
|---|---|---|
| 1 | `dp_shard=8, tp=1` | trains, 156 s |
| 2 | `dp_shard=4, tp=2` | trains, 90 s |

Each cell verified from the log, not from the exit code: the
`DeviceMesh(...)` line confirms TP was actually applied (a silent
downgrade to `tp=1` would otherwise read as a pass), plus plots written
and zero watchdog lines.

**No `NotImplementedError: ColwiseParallel currently only support
nn.Linear and nn.Embedding!`** — the specific failure expected if
`lora_tp_plan`'s key rewrite were wrong. So that rewrite, which had
only ever run at `world_size=1`, works on real multi-rank 2D meshes on
**both** collectives backends.

Two gaps closed at once: multi-node XPU (Sunspot had only ever run
single-node) and TP>1 with LoRA anywhere.

## The matrix to run

Per machine, multi-node, at the world size each provides:

| cell | `tp` | what it proves |
|---|---|---|
| `tp1-baseline` | 1 | control — the known-good path, same allocation |
| `tp2` | 2 | `lora_tp_plan` rewrite works on a real 2D mesh |
| `tp4` | 4 | TP degree beyond the minimum |

`--lora-rank 18` throughout: on Perlmutter that avoids #239, and
everywhere else it is simply a working rank, so a hang would be new
information rather than a rediscovery.

Run `tp1-baseline` **first** in every job. If the control fails, the
`tp>1` cells say nothing about TP.

## Expected world sizes

| | nodes | per node | `world_size` | valid `tp` |
|---|---|---|---|---|
| Polaris | 2 | 4 | 8 | 1, 2, 4 |
| Sunspot | 2 | 12 | 24 | 1, 2, 4 (dp must divide) |
| Aurora | 2 | 12 | 24 | 1, 2, 4 |

`dp_shard * tp == world_size` must hold, so on 24 ranks `tp=4` gives
`dp_shard=6`.
