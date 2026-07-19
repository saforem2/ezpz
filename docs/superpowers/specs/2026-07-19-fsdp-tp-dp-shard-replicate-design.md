# `--dp-shard` / `--dp-replicate` (HSDP) for `fsdp_tp.py`

**Date:** 2026-07-19
**Status:** Design — approved (pending spec review)
**Scope:** Single-file feature (`src/ezpz/examples/fsdp_tp.py`) + tests + a docs note.

## Goal

Add explicit control over the data-parallel topology in
`ezpz.examples.fsdp_tp` via two CLI flags — `--dp-replicate` and
`--dp-shard` — enabling **HSDP** (Hybrid Sharded Data Parallel): replicate
model weights across `dp_replicate` groups, shard within each `dp_shard`
group. Mirrors torchtitan's `data_parallel_replicate_degree` /
`data_parallel_shard_degree` so configs port 1:1.

## Background / current state

- `fsdp_tp` builds a **2D device mesh** `("dp", "tp")` at
  `fsdp_tp.py:1934`, where `dp = world_size // tp` — a single flat
  data-parallel dim.
- FSDP2 (`fully_shard`) is applied over the full `dp` mesh → pure ZeRO-style
  sharding. There is no way to request replication (HSDP) today.
- The legacy `--sharding-strategy hybrid_shard` / `hybrid_shard_zero2`
  names exist but a code comment (`fsdp_tp.py:1243-1246`) states FSDP1
  hybrid has no one-flag FSDP2 equivalent — they only map to
  `reshard_after_forward`, i.e. they do **not** actually do HSDP.
- torchtitan precedent (`torchtitan/distributed/parallel_dims.py`):
  `dp_replicate` (default 1), `dp_shard` (default -1 = use all remaining),
  constraint `dp_replicate * dp_shard * cp * tp * pp == world_size`, mesh
  dims split into `("dp_replicate", "dp_shard")`. FSDP2 reads a 2D DP
  submesh as HSDP automatically.
- Verified in the pinned torch (2.12.1): `DeviceMesh._flatten` exists and
  `init_device_mesh_safe` forwards N-D `mesh_shape` generically.

## Decisions (locked)

1. **Full HSDP** — split the flat `dp` dim into a 2D
   `(dp_replicate, dp_shard)` submesh; use FSDP2's native HSDP.
2. **Defaults:** `--dp-replicate 1`, `--dp-shard -1` (auto = all
   remaining). This reproduces today's behavior *exactly* when neither is
   passed (replicate=1 → pure FSDP shard over the whole DP dim). Matches
   torchtitan.
3. **Mesh shape:** always build a 3D mesh
   `("dp_replicate", "dp_shard", "tp")` and **flatten** the two DP dims
   into a single named `"dp"` dim, so every existing `dp` consumer
   (DistributedSampler, loss group) keeps working unchanged with correct
   rank math from torch's tested flattening. Only the `fully_shard` mesh
   arg changes to use the DP submesh.

## Architecture

### CLI surface (new args, near `--tp` at `fsdp_tp.py:1217`)

```
--dp-replicate  int  default 1   # replicate weights across N groups (HSDP outer dim)
--dp-shard      int  default -1  # shard within groups; -1 = world_size // (dp_replicate * tp)
```

Help text names them as the real HSDP knobs; add a one-line pointer from
`--sharding-strategy`'s help to `--dp-replicate`/`--dp-shard` for HSDP.

### Degree resolution + validation (replaces `dpsize = world_size // args.tp` at 1923)

```python
dp_replicate = args.dp_replicate
dp_shard = args.dp_shard
if dp_shard < 0:  # -1 → use all remaining ranks
    dp_shard = world_size // (dp_replicate * args.tp)
assert dp_replicate >= 1, "--dp-replicate must be >= 1"
assert dp_shard >= 1, "--dp-shard must be >= 1 (or -1 for auto)"
assert dp_replicate * dp_shard * args.tp == world_size, (
    f"dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * tp({args.tp}) "
    f"!= WORLD_SIZE({world_size})"
)
dpsize = dp_replicate * dp_shard  # == flattened "dp" size; keeps all downstream math correct
```

### Mesh construction (replaces `fsdp_tp.py:1934`)

```python
device_mesh = ezpz.init_device_mesh_safe(
    str(ezpz.get_torch_device()),
    (dp_replicate, dp_shard, args.tp),
    mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
)
# Flatten the two DP dims into one named "dp" dim so existing consumers
# (device_mesh["dp"], get_group("dp"), get_local_rank("dp"),
# num_replicas=dpsize) work unchanged with correct rank arithmetic.
device_mesh[("dp_replicate", "dp_shard")]._flatten("dp")
```

### `fully_shard` mesh selection (in `parallelize`, around `fsdp_tp.py:1638`)

The `fsdp_kwargs["mesh"]` currently uses `dp_mesh = device_mesh["dp"]`
(1543). Change to select based on replication:

```python
if dp_replicate > 1:
    fsdp_dp_mesh = device_mesh[("dp_replicate", "dp_shard")]  # 2D → HSDP
else:
    fsdp_dp_mesh = device_mesh["dp_shard"]                    # 1D → plain FSDP
```

- `dp_replicate == 1` → 1D `dp_shard` mesh → **byte-identical to today**
  (a 1-sized replicate dim would otherwise be a needless HSDP wrap).
- `dp_replicate > 1` → 2D submesh → FSDP2 HSDP.

`parallelize()` (signature at 1525) already receives `device_mesh`, so it
reads `device_mesh["dp_replicate"].size()` to branch — **no new
parameter**. The `dp_mesh = device_mesh["dp"]` line (1543) stays for the
non-FSDP dp uses; only the `fsdp_kwargs["mesh"]` selection branches on the
replicate size.

### The single flattened-`dp` invariant

Every other DP consumer stays as-is because they read the flattened dim:
- `device_mesh["dp"]` (the FSDP2 optimizer/loss-dp uses at 2175)
- `num_replicas=dpsize` + `rank=device_mesh.get_local_rank("dp")`
  (DistributedSampler, 2372-2373 and 2406-2407)
- `device_mesh.get_group("dp")` (loss dp_group, 2389)

These are semantically correct only against the *flattened* dp (a data
sample shards/replicates across the entire dp product, not one sub-dim),
which is exactly what `_flatten("dp")` provides.

## Data flow

```
args.dp_replicate, args.dp_shard, args.tp
   └─ resolve dp_shard (-1 → auto) + assert product == world_size
   └─ init_device_mesh_safe((dp_replicate, dp_shard, tp), names=(dp_replicate, dp_shard, tp))
   └─ mesh[(dp_replicate, dp_shard)]._flatten("dp")
        ├─ fully_shard: mesh = dp_shard (1D) if dp_replicate==1 else (dp_replicate,dp_shard) (2D→HSDP)
        └─ everything else: device_mesh["dp"] / get_group("dp") / get_local_rank("dp") / dpsize (unchanged)
```

## Error handling

- Non-divisible config → `AssertionError` with the exact
  `dp_replicate * dp_shard * tp != WORLD_SIZE` message (fail fast at setup).
- `dp_replicate < 1` or `dp_shard < 1` (after -1 resolution) → assert with
  a clear message.
- HF-model branch already forces `args.tp = 1` (2015-2024); dp flags are
  independent of that and need no special-casing (the product check just
  uses the final `args.tp`).

## Testing

CPU-only, no accelerator (match existing `fsdp_tp` test style —
`tests/test_fsdp_tp_*.py` parse args / assert topology without launching):

1. **Defaults reproduce today:** `--dp-replicate` default 1, `--dp-shard`
   default -1; resolved `dp_shard == world_size // tp`, `dpsize` unchanged.
2. **-1 auto-fill math:** for a mocked `world_size` and `tp`,
   `dp_shard = world_size // (dp_replicate * tp)`.
3. **Product validation:** a config where
   `dp_replicate * dp_shard * tp != world_size` raises AssertionError with
   the documented message.
4. **fully_shard mesh selection:** `dp_replicate == 1` → 1D `dp_shard`
   mesh path; `dp_replicate > 1` → 2D submesh path. (Assert on the mesh
   dim(s) passed, mocking `init_device_mesh_safe` / `fully_shard` so no
   real process group is needed.)
5. **Arg parsing:** `--dp-replicate`/`--dp-shard` parse as ints; `--help`
   lists them.

Where a real mesh is impractical CPU-side, mock `init_device_mesh_safe`
to return a stub whose `[...]`/`_flatten`/`size()` are inspectable — the
same mocking approach the existing fsdp_tp tests use for topology.

## Documentation

- Update `docs/examples/fsdp-tp.md` with an HSDP example
  (`--dp-replicate 2 --dp-shard 4 --tp ...`) and the product constraint.
- Update the module docstring's mesh description (`fsdp_tp.py:20-21`,
  which says "Data Parallel (dp) across hosts / Tensor Parallel (tp)")
  to mention the replicate/shard split.
- Docs updated in the same pass as code (project convention).

## Out of scope (YAGNI)

- No changes to `--pp` / `--cp` (fsdp_tp doesn't expose them as active
  dims today; dp flags compose with `tp` only, matching current mesh).
- No `ezpz.tp` / `distributed.py` HSDP plumbing — this is confined to the
  `fsdp_tp` example's own mesh build.
- No removal/renaming of the legacy `--sharding-strategy hybrid_shard*`
  values (kept for back-compat; just cross-referenced).

## Release

Single `release:patch` PR (additive; default behavior unchanged when the
flags aren't set). Merge as a merge commit, not squash (project convention).
On-node HSDP validation (dp_replicate>1 across real nodes) is a follow-up;
unit tests cover the topology math + mesh selection.
