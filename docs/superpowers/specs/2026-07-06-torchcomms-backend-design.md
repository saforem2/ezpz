# Wire in torchcomms backend via environment variable

**Date:** 2026-07-06
**Status:** Design — approved (pending spec review)
**Scope:** Small, single-file mechanism + docs + tests

## Goal

Let ezpz route `torch.distributed` through
[torchcomms](https://github.com/meta-pytorch/torchcomms) — Meta's experimental
PyTorch communications API — via a single environment variable, without
disturbing the existing backend-selection path or any downstream
DeviceMesh / FSDP2 / DTensor code.

## Background / key facts

- torchcomms is **not** a c10d backend string. It is a *mode over* a transport:
  you keep selecting `nccl` / `xccl` / `gloo` as the wire backend, and a global
  switch makes `init_process_group` build its process groups on torchcomms
  rails. The standard c10d PG API is preserved, so DeviceMesh, FSDP2, and
  DTensor continue to work unchanged.
- The activation mechanism (confirmed against torchtitan's
  `torchtitan/distributed/utils.py` `init_distributed`, and the torchcomms
  README) is exactly three things:
  1. `import torchcomms`
  2. set `torch.distributed.config.use_torchcomms = True`
  3. pass `device_id=torch.device(device_type, local_rank)` to the normal
     `torch.distributed.init_process_group(...)`.
- ezpz is already positioned for this:
  - `get_torch_backend()` (`src/ezpz/distributed.py:411`) already reads
    `TORCH_BACKEND` and otherwise auto-probes nccl/xccl/gloo. This stays the
    transport selector.
  - `_setup_ddp()` (`src/ezpz/distributed.py:~1766`) already resolves and passes
    `device_id=` to `init_process_group` (lines ~1800-1811). No change needed
    there to satisfy torchcomms' device requirement.
  - `torchcomms` is **already** declared in the `torch` optional-dependencies
    group in `pyproject.toml` (`torch = [..., "torchcomms"]`).
  - `TORCH_BACKEND` is already documented in `docs/configuration.md`
    (env-var table).

## Decisions (locked)

1. **Activation:** dedicated boolean env var `EZPZ_USE_TORCHCOMMS`. torchcomms is
   a mode over a transport; `TORCH_BACKEND` still names the wire backend
   (nccl/xccl). Selected over overloading `TORCH_BACKEND=torchcomms` because
   that would conflate "which transport" with "use the torchcomms path".
2. **Fallback:** warn + fall back to the standard PG. If `EZPZ_USE_TORCHCOMMS`
   is set but torchcomms cannot be used (package not importable, or the
   installed torch lacks `torch.distributed.config.use_torchcomms`), log a
   single rank-0 warning naming the reason and continue on the standard
   backend. The flag is best-effort; it never breaks a run.
3. **Reporting:** one rank-0 setup log line **plus** a `TORCHCOMMS` field in the
   existing `get_dist_info()` diagnostics dict (so it appears in the startup
   banner and any run metadata / W&B config). No `ezpz doctor` change (YAGNI).

## Architecture

### New helper: `use_torchcomms()` (in `src/ezpz/distributed.py`)

```python
def use_torchcomms() -> bool:
    """Whether to route torch.distributed through torchcomms.

    Gated on EZPZ_USE_TORCHCOMMS (truthy: 1/true/yes/on, case-insensitive).
    Best-effort: returns False if the torchcomms package or the torch
    switch (torch.distributed.config.use_torchcomms) is unavailable. The
    unavailability reason is cached so the caller can emit exactly one
    rank-0 warning.
    """
```

- Truthy parse reuses the existing ezpz idiom
  (`value.lower() in ("1", "true", "yes", "on")`, cf. `distributed.py:456`).
- Availability probe: `import torchcomms` **and**
  `hasattr(torch.distributed.config, "use_torchcomms")`.
- **Caching:** the (requested, available, reason) result is computed once and
  stored in a module-level variable, so the import probe does not re-run on
  repeated calls and the warning path is idempotent. A private
  `_torchcomms_unavailable_reason()` (or the cached tuple) exposes the reason
  string for the warning.
- Semantics: `use_torchcomms()` returns `True` only when the env is truthy AND
  torchcomms is usable. "Requested but unavailable" is distinguishable (env
  truthy + returns False + non-empty reason) so the caller can warn.

### Touch point A — `_setup_ddp()`, immediately before `init_process_group`

```python
if use_torchcomms():
    import torch.distributed.config as dist_config
    dist_config.use_torchcomms = True
    if rank == 0:
        logger.info("Using torchcomms over backend=%s", backend)
elif os.environ.get("EZPZ_USE_TORCHCOMMS") and rank == 0:
    logger.warning(
        "EZPZ_USE_TORCHCOMMS set but torchcomms unavailable (%s); "
        "using standard %s backend.",
        _torchcomms_unavailable_reason(),
        backend,
    )
```

- Inserted after `backend` and `rank` are resolved (both already computed at the
  top of `_setup_ddp`, ~lines 1706-1709) and before the
  `if not torch.distributed.is_initialized():` block that calls
  `init_process_group` (~line 1766).
- The existing `device_id=` resolution (lines ~1800-1811) is unchanged and
  already satisfies torchcomms' device-bound-PG requirement.
- The single-device fast path in `setup_torch` (WORLD_SIZE==1, returns before
  `_setup_ddp`) is unaffected — torchcomms only matters for real multi-rank
  init.

### Touch point B — `get_dist_info()` diagnostics dict (~line 1032)

Add one entry next to `DISTRIBUTED_BACKEND`:

```python
"TORCHCOMMS": use_torchcomms(),
```

Flows into the rank-0 startup banner and run metadata / W&B config.

## Data flow

```
setup_torch()
  └─ get_torch_backend()            # TORCH_BACKEND → nccl/xccl/gloo (transport)
  └─ _setup_ddp(backend=...)
       ├─ use_torchcomms()          # EZPZ_USE_TORCHCOMMS + availability probe
       │    ├─ True  → dist_config.use_torchcomms = True; log "over backend=X"
       │    └─ set-but-unusable → rank-0 warn(reason); proceed standard
       └─ init_process_group(backend=..., device_id=..., ...)  # unchanged call
```

## Error handling

- **Package missing / torch too old:** `use_torchcomms()` returns False; rank-0
  warning naming the reason; run proceeds on the standard backend. No raise.
- **Env unset:** `use_torchcomms()` returns False silently; zero behavior change
  from today.
- **Idempotency:** availability + reason cached at module scope; warning logic
  keyed on rank 0 so it prints once per job.

## Testing

Unit tests (CPU-only, no accelerator, matching existing `tests/` conventions):

1. `EZPZ_USE_TORCHCOMMS` unset → `use_torchcomms()` is False; no
   `dist_config.use_torchcomms` mutation.
2. Truthy variants (`1`, `true`, `TRUE`, `yes`, `on`) all parse True at the env
   layer; falsy / absent parse False.
3. `EZPZ_USE_TORCHCOMMS=1` with torchcomms **not importable** (simulate via
   `monkeypatch` on the import / `sys.modules`) → `use_torchcomms()` False and a
   non-empty unavailability reason is recorded (the warning path is reachable).
4. `EZPZ_USE_TORCHCOMMS=1` with torch lacking
   `torch.distributed.config.use_torchcomms` (monkeypatch `delattr`/absent) →
   False + reason.
5. `get_dist_info()` includes a `TORCHCOMMS` key (bool).
6. Availability probe result is cached (probe invoked once across repeated
   `use_torchcomms()` calls) — assert via a counter/spy on the probe.

Use `monkeypatch.setenv` / `monkeypatch.setattr` so environment and module state
are restored automatically (cf. the `NO_COLOR` test-hygiene fix in PR #177).

## Documentation

- Add `EZPZ_USE_TORCHCOMMS` to the env-var table in `docs/configuration.md`
  (next to `TORCH_BACKEND`), describing it as "route torch.distributed through
  torchcomms over the selected `TORCH_BACKEND` transport; best-effort, warns and
  falls back if unavailable."
- One example line: `EZPZ_USE_TORCHCOMMS=1 TORCH_BACKEND=nccl ezpz launch ...`.
- Update docs in the same pass as the code (per project convention — do not
  defer docs).

## Out of scope (YAGNI)

- No `ezpz doctor` torchcomms probe.
- No per-parallel-group (TP/PP/CP/DP) torchcomms toggles — the global switch
  covers all PGs built by the standard path.
- No overloading of `TORCH_BACKEND` with a `torchcomms` value / prefix.
- No torchcomms direct-API (`new_comm()`) usage — we use the c10d-integrated
  path only, so DeviceMesh/FSDP2/DTensor are untouched.

## Validation (must happen on-node; NOT satisfied by design review)

The mechanism is validated against torchtitan's source and the torchcomms docs,
but has **not** been executed — no torch on the dev host; real runs go through
qsub on ALCF. Required real-cluster confirmation before claiming it works:

- On Aurora (XPU): `EZPZ_USE_TORCHCOMMS=1 TORCH_BACKEND=xccl` launch of
  `python3 -m ezpz.examples.test` (and ideally `fsdp_tp` tp=2) comes up, trains
  a few steps rc=0, and the startup banner shows `TORCHCOMMS: True`.
- On a CUDA host if available: same with `TORCH_BACKEND=nccl`.
- Confirm the warn-and-fallback path on a node where torchcomms is not
  installed: banner shows `TORCHCOMMS: False` + the rank-0 warning, run still
  succeeds.

## Release

Single `release:patch` PR (additive, backward-compatible; default behavior
unchanged when the env var is unset). Merge as a merge commit, not squash (per
project convention).
