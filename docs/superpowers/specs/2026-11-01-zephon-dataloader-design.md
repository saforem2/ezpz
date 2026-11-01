# ezpz + Zephon — elastic dataloader integration (draft spec)

## Why

Current `fsdp_tp` dataloader is `HF datasets` → `DistributedSampler` → `DataLoader`. It is non-elastic: change DP degree on resume and shard assignments drift (packing depends on prior samples). Aurora/Polaris jobs die mid-run and swap spares — the dataloader should survive that deterministically.

Zephon (DatologyAI, Apache 2.0, `uv pip install zephon`) is elastically deterministic: packing + mixtures + ordering survive DP-size changes across checkpoint/resume, because the pipeline is checkpointed as a whole, not per-rank.

Upstream proof: `datologyai/torchtitan-zephon` wires Zephon into torchtitan via a leaf adapter `torchtitan_recipes/zephon/dataloader.py` (TOML recipe → ZephonDataLoader) without touching the trainer/model stack. We copy that pattern.

## What we ship

- **Thin adapter** `src/ezpz/data/zephon.py` → `ZephonDataLoader(Module)` that conforms to the same surface our HF loader does (`__iter__` yields `TrainingMicrobatch`, `save_state`/`load_state` for checkpoints).
- **TOML recipe** `examples/zephon/local_jsonl.toml` analog + one per-mix config for our ALCF corpora.
- **Smoke** `examples/zephon/elastic_resume_demo.py` analog (CPU-only, no model) + `run_training_smoke.sh` (GPU, via `ezpz launch`).
- **Flag** `--dataloader {hf,zephon,auto}` on the 3 from-scratch examples (`fsdp_tp`, `fsdp`, `vit`); `hf` default so nothing breaks.

## Integration shape

```
configs:
  dataloader = ZephonDataLoader.Config.from_toml("path/to/recipe.toml",
    aggregate_dir=shared_fs, aggregate_store="torch://", run_id=...)

training loop:
  for microbatch in dataloader:
      train_step(microbatch)
      if step % ckpt_interval == 0:
          ckpt.save(dataloader_state=dataloader.save_state())
  # resume:
  dataloader.load_state(ckpt["dataloader"])
```

Ranks aggregate Zephon checkpoint state through `torchrun`'s TCPStore (already present in ezpz's `launch`) or a shared directory on Aurora/Polaris `/eagle`.

## Non-goals

- No auto-migration of existing HF checkpoints.
- No grain wrapper beyond the already-documented grain row-group pattern — we keep `hf` default.
- No changes to `ezpz.submit` / `ezpz launch` beyond passing through `ZEPHON_*` env vars.

## Risks & mitigations

- Zephon is beta (0.2.1, 85 stars) — pin via `pyproject.toml` `zephon==0.2.*`, behind an optional extra `ezpz[zephon]` so main deps don't carry it.
- Requires Python ≥3.10 (we already are).
