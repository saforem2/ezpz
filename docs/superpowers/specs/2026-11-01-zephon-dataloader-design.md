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

## Tradeoffs vs HF-AOT

Three ways to feed tokens into fsdp_tp, each with a different cost curve.
The right choice depends on dataset size, job churn, and whether you need
resume determinism.

| Dimension | HF datasets + DistributedSampler (current default) | AOT tokenize → Arrow/Parquet | Zephon (this spec) |
|---|---|---|---|
| Setup | `pip install ezpz` (no extra) | Same + one-off offline tokenize+save | `pip install ezpz[zephon]` + TOML recipe |
| First-batch latency (smoke, 16 docs, seq 32) | download+load_dataset+map+cache ≈ 5–30 s (cold), ~1 s (warm, cached) | same cold cost, then zero on reload (Arrow mmap) | import ~0.9 s + first batch ~3 s (tokenize on the fly), subsequent ~0.1 ms/batch |
| Throughput (steady state) | memcpy from Arrow shards; ~CPU bound by sampler shuffle | fastest (direct Arrow mmap, no tokenize) | tokenize + pack per step; hidden by prefetch but still CPU-bound on tiny fixtures |
| Checkpoint resume | re-reads dataset, re-shards by new DP degree — packing that depended on previous sharding silently changes (wrong tokens in packed sequences) | same as HF (sampler owns sharding) | elastic: checkpoint is pipeline-wide, survives DP resize (spare-swap on Aurora) |
| Disk | HF cache (fingerprint-named Arrow) + tokenizer cache; unbounded per job | Arrow/Parquet materialization (one copy per tokenizer) | shard cache (shared per node, bounded at 500 GB in integration) |
| Mixtures/mixtures | not natively; hack via ConcatDataset + custom sampler | same | first-class: [[sources]] with weight, token-aware mixing |
| When to use | smoke tests, quick finetunes on gated HF repos (<10 K rows after --hf-limit) | stable corpora you re-read unchanged (pretrained vocab fixed) | Aurora/Polaris pretraining where spare-swap resume determinism matters; mixture ablations |

Rule of thumb: keep HF default for smoke/eval; use AOT for locked-vocab throughput records; use Zephon when resume-determinism or mixture experimentation is a requirement.

## Risks & mitigations

- Zephon is beta (0.2.1, 85 stars) — pin via `pyproject.toml` `zephon==0.2.*`, behind an optional extra `ezpz[zephon]` so main deps don't carry it.
- Requires Python ≥3.10 (we already are).
