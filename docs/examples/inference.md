# 🔮 `ezpz.examples.inference`

Distributed inference over a HuggingFace model + dataset.

Each rank loads the model, processes a disjoint shard of prompts via
data-parallel sharding, and writes its predictions to a per-rank
JSONL file. Per-batch latency, throughput (tokens/sec), TFLOPS, and
MFU are tracked through `ezpz.History` and aggregated in the
finalized run.

## Quick Start

```bash
# Single node, default model + dataset
ezpz launch python3 -m ezpz.examples.inference

# Custom model + dataset
ezpz launch python3 -m ezpz.examples.inference \
    --model meta-llama/Llama-3.2-1B \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --max-samples 256 --max-new-tokens 32
```

## How sharding works

Prompts are loaded once on every rank, then each rank processes a
contiguous block via [`shard_indices`](#sharding-helper):

```
total = 10 samples, world_size = 4

rank 0: [0, 1, 2]
rank 1: [3, 4, 5]
rank 2: [6, 7, 8]
rank 3: [9]
```

This is **data parallelism** — each rank holds the full model.
There is no model-parallel sharding, so the model must fit in a
single device's memory. For larger models, use a model-parallel
inference framework (vLLM, DeepSpeed-Inference) instead.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `meta-llama/Llama-3.2-1B` | HF model name or local path |
| `--dataset` | `wikitext` | HF dataset name or local path |
| `--dataset-config` | `wikitext-2-raw-v1` | Dataset configuration / subset |
| `--dataset-split` | `test` | Split (`train`/`validation`/`test`) |
| `--text-column` | `text` | Column containing the prompt |
| `--max-samples` | `128` | Total samples across all ranks |
| `--batch-size` | `4` | Per-rank batch size |
| `--max-input-tokens` | `512` | Truncate prompts to this many tokens |
| `--max-new-tokens` | `64` | Tokens to generate per sample |
| `--dtype` | `bfloat16` | Model dtype (`float32`/`float16`/`bfloat16`) |
| `--do-sample` | off | Sample instead of greedy decoding |
| `--temperature` | `1.0` | Sampling temperature (with `--do-sample`) |
| `--top-p` | `1.0` | Nucleus sampling cutoff (with `--do-sample`) |
| `--no-save-predictions` | save on | Skip writing per-sample JSONL |

## Outputs

```
outputs/ezpz.examples.inference/<timestamp>/
├── predictions-rank0.jsonl       # one row per generated sample
├── predictions-rank1.jsonl
├── ...
├── inference.h5                  # finalized History dataset
├── report-inference.md           # markdown summary
└── plots/                        # auto-generated metric plots
```

Each `predictions-rank<N>.jsonl` row:

```json
{"rank": 0, "prompt": "...", "completion": "..."}
```

## MFU Tracking

Per-batch MFU is approximated as `n_new_tokens × forward_flops / dt`,
where `forward_flops` is `1/3` of the `try_estimate` fwd+bwd estimate
(rule-of-thumb: backward is ~2× forward). See
[`ezpz.flops`](../python/Code-Reference/flops.md) for the per-device
MFU formula.

## See Also

- [`ezpz.examples.hf`](hf.md) — fine-tuning loop (the training
  counterpart to this inference example)
- [Recipes › MFU Tracking](../recipes.md#mfu-tracking)
