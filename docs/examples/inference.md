# 🔮 `ezpz.examples.inference`

Distributed inference over a HuggingFace model + dataset, with three
distinct modes: **benchmark**, **generate**, and **eval**.

Each rank loads the model and processes a disjoint shard of inputs
(data parallelism). Per-batch latency, throughput (tokens/sec),
TFLOPS, and MFU are tracked through `ezpz.History`. With `--mode eval`
an accuracy metric is added.

## Source

<details closed><summary><code>src/ezpz/examples/inference.py</code></summary>

```python title="src/ezpz/examples/inference.py"
--8<-- "src/ezpz/examples/inference.py"
```

</details>

## Modes

```bash
# Throughput benchmark — synthetic random tokens, no dataset
ezpz launch python3 -m ezpz.examples.inference --mode benchmark

# Synthetic data generation — dataset prompts → completions JSONL
ezpz launch python3 -m ezpz.examples.inference --mode generate

# Evaluation — generate, then score against gold labels
ezpz launch python3 -m ezpz.examples.inference --mode eval \
    --dataset gsm8k --dataset-config main --dataset-split test \
    --text-column question --label-column answer
```

| Mode | Source | Saves predictions? | Reports accuracy? |
|------|--------|--------------------|-------------------|
| `benchmark` | random tokens | no | no |
| `generate` *(default)* | dataset prompts | yes | no |
| `eval` | dataset prompts + labels | yes (with `label`) | yes |

### `--mode benchmark`

Pure throughput measurement. Skips the tokenizer entirely and feeds
random `input_ids` of shape `(batch_size, max_input_tokens)` into
the model. Configurable via:

- `--benchmark-iters` (default `20`) — number of forward passes
- `--benchmark-warmup` (default `3`) — warmup iters excluded from totals

Use this for hardware/configuration comparisons (different
`--batch-size`, `--dtype`, `--world-size`, `--max-input-tokens`).
The reported tokens/sec and MFU are uncontaminated by tokenizer
overhead or dataset variance.

### `--mode generate`

Reads prompts from a HuggingFace dataset, generates completions, and
writes them to `predictions-rank<N>.jsonl`. Useful for:

- **Synthetic data generation** for downstream training
- **Distillation** — generate teacher completions to train a student
- **Spot-checking** model behavior on a known corpus

### `--mode eval` (with `--label-column`)

Same as `generate`, but also extracts a gold label from
`--label-column` and compares the completion. The match rule is
"normalized exact-match OR substring": both completion and label
are lowercased, whitespace-collapsed, then checked. The label
counts as correct if it appears as a substring of the normalized
completion (handles "the answer is 42" vs gold "42").

Reports `accuracy = correct / labeled` per batch (in `History`)
and overall (in the final log line).

### `--mode eval` (without `--label-column`)

If you omit `--label-column`, eval falls back to **next-token
prediction scoring** — useful on any text dataset, no labels needed:

```bash
ezpz launch python3 -m ezpz.examples.inference --mode eval \
    --dataset wikitext --dataset-config wikitext-2-raw-v1
```

This path:

- Runs **one forward pass** over the full prompt batch (not generation)
- Computes argmax accuracy at each position (token `i+1` predicted
  from logits at position `i`)
- Computes per-token cross-entropy → **perplexity** = `exp(NLL/token)`
- Reports `next_token_accuracy` and `perplexity` per batch + overall

Much faster than the labeled path (one forward pass vs autoregressive
generation), and gives the standard "language modeling perplexity"
metric you'd see in eval-harness tools — without needing a labeled
dataset.

The per-batch log line shows `tokens_scored=` instead of `new_tokens=`
to make the difference obvious.

## How sharding works

Inputs are loaded once on every rank, then each rank processes a
contiguous block via `shard_indices`:

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
| `--mode` | `generate` | `benchmark` / `generate` / `eval` |
| `--model` | `meta-llama/Llama-3.2-1B` | HF model name or local path |
| `--dataset` | `wikitext` | HF dataset (ignored in benchmark mode) |
| `--dataset-config` | `wikitext-2-raw-v1` | Dataset configuration / subset |
| `--dataset-split` | `test` | Split (`train`/`validation`/`test`) |
| `--text-column` | `text` | Column containing the prompt |
| `--label-column` | — | Gold label column (optional for `--mode eval`; without it, scores next-token prediction) |
| `--max-samples` | `128` | Total samples across all ranks |
| `--batch-size` | `4` | Per-rank batch size |
| `--max-input-tokens` | `512` | Truncate prompts to this many tokens |
| `--max-new-tokens` | `64` | Tokens to generate per sample |
| `--dtype` | `bfloat16` | Model dtype (`float32`/`float16`/`bfloat16`) |
| `--do-sample` | off | Sample instead of greedy decoding |
| `--temperature` | `1.0` | Sampling temperature (with `--do-sample`) |
| `--top-p` | `1.0` | Nucleus sampling cutoff (with `--do-sample`) |
| `--benchmark-iters` | `20` | Iterations (only `--mode benchmark`) |
| `--benchmark-warmup` | `3` | Warmup iters excluded from totals |
| `--flops` | off | Measure real per-batch FLOPS via `FlopCounterMode`. Without this flag, `tflops` and `mfu` are not reported (rather than reporting approximated values). |
| `--flops-every-n-steps` | `1` | When `--flops` is set, measure every N steps to amortize the overhead |
| `--no-save-predictions` | save on | Skip writing per-sample JSONL |

## Outputs

```
outputs/ezpz.examples.inference/<timestamp>/
├── predictions-rank0.jsonl       # not written in benchmark mode
├── predictions-rank1.jsonl
├── ...
├── inference.h5                  # finalized History dataset
├── report-inference.md           # markdown summary
└── plots/                        # auto-generated metric plots
```

Each `predictions-rank<N>.jsonl` row:

```json
// generate mode
{"rank": 0, "prompt": "...", "completion": "..."}

// eval mode
{"rank": 0, "prompt": "...", "completion": "...", "label": "42"}
```

## MFU Tracking (opt-in)

`tflops` and `mfu` are **off by default** because the only honest
measurement is via `FlopCounterMode`, which adds ~15-40% per-step
overhead. Approximated values (linear-scaled startup estimates,
`n_tokens × forward_flops`, etc.) tend to be misleading — they ignore
attention's `O(seq²)` cost and KV-cache savings, and have produced
MFU values >100% in practice.

To opt in:

```bash
# Measure real FLOPS on every batch
ezpz launch python3 -m ezpz.examples.inference --mode eval --flops

# Or amortize the overhead — measure every 10th batch
ezpz launch python3 -m ezpz.examples.inference \
    --mode eval --flops --flops-every-n-steps 10
```

Behavior:

- **Without `--flops`**: `tflops` and `mfu` keys are absent from
  `metrics`. Per-batch logs show timing/throughput/eval metrics only.
- **With `--flops` and `--flops-every-n-steps 1`** (default): every
  batch is profiled. ~15-40% slower overall, exact MFU on every step.
- **With `--flops` and `--flops-every-n-steps N` (N > 1)**: every Nth
  batch is profiled, others run normally. Profiled batches get
  `metrics["flops_measured"] = True` so post-analysis can filter
  to the trustworthy points.

See [`ezpz.flops`](../python/Code-Reference/flops.md) for the
per-device MFU formula.

## See Also

- [`ezpz.examples.hf`](hf.md) — fine-tuning loop (the training
  counterpart to this inference example)
- [Recipes › MFU Tracking](../recipes.md#mfu-tracking)
