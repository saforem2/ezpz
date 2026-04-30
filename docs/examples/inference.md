# 🔮 `ezpz.examples.inference`

Distributed inference over a HuggingFace model + dataset, with three
distinct modes: **benchmark**, **generate**, and **eval**.

Each rank loads the model and processes a disjoint shard of inputs
(data parallelism). Per-batch latency and throughput (tokens/sec)
are tracked through `ezpz.History`. With `--mode eval` an accuracy
metric is added. **TFLOPS and MFU are opt-in via `--flops`** —
without that flag the columns are simply absent rather than
approximated (see [MFU Tracking (opt-in)](#mfu-tracking-opt-in)).

## Source

<details closed><summary><code>src/ezpz/examples/inference.py</code></summary>

```python title="src/ezpz/examples/inference.py"
--8<-- "src/ezpz/examples/inference.py"
```

</details>

## Code Walkthrough

<details closed markdown><summary><strong><code>shard_indices</code> — data-parallel input partitioning</strong></summary>

Each rank computes its slice of the input array; the helper handles
uneven splits and the "more ranks than samples" edge case.

```python title="src/ezpz/examples/inference.py:202:215"
--8<-- "src/ezpz/examples/inference.py:202:215"
```

</details>

<details closed markdown><summary><strong><code>_run_with_optional_flops</code> — opt-in FLOPS measurement</strong></summary>

The eval-unlabeled forward path and the `model.generate` path share
this wrapper.  When `measure=False` (the default — no `--flops`),
the call is a passthrough; otherwise it runs inside `FlopCounterMode`
and returns the measured FLOP count alongside the result.

```python title="src/ezpz/examples/inference.py:232:254"
--8<-- "src/ezpz/examples/inference.py:232:254"
```

</details>

<details closed markdown><summary><strong>Per-batch metrics — measured vs absent</strong></summary>

`tflops` and `mfu` are only added to the metrics dict when the step
actually ran `FlopCounterMode`.  No approximation is reported on
unmeasured batches; downstream filters can rely on
`flops_measured == True` to identify trustworthy points.

The unlabeled-eval path emits `tokens_scored` (one forward pass over
the prompt batch); the generate path emits `new_tokens` (one forward
per autoregressive token).  Never both.

</details>

<details closed markdown><summary><strong>Eval-unlabeled scoring — perplexity from logits</strong></summary>

Without `--label-column`, eval mode runs a single forward pass per
batch and scores next-token prediction at every position.  Argmax
accuracy and per-token cross-entropy go into the metrics; perplexity
= `exp(NLL/token)` is reported per batch and overall.

This is much cheaper than autoregressive generation and matches the
"language modeling perplexity" you'd get from eval-harness tools.

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
| `--seed` | `0` | Random seed for token sampling and shard generation |
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

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ ezpz launch python3 -m ezpz.examples.inference --help
usage: ezpz.examples.inference [-h] [--mode {benchmark,generate,eval}]
                               [--model MODEL] [--dataset DATASET]
                               [--dataset-config DATASET_CONFIG]
                               [--dataset-split DATASET_SPLIT]
                               [--text-column TEXT_COLUMN]
                               [--label-column LABEL_COLUMN]
                               [--benchmark-iters BENCHMARK_ITERS]
                               [--benchmark-warmup BENCHMARK_WARMUP]
                               [--max-samples MAX_SAMPLES]
                               [--batch-size BATCH_SIZE]
                               [--max-input-tokens MAX_INPUT_TOKENS]
                               [--max-new-tokens MAX_NEW_TOKENS]
                               [--dtype {float32,float16,bfloat16}] [--flops]
                               [--flops-every-n-steps FLOPS_EVERY_N_STEPS]
                               [--do-sample] [--temperature TEMPERATURE]
                               [--top-p TOP_P] [--seed SEED]
                               [--save-predictions] [--no-save-predictions]

Distributed inference over a HuggingFace model + dataset. Three modes: --mode
benchmark (throughput), generate (synthetic data corpus), eval (accuracy on
labeled data). Each rank processes a disjoint shard of prompts.

options:
  -h, --help            show this help message and exit
  --mode {benchmark,generate,eval}
                        Inference mode: 'benchmark' = synthetic random tokens,
                        no dataset, focus on tokens/sec/MFU. 'generate' =
                        dataset prompts → completions, save to JSONL
                        (synthetic data / distillation use case). 'eval' =
                        dataset prompts + gold labels, compare generated text
                        to label, report accuracy.
  --model MODEL         HuggingFace model name or local path
  --dataset DATASET     HuggingFace dataset name or local path (ignored in
                        --mode benchmark)
  --dataset-config DATASET_CONFIG
                        Dataset configuration (subset name)
  --dataset-split DATASET_SPLIT
                        Dataset split (train/validation/test)
  --text-column TEXT_COLUMN
                        Dataset column containing the prompt text
  --label-column LABEL_COLUMN
                        Dataset column containing the gold label (required for
                        --mode eval)
  --benchmark-iters BENCHMARK_ITERS
                        Number of benchmark iterations (only with --mode
                        benchmark)
  --benchmark-warmup BENCHMARK_WARMUP
                        Warmup iterations to skip when reporting (only with
                        --mode benchmark)
  --max-samples MAX_SAMPLES
                        Maximum number of samples to process across all ranks
  --batch-size BATCH_SIZE
                        Per-rank batch size
  --max-input-tokens MAX_INPUT_TOKENS
                        Truncate prompts to this many tokens
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum tokens to generate per sample
  --dtype {float32,float16,bfloat16}
                        Model dtype
  --flops               Measure exact per-batch FLOPS via FlopCounterMode and
                        report tflops + mfu in metrics. Off by default —
                        without this flag, MFU/TFLOPS columns are simply
                        omitted (rather than reporting approximated values).
                        Adds ~15-40% overhead per step.
  --flops-every-n-steps FLOPS_EVERY_N_STEPS
                        When --flops is set, measure FLOPS every N steps
                        (default 1 = every step). Use a higher value to
                        amortize the overhead across batches.
  --do-sample           Use sampling instead of greedy decoding
  --temperature TEMPERATURE
                        Sampling temperature (only used with --do-sample)
  --top-p TOP_P         Nucleus sampling threshold (only used with --do-
                        sample)
  --seed SEED           Random seed
  --save-predictions    Write per-sample predictions to JSONL
  --no-save-predictions
                        Skip writing per-sample predictions
```

</details>

## See Also

- [`ezpz.examples.hf`](hf.md) — fine-tuning loop (the training
  counterpart to this inference example)
- [Recipes › MFU Tracking](../recipes.md#mfu-tracking)
