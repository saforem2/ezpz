# Train Transformer with FSDP and TP on HF Datasets

Use this example when your model is large enough that FSDP alone isn't
sufficient — 2D parallelism combines tensor parallelism (splitting individual
layers across GPUs within a node) with FSDP (sharding parameters across
nodes). This is the approach for training very large transformer models where
both memory and communication efficiency matter.

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`wrap_model()`][ezpz.distributed.wrap_model] — Wrap model for FSDP
    - [`ezpz.tp`](../python/Code-Reference/tp/index.md) — Tensor parallelism utilities

See:

- 📘 [examples/FSDP TP](../python/Code-Reference/examples/fsdp_tp.md)
- 🐍 [src/ezpz/examples/fsdp_tp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp \
    --tp=2 \
    --epochs=5 \
    --batch-size=2 \
    --dataset=eliplutchok/fineweb-small-sample \
```

## Common modifications

- **Pick a model size** — pass `--model {debug,small,medium,large,xl,xxl,xxxl}`.
  Each `xN` size also accepts long-form aliases (`xlarge`/`extra-large` etc.).
  Sizes climb in roughly Llama-1.5B / 7B / 13B parameter targets at
  `xl`/`xxl`/`xxxl`.
- **AuroraGPT presets** — `--model agpt-2b` or `--model agpt-20b` reproduce
  torchtitan's `agpt_configs` registry exactly (`dim`, `n_layers`, `n_heads`,
  `n_kv_heads`, `vocab_size`, `hidden_dim`, `rope_theta` all match). Useful
  for A/B'ing against a torchtitan agpt run on identical architecture.
  Aliases: `agpt2b`, `agpt_2b`, `AGPT-2B` (and 20b counterparts).
- **Load a HuggingFace model** — pass any `--model owner/repo` (e.g.
  `--model meta-llama/Llama-3.2-1B`). The `/` is the sentinel — anything
  with one is treated as a HF repo id, pulled via `AutoModelForCausalLM`,
  and wrapped with FSDP. `--tokenizer_name` auto-defaults to the same repo.
  **Note**: HF mode forces `--tp 1` (the TP plan is hardcoded to ezpz's own
  Transformer module names; doesn't apply to HF's `LlamaDecoderLayer`). See
  [HuggingFace models](#huggingface-models) for details.
- **Activation checkpointing** — `--ac {none,block,full,selective}` (or
  `--activation-checkpoint`) trades compute for memory during training.
  `block`/`full` wraps each TransformerBlock (~30-40 pct activation-memory
  reduction, ~20 pct throughput hit); matches torchtitan's default for
  agpt-2b/agpt-20b. See [Activation checkpointing](#activation-checkpointing).
- **Compile with torch.compile** — pass `--compile` to wrap the model with
  `torch.compile()` after FSDP/DDP wrap. Tune the mode with
  `--compile-mode {default,reduce-overhead,max-autotune}` (default: `default`).
  Use `reduce-overhead` for cudagraphs on small models / large batches;
  `max-autotune` for the slowest startup / fastest steady-state.

> Note: combining `--compile` with `--ac` on HuggingFace models can trigger a
> `CheckpointError: tensor count mismatch` due to HF's DynamicCache. Drop one
> if you hit it.

## Source

<details closed><summary><code>src/ezpz/examples/fsdp_tp.py</code></summary>

```python title="src/ezpz/examples/fsdp_tp.py"
--8<-- "src/ezpz/examples/fsdp_tp.py"
```

</details>

## Code Walkthrough


<details closed markdown><summary><strong>Module Docstring: Parallel Layout</strong></summary>

The file opens with a docstring describing the 2D parallelism layout --
TP (tensor parallel) within each host, FSDP (data parallel) across hosts.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="1"
"""
ezpz/examples/fsdp_tp.py

2D tensor/sequence parallel + FSDP training demo on a Llama-style model.

Sam Foreman
2025-09-08

Modified from:
<https://pytorch.org/tutorials/intermediate/TP_tutorial.html>


This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

We use a simple diagram to illustrate below:

+-----.-----+-----+-----+
|  0  |  1  |  2  |  3  |
|     |     |     |     |
+-----+-----+-----+-----+
|  4  |  5  |  6  |  7  |
|     |     |     |     |
+-----+-----+-----+-----+
|  8  |  9  | 10  | 11  |
|     |     |     |     |
+-----+-----+-----+-----+


+----------+        +------------+       +----------+       +------------+
| Host 1   |        | Host 2     |       |          |       |  Host N    |
| 8 GPUs   |        | 8 GPUs     |       |          |       |  8 GPUs    |
|          |        |            |       |    ...   |       |            |
| (TP)     |        | (TP)       |       |          |       |  (TP)      |
|[0,1,..,7]|        | [8,9..,15] |       |          |       | [8N-8,8N-7 |
|          |        |            |       |          |       |  .., 8N-1] |
|          |        |            |       |          |       |            |
+----------+        +------------+       +----------+       +------------+

- FSDP:

  [0, 8, ..., 8N-8],
  [1, 9, ..., 8N-7],
  ...,
  [7, 15, ..., 8N-1]
"""
```

</details>

<details closed markdown><summary><strong>Imports</strong></summary>

Standard library, PyTorch, and `ezpz` imports. The key distributed
primitives -- `DeviceMesh`, `FSDP`, and `parallelize_module` -- are
all pulled in here.

```python title="src/ezpz/examples/fsdp_tp.py:118:158"
--8<-- "src/ezpz/examples/fsdp_tp.py:118:158"
```

A logger is set up and W&B is optionally imported.

```python title="src/ezpz/examples/fsdp_tp.py:160:170"
--8<-- "src/ezpz/examples/fsdp_tp.py:160:170"
```

</details>

<details closed markdown><summary><strong>Model Presets</strong></summary>

`MODEL_PRESETS` defines canned configurations that override default CLI
values for quick experimentation:

- Toy / smoke-test sizes: `debug`, `small`, `medium`, `large`.
- Production-scale sizes: `xl`, `xxl`, `xxxl` — chosen to map roughly to
  Llama-1.5B / Llama-7B / Llama-13B (`dim 2048/4096/5120`,
  `n_layers 24/32/40`, `n_heads 32/32/40`, `n_kv_heads 8`).
- **AuroraGPT (agpt) sizes**: `agpt-2b`, `agpt-20b` — verbatim from
  torchtitan's `agpt_configs` registry. agpt-2b is
  `dim=2048, n_layers=12, n_heads=16, n_kv_heads=4, vocab_size=256128, hidden_dim=11008, rope_theta=50000`.
  agpt-20b is `dim=5120, n_layers=64, n_heads=40, n_kv_heads=8, vocab_size=256128, hidden_dim=14336, rope_theta=500000`.
  These exist so an ezpz `fsdp_tp` run can be A/B'd against a torchtitan
  agpt run on the same architecture.

Each `xN` size also accepts long-form aliases via `MODEL_ALIASES`
(`xlarge` / `extra-large` → `xl`, `xxlarge` / `extra-extra-large` →
`xxl`, `xxxlarge` / `extra-extra-extra-large` → `xxxl`), so spelling
them out works too. agpt presets accept the natural variants
`agpt2b` / `agpt_2b` / `AGPT-2B` (and the 20b counterparts).

### HuggingFace models

`--model` also accepts a HuggingFace repo id whenever the value contains
a `/`:

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp --model meta-llama/Llama-3.2-1B
```

The HF path:

- Pulls both the architecture (config) AND the pretrained weights via
  `AutoModelForCausalLM.from_pretrained(...)`.
- Auto-defaults `--tokenizer_name` to the same repo id (override with
  `--tokenizer_name <other>` if you want a different one).
- Reads `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN` for gated repos; you can
  also `huggingface-cli login` once and skip the env var.
- **Forces `--tp 1`** (FSDP-only). The existing TP `parallelize()` plan is
  hardcoded to ezpz's own `TransformerBlock` module names (`attention.wq`,
  `feed_forward.w1`, etc.) and won't apply cleanly to HF's
  `LlamaDecoderLayer` / `GemmaDecoderLayer` / etc. The example logs a
  warning if you pass `--tp > 1` along with a HF model.
- Wraps the model with FSDP using a transformer-block auto-wrap policy
  derived from the HF model's own `ModuleList` children — so each
  decoder layer becomes its own FSDP unit.

### Activation checkpointing

Pass `--activation-checkpoint` (alias `--ac`) to trade compute for
memory during training:

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp --model agpt-2b --ac block
```

Modes:

- **`none`** (default) — keeps all forward activations in memory.
  Lowest latency, highest memory.
- **`block`** (alias: **`full`**) — wraps each TransformerBlock's
  forward with `torch.utils.checkpoint`. The block re-runs its
  forward during backward instead of caching intermediate
  activations. Typical **30-40% activation-memory reduction**,
  **~20% throughput hit**. Matches torchtitan's default for
  agpt-2b / agpt-20b. `--ac full` is accepted as a compatibility
  alias with torchtitan's CLI.
- **`selective`** — checkpoints only the attention computation
  inside each block. Smaller memory savings (~15-20%), smaller
  throughput hit (~10%). Less robust than `block` for arbitrary
  architectures.

AC is applied **after** the TP/FSDP wrap so the checkpoint envelope
contains FSDP's unshard/reshard bookkeeping; reversed order would
double-shard and corrupt grads.

Caveat: AC only helps with **training-time** activation memory. It
will NOT fix init-time OOMs (every rank holds the full unsharded
model momentarily during `model.to(device)` before FSDP shards). If
the model OOMs during init, raise `--tp` (halves per-rank weight
memory for each doubling) or use a smaller preset.

```python title="src/ezpz/examples/fsdp_tp.py:172:303"
--8<-- "src/ezpz/examples/fsdp_tp.py:172:303"
```

</details>

<details closed markdown><summary><strong>Sharding Strategies</strong></summary>

Maps user-facing string names to PyTorch `ShardingStrategy` enum values.

```python title="src/ezpz/examples/fsdp_tp.py:226:232"
--8<-- "src/ezpz/examples/fsdp_tp.py:226:232"
```

</details>

<details closed markdown><summary><strong>Sequence Parallel Label Slicing</strong></summary>

When sequence parallelism is active, each TP rank only sees a slice of
the sequence dimension. This helper narrows the label tensor to match
the local shard so `cross_entropy` computes the correct loss.

```python title="src/ezpz/examples/fsdp_tp.py:235:290"
--8<-- "src/ezpz/examples/fsdp_tp.py:235:290"
```

</details>

<details closed markdown><summary><strong>Argument Parsing</strong></summary>

`parse_args` defines every CLI flag. The `--model` flag selects a preset
from `MODEL_PRESETS`; any flag the user provides explicitly takes
precedence over the preset via `apply_model_preset`.

```python title="src/ezpz/examples/fsdp_tp.py:444:524"
--8<-- "src/ezpz/examples/fsdp_tp.py:444:524"
```

</details>

<details closed markdown><summary><strong><code>parallelize</code>: TP Parallelization + FSDP Wrapping</strong></summary>

This is the core of the 2D parallelism setup. It takes the model and
device mesh, applies tensor/sequence parallelism along the `"tp"` mesh
dimension, then wraps the result with FSDP along the `"dp"` dimension.

```python title="src/ezpz/examples/fsdp_tp.py:527:541"
--8<-- "src/ezpz/examples/fsdp_tp.py:527:541"
```

**Top-level TP plan.** The embedding is row-sharded, the final output
projection is column-sharded, and the RMS norm between them uses
`SequenceParallel`.

```python title="src/ezpz/examples/fsdp_tp.py:542:558"
--8<-- "src/ezpz/examples/fsdp_tp.py:542:558"
```

**Per-layer TP plan.** Each transformer block's attention and FFN
sub-modules are parallelized: Q/K/V projections are column-sharded,
output projections are row-sharded, and norms use `SequenceParallel`.
Attention head counts are divided by the TP mesh size.

```python title="src/ezpz/examples/fsdp_tp.py:560:589"
--8<-- "src/ezpz/examples/fsdp_tp.py:560:589"
```

**FSDP wrapping.** After TP is applied, the entire model is wrapped with
FSDP on the `"dp"` sub-mesh.

```python title="src/ezpz/examples/fsdp_tp.py:598:606"
--8<-- "src/ezpz/examples/fsdp_tp.py:598:606"
```

</details>

<details closed markdown><summary><strong><code>train</code>: Device Mesh, Data Loading, and Training Loop</strong></summary>

`train` orchestrates the full run. It first creates the 2D device mesh,
loads data, then runs the epoch loop.

**Device mesh creation.** World size is split into `dp` x `tp`
dimensions.

```python title="src/ezpz/examples/fsdp_tp.py:681:695"
--8<-- "src/ezpz/examples/fsdp_tp.py:681:695"
```

**HuggingFace dataset loading.** If `--dataset` is not `"mnist"` or
`"random"`, a tokenized HF text dataset is loaded and the vocab size is
synced to the tokenizer.

```python title="src/ezpz/examples/fsdp_tp.py:697:718"
--8<-- "src/ezpz/examples/fsdp_tp.py:697:718"
```

**Model construction and parallelization.** A `Transformer` is built
from `ModelArgs`, moved to the device, optionally given a
`MixedPrecision` config, and then handed to `parallelize`.

```python title="src/ezpz/examples/fsdp_tp.py:720:729"
--8<-- "src/ezpz/examples/fsdp_tp.py:720:729"
```

```python title="src/ezpz/examples/fsdp_tp.py:753:782"
--8<-- "src/ezpz/examples/fsdp_tp.py:753:782"
```

**DataLoader setup.** Three branches: MNIST, random synthetic data, or
HuggingFace datasets. For HF data, a `DistributedSampler` partitions
across the DP dimension, and `TPBroadcastDataLoader` replicates batches
within each TP group.

```python title="src/ezpz/examples/fsdp_tp.py:840:862"
--8<-- "src/ezpz/examples/fsdp_tp.py:840:862"
```

**Metric tracking.** An `ezpz.history.History` object is created for
JSONL metric logging and optional distributed aggregation.

```python title="src/ezpz/examples/fsdp_tp.py:870:885"
--8<-- "src/ezpz/examples/fsdp_tp.py:870:885"
```

**Training loop.** Each batch is moved to device, split into
`inp`/`labels`, run through the model, and the loss is computed with
`cross_entropy`. Labels are narrowed for sequence parallelism when
needed. Gradient clipping is applied before `optimizer.step()`.

```python title="src/ezpz/examples/fsdp_tp.py:893:922"
--8<-- "src/ezpz/examples/fsdp_tp.py:893:922"
```

```python title="src/ezpz/examples/fsdp_tp.py:958:962"
--8<-- "src/ezpz/examples/fsdp_tp.py:958:962"
```

```python title="src/ezpz/examples/fsdp_tp.py:978:985"
--8<-- "src/ezpz/examples/fsdp_tp.py:978:985"
```

After each step, timing and loss metrics are collected into a dict and
passed to `history.update` and `history.log_metrics`.

```python title="src/ezpz/examples/fsdp_tp.py:989:997"
--8<-- "src/ezpz/examples/fsdp_tp.py:989:997"
```

At the end of training, activation hooks are removed, a barrier syncs
all ranks, and `history.finalize` writes the summary dataset on rank 0.

```python title="src/ezpz/examples/fsdp_tp.py:1058:1101"
--8<-- "src/ezpz/examples/fsdp_tp.py:1058:1101"
```

</details>

<details closed markdown><summary><strong><code>main</code> and Entrypoint</strong></summary>

`main` calls `ezpz.distributed.setup_torch` to initialize the
distributed backend (including TP groups), determines the output
directory, and dispatches to `train`.

```python title="src/ezpz/examples/fsdp_tp.py:1066:1087"
--8<-- "src/ezpz/examples/fsdp_tp.py:1066:1087"
```

The `if __name__ == "__main__"` block parses args, runs `main`, cleans
up distributed state, and exits.

```python title="src/ezpz/examples/fsdp_tp.py:1104:1108"
--8<-- "src/ezpz/examples/fsdp_tp.py:1104:1108"
```

</details>

## MFU Tracking

Model FLOPS are estimated via [`try_estimate`](../recipes.md#mfu-tracking)
on the unwrapped model **before** the 2D FSDP+TP parallelization.
Per-step **TFLOPS** and **MFU** are reported under `train/tflops`
and `train/mfu`.

```python
_model_flops = try_estimate(model, (args.batch_size, args.seq_len))
# ... per step:
metrics["train/tflops"] = _model_flops / (t2 - t0) / 1e12
metrics["train/mfu"] = compute_mfu(_model_flops, t2 - t0)
```

See [`ezpz.flops`](../python/Code-Reference/flops.md) for details.

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.fsdp_tp --help
usage: fsdp_tp.py [-h] [--dim DIM] [--n-layers N_LAYERS] [--n-heads N_HEADS]
                  [--n-kv-heads N_KV_HEADS] [--multiple-of MULTIPLE_OF]
                  [--ffn-dim-multiplier FFN_DIM_MULTIPLIER]
                  [--hidden-dim HIDDEN_DIM] [--rope-theta ROPE_THETA]
                  [--norm-eps NORM_EPS] [--vocab-size VOCAB_SIZE] [--lr LR]
                  [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--model MODEL]
                  [--test-batch-size TEST_BATCH_SIZE]
                  [--num-workers NUM_WORKERS] [--seed SEED] [--tp TP]
                  [--sharding-strategy SHARDING_STRATEGY]
                  [--activation-checkpoint {none,block,full,selective}]
                  [--max-grad-norm MAX_GRAD_NORM] [--outdir OUTDIR]
                  [--dataset DATASET] [--tokenizer_name TOKENIZER_NAME]
                  [--model_name_or_path MODEL_NAME_OR_PATH]
                  [--hf-split HF_SPLIT] [--hf-text-column HF_TEXT_COLUMN]
                  [--hf-limit HF_LIMIT] [--seq-len SEQ_LEN]
                  [--max-seq-len MAX_SEQ_LEN] [--depth-init DEPTH_INIT]
                  [--fp32]

2D Parallel Training

options:
  -h, --help            show this help message and exit
  --dim DIM
  --n-layers N_LAYERS
  --n-heads N_HEADS
  --n-kv-heads N_KV_HEADS
  --multiple-of MULTIPLE_OF
  --ffn-dim-multiplier FFN_DIM_MULTIPLIER
  --hidden-dim HIDDEN_DIM
                        Override SwiGLU FFN hidden dim. When None (default),
                        TransformerBlock derives it as `4 * dim` and
                        FeedForward applies the 2/3 + ffn_dim_multiplier +
                        multiple_of pipeline. Set this to a concrete value
                        (e.g. 11008 for agpt-2b, 14336 for agpt-20b) to bypass
                        the formula and hit a published architecture exactly.
                        (default: None)
  --rope-theta ROPE_THETA
                        Base frequency for RoPE positional embeddings.
                        Llama1/2 used 10000 (the default); Llama3 uses 500000;
                        agpt-2b uses 50000. (default: 10000.0)
  --norm-eps NORM_EPS
  --vocab-size VOCAB_SIZE
  --lr LR
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  --model MODEL         Model size preset (overrides dim/layer defaults).
                        Presets:
                        debug/small/medium/large/xl/xxl/xxxl/agpt-2b/agpt-20b.
                        xl/xxl/xxxl accept long-form aliases (`xlarge`/`extra-
                        large`, etc). agpt presets accept `agpt2b`/`agpt_2b`
                        etc. Pass a HuggingFace repo id with a `/` (e.g.
                        `meta-llama/Llama-3.2-1B`) to load HF weights instead
                        — that path forces --tp 1 (FSDP-only). (default: None)
  --test-batch-size TEST_BATCH_SIZE
  --num-workers NUM_WORKERS
  --seed SEED
  --tp TP
  --sharding-strategy SHARDING_STRATEGY
  --activation-checkpoint {none,block,full,selective}, --ac {none,block,full,selective}
                        Activation checkpointing strategy. `none` (default)
                        keeps all forward activations in memory. `block`
                        (alias: `full`) wraps each TransformerBlock — typical
                        30-40 pct activation memory reduction, ~20 pct
                        throughput hit (matches torchtitan's default for
                        agpt-2b/agpt-20b). `selective` checkpoints only the
                        attention computation inside each block — ~15-20 pct
                        memory reduction, ~10 pct throughput hit. Trade
                        activation memory for recomputation cost — useful when
                        OOM-ing during training (NOT during init; for init-
                        time OOM consider increasing --tp or reducing
                        --seq-len). (default: none)
  --max-grad-norm MAX_GRAD_NORM
  --outdir OUTDIR
  --dataset DATASET
  --tokenizer_name TOKENIZER_NAME
  --model_name_or_path MODEL_NAME_OR_PATH
  --hf-split HF_SPLIT, --hf_split HF_SPLIT
                        Dataset split to load. (default: train)
  --hf-text-column HF_TEXT_COLUMN, --hf_text_column HF_TEXT_COLUMN
                        Column containing raw text in the dataset. (default:
                        text)
  --hf-limit HF_LIMIT, --hf_limit HF_LIMIT
                        Maximum number of rows to sample from the HF dataset.
                        0 (default) = no limit (use the full dataset). Pass a
                        positive value (e.g. `--hf-limit 512`) to subsample
                        for smoke tests. Subsampling is deterministic given
                        $EZPZ_HF_SAMPLE_SEED. (default: 0)
  --seq-len SEQ_LEN
  --max-seq-len MAX_SEQ_LEN
  --depth-init DEPTH_INIT
  --fp32                Disable mixed precision (use fp32) for debugging NaNs.
                        (default: False)
```

</details>

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
$ ezpz launch python3 -m ezpz.examples.fsdp_tp
```

/embed `ezpz-fsdp-tp.html`


</details>
