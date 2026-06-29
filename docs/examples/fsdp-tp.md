# Train Transformer with FSDP and TP on HF Datasets

Use this example when your model is large enough that FSDP alone isn't
sufficient — 2D parallelism combines tensor parallelism (splitting individual
layers across GPUs within a node) with FSDP (sharding parameters across
nodes). This is the approach for training very large transformer models where
both memory and communication efficiency matter.

!!! note "FSDP2 (`fully_shard`)"

    This example uses **FSDP2** — each module group (embedding, every
    `TransformerBlock`, `[norm, output]`, then the root) is sharded
    independently with `fully_shard`. Per-module sharding keeps backward
    memory bounded — in particular for the 256K-vocab embedding + output
    projection — where FSDP1's single flat-parameter wrap OOM'd at long
    sequence length. Tensor parallelism is only applied when `--tp > 1`
    (at `--tp 1` the model stays in plain tensors, no SequenceParallel
    overhead). The first-step logits debug probe is gated behind
    `EZPZ_TRACK_LOGITS=1` (off by default — it allocates a full-logits
    tensor that can itself OOM a large-vocab run).

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

- **Pick a model size** — pass `--model {debug,s,m,l,xl,xxl,xxxl}`. Each
  short-name size also accepts long-form aliases (`small`/`medium`/`large`/
  `xlarge`/`extra-large`/etc). Shared size ladder across all 5 example
  modules:

  | Preset | Llama-arch (vocab=32k) |
  |---|---|
  | `debug` | ~10K (laptop smoke test) |
  | `s` (small) | **~125M** |
  | `m` (medium) | **~246M** |
  | `l` (large) | **~495M** |
  | `xl` | **~1.21B** (Llama-1.5B-ish) |
  | `xxl` | **~5.93B** (Llama-7B-ish) |
  | `xxxl` | **~11.34B** (Llama-13B-ish) |

  > **Breaking change**: `--model small` now resolves to ~125M (was a
  > toy ~6M). Use `--model debug` for laptop-runnable smoke tests.
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
- **Compile with torch.compile** — pass `--compile` to compile each
  TransformerBlock after the FSDP/TP wrap. Tune the mode with
  `--compile-mode {default,reduce-overhead,max-autotune}` (default: `default`).
  Use `reduce-overhead` for cudagraphs on small models / large batches;
  `max-autotune` for the slowest startup / fastest steady-state. See
  [torch.compile](#torchcompile) for the per-block rationale and the
  `--tp`/`--ac`/`--compile` interaction caveat.
- **Cross-entropy implementation** —
  `--loss-impl {eager,chunked,chunked-backward,compiled,fused-linear,loss-parallel}`
  (default `eager`). At a large vocab (agpt's 256K) and long sequence, eager
  `F.cross_entropy` materializes a multi-GB `(B·T, vocab)` fp32 logits +
  gradient transient in backward that OOMs even when the model fits. The
  large-vocab output path is *the* memory bottleneck; pick by need (numbers
  measured on agpt-2b, tp=1, bs=2, seq=8192):
    - `eager` — full logits; OOMs at this scale. Small vocab/seq only.
    - `chunked` — chunks the **forward** only (`--loss-chunk-size`); does
      **not** bound backward, still OOMs at large vocab. Rarely useful.
    - `chunked-backward` — custom autograd Function that also bounds the
      backward **graph** (recomputes each chunk's grad), saving ~one full
      logits buffer vs eager. General + model-agnostic: works for **HF
      models** (where `fused-linear` can't) and needs **no `torch.compile`** —
      good at *moderate* vocab/seq or when compile is unavailable. Still holds
      two logit-sized buffers, so it does **not** fix the very-large-vocab OOM
      (use `fused-linear`/`compiled` there).
    - `compiled` — `torch.compile` fuses log-softmax + NLL + backward so the
      full transient never materializes (torchtitan's approach). Fits
      (~45 GiB) and is the **fastest that fits** (~28% MFU). Best default
      when it fits + compile works.
    - `fused-linear` — runs the output projection per row-chunk so the full
      `(B·T, vocab)` logits/grad are **never built** (Liger/Cut-CE style);
      bounds **both** the row and vocab dims. **Lowest memory** (~32 GiB,
      below compiled) at ~24% MFU — trade a little speed for headroom (bigger
      batch/seq). ezpz `Transformer` + tp=1 only (HF / tp>1 fall back to
      compiled).
    - `loss-parallel` — vocab-parallel CE sharding the vocab across TP ranks
      (each holds `vocab/tp`) via TP all-reduces. Bounds the **vocab** dim;
      only helps at tp>1 (falls back to eager at tp=1). At tp>1 it is also the
      **only correct path** (plain CE hits a Tensor/DTensor mismatch on the
      replicated logits). ~23 GiB/rank, ~34% MFU at tp=2.

  See [Matching torchtitan](#matching-torchtitan).
- **Activation-memory budget** — `--act-mem-budget <float>` (default `1.0`,
  only active with `--compile`). Sets
  `torch._functorch.config.activation_memory_budget`: `1.0` saves every
  activation, lower values let the inductor partitioner recompute a fraction
  in backward to cut peak memory. This is the knob that lets larger batches
  fit. See [Matching torchtitan](#matching-torchtitan).

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

```python title="src/ezpz/examples/fsdp_tp.py:115:158"
--8<-- "src/ezpz/examples/fsdp_tp.py:115:158"
```

A logger is set up and W&B is optionally imported.

```python title="src/ezpz/examples/fsdp_tp.py:160:169"
--8<-- "src/ezpz/examples/fsdp_tp.py:160:169"
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
- Wraps the model with FSDP2: each block in the HF model's own decoder
  `ModuleList` gets its own `fully_shard` unit, then the root — so each
  decoder layer is sharded independently.

### Activation checkpointing

Pass `--activation-checkpoint` (alias `--ac`) to trade compute for
memory during training:

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp --model agpt-2b --ac block
```

Modes:

- **`none`** (default) — keeps all forward activations in memory.
  Lowest latency, highest memory.
- **`block`** (alias: **`full`**) — wraps each TransformerBlock with
  the compile-aware `checkpoint_wrapper`
  (`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`). The
  block re-runs its forward during backward instead of caching
  intermediate activations. Typical **30-40% activation-memory reduction**,
  **~20% throughput hit**. Matches torchtitan's default for
  agpt-2b / agpt-20b. `--ac full` is accepted as a compatibility
  alias with torchtitan's CLI.
- **`selective`** — checkpoints only the attention computation
  inside each block. Smaller memory savings (~15-20%), smaller
  throughput hit (~10%). Less robust than `block` for arbitrary
  architectures.

Under FSDP2, AC is applied to each block **before** `fully_shard` (the
checkpoint wrapper lives inside the FSDP2 unit) — this is torchtitan's
ordering. (This is the reverse of the FSDP1 order, where AC wrapped the
already-sharded module.)

Caveat: AC only helps with **training-time** activation memory. It
will NOT fix init-time OOMs (every rank holds the full unsharded
model momentarily during `model.to(device)` before FSDP shards). If
the model OOMs during init, raise `--tp` (halves per-rank weight
memory for each doubling) or use a smaller preset.

```python title="src/ezpz/examples/fsdp_tp.py:170:297"
--8<-- "src/ezpz/examples/fsdp_tp.py:170:297"
```

</details>

<details closed markdown><summary><strong>Sharding Strategies</strong></summary>

Maps the legacy FSDP1 strategy names (kept as the CLI surface so existing
scripts don't break) to FSDP2's `reshard_after_forward` policy: `True`
reshards params after forward (ZeRO-3-like), `False` keeps them gathered
(ZeRO-2-like).

```python title="src/ezpz/examples/fsdp_tp.py:327:333"
--8<-- "src/ezpz/examples/fsdp_tp.py:327:333"
```

</details>

<details closed markdown><summary><strong>Sequence Parallel Label Slicing</strong></summary>

When sequence parallelism is active, each TP rank only sees a slice of
the sequence dimension. This helper narrows the label tensor to match
the local shard so `cross_entropy` computes the correct loss.

```python title="src/ezpz/examples/fsdp_tp.py:343:399"
--8<-- "src/ezpz/examples/fsdp_tp.py:343:399"
```

</details>

<details closed markdown><summary><strong>Argument Parsing</strong></summary>

`parse_args` defines every CLI flag. The `--model` flag selects a preset
from `MODEL_PRESETS`; any flag the user provides explicitly takes
precedence over the preset via `apply_model_preset`.

```python title="src/ezpz/examples/fsdp_tp.py:690:1086"
--8<-- "src/ezpz/examples/fsdp_tp.py:690:1086"
```

</details>

<details closed markdown><summary><strong><code>parallelize</code>: TP Parallelization + FSDP Wrapping</strong></summary>

This is the core of the 2D parallelism setup. It takes the model and
device mesh, applies tensor/sequence parallelism along the `"tp"` mesh
dimension, then wraps the result with FSDP along the `"dp"` dimension.

```python title="src/ezpz/examples/fsdp_tp.py:1130:1147"
--8<-- "src/ezpz/examples/fsdp_tp.py:1130:1147"
```

**Top-level TP plan.** Applied only when `--tp > 1` (at `--tp 1` the whole
TP plan is skipped — no SequenceParallel overhead). The embedding is
row-sharded, the final output projection is column-sharded, and the RMS
norm between them uses `SequenceParallel`.

```python title="src/ezpz/examples/fsdp_tp.py:1148:1198"
--8<-- "src/ezpz/examples/fsdp_tp.py:1148:1198"
```

**Per-layer TP plan.** Each transformer block's attention and FFN
sub-modules are parallelized: Q/K/V projections are column-sharded,
output projections are row-sharded, and norms use `SequenceParallel`.
Attention head counts are divided by the TP mesh size.

```python title="src/ezpz/examples/fsdp_tp.py:1200:1230"
--8<-- "src/ezpz/examples/fsdp_tp.py:1200:1230"
```

**FSDP2 wrapping.** After TP is applied, each module group is sharded
independently on the `"dp"` sub-mesh with `fully_shard` — the embedding,
each TransformerBlock, then `[norm, output]`, then the root last. A final
`_configure_fsdp_gradient_division` call forces SUM reduction for gradient
comms on CCL/XPU (matching torchtitan).

```python title="src/ezpz/examples/fsdp_tp.py:1239:1264"
--8<-- "src/ezpz/examples/fsdp_tp.py:1239:1264"
```

</details>

<details closed markdown><summary><strong><code>train</code>: Device Mesh, Data Loading, and Training Loop</strong></summary>

`train` orchestrates the full run. It first creates the 2D device mesh,
loads data, then runs the epoch loop.

**Device mesh creation.** World size is split into `dp` x `tp`
dimensions.

```python title="src/ezpz/examples/fsdp_tp.py:1517:1522"
--8<-- "src/ezpz/examples/fsdp_tp.py:1517:1522"
```

**HuggingFace dataset loading.** If `--dataset` is not `"mnist"` or
`"random"`, a tokenized HF text dataset is loaded and the vocab size is
synced to the tokenizer.

```python title="src/ezpz/examples/fsdp_tp.py:1524:1546"
--8<-- "src/ezpz/examples/fsdp_tp.py:1524:1546"
```

**Model construction and parallelization.** A `Transformer` is built
from `ModelArgs`, moved to the device, optionally given a
`MixedPrecision` config, and then handed to `parallelize`.

```python title="src/ezpz/examples/fsdp_tp.py:1552:1632"
--8<-- "src/ezpz/examples/fsdp_tp.py:1552:1632"
```

```python title="src/ezpz/examples/fsdp_tp.py:1743:1749"
--8<-- "src/ezpz/examples/fsdp_tp.py:1743:1749"
```

**DataLoader setup.** Three branches: MNIST, random synthetic data, or
HuggingFace datasets. For HF data, a `DistributedSampler` partitions
across the DP dimension, and `TPBroadcastDataLoader` replicates batches
within each TP group.

```python title="src/ezpz/examples/fsdp_tp.py:1879:1939"
--8<-- "src/ezpz/examples/fsdp_tp.py:1879:1939"
```

**Metric tracking.** An `ezpz.history.History` object is created for
JSONL metric logging and optional distributed aggregation.

```python title="src/ezpz/examples/fsdp_tp.py:1950:1962"
--8<-- "src/ezpz/examples/fsdp_tp.py:1950:1962"
```

**Training loop.** Each batch is moved to device, split into
`inp`/`labels`, run through the model, and the loss is computed with
`cross_entropy`. Labels are narrowed for sequence parallelism when
needed. Gradient clipping is applied before `optimizer.step()`.

```python title="src/ezpz/examples/fsdp_tp.py:1970:2148"
--8<-- "src/ezpz/examples/fsdp_tp.py:1970:2148"
```

```python title="src/ezpz/examples/fsdp_tp.py:2067:2074"
--8<-- "src/ezpz/examples/fsdp_tp.py:2067:2074"
```

```python title="src/ezpz/examples/fsdp_tp.py:2078:2090"
--8<-- "src/ezpz/examples/fsdp_tp.py:2078:2090"
```

After each step, timing and loss metrics are collected into a dict and
passed to `history.update` and `history.log_metrics`.

```python title="src/ezpz/examples/fsdp_tp.py:2141:2148"
--8<-- "src/ezpz/examples/fsdp_tp.py:2141:2148"
```

At the end of training, activation hooks are removed, a barrier syncs
all ranks, and `history.finalize` writes the summary dataset on rank 0.

```python title="src/ezpz/examples/fsdp_tp.py:2151:2156"
--8<-- "src/ezpz/examples/fsdp_tp.py:2151:2156"
```

</details>

<details closed markdown><summary><strong><code>main</code> and Entrypoint</strong></summary>

`main` calls `ezpz.distributed.setup_torch` to initialize the
distributed backend (including TP groups), determines the output
directory, and dispatches to `train`.

```python title="src/ezpz/examples/fsdp_tp.py:2160:2198"
--8<-- "src/ezpz/examples/fsdp_tp.py:2160:2198"
```

The `if __name__ == "__main__"` block parses args, runs `main`, cleans
up distributed state, and exits.

```python title="src/ezpz/examples/fsdp_tp.py:2201:2206"
--8<-- "src/ezpz/examples/fsdp_tp.py:2201:2206"
```

</details>

## MFU Tracking

Model FLOPs are estimated on the unwrapped model **before** the 2D
FSDP+TP parallelization, and per-step **TFLOPS** and **MFU** are reported
under `train/tflops` and `train/mfu`.

The example prefers an **exact** count via
[`try_estimate_fake`](../python/Code-Reference/flops.md): it runs a
forward+backward under `FakeTensorMode` (shape-only tensors — no
allocations, so it never OOMs at the real `(batch, seq)` shape) with
`sdpa_kernel(MATH)` forced so attention decomposes into matmuls the FLOP
counter can see. If that path fails it falls back to the older
linear-scaling probe (`try_estimate` at a tiny seq, scaled by the token
ratio).

> **Why the fake-tensor path matters.** The linear-scaling probe
> under-counts the `O(seq²)` attention term — and on CPU and the fused
> SDPA backends `FlopCounterMode` reports *zero* for the attention op
> entirely, so both the probe and the real shape silently drop it. For
> agpt-xl at `seq=8192` that made reported MFU ~36% **low**. The exact
> path includes attention, so when it falls back the printed MFU is a
> *lower bound*, not an upper bound.

```python
# exact (preferred), with a graceful fallback:
_model_flops = try_estimate_fake(model, (args.batch_size, args.seq_len))
# ... per step:
metrics["train/tflops"] = _model_flops / (t2 - t0) / 1e12
metrics["train/mfu"] = compute_mfu(_model_flops, t2 - t0)
```

See [`ezpz.flops`](../python/Code-Reference/flops.md) for details.

## torch.compile

Pass `--compile` to apply `torch.compile` after the FSDP/TP wrap. The
example compiles **each `TransformerBlock` individually** (matching
torchtitan's `apply_compile`) rather than wrapping the whole model in a
single `torch.compile` call:

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp --model agpt-2b --tp 2 --compile
```

- Per-block compile dodges a Dynamo graph break that whole-model compile
  hits on the TP-wrapped `tok_embeddings` (the embedding's
  `RowwiseParallel` output transform redistributes a `_MaskPartial`
  DTensor, which Dynamo can't trace under fake tensors), and it amortizes
  compile cost across the `N` identical layers.
- `--compile-mode {default,reduce-overhead,max-autotune}` selects the
  compile mode (default: `default`).
- The example uses **FSDP2** (`fully_shard`), which has no flat
  `FlatParameter` and so no `use_orig_params` knob — the Dynamo refusal that
  FSDP1 hit there is moot. Activation checkpointing still uses the
  compile-aware `checkpoint_wrapper` (see
  [Activation checkpointing](#activation-checkpointing)).

> **Known limitation.** Combining all three of `--tp > 1`,
> `--activation-checkpoint`, and `--compile` at once trips an upstream
> AOTAutograd assertion (a `DeviceMesh` reaches the saved-tensors check).
> Drop any one of the three until it's fixed upstream. Any two together
> work.

## Matching torchtitan

This example targets parity with [torchtitan](https://github.com/pytorch/torchtitan)
on the same AuroraGPT model. On Sunspot (Intel PVC), agpt-2b at
`--batch-size 2 --seq-len 7320 --tp 1`, full-shard, 2 nodes (dp=24), the
recipe below reproduces torchtitan's throughput within noise:

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp \
  --model agpt-2b --seq-len 7320 --batch-size 2 --tp 1 \
  --sharding-strategy full_shard \
  --compile --loss-impl compiled --act-mem-budget 0.5
```

| metric | torchtitan | this example |
|---|---|---|
| MFU | ~26% | ~26% |
| TFLOP/s (per rank) | ~79 | ~77 |
| tokens/s (per rank) | ~7,300 | ~7,180 |

Three pieces close the gap, all matching what torchtitan does:

- **`--act-mem-budget 0.5`** — the decisive one for memory. torchtitan's
  `MemoryBudgetAC` sets `activation_memory_budget = 0.5`; this example
  defaulted to `1.0` (save all activations) and so OOM'd at `bs=2` where
  torchtitan fit. The budget only takes effect with `--compile` (it drives
  the inductor min-cut partitioner). `--loss-impl` alone is **not** enough —
  the eager `(B·T, vocab)` cross-entropy transient is only one contributor;
  the activation budget is what actually makes `bs=2` fit.
- **`--loss-impl compiled`** — torchtitan compiles its loss
  (`compile.components = ["model", "loss"]`); this fuses the large-vocab
  cross-entropy so its logits/grad transient never fully materializes.
- **Fused AdamW + FSDP gradient division** — the optimizer now uses
  `fused=True` with torchtitan's betas/eps/wd, and FSDP2 gradient comms force
  SUM reduction on CCL/XPU (mirroring torchtitan's
  `disable_fsdp_gradient_division`). Both are automatic — no flags.

!!! warning "Reading the memory number: allocated vs reserved"

    This example's `memory=alloc/peak` line reports **peak *allocated***
    bytes; torchtitan's `memory:` line reports **peak *reserved*** bytes,
    reset per step. They are different metrics, so don't compare them
    directly. The **allocated** peaks match (~44 GiB for this recipe vs
    torchtitan's ~44 GiB). This example's cumulative *reserved* peak looks
    higher only because it includes the one-time compile-warmup
    high-water mark that torchtitan's per-step reset discards — it is not
    extra real usage.

Verified batch/loss/budget sweep (agpt-2b, seq=8192, tp=1, full_shard,
dp=48, `--compile`):

| batch | `--loss-impl` | `--act-mem-budget` | result |
|---|---|---|---|
| 1 | eager | 1.0 | fits (peak ~41 GiB alloc) |
| 2 | eager | 1.0 | OOM in `loss.backward()` |
| 2 | compiled | 1.0 | OOM (FSDP reduce-scatter) |
| 2 | chunked | 1.0 | OOM in backward |
| 2 | compiled | 0.5 | **fits** (peak ~44 GiB alloc) |

Lower the budget further (e.g. `0.3`) to trade more recompute for less
memory if you need headroom for a bigger batch.

## `compiled` vs `fused-linear` benchmark (agpt-2b, 128K vocab)

A head-to-head of the two production-scale `--loss-impl` modes on Sunspot
(Intel PVC). To keep the run in a regime where both modes comfortably fit —
so the comparison measures *speed vs memory*, not who hits the OOM cliff
first — the agpt-2b architecture is paired with the **`meta-llama/Llama-3.2-1B`
tokenizer**, which drops the vocab from agpt's native 256,128 to **128,000**
(the `(B·T, vocab)` logits/grad buffers roughly halve). The example does this
override automatically: pass a real HF dataset + `--tokenizer_name` and the
tokenizer's `vocab_size` replaces the preset's (logged as `Overriding
vocab_size from 256128 to tokenizer vocab_size=128000`).

Setup: agpt-2b (`dim=2048`, `n_layers=12`, `hidden_dim=11008`),
`--tokenizer_name meta-llama/Llama-3.2-1B` (vocab 128,000), `--seq-len 8192`,
`--batch-size 2`, `--tp 1`, `--sharding-strategy full_shard`, 2× Sunspot nodes
(dp=24), dataset `eliplutchok/fineweb-small-sample`. Each config runs ~31
iters; metrics are the steady-state mean over 26 iters (warmup + the final
eval-pass step dropped).

```bash
# compiled (fastest that fits)
ezpz launch python3 -m ezpz.examples.fsdp_tp \
  --model agpt-2b --tokenizer_name meta-llama/Llama-3.2-1B \
  --seq-len 8192 --batch-size 2 --tp 1 --sharding-strategy full_shard \
  --compile --loss-impl compiled --act-mem-budget 0.5

# fused-linear (lowest memory)
ezpz launch python3 -m ezpz.examples.fsdp_tp \
  --model agpt-2b --tokenizer_name meta-llama/Llama-3.2-1B \
  --seq-len 8192 --batch-size 2 --tp 1 --sharding-strategy full_shard \
  --loss-impl fused-linear
```

| metric | `compiled` | `fused-linear` | Δ (fused vs compiled) |
|---|---|---|---|
| MFU | **33.4 %** | 28.8 % | −13.8 % |
| TFLOP/s (per rank) | **99.5** | 85.8 | −13.8 % |
| tokens/s (global) | **248,400** | 214,200 | −13.8 % |
| tokens/s (per GPU) | 10,351 | 8,926 | −13.8 % |
| step time | 1.58 s | 1.84 s | +16 % |
| memory (peak reserved) | 28.64 GiB | 29.10 GiB | +1.6 % |
| memory (allocated) | 8.76 GiB | **1.50 GiB** | **−83 %** |

Takeaways:

- **`compiled` is ~14% faster** on every throughput metric. Compiling the
  block + the fused cross-entropy keeps the GPU busy; this is the mode to use
  when it fits.
- **`fused-linear`'s win is allocated memory: 1.50 GiB vs 8.76 GiB (−83%).**
  It runs the output projection per row-chunk so the full `(B·T, vocab)`
  logits/grad are never materialized — exactly its design goal. The *reserved*
  peak looks ~equal only because at 128K vocab the activation high-water mark
  dominates the allocator arena; the headroom shows up in **allocated** bytes,
  which is what lets you push a bigger batch or sequence before OOM.
- Net guidance: **`compiled` for raw speed when it fits; `fused-linear` for
  memory headroom.** At this 128K-vocab / seq-8192 operating point both fit
  easily and `compiled` wins on speed. `fused-linear`'s advantage widens as
  vocab × seq grows and `compiled` approaches the OOM cliff (cf. agpt's native
  256K vocab, where the eager/compiled transient is the binding constraint).

> Loss reads `nan` during these runs — expected, since the model is
> randomly initialized with no real LR schedule. The benchmark measures
> throughput and memory only, not convergence.

## Help

<details closed><summary><code>--help</code></summary>

```bash
usage: fsdp_tp.py [-h] [--dim DIM] [--n-layers N_LAYERS]
                  [--n-heads N_HEADS]
                  [--n-kv-heads N_KV_HEADS]
                  [--multiple-of MULTIPLE_OF]
                  [--ffn-dim-multiplier FFN_DIM_MULTIPLIER]
                  [--hidden-dim HIDDEN_DIM]
                  [--rope-theta ROPE_THETA]
                  [--norm-eps NORM_EPS]
                  [--vocab-size VOCAB_SIZE] [--lr LR]
                  [--epochs EPOCHS]
                  [--batch-size BATCH_SIZE]
                  [--model MODEL]
                  [--test-batch-size TEST_BATCH_SIZE]
                  [--num-workers NUM_WORKERS]
                  [--seed SEED] [--tp TP]
                  [--sharding-strategy {no_shard,full_shard,shard_grad_op,hybrid_shard,hybrid_shard_zero2}]
                  [--activation-checkpoint {none,block,full,selective}]
                  [--max-grad-norm MAX_GRAD_NORM]
                  [--outdir OUTDIR] [--dataset DATASET]
                  [--tokenizer_name TOKENIZER_NAME]
                  [--hf-split HF_SPLIT]
                  [--hf-text-column HF_TEXT_COLUMN]
                  [--hf-limit HF_LIMIT]
                  [--seq-len SEQ_LEN]
                  [--max-seq-len MAX_SEQ_LEN] [--fp32]
                  [--compile]
                  [--compile-mode {default,reduce-overhead,max-autotune}]
                  [--act-mem-budget ACT_MEM_BUDGET]
                  [--loss-impl {eager,chunked,chunked-backward,compiled,fused-linear,loss-parallel}]
                  [--loss-chunk-size LOSS_CHUNK_SIZE]

2D Parallel Training

options:
  -h, --help            show this help message and exit
  --dim DIM             Model hidden / embedding dimension (a.k.a. d_model).
                        Overridden when --model selects a preset. (default:
                        256)
  --n-layers N_LAYERS   Number of TransformerBlocks stacked in the model.
                        Overridden when --model selects a preset. (default:
                        32)
  --n-heads N_HEADS     Number of attention heads per layer. Must divide
                        --dim. Overridden when --model selects a preset.
                        (default: 32)
  --n-kv-heads N_KV_HEADS
                        Number of key/value heads for grouped-query attention
                        (GQA). Must divide --n-heads. Set equal to --n-heads
                        for standard MHA. Overridden when --model selects a
                        preset. (default: 4)
  --multiple-of MULTIPLE_OF
                        Round the SwiGLU FFN hidden dim up to a multiple of
                        this value (for hardware-friendly shapes). Ignored
                        when --hidden-dim is set explicitly. (default: 360)
  --ffn-dim-multiplier FFN_DIM_MULTIPLIER
                        Scale factor applied to the SwiGLU FFN hidden dim
                        before the --multiple-of rounding step. None (default)
                        means no extra scaling; Llama2-style models use 1.3.
                        Ignored when --hidden-dim is set explicitly. (default:
                        None)
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
  --norm-eps NORM_EPS   Epsilon added to RMSNorm denominators for numerical
                        stability. (default: 1e-05)
  --vocab-size VOCAB_SIZE
                        Tokenizer vocabulary size. Sets the embedding table
                        and output projection sizes; must match the tokenizer
                        used for the dataset. (default: 32000)
  --lr LR               Peak learning rate for the AdamW optimizer. (default:
                        0.003)
  --epochs EPOCHS       Number of passes over the training dataset. (default:
                        5)
  --batch-size BATCH_SIZE
                        Per-DP-rank training batch size (a.k.a. micro-batch).
                        Global batch = --batch-size * (world_size / --tp).
                        (default: 1)
  --model MODEL         Model size preset (overrides dim/layer defaults).
                        Presets:
                        debug/small/medium/large/xl/xxl/xxxl/agpt-2b/agpt-20b.
                        xl/xxl/xxxl accept long-form aliases (`xlarge`/`extra-
                        large`, etc). agpt presets accept `agpt2b`/`agpt_2b`
                        etc. Pass a HuggingFace repo id with a `/` (e.g.
                        `meta-llama/Llama-3.2-1B`) to load HF weights instead
                        — that path forces --tp 1 (FSDP-only). (default: None)
  --test-batch-size TEST_BATCH_SIZE
                        Per-DP-rank batch size for the eval/test loader. Only
                        consumed by the MNIST data path; ignored for random
                        and HF datasets. (default: 1000)
  --num-workers NUM_WORKERS
                        Subprocess workers for the DataLoader. 0 (default)
                        loads in-process — fine for tokenized HF datasets;
                        bump for image pipelines or heavy on-the-fly
                        preprocessing. (default: 0)
  --seed SEED           Seed for torch/numpy/python RNGs (forwarded to
                        ezpz.setup_torch). None (default) leaves the RNGs
                        unseeded for non-deterministic runs. (default: None)
  --tp TP               Tensor-parallel degree (a.k.a. TP / Megatron-style
                        sharding). Must divide WORLD_SIZE. The remaining
                        dimension (WORLD_SIZE / --tp) is used for FSDP data
                        parallelism. Set to 1 for FSDP-only. Forced to 1 when
                        --model is a HF repo id. (default: 2)
  --sharding-strategy {no_shard,full_shard,shard_grad_op,hybrid_shard,hybrid_shard_zero2}
                        FSDP sharding behavior. The model uses FSDP2
                        (`fully_shard`), which controls sharding via
                        `reshard_after_forward` rather than a strategy enum;
                        the legacy FSDP1 names below are kept as the CLI
                        surface and mapped to the nearest FSDP2 policy.
                        `full_shard` (default, ZeRO-3): reshard params after
                        forward — lowest memory, params re-all-gathered in
                        backward. `shard_grad_op` (ZeRO-2): keep params
                        unsharded after forward — more memory, avoids the
                        backward all-gather. `no_shard`: mapped to keep-
                        unsharded (FSDP2 has no true per-module no-shard).
                        `hybrid_shard`/`hybrid_shard_zero2`: FSDP1 hybrid has
                        no one-flag FSDP2 equivalent — `hybrid_shard` maps to
                        reshard-after-forward, `hybrid_shard_zero2` to keep-
                        unsharded (mirroring their ZeRO-3/ZeRO-2 variants).
                        (default: full_shard)
  --activation-checkpoint, --ac {none,block,full,selective}
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
                        time OOM consider increasing --tp or reducing --seq-
                        len). NOTE: cannot be combined with --compile
                        (upstream AOTAutograd DeviceMesh-in-saved-tensors bug
                        — see the --compile warning). With FSDP2 you usually
                        don't need --ac anyway; it was a workaround for the
                        FSDP1 backward-memory OOM that FSDP2 fixes. (default:
                        none)
  --max-grad-norm MAX_GRAD_NORM
                        Clip gradients to this L2 norm before the optimizer
                        step. Set to 0 (or negative) to disable gradient
                        clipping. (default: 1.0)
  --outdir OUTDIR       Base directory for checkpoints and logs. None
                        (default) writes under the current working directory.
                        (default: None)
  --dataset DATASET     Training dataset. Special values: `mnist` (image debug
                        dataset) and `random` (synthetic tokens, no IO).
                        Anything else is treated as a HuggingFace dataset repo
                        id. (default: eliplutchok/fineweb-small-sample)
  --tokenizer_name TOKENIZER_NAME
                        HuggingFace tokenizer repo id used to tokenize the HF
                        dataset. Auto-overridden to --model when --model is a
                        HF repo id and --tokenizer_name wasn't passed
                        explicitly. (default: meta-llama/llama-2-7b-hf)
  --hf-split, --hf_split HF_SPLIT
                        Dataset split to load. (default: train)
  --hf-text-column, --hf_text_column HF_TEXT_COLUMN
                        Column containing raw text in the dataset. (default:
                        text)
  --hf-limit, --hf_limit HF_LIMIT
                        Maximum number of rows to sample from the HF dataset.
                        0 (default) = no limit (use the full dataset). Pass a
                        positive value (e.g. `--hf-limit 512`) to subsample
                        for smoke tests. Subsampling is deterministic given
                        $EZPZ_HF_SAMPLE_SEED. (default: 0)
  --seq-len SEQ_LEN     Training sequence length (tokens per sample). Defaults
                        to $SEQ_LEN if set, otherwise 1024. Must be <= --max-
                        seq-len. (default: 1024)
  --max-seq-len MAX_SEQ_LEN
                        Maximum sequence length the model is built to support
                        — sets the RoPE frequency table size and the attention
                        scratch budget. Increase if you raise --seq-len.
                        (default: 32768)
  --fp32                Disable mixed precision (use fp32) for debugging NaNs.
                        (default: False)
  --compile             Compile each TransformerBlock with torch.compile after
                        FSDP/TP wrap (matches torchtitan's apply_compile
                        pattern). Per-block compile dodges the Dynamo +
                        DTensor _MaskPartial graph break that whole-model
                        compile hits on TP-wrapped tok_embeddings, and
                        amortizes compile cost across N layers. (default:
                        False)
  --compile-mode {default,reduce-overhead,max-autotune}
                        torch.compile mode (only used when --compile is set).
                        `default` is safest. `reduce-overhead` enables
                        cudagraphs for small models / large batches. `max-
                        autotune` does extensive kernel search — slow startup,
                        fastest steady state. (default: default)
  --act-mem-budget ACT_MEM_BUDGET
                        Activation-memory budget for the inductor min-cut
                        partitioner (sets
                        torch._functorch.config.activation_memory_budget).
                        Only takes effect with --compile. 1.0 (default) saves
                        ALL activations (no recompute); lower values let the
                        compiler recompute activations in backward to cut peak
                        memory — e.g. 0.5 keeps ~half. This is how torchtitan
                        fits larger batches for the same model (its
                        MemoryBudgetAC sets 0.5). Try 0.5 if you OOM in
                        backward at a batch size that should fit. (default:
                        1.0)
  --loss-impl {eager,chunked,chunked-backward,compiled,fused-linear,loss-parallel}
                        Cross-entropy implementation. `eager` (default) is the
                        plain F.cross_entropy over the full (B*T, vocab)
                        logits — simplest, but at large vocab (e.g. agpt's
                        256K) and long seq it materializes a multi-GB fp32
                        logits+grad transient in loss.backward() that can OOM
                        a GPU tile (UR_RESULT_ERROR_OUT_OF_RESOURCES) even
                        when the model itself fits. `chunked` computes CE over
                        row-chunks (see --loss-chunk-size) so only one chunk's
                        logits/grad exist at once — pure eager, no
                        torch.compile needed. `compiled` wraps CE in
                        torch.compile so inductor fuses
                        log_softmax+NLL+backward and never materializes the
                        full transient (torchtitan's approach). NOTE:
                        `--compile` only compiles the transformer blocks, NOT
                        the loss, so it does NOT fix this — use --loss-impl
                        for the loss transient. (default: eager)
  --loss-chunk-size LOSS_CHUNK_SIZE
                        Row-chunk size for --loss-impl=chunked (number of
                        (B*T) token rows per cross-entropy chunk). Smaller =
                        lower peak memory, more kernel launches. Ignored for
                        other --loss-impl. (default: 1024)
```

</details>

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
$ ezpz launch python3 -m ezpz.examples.fsdp_tp
```

/embed `ezpz-fsdp-tp.html`


</details>
