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

`MODEL_PRESETS` defines canned configurations (`debug`, `small`,
`medium`, `large`) that override default CLI values for quick
experimentation.

```python title="src/ezpz/examples/fsdp_tp.py:172:213"
--8<-- "src/ezpz/examples/fsdp_tp.py:172:213"
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
_model_flops = try_estimate(model, (args.batch_size, args.seq_length))
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
                  [--norm-eps NORM_EPS] [--vocab-size VOCAB_SIZE]
                  [--seq-length SEQ_LENGTH] [--lr LR] [--epochs EPOCHS]
                  [--batch-size BATCH_SIZE]
                  [--model {debug,large,medium,small}]
                  [--test-batch-size TEST_BATCH_SIZE]
                  [--num-workers NUM_WORKERS] [--seed SEED] [--tp TP]
                  [--sharding-strategy SHARDING_STRATEGY]
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
  --norm-eps NORM_EPS
  --vocab-size VOCAB_SIZE
  --seq-length SEQ_LENGTH
  --lr LR
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  --model {debug,large,medium,small}
                        Model size preset (overrides dim/layer defaults)
                        (default: None)
  --test-batch-size TEST_BATCH_SIZE
  --num-workers NUM_WORKERS
  --seed SEED
  --tp TP
  --sharding-strategy SHARDING_STRATEGY
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
                        Number of rows to sample from the HF dataset for quick
                        experiments. (default: 512)
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
