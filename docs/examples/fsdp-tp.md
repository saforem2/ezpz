# Train Transformer with FSDP and TP on HF Datasets

FSDP Example with Tensor Parallelism

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

## Code Walkthrough

### Device Mesh and 2D Parallelism

A 2D device mesh separates TP (within-node) from FSDP (across-node)
parallelism. From the source:

```python title="src/ezpz/examples/fsdp_tp.py"
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)
```

The mesh is initialized with `("dp", "tp")` dimensions, then attention
and FFN layers are parallelized along the TP axis while FSDP shards
across the DP axis.

### Parallel Layout

The grid below shows how GPUs are organized. Each **row** is a TP group
(within one host), and each **column** is an FSDP group (across hosts):

```
┌──────────┬──────────┬──────────┬──────────┐
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │  ← Host 1 (TP group)
├──────────┼──────────┼──────────┼──────────┤
│  GPU 4   │  GPU 5   │  GPU 6   │  GPU 7   │  ← Host 2 (TP group)
├──────────┼──────────┼──────────┼──────────┤
│  GPU 8   │  GPU 9   │  GPU 10  │  GPU 11  │  ← Host 3 (TP group)
└──────────┴──────────┴──────────┴──────────┘
     ↑          ↑          ↑          ↑
   FSDP       FSDP       FSDP       FSDP
  group 0    group 1    group 2    group 3
```

## Help

<details closed><summary><code>--help</code></summary>

```bash
usage: fsdp_tp.py [-h] [--dim DIM] [--n-layers N_LAYERS] [--n-heads N_HEADS]
                [--n-kv-heads N_KV_HEADS] [--multiple-of MULTIPLE_OF]
                [--ffn-dim-multiplier FFN_DIM_MULTIPLIER]
                [--norm-eps NORM_EPS] [--vocab-size VOCAB_SIZE]
                [--seq-length SEQ_LENGTH] [--lr LR] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE]
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
                        Dataset split to load.
--hf-text-column HF_TEXT_COLUMN, --hf_text_column HF_TEXT_COLUMN
                        Column containing raw text in the dataset.
--hf-limit HF_LIMIT, --hf_limit HF_LIMIT
                        Number of rows to sample from the HF dataset for
                        quick experiments.
--seq-len SEQ_LEN
--max-seq-len MAX_SEQ_LEN
--depth-init DEPTH_INIT
--fp32                Disable mixed precision (use fp32) for debugging NaNs.
```

</details>

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
$ ezpz launch python3 -m ezpz.examples.fsdp_tp
```

/embed `ezpz-fsdp-tp.html`


</details>
