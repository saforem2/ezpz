# Train Transformer with FSDP and TP on HF Datasets

FSDP Example with Tensor Parallelism

See:

- 📘 [examples/FSDP TP](../python/Code-Reference/examples/fsdp_tp.md)
- 🐍 [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp \
    --tp=2 \
    --epochs=5 \
    --batch-size=2 \
    --dataset=eliplutchok/fineweb-small-sample \
```

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

<details closed><summary>Output on Sunspot:</summary>

```bash

```

</details>
