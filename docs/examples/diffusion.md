# Train Diffusion LLM with FSDP on HF Datasets

See:

- 📘 [examples/Diffusion](../python/Code-Reference/examples/diffusion.md)
- 🐍 [src/ezpz/examples/diffusion.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py)

```bash
ezpz launch python3 -m ezpz.examples.diffusion --batch_size 1 --hf_dataset stanfordnlp/imdb
```

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.diffusion --help
usage: diffusion.py [-h] [--batch-size BATCH_SIZE] [--dtype DTYPE]
                    [--extra-text [EXTRA_TEXT ...]] [--fsdp]
                    [--fsdp-mixed-precision] [--hidden HIDDEN]
                    [--hf-dataset HF_DATASET] [--hf-split HF_SPLIT]
                    [--hf-text-column HF_TEXT_COLUMN] [--hf-limit HF_LIMIT]
                    [--log_freq LOG_FREQ] [--outdir OUTDIR]
                    [--samples SAMPLES] [--seed SEED] [--seq-len SEQ_LEN]
                    [--timesteps TIMESTEPS] [--train-steps TRAIN_STEPS]
                    [--lr LR]

Tiny diffusion example for text generation.

options:
-h, --help            show this help message and exit
--batch-size BATCH_SIZE
--dtype DTYPE
--extra-text [EXTRA_TEXT ...]
                        Additional sentences to add to the tiny corpus.
--fsdp                Enable FSDP wrapping (requires WORLD_SIZE>1 and
                        torch.distributed init).
--fsdp-mixed-precision
                        Use bfloat16 parameters with FSDP for speed (defaults
                        to float32).
--hidden HIDDEN
--hf-dataset HF_DATASET
                        Optional Hugging Face dataset name (e.g., 'ag_news').
                        When set, replaces the toy corpus.
--hf-split HF_SPLIT   Dataset split to load.
--hf-text-column HF_TEXT_COLUMN
                        Column containing raw text in the dataset.
--hf-limit HF_LIMIT   Number of rows to sample from the HF dataset for quick
                        experiments.
--log_freq LOG_FREQ
--outdir OUTDIR
--samples SAMPLES
--seed SEED
--seq-len SEQ_LEN
--timesteps TIMESTEPS
--train-steps TRAIN_STEPS
--lr LR
```

</details>
