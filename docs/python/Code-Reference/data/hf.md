# `ezpz.data.hf`

- See [ezpz/data/`hf.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/data/hf.py)

::: ezpz.data.hf

## Remote streaming via `globusfs`

`storage_options` on `HfDataTrainingArguments` is forwarded to every
`datasets.load_dataset(..., storage_options=...)` call. This lets any
`fsspec` filesystem back the streaming path — including `globus://` via
[`globusfs`](https://github.com/saforem2/globusfs).

```python
from ezpz.configs import HfDataTrainingArguments
from ezpz.data.hf import get_hf_datasets

# Public collection — no auth
args = HfDataTrainingArguments(
    train_file="globus://isaac/ability/ALL_2007-01.parquet",
    streaming=True,
    storage_options={
        "collection_id": "isaac",
        "https_url": "https://g-05a4b6.2d513.8443.data.globus.org",
    },
)
datasets = get_hf_datasets(args, tokenizer_name="gpt2", model_name_or_path="gpt2")

# Authenticated (Eagle) — log in once, share token_path
# python -c "import globusfs; globusfs.login('alcf#dtn_eagle', token_path='/eagle/projects/MyProj/.globus-tokens.json')"
args = HfDataTrainingArguments(
    train_file="globus://05d2c76a-e867-4f67-aa57-76edeb0beda0/my-dataset/train-00000.parquet",
    streaming=True,
    storage_options={"token_path": "/eagle/projects/MyProj/.globus-tokens.json"},
)
```

The CLI form accepts a JSON string:

```bash
--storage_options '{"collection_id": "isaac", "https_url": "https://g-05a4b6.2d513.8443.data.globus.org"}'
```

Also works for `s3://`, `gs://`, or any other `fsspec` protocol — same
field, different keys. See the [LLM fine-tuning guide](../../guides/llm-finetuning.md#streaming-from-remote-storage-via-globusfs)
for when to stream vs. stage.
