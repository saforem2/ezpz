# `ezpz.data`

- See [ezpz/data](https://github.com/saforem2/ezpz/tree/main/src/ezpz/data)

Data loading utilities for distributed and tensor-parallel training. Provides
specialized DataLoaders, iterable datasets, and integrations with HuggingFace
datasets.

## TP-Aware Data Loading

### `TPBroadcastDataLoader`

Wraps an existing DataLoader to broadcast batches from the TP group leader to
all TP ranks, ensuring all ranks in a tensor-parallel group see the same data:

```python
from ezpz.data.distributed import TPBroadcastDataLoader

# Wrap an existing dataloader for TP-aware broadcasting
tp_loader = TPBroadcastDataLoader(dataloader, tp_group=tp_group)

for batch in tp_loader:
    output = model(batch)
```

## Distributed Datasets

### `DataParallelIterableDataset`

An `IterableDataset` that automatically shards data across data-parallel ranks:

```python
from ezpz.data.hf import DataParallelIterableDataset

dataset = DataParallelIterableDataset(hf_dataset)
for sample in dataset:
    process(sample)
```

### `ConstantLengthDataset`

An `IterableDataset` that packs tokenized text into constant-length sequences,
useful for language model training:

```python
from ezpz.data.text import ConstantLengthDataset

dataset = ConstantLengthDataset(
    tokenizer=tokenizer,
    dataset=hf_dataset,
    seq_length=2048,
    num_of_sequences=1024,
)

for batch in dataset:
    output = model(batch)
```

## LLaMA Data Loading

### `LlamaDataLoader`

A DataLoader tailored for LLaMA-style language model training with HuggingFace
datasets:

```python
from ezpz.data.llama import LlamaDataLoader

loader = LlamaDataLoader(
    dataset_repo="wikitext",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    max_length=2048,
    batch_size=4,
)
dataloader = loader.get_data_loader()
```

## Vision Datasets

Pre-configured loaders for common vision datasets with distributed samplers:

??? example "MNIST"

    ```python
    from ezpz.data.vision import get_mnist

    data = get_mnist(
        train_batch_size=64,
        test_batch_size=1000,
        num_replicas=world_size,
        rank=rank,
    )
    train_loader = data["training"]["dataloader"]
    test_loader = data["testing"]["dataloader"]
    ```

??? example "FakeImageDataset"

    ```python
    from ezpz.data.vision import FakeImageDataset

    dataset = FakeImageDataset(size=1000, dtype=torch.float32)
    ```

## HuggingFace Integration

### `get_hf_datasets()`

Full HuggingFace dataset pipeline with tokenization and grouping:

```python
from ezpz.data.hf import get_hf_datasets

datasets = get_hf_datasets(
    data_args=data_args,
    tokenizer_name="gpt2",
    model_name_or_path="gpt2",
)
```

### `load_hf_texts()`

Load a limited number of text samples from a HuggingFace dataset:

```python
from ezpz.data.hf import load_hf_texts

texts = load_hf_texts(
    dataset_name="wikitext",
    split="train",
    text_column="text",
    limit=1000,
)
```

::: ezpz.data
