# `ezpz.configs`

- See [ezpz/`configs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/configs.py)

Configuration dataclasses and utility functions for training, DeepSpeed, and
HuggingFace integration.

## `TrainConfig`

High-level training configuration:

```python
from ezpz.configs import TrainConfig

config = TrainConfig(
    seed=42,
    dtype="bf16",
    use_wandb=True,
    wandb_project_name="my-project",
    ds_config_path="./ds_config.json",
)
```

| Field                 | Type            | Default    | Description                          |
| --------------------- | --------------- | ---------- | ------------------------------------ |
| `gas`                 | `int`           | `1`        | Gradient accumulation steps          |
| `use_wandb`           | `bool`          | `False`    | Enable Weights & Biases logging      |
| `seed`                | `int \| None`   | `None`     | Random seed                          |
| `port`                | `str \| None`   | `None`     | Rendezvous port                      |
| `dtype`               | `Any \| None`   | `None`     | Data type for training               |
| `load_from`           | `str \| None`   | `None`     | Path to load checkpoint from         |
| `save_to`             | `str \| None`   | `None`     | Path to save checkpoint to           |
| `ds_config_path`      | `str \| None`   | `None`     | Path to DeepSpeed config             |
| `wandb_project_name`  | `str \| None`   | `None`     | W&B project name                     |
| `ngpus`               | `int \| None`   | `None`     | Number of GPUs to use                |

Unknown keyword arguments passed to the constructor are captured in
`self.extras` rather than raising an error.

## `ZeroConfig`

DeepSpeed ZeRO optimizer configuration with all ZeRO stage options:

```python
from ezpz.configs import ZeroConfig

zero = ZeroConfig(stage=2, overlap_comm=True, contiguous_gradients=True)
```

## HuggingFace Configs

### `HfModelArguments`

Configuration for HuggingFace model loading:

```python
from ezpz.configs import HfModelArguments

model_args = HfModelArguments(
    model_name_or_path="gpt2",
    torch_dtype="bfloat16",
    use_fast_tokenizer=True,
)
```

### `HfDataTrainingArguments`

Configuration for HuggingFace dataset loading and preprocessing:

```python
from ezpz.configs import HfDataTrainingArguments

data_args = HfDataTrainingArguments(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    block_size=1024,
)
```

## Vision Transformer Configs

### `ViTConfig`

Standard Vision Transformer configuration:

```python
from ezpz.configs import ViTConfig

vit = ViTConfig(
    img_size=224,
    patch_size=16,
    depth=12,
    num_heads=12,
    hidden_dim=768,
    num_classes=10,
)
```

### `timmViTConfig`

Timm-compatible Vision Transformer configuration with additional training
parameters:

```python
from ezpz.configs import timmViTConfig

vit = timmViTConfig(batch_size=128, head_dim=64)
```

## `TrainArgs`

Training hyperparameters dataclass used by the example scripts:

```python
from ezpz.configs import TrainArgs

args = TrainArgs(
    batch_size=32,
    max_iters=1000,
    fsdp=True,
    dtype="bf16",
    compile=True,
)
```

## Utility Functions

### `get_scheduler()`

Detect the active job scheduler from environment variables:

```python
from ezpz.configs import get_scheduler

scheduler = get_scheduler()  # Returns "PBS", "SLURM", or falls back to hostname-based detection
```

### `print_config_tree()`

Display an OmegaConf `DictConfig` as a rich tree in the terminal:

```python
from ezpz.configs import print_config_tree

print_config_tree(cfg, resolve=True, style="tree")
```

::: ezpz.configs
    options:
      filters:
        - "!BaseConfig"
