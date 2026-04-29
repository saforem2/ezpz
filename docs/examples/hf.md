# Fine-Tune a Causal LM (HuggingFace + Accelerate)

Use this example when you want to fine-tune a pretrained HuggingFace language
model with full control over the training loop. Unlike the `hf_trainer`
example which uses the HF Trainer abstraction, this writes the loop
explicitly — giving you control over gradient accumulation, custom
evaluation, and learning rate scheduling while still using Accelerate for
distributed coordination and ezpz for setup and metrics.

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`History`][ezpz.history.History] — Track training and eval metrics
    - [`setup_wandb()`][ezpz.distributed.setup_wandb] — W&B integration
    - [`HfModelArguments`][ezpz.configs.HfModelArguments] / [`HfDataTrainingArguments`][ezpz.configs.HfDataTrainingArguments] — Config dataclasses

See:

- 📘 [examples/hf](../python/Code-Reference/examples/hf.md)
- 🐍 [src/ezpz/examples/hf.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf.py)

```bash
ezpz launch python3 -m ezpz.examples.hf \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2 \
    --output_dir ./output-hf
```

## Source

<details closed><summary><code>src/ezpz/examples/hf.py</code></summary>

```python title="src/ezpz/examples/hf.py"
--8<-- "src/ezpz/examples/hf.py"
```

</details>

## Code Walkthrough


<details closed markdown><summary><strong>Imports</strong></summary>

Standard library, HuggingFace, and ezpz imports. The `Accelerator` and
`FullyShardedDataParallelPlugin` from `accelerate` are loaded in a
try/except so the error message is clear if the package is missing.

```python title="src/ezpz/examples/hf.py" linenums="1"
#!/usr/bin/env python
"""
Fine-tune a causal LM with a hand-rolled training loop.

This mirrors the dataset/model setup used in ``ezpz.examples.hf_trainer`` while
keeping an explicit training loop like the other examples.
"""

from __future__ import annotations

# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false

import json
import math
import os
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Optional, cast

import datasets
import torch
import transformers
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import ezpz
from ezpz.configs import HfDataTrainingArguments, HfModelArguments

logger = ezpz.get_logger(__name__)

try:
    from accelerate import Accelerator, FullyShardedDataParallelPlugin  # noqa: E402 type:ignore
    from accelerate.utils import set_seed
except ImportError as exc:
    logger.error(
        "Please install accelerate to run this script: `pip install accelerate`"
    )
    raise exc

require_version(
    "datasets>=2.14.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
```

</details>

<details closed markdown><summary><strong><code>parse_args</code></strong></summary>

Uses HuggingFace's `HfArgumentParser` to parse three dataclass groups in
one pass. Supports loading arguments from a JSON file when the sole CLI
argument is a `.json` path. Logging verbosity is restricted to rank 0.

```python title="src/ezpz/examples/hf.py" linenums="61"
def parse_args(
    ) -> tuple[HfModelArguments, HfDataTrainingArguments, TrainingArguments]:
    """Parse Hugging Face model, data, and training arguments.

    Returns:
        Mapping with ``model``, ``data``, and ``training`` argument objects.
    """
    parser = HfArgumentParser(
        (HfModelArguments, HfDataTrainingArguments, TrainingArguments)  # type:ignore
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = (
            parser.parse_args_into_dataclasses()
        )

    if training_args.should_log:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_info()

    rank = ezpz.get_rank()
    log_level_info = 20
    log_level_critical = 50
    log_level = log_level_info if rank == 0 else log_level_critical
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if rank == 0:
        logger.info("Training/evaluation parameters %s", training_args)

    return model_args, data_args, training_args
```

</details>

<details closed markdown><summary><strong><code>split_dataset</code></strong></summary>

Loads a named dataset from the HuggingFace Hub, splitting it into
train/validation by percentage. Falls back gracefully when the requested
split syntax is not supported by the dataset.

```python title="src/ezpz/examples/hf.py" linenums="100"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def split_dataset(
    model_args: HfModelArguments,
    data_args: HfDataTrainingArguments,
    train_split_name: str = "train",
    validation_split_name: Optional[str] = None,
) -> datasets.IterableDatasetDict | datasets.DatasetDict:
    """Split a Hugging Face dataset into train/validation splits.

    Args:
        model_args: Model configuration arguments for cache/token settings.
        data_args: Data-related arguments for dataset selection.
        train_split_name: Name of the training split.
        validation_split_name: Name of the validation split (if any).

    Returns:
        Dataset dictionary with requested splits.
    """
    dataset_name = data_args.dataset_name
    assert dataset_name is not None, (
        "dataset_name must be provided to split the dataset."
    )
    dsets: dict[str, datasets.Dataset | datasets.IterableDataset] = {}
    if validation_split_name is not None:
        try:
            dsets[validation_split_name] = datasets.load_dataset(  # type:ignore
                dataset_name,
                data_args.dataset_config_name,
                split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            dsets[train_split_name] = datasets.load_dataset(  # type: ignore
                dataset_name,
                data_args.dataset_config_name,
                split=f"{train_split_name}[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
        except ValueError:
            dsets[validation_split_name] = datasets.load_dataset(  # type:ignore
                dataset_name,
                data_args.dataset_config_name,
                split=train_split_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            try:
                dsets[train_split_name] = datasets.load_dataset(  # type:ignore
                    dataset_name,
                    data_args.dataset_config_name,
                    split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
            except Exception:
                dsets[train_split_name] = datasets.load_dataset(  # type:ignore
                    dataset_name,
                    data_args.dataset_config_name,
                    split=train_split_name,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )

    if data_args.streaming:
        return datasets.IterableDatasetDict(  # type: ignore
            cast(dict[str, datasets.IterableDataset], dsets)
        )
    return datasets.DatasetDict(  # type: ignore
        cast(dict[str, datasets.Dataset], dsets)
    )
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Distributed Setup</strong></summary>

`setup_torch()` initializes the process group and returns the local rank.
Arguments are parsed, and the output directory is established.

```python title="src/ezpz/examples/hf.py" linenums="183"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def main() -> None:
    """Entrypoint for standalone HF causal LM fine-tuning without Trainer."""
    t0 = time.perf_counter()
    rank = ezpz.setup_torch()
    model_args, data_args, training_args = parse_args()

    output_dir = training_args.output_dir or os.getcwd()
    wandb = None
    report_to = training_args.report_to
```

</details>

<details closed markdown><summary><strong><code>main</code> -- FSDP Plugin</strong></summary>

When `ACCELERATE_USE_FSDP=true`, an explicit `FullyShardedDataParallelPlugin`
is constructed with a bf16 mixed-precision policy, bypassing Accelerate's
env-var machinery which can pick up stale defaults.

```python title="src/ezpz/examples/hf.py" linenums="196"
    fsdp_plugin = None
    use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true"
    if use_fsdp:
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            MixedPrecision,
            ShardingStrategy,
        )

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ) if training_args.bf16 else None
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision_policy=mp_policy,
            use_orig_params=True,
            sync_module_states=False,
            cpu_ram_efficient_loading=False,
            limit_all_gathers=True,
        )
        logger.info("[rank %d] using explicit FSDP plugin: %s", rank, fsdp_plugin)
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Accelerator</strong></summary>

The `Accelerator` is created with gradient accumulation and the optional
FSDP plugin. W&B logging is handled separately via `ezpz.setup_wandb()`,
so it is not passed to Accelerate.

```python title="src/ezpz/examples/hf.py" linenums="221"
    # Don't let Accelerator manage wandb — we handle it via ezpz.setup_wandb()
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
    )
    t_setup = time.perf_counter()
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Dataset Loading</strong></summary>

Datasets are loaded from the Hub via `split_dataset` (or from local
files), then the text column is identified for tokenization.

```python title="src/ezpz/examples/hf.py" linenums="280"
    train_split_name = data_args.train_split_name or "train"
    validation_split_name = data_args.validation_split_name or "validation"
    if data_args.dataset_name is not None:
        raw_datasets = split_dataset(
            model_args,
            data_args,
            train_split_name=train_split_name,
            validation_split_name=validation_split_name,
        )
    else:
        data_files: dict[str, str] = {}
        dataset_args: dict[str, object] = {}
        if data_args.train_file is not None:
            data_files[train_split_name] = data_args.train_file
        if data_args.validation_file is not None:
            data_files[validation_split_name] = data_args.validation_file
        if data_args.train_file is not None:
            extension = data_args.train_file.split(".")[-1]
        elif data_args.validation_file is not None:
            extension = data_args.validation_file.split(".")[-1]
        else:
            raise ValueError("Expected a train or validation file.")
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = datasets.load_dataset(  # type: ignore[arg-type]
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        if validation_split_name not in raw_datasets.keys():
            raw_datasets[validation_split_name] = datasets.load_dataset(  # type:ignore
                extension,
                data_files=data_files,
                split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets[train_split_name] = datasets.load_dataset(  # type:ignore
                extension,
                data_files=data_files,
                split=f"{train_split_name}[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Model & Tokenizer</strong></summary>

Config, tokenizer, and model are resolved from `AutoConfig`,
`AutoTokenizer`, and `AutoModelForCausalLM`. The embedding layer is
resized if the tokenizer vocabulary is larger than the model's embedding
matrix.

```python title="src/ezpz/examples/hf.py" linenums="330"
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info("Overriding config: %s", model_args.config_overrides)
            config.update_from_string(model_args.config_overrides)
            logger.info("New config: %s", config)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(  # type:ignore
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(  # type:ignore
            config, trust_remote_code=model_args.trust_remote_code
        )

    if callable(getattr(model, "get_input_embeddings")):
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
```

</details>

<details closed markdown><summary><strong><code>main</code> -- W&B Setup</strong></summary>

Weights & Biases is initialized on rank 0 only via `ezpz.setup_wandb()`.
A custom metric axis (`num_input_tokens_seen`) is defined and the full
config is uploaded to the run.

```python title="src/ezpz/examples/hf.py" linenums="401"
    if rank == 0:
        try:
            import wandb

            if (
                wandb is not None
                and report_to is not None
                and report_to != "none"
                and not os.environ.get("WANDB_DISABLED", False)
            ):
                wbproj_name = (
                    model_args.wandb_project_name
                    if model_args.wandb_project_name is not None
                    else model_args.model_name_or_path
                )
                if wbproj_name is None:
                    wbproj_name = "ezpz-hf-default-project"
                wbproj_name = f"ezpz-hf-{wbproj_name}".replace("/", "-")
                run = ezpz.setup_wandb(project_name=wbproj_name)
                if run is not None and run is wandb.run:
                    wandb.define_metric("num_input_tokens_seen")
                    run.config.update(
                        {
                            "model": model_args.__dict__,
                            "data": data_args.__dict__,
                            "training": training_args.to_dict(),
                            "ezpz.dist_info": ezpz.get_dist_info(),
                        }
                    )
        except Exception:
            logger.info("W&B setup skipped")

    ezpz.barrier()  # sync all ranks after rank-0 wandb setup
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Tokenization</strong></summary>

Raw text is tokenized with `dataset.map()`. The main process runs the map
first (via `main_process_first`) so other ranks can reuse the cache.

```python title="src/ezpz/examples/hf.py" linenums="436"
    if training_args.do_train:
        column_names = list(raw_datasets[train_split_name].features)  # type:ignore
    else:
        column_names = list(raw_datasets[validation_split_name].features)  # type:ignore
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples: dict[str, object]) -> dict[str, object]:
        """Tokenize raw text using the configured tokenizer."""
        return tokenizer(examples[text_column_name])

    logger.info("[rank %d] entering tokenization", rank)
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,  # type:ignore
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,  # type:ignore
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    logger.info("[rank %d] tokenization done", rank)
```

</details>

<details closed markdown><summary><strong><code>main</code> -- <code>group_texts</code></strong></summary>

Tokenized sequences are concatenated end-to-end, then sliced into
fixed-length `block_size` chunks. This maximizes token utilization by
avoiding padding. Labels are a copy of `input_ids` (standard causal LM
objective).

```python title="src/ezpz/examples/hf.py" linenums="497"
    def group_texts(
        examples: dict[str, list[list[int]]]
    ) -> dict[str, list[list[int]]]:
        """Concatenate and chunk tokenized text into fixed-size blocks."""
        concatenated_examples = {
            k: [int(x) for x in chain(*examples[k])]
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("[rank %d] entering group_texts", rank)
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,  # type:ignore
                load_from_cache_file=not data_args.overwrite_cache,  # type:ignore
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
    logger.info("[rank %d] group_texts done", rank)
```

</details>

<details closed markdown><summary><strong><code>main</code> -- DataLoaders</strong></summary>

Train and eval `DataLoader`s are built from the processed datasets. The
train loader shuffles when not streaming.

```python title="src/ezpz/examples/hf.py" linenums="560"
    assert train_dataset is not None
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=not data_args.streaming,
        collate_fn=default_data_collator,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
        )
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Optimizer & LR Scheduler</strong></summary>

AdamW is used with separate weight-decay groups (bias and LayerNorm
weights are excluded). The LR scheduler is created from the HuggingFace
`get_scheduler` helper.

```python title="src/ezpz/examples/hf.py" linenums="577"
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=training_args.learning_rate
    )
```

```python title="src/ezpz/examples/hf.py" linenums="619"
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps
        * accelerator.num_processes,
        num_training_steps=training_args.max_steps
        if overrode_max_train_steps
        else training_args.max_steps * accelerator.num_processes,
    )
```

</details>

<details closed markdown><summary><strong><code>main</code> -- <code>accelerator.prepare</code></strong></summary>

All training objects are wrapped by Accelerate in one call. This handles
DDP/FSDP wrapping, optimizer state sharding, and dataloader distribution.

```python title="src/ezpz/examples/hf.py" linenums="629"
    logger.info("[rank %d] calling accelerator.prepare() ...", rank)
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    logger.info("[rank %d] accelerator.prepare() complete", rank)
```

</details>

<details closed markdown><summary><strong><code>main</code> -- History & Checkpointing Setup</strong></summary>

An `ezpz.history.History` object is created for metric tracking.
Checkpoint resumption is handled by detecting `step_*` or `epoch_*`
directories and calling `accelerator.load_state`.

```python title="src/ezpz/examples/hf.py" linenums="700"
    logging_steps = max(1, int(training_args.logging_steps))
    outdir = Path(training_args.output_dir) if training_args.output_dir else Path.cwd() / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    history = ezpz.history.History(
        report_dir=outdir,
        report_enabled=True,
        jsonl_path=outdir / "metrics.jsonl",
        jsonl_overwrite=True,
    )
    completed_steps = 0
    starting_epoch = 0
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Training Loop</strong></summary>

Each epoch iterates over the dataloader inside `accelerator.accumulate`,
which handles gradient accumulation transparently. After each optimizer
step, per-step metrics (loss, perplexity, tokens/sec) are recorded via
`history.update`.

```python title="src/ezpz/examples/hf.py" linenums="752"
    total_loss = 0
    perplexity: Optional[float] = None
    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0
        if (
            training_args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for _, batch in enumerate(active_dataloader):
            t0step = time.perf_counter()
            t1step = 0.0
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                t1step = time.perf_counter() - t0step
                completed_steps += 1
                total_loss += loss.detach().float().item()

                tokens_per_step = total_batch_size * block_size
                tps = tokens_per_step / t1step if t1step > 0 else 0.0
                step_loss = loss.detach().float().item()
                try:
                    step_ppl = math.exp(step_loss)
                except OverflowError:
                    step_ppl = float("inf")

                metrics = {
                    "step": completed_steps,
                    "epoch": epoch,
                    "loss": step_loss,
                    "perplexity": step_ppl,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "dts": t1step,
                    "tokens_per_sec": tps,
                }

                if completed_steps % logging_steps == 0:
                    summary = history.update(metrics)
                    logger.info(summary)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(
                            training_args.output_dir, output_dir
                        )
                    accelerator.save_state(output_dir)

            if completed_steps >= training_args.max_steps:
                break
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Evaluation Loop</strong></summary>

At the end of each epoch, an eval pass gathers losses across all ranks
and computes perplexity. Results are logged through `history.update`.

```python title="src/ezpz/examples/hf.py" linenums="817"
        if eval_dataloader is not None:
            model.eval()
            losses = []
            for _, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(
                    accelerator.gather_for_metrics(
                        loss.repeat(training_args.per_device_eval_batch_size)
                    )
                )

            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            avg_train_loss = float(total_loss) / max(completed_steps, 1)
            eval_metrics = {
                "step": completed_steps,
                "epoch": epoch,
                "eval_loss": float(eval_loss),
                "eval_perplexity": perplexity,
                "train_loss": avg_train_loss,
            }
            summary = history.update(eval_metrics)
            logger.info(summary)
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Epoch Checkpointing & Hub Upload</strong></summary>

If `push_to_hub` is enabled, the model and tokenizer are saved and
uploaded after each epoch. Per-epoch state can also be saved locally when
`save_strategy="epoch"`.

```python title="src/ezpz/examples/hf.py" linenums="849"
        if training_args.push_to_hub and epoch < training_args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process and api is not None and repo_id is not None:
                tokenizer.save_pretrained(output_dir)
                api.upload_folder(  # type: ignore[arg-type]
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=training_args.hub_token,
                )

        if training_args.save_strategy == "epoch":
            output_dir = f"epoch_{epoch}"
            output_dir = os.path.join(output_dir, output_dir)
            accelerator.save_state(output_dir)
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Finalization & Save</strong></summary>

After all epochs complete, the unwrapped model is saved. Timing
information is collected and `history.finalize()` generates summary
plots and writes final metrics. W&B timings are logged as a last step.

```python title="src/ezpz/examples/hf.py" linenums="872"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        if training_args.push_to_hub and api is not None and repo_id is not None:
            api.upload_folder(  # type: ignore[arg-type]
                commit_message="End of training",
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                token=training_args.hub_token,
            )
        if perplexity is not None:
            with open(
                os.path.join(output_dir, "all_results.json"),
                "w",
            ) as f:
                json.dump({"perplexity": perplexity}, f)

    train_end = time.perf_counter()
    timings = {
        "main/setup_torch": t_setup - t0,
        "main/train": train_end - train_start,
        "main/total": train_end - t0,
        "timings/training_start": train_start - t0,
        "timings/train_duration": train_end - train_start,
        "timings/end-to-end": train_end - t0,
    }
    logger.info("Timings: %s", timings)

    if accelerator.is_main_process:
        history.finalize(
            run_name="ezpz.examples.hf",
            dataset_fname="train",
            warmup=0,
            save=True,
            plot=True,
            outdir=outdir,
            timings=timings,
        )

    if wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            wandb.log(
                {
                    (f"timings/{k}" if not k.startswith("timings/") else k): v
                    for k, v in timings.items()
                }
            )
        except Exception:
            logger.warning("Failed to log timings to wandb")
```

</details>

<details closed markdown><summary><strong><code>__main__</code> guard</strong></summary>

```python title="src/ezpz/examples/hf.py" linenums="930"
if __name__ == "__main__":
    main()
```

</details>

## MFU Tracking

`hf.py` estimates model FLOPS via [`try_estimate`](../recipes.md#mfu-tracking)
**before** `accelerator.prepare()` (FlopCounterMode can't run through
DDP/FSDP wrappers). Per-step **TFLOPS** and **MFU** are reported as
`train/tflops` and `train/mfu`.

```python
_model_flops = try_estimate(
    model, (training_args.per_device_train_batch_size, block_size),
)
# ... per step:
metrics["train/tflops"] = _model_flops / t1step / 1e12
metrics["train/mfu"] = compute_mfu(_model_flops, t1step)
```

For HF causal LMs, `estimate_model_flops` extracts `output.logits.sum()`
as the backward target since `output.loss` is `None` without labels.
See [`ezpz.flops`](../python/Code-Reference/flops.md) for details.

## Metric Keys, Logging, and Output

- **Prefixed keys** — Train metrics use `train/` prefix (`train/loss`,
  `train/perplexity`, `train/tflops`, `train/mfu`); eval metrics use
  `eval/` prefix. This makes `History.finalize()` produce separate
  `train.h5` / `eval.h5` datasets and grouped plots, instead of
  flattening everything into one column-shared table.
- **Log line cleanup** — Each log line is tagged `[train]` or `[eval]`,
  so the prefix is stripped from the per-line summary to reduce noise:
  ```python
  logger.info("[train] %s", summary.replace("train/", ""))
  ```
- **HTTP log suppression** — `httpx`, `huggingface_hub`, and `filelock`
  are silenced to `WARNING` at startup. Without this, every Hub HEAD/GET
  produced an `INFO` log line per rank — hundreds of lines on multi-rank
  jobs.

## Robustness

- **Safetensors fallback** — On some parallel filesystems (Lustre),
  `safetensors` raises `Argument list too long` (OS error 7) during
  `save_pretrained`. The example catches this and retries with
  `safe_serialization=False` to write `.bin` instead — the model still
  gets saved.
- **`max_steps` exits early** — The HF training loop's `completed_steps`
  is global across epochs. Without an outer-loop break, epoch 1 would
  run a single wasted step before hitting `max_steps`. The example
  breaks out of the epoch loop too.

## Comparison with `hf_trainer.py`

This example (`hf.py`) uses an **explicit training loop** — you control the
forward/backward/optimizer step directly, giving maximum flexibility for
custom metrics, gradient manipulation, or unconventional schedules.

The companion [`hf_trainer.py`](hf-trainer/index.md) uses the
**HuggingFace Trainer abstraction**, which handles the loop internally.
Use the Trainer version when you want standard training with minimal code;
use this version when you need full control.

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.hf --help
# Accepts all HuggingFace TrainingArguments plus:
#   --model_name_or_path    Pretrained model name or path
#   --dataset_name          HuggingFace dataset name
#   --dataset_config_name   Dataset configuration
#   --do_train / --do_eval  Enable training / evaluation
#   --block_size            Token block size for grouping
#   --wandb_project_name    Custom W&B project name
# See HfModelArguments and HfDataTrainingArguments for full list.
```

</details>
