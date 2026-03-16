# Fine-Tune a Causal LM (HuggingFace + Accelerate)

Fine-tune a HuggingFace causal language model with an explicit training loop.
Uses Accelerate for distributed training and ezpz for setup, metrics, and
W&B integration.

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

## What to Expect

You'll see per-step loss, perplexity, learning rate, and tokens/sec.
At the end of each epoch, an evaluation pass reports eval loss and
perplexity. The final model checkpoint is saved to `--output_dir`.

## Code Walkthrough

### Setup & Config

The script uses ezpz for distributed init, then HuggingFace's argument
parser for model/data/training configuration:

```python
rank = ezpz.setup_torch()
model_args, data_args, training_args = parse_args()

# Build FSDP plugin when requested
fsdp_plugin = None
if os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true":
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        ...
    )

accelerator = Accelerator(
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    fsdp_plugin=fsdp_plugin,
)
```

`setup_torch()` handles device detection and `init_process_group`, while
Accelerate manages the model/optimizer wrapping.

### Data Pipeline

Datasets are loaded from HuggingFace Hub, tokenized, then grouped into
fixed-size blocks for causal LM training:

```python
# Load and tokenize
raw_datasets = split_dataset(model_args, data_args, ...)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Group into fixed-size blocks
def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result
```

This concatenate-then-chunk approach maximizes token utilization.

### Training Loop

The training loop uses Accelerate's gradient accumulation and ezpz's
`History` for metric tracking:

```python
history = ezpz.history.History(report_dir=outdir, ...)

for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
    model.train()
    for _, batch in enumerate(active_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            completed_steps += 1
            metrics = {
                "step": completed_steps,
                "loss": loss.detach().float().item(),
                "tokens_per_sec": tokens_per_step / dt,
                ...
            }
            summary = history.update(metrics)
```

### Evaluation

At the end of each epoch, an eval pass computes perplexity:

```python
model.eval()
losses = []
for _, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)
    losses.append(accelerator.gather_for_metrics(
        loss.repeat(training_args.per_device_eval_batch_size)
    ))
eval_loss = torch.mean(torch.cat(losses))
perplexity = math.exp(eval_loss)
```

### Finalization

After all epochs, metrics are finalized and the model checkpoint is saved:

```python
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, ...)

if accelerator.is_main_process:
    history.finalize(run_name="ezpz.examples.hf", ...)
```

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
