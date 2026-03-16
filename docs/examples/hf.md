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

## Code Walkthrough

### Setup & Config

`setup_torch()` handles device detection and `init_process_group`, while
Accelerate manages model/optimizer wrapping. An explicit FSDP plugin is
constructed when `ACCELERATE_USE_FSDP=true`:

```python title="src/ezpz/examples/hf.py" linenums="186"
    rank = ezpz.setup_torch()
    model_args, data_args, training_args = parse_args()

    output_dir = training_args.output_dir or os.getcwd()
    wandb = None
    report_to = training_args.report_to

    # Build FSDP plugin explicitly when --fsdp is requested, bypassing
    # the env-var machinery which can pick up stale/conflicting defaults.
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

    # Don't let Accelerator manage wandb — we handle it via ezpz.setup_wandb()
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
    )
```

### Data Pipeline

Datasets are loaded from HuggingFace Hub, tokenized, then concatenated
and chunked into fixed-size blocks — maximizing token utilization:

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
```

### Training Loop

The training loop uses Accelerate's gradient accumulation and ezpz's
`History` for metric tracking:

```python title="src/ezpz/examples/hf.py" linenums="754"
    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0
        ...
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
                ...
                if completed_steps % logging_steps == 0:
                    summary = history.update(metrics)
                    logger.info(summary)
```

### Evaluation

At the end of each epoch, an eval pass computes perplexity:

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
```

### Finalization

After all epochs, metrics are finalized and the model checkpoint is saved:

```python title="src/ezpz/examples/hf.py" linenums="872"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    ...
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
