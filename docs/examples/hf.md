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

```python title="src/ezpz/examples/hf.py:1:59"
--8<-- "src/ezpz/examples/hf.py:1:59"
```

</details>

<details closed markdown><summary><strong><code>parse_args</code></strong></summary>

Uses HuggingFace's `HfArgumentParser` to parse three dataclass groups in
one pass. Supports loading arguments from a JSON file when the sole CLI
argument is a `.json` path. Logging verbosity is restricted to rank 0.

```python title="src/ezpz/examples/hf.py:62:98"
--8<-- "src/ezpz/examples/hf.py:62:98"
```

</details>

<details closed markdown><summary><strong><code>split_dataset</code></strong></summary>

Loads a named dataset from the HuggingFace Hub, splitting it into
train/validation by percentage. Falls back gracefully when the requested
split syntax is not supported by the dataset.

```python title="src/ezpz/examples/hf.py:101:181"
--8<-- "src/ezpz/examples/hf.py:101:181"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Distributed Setup</strong></summary>

`setup_torch()` initializes the process group and returns the local rank.
Arguments are parsed, and the output directory is established.

```python title="src/ezpz/examples/hf.py:184:196"
--8<-- "src/ezpz/examples/hf.py:184:196"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- FSDP Plugin</strong></summary>

When `ACCELERATE_USE_FSDP=true`, an explicit `FullyShardedDataParallelPlugin`
is constructed with a bf16 mixed-precision policy, bypassing Accelerate's
env-var machinery which can pick up stale defaults.

```python title="src/ezpz/examples/hf.py:200:223"
--8<-- "src/ezpz/examples/hf.py:200:223"
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

```python title="src/ezpz/examples/hf.py:294:342"
--8<-- "src/ezpz/examples/hf.py:294:342"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Model & Tokenizer</strong></summary>

Config, tokenizer, and model are resolved from `AutoConfig`,
`AutoTokenizer`, and `AutoModelForCausalLM`. The embedding layer is
resized if the tokenizer vocabulary is larger than the model's embedding
matrix.

```python title="src/ezpz/examples/hf.py:344:413"
--8<-- "src/ezpz/examples/hf.py:344:413"
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

```python title="src/ezpz/examples/hf.py:417:443"
--8<-- "src/ezpz/examples/hf.py:417:443"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- <code>group_texts</code></strong></summary>

Tokenized sequences are concatenated end-to-end, then sliced into
fixed-length `block_size` chunks. This maximizes token utilization by
avoiding padding. Labels are a copy of `input_ids` (standard causal LM
objective).

```python title="src/ezpz/examples/hf.py:478:512"
--8<-- "src/ezpz/examples/hf.py:478:512"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- DataLoaders</strong></summary>

Train and eval `DataLoader`s are built from the processed datasets. The
train loader shuffles when not streaming.

```python title="src/ezpz/examples/hf.py:541:556"
--8<-- "src/ezpz/examples/hf.py:541:556"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Optimizer & LR Scheduler</strong></summary>

AdamW is used with separate weight-decay groups (bias and LayerNorm
weights are excluded). The LR scheduler is created from the HuggingFace
`get_scheduler` helper.

```python title="src/ezpz/examples/hf.py:558:579"
--8<-- "src/ezpz/examples/hf.py:558:579"
```

```python title="src/ezpz/examples/hf.py:600:608"
--8<-- "src/ezpz/examples/hf.py:600:608"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- <code>accelerator.prepare</code></strong></summary>

All training objects are wrapped by Accelerate in one call. This handles
DDP/FSDP wrapping, optimizer state sharding, and dataloader distribution.

```python title="src/ezpz/examples/hf.py:614:624"
--8<-- "src/ezpz/examples/hf.py:614:624"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- History & Checkpointing Setup</strong></summary>

An `ezpz.history.History` object is created for metric tracking.
Checkpoint resumption is handled by detecting `step_*` or `epoch_*`
directories and calling `accelerator.load_state`.

```python title="src/ezpz/examples/hf.py:685:703"
--8<-- "src/ezpz/examples/hf.py:685:703"
```

</details>

<details closed markdown><summary><strong><code>main</code> -- Training Loop</strong></summary>

Each epoch iterates over the dataloader inside `accelerator.accumulate`,
which handles gradient accumulation transparently. After each optimizer
step, per-step metrics (loss, perplexity, tokens/sec) are recorded via
`history.update`.

```python title="src/ezpz/examples/hf.py:745:816"
--8<-- "src/ezpz/examples/hf.py:745:816"
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

```python title="src/ezpz/examples/hf.py:850:871"
--8<-- "src/ezpz/examples/hf.py:850:871"
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

```python title="src/ezpz/examples/hf.py:941:942"
--8<-- "src/ezpz/examples/hf.py:941:942"
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
