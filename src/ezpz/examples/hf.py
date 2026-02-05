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
from tqdm.auto import tqdm
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
    from accelerate import Accelerator  # noqa: E402 type:ignore
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


def main() -> None:
    """Entrypoint for standalone HF causal LM fine-tuning without Trainer."""
    t0 = time.perf_counter()
    rank = ezpz.setup_torch()
    model_args, data_args, training_args = parse_args()

    output_dir = training_args.output_dir or os.getcwd()
    wandb = None
    report_to = training_args.report_to
    log_with = report_to if report_to is not None and report_to != "none" else None
    project_dir = output_dir if log_with is not None else None
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=log_with,
        project_dir=project_dir,
    )
    t_setup = time.perf_counter()

    logger.warning(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if training_args.seed is not None:
        set_seed(training_args.seed)

    api = None
    repo_id = None
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(
                repo_name, exist_ok=True, token=training_args.hub_token
            ).repo_id

            with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        else:
            os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    last_checkpoint = None
    if (
        os.path.isdir(output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(output_dir)) > 0
        ):
            raise ValueError(
                "Output directory already exists and is not empty. Use --overwrite_output_dir to train from scratch."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. To avoid this behavior, change the output_dir or add --overwrite_output_dir.",
                last_checkpoint,
            )

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

    if training_args.do_train:
        column_names = list(raw_datasets[train_split_name].features)  # type:ignore
    else:
        column_names = list(raw_datasets[validation_split_name].features)  # type:ignore
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples: dict[str, object]) -> dict[str, object]:
        """Tokenize raw text using the configured tokenizer."""
        return tokenizer(examples[text_column_name])

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

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        max_pos_embeddings = 1024
        logger.warning(
            "Config %s does not have 'max_position_embeddings'; using %s.",
            config,
            max_pos_embeddings,
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                "The tokenizer picked seems to have a very large `model_max_length` (%s). "
                "Using block_size=%s instead. You can change that default value by passing --block_size xxx.",
                tokenizer.model_max_length,
                min(1024, max_pos_embeddings),
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                "The block_size passed (%s) is larger than the maximum length for the model (%s). Using block_size=%s.",
                data_args.block_size,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

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

    train_dataset = None
    if training_args.do_train:
        if train_split_name not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets[train_split_name]  # type:ignore
        if data_args.max_train_samples is not None:
            if isinstance(train_dataset, datasets.IterableDataset):
                train_dataset = train_dataset.take(data_args.max_train_samples)
            else:
                max_train_samples = min(
                    train_dataset.num_rows, data_args.max_train_samples
                )
                train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if validation_split_name not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[validation_split_name]  # type:ignore
        if data_args.max_eval_samples is not None:
            if isinstance(eval_dataset, datasets.IterableDataset):
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                eval_dataset = eval_dataset.select(
                    range(data_args.max_eval_samples)
                )  # type:ignore

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

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    if training_args.max_steps <= 0:
        training_args.max_steps = int(
            training_args.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps
        * accelerator.num_processes,
        num_training_steps=training_args.max_steps
        if overrode_max_train_steps
        else training_args.max_steps * accelerator.num_processes,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        training_args.max_steps = int(
            training_args.num_train_epochs * num_update_steps_per_epoch
        )
    training_args.num_train_epochs = math.ceil(
        training_args.max_steps / num_update_steps_per_epoch
    )

    checkpointing_steps = training_args.save_steps

    if report_to is not None and report_to != "none":
        experiment_config = {
            "model": model_args.__dict__,
            "data": data_args.__dict__,
            "training": training_args.to_dict(),
        }
        experiment_config["lr_scheduler_type"] = experiment_config[
            "training"
        ].get("lr_scheduler_type")
        accelerator.init_trackers("ezpz-hf", experiment_config)

    train_start = time.perf_counter()
    total_batch_size = (
        training_args.per_device_train_batch_size
        * accelerator.num_processes
        * training_args.gradient_accumulation_steps
    )

    logger.info("***** Model *****")
    logger.info("  Model = %s", model)
    logger.info("***** Args *****")
    logger.info(
        json.dumps(
            {
                "model": model_args.__dict__,
                "data": data_args.__dict__,
                "training": training_args.to_dict(),
            },
            indent=4,
            sort_keys=True,
        )
    )
    logger.info("***** Running training *****")
    logger.info("  Num processes = %s", accelerator.num_processes)
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  Num Epochs = %s", training_args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per device = %s",
        training_args.per_device_train_batch_size,
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %s",
        total_batch_size,
    )
    logger.info(
        "  Gradient Accumulation steps = %s",
        training_args.gradient_accumulation_steps,
    )
    logger.info("  Total optimization steps = %s", training_args.max_steps)

    progress_bar = tqdm(
        range(training_args.max_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0

    if training_args.resume_from_checkpoint:
        if (
            training_args.resume_from_checkpoint is not None
            or training_args.resume_from_checkpoint != ""
        ):
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = (
                int(training_difference.replace("step_", ""))
                * training_args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    else:
        resume_step = None

    progress_bar.update(completed_steps)
    total_loss = 0
    perplexity: Optional[float] = None
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        if report_to is not None and report_to != "none":
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
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(
                            training_args.output_dir, output_dir
                        )
                    accelerator.save_state(output_dir)

            logging_steps = max(1, int(training_args.logging_steps))
            if completed_steps % logging_steps == 0:
                if report_to is not None and report_to != "none":
                    logs = {
                        "epoch": epoch,
                        "step": completed_steps,
                        "loss": loss.detach().float(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "dts": t1step,
                    }
                    accelerator.log(logs, step=completed_steps)
                    if wandb is not None and getattr(wandb, "run", None) is not None:
                        wandb.log(logs)

            if completed_steps >= training_args.max_steps:
                break

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

            logger.info(
                "epoch %s: perplexity: %s eval_loss: %s",
                epoch,
                perplexity,
                eval_loss,
            )

            if report_to is not None and report_to != "none":
                accelerator.log(
                    {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": float(total_loss) / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

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

    if report_to is not None and report_to != "none":
        accelerator.end_training()

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
    }
    logger.info("Timings: %s", timings)
    if wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            wandb.log(timings)
        except Exception:
            logger.warning("Failed to log timings to wandb")


if __name__ == "__main__":
    main()
