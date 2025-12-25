#!/usr/bin/env python
"""
hf_trainer.py

Complete, self-contained script for fine-tuning a model on a text file or a dataset for causal language modeling.

Modified from:
https://github.com/huggingface/transformers/blob/51ed61e2f05176f81fa7c9decba10cc28e138f61/examples/pytorch/language-modeling/run_clm.py

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import math
import os
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Optional

import datasets
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    # set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import ezpz
import ezpz.configs
from ezpz.configs import HfDataTrainingArguments, HfModelArguments

logger = ezpz.get_logger(__name__)


def parse_args() -> dict:
    """Returns dictionary of

    `{"model": model_args, "data": data_args, "training": training_args}`
    """
    # NOTE:
    #   See all possible arguments by passing the --help flag to this script.
    #   We now keep distinct sets of args, for a cleaner separation of concerns.
    rank = ezpz.get_rank()
    parser = HfArgumentParser(
        (HfModelArguments, HfDataTrainingArguments, TrainingArguments)  # type:ignore
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = (
            parser.parse_args_into_dataclasses()
        )

    try:
        import wandb
    except (ImportError, ModuleNotFoundError):
        wandb = None  # type:ignore

    if (
        wandb is not None
        and rank == 0
        and not os.environ.get("WANDB_DISABLED", False)
    ):
        if (
            model_args.wandb_project_name is None
            and model_args.model_name_or_path is None
        ):
            wbproj_name = "ezpz-hf_trainer-default-project"
        else:
            wbproj_name = (
                model_args.wandb_project_name
                if model_args.wandb_project_name is not None
                else model_args.model_name_or_path
            )
            wbproj_name = f"ezpz-hf_trainer-{wbproj_name}"
        run = ezpz.setup_wandb(project_name=wbproj_name.replace("/", "-"))
        wandb.define_metric("num_input_tokens_seen")
        wandb.define_metric("train/", step_metric="num_input_tokens_seen")
        # wandb.define_metric("train/epoch")
        # wandb.define_metric("train/step")
        wandb.define_metric("eval/", step_metric="num_input_tokens_seen")
        # wandb.define_metric()
        # wandb.define_metric(
        #     name="num_input_tokens_seen",
        #     step_metric="num_input_tokens_seen",
        # )  # Allow us to track the number of tokens seen during training
        if run is not None:
            # wandb.log({"train_iter": 0}, step=0)
            run.config.update(
                {
                    "model": model_args.__dict__,
                    "data": data_args.__dict__,
                    "training": training_args.to_dict(),
                    "ezpz.dist_info": ezpz.get_dist_info(),
                }
            )

    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_info()

    # Log on each process the small summary:
    logger.warning(
        ", ".join(
            [
                f"Process rank: {rank}",
                f"device: {training_args.device}",
                f"n_gpu: {training_args.n_gpu}",
                f"distributed training: {training_args.parallel_mode.value == 'distributed'}",
            ]
        )
    )

    log_level_info = 20  # "INFO"
    log_level_critical = 50  # "CRITICAL"
    log_level = log_level_info if rank == 0 else log_level_critical
    import datasets.utils.logging
    import transformers.utils.logging

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, "
    #     f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    #     + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    # )
    if rank == 0:
        logger.info(f"Training/evaluation parameters {training_args}")

    return {"model": model_args, "data": data_args, "training": training_args}


# Write a function below to resolve possibility of optimizer being defined both as:
#
# 1. `--optim=adamw` at the CLI
#
# and
#
# 2. `"optimizer": { ... }` in a `--deepspeed=config.json` file.


def resolve_optimizer(
    optimizer: Optional[str], deepspeed_config: Optional[dict]
) -> str:
    """
    Resolve the optimizer to use based on the command line argument and the deepspeed config.

    Args:
        optimizer (str): The optimizer specified in the command line argument.
        deepspeed_config (dict): The deepspeed config dictionary.

    Returns:
        str: The resolved optimizer.
    """
    if optimizer is not None:
        return optimizer
    elif deepspeed_config is not None and "optimizer" in deepspeed_config:
        return deepspeed_config["optimizer"]["type"]
    else:
        return "adamw"


def decode_predictions(
    tokenizer: transformers.PreTrainedTokenizer,
    predictions: transformers.EvalPrediction,  # | list[int] | list[list[int]] | torch.Tensor,
) -> dict[str, list[str]]:
    """
    Decode the predictions from the model into text labels.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to decode the predictions.
        predictions (transformers.TrainerPredictionOutput): The output from the Trainer containing predictions.
    """
    labels = tokenizer.batch_decode(list(predictions.label_ids))
    logits = torch.Tensor(predictions.predictions).argmax(-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


def split_dataset(
    model_args: HfModelArguments,
    data_args: HfDataTrainingArguments,
    train_split_name: str = "train",
    validation_split_name: Optional[str] = None,
) -> datasets.IterableDatasetDict | datasets.DatasetDict:
    """
    Splits the dataset into training and validation sets based on the provided split names.

    Args:
    """
    dsets = {}
    # if (
    #     validation_split_name not in raw_datasets.keys() and training_args.do_eval
    # ):  # type:ignore
    # assert data_args.dataset_name is not None, (
    #     "dataset_name must be provided to split the dataset."
    # )
    dataset_name = data_args.dataset_name
    assert dataset_name is not None, (
        "dataset_name must be provided to split the dataset."
    )
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
            # In some cases, the dataset doesn't support slicing.
            # In this case, we just use the full training set as validation set.
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
                # In some cases, the dataset doesn't support slicing.
                # In this case, we just use the full training set as validation set.
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
        return datasets.IterableDatasetDict(dsets)
    return datasets.DatasetDict(dsets)


def main() -> int:
    """
    Main function to run the training and evaluation of a causal language model.

    This function sets up the training environment, loads the datasets, tokenizes the data,
    initializes the model and tokenizer, and runs the training and evaluation loops.

    It also handles distributed training setup, logging, and configuration loading.

    """
    # hfloglevel = "INFO" if rank == 0 else "ERROR"
    # logging.getLogger("datasets").setLevel(hfloglevel)
    import ezpz.dist
    rank = ezpz.dist.setup_torch()
    # rank = ezpz.dist.setup_torch(
    #     # seed=training_args.seed,
    #     # device_id=int(devid) if devid is not None else devid,
    # )


    args = parse_args()
    # args: dict[str, HfModelArguments| HfDataTrainingArguments| TrainingArguments]
    # def main(args: dict[str, HfModelArguments| HfDataTrainingArguments| TrainingArguments]) -> int:
    assert "data" in args and "model" in args and "training" in args
    data_args: HfDataTrainingArguments = args["data"]
    model_args: HfModelArguments = args["model"]
    training_args : TrainingArguments = args["training"]

    try:
        import wandb
    except (ImportError, ModuleNotFoundError):
        wandb = None  # type: ignore

    try:
        import evaluate
    except (ImportError, ModuleNotFoundError):
        evaluate = None  # type: ignore
        print(
            '"evaluate" library is not installed. '
            "We will continue without running evaluations. "
            'Please install it using "pip install evaluate" to run evaluations'
        )


    dsconfig_fp = (
        Path(training_args.deepspeed) if training_args.deepspeed else None
    )
    ds_config = ezpz.configs.load_ds_config(dsconfig_fp)
    if training_args.optim is not None and "optimizer" in ds_config:
        logger.warning(
            f"Overriding optimizer in deepspeed config with {training_args.optim}"
        )
        _ = ds_config.pop("optimizer")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    # from datasets import datasets.load_dataset

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the datasets.load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    train_split_name = data_args.train_split_name
    validation_split_name = data_args.validation_split_name
    test_split_name = data_args.test_split_name
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # dataset = datasets.load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir=model_args.cache_dir,
        #     token=model_args.token,
        #     streaming=data_args.streaming,
        #     trust_remote_code=model_args.trust_remote_code,
        # )
        raw_datasets = split_dataset(
            model_args,
            data_args,
            train_split_name=train_split_name,
            validation_split_name=validation_split_name,
        )

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files[train_split_name] = data_args.train_file
        if data_args.validation_file is not None:
            data_files[validation_split_name] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if validation_split_name not in raw_datasets.keys():  # type:ignore
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

    # See more about loading any type of standard or custom dataset (from
    # files, python dict, pandas DataFrame, etc) at:
    # https://huggingface.co/docs/datasets/loading_datasets.
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
        logger.warning(
            "You are instantiating a new config instance from scratch."
        )
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # mm = AutoModelForCausalLM.
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
        model = AutoModelForCausalLM.from_config(  # type:ignore
            config, trust_remote_code=model_args.trust_remote_code
        )
        n_params = sum(
            {p.data_ptr(): p.numel() for p in model.parameters()}.values()
        )
        logger.info(
            f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params"
        )

    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.watch(model, log="all")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if callable(getattr(model, "get_input_embeddings")):
        embedding_size = (
            model.get_input_embeddings().weight.shape[0]  # type:ignore
        )
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets[train_split_name].features)  # type:ignore
    else:
        column_names = list(raw_datasets[validation_split_name].features)  # type:ignore
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger(
    #     "transformers.tokenization_utils_base"
    # )

    def tokenize_function(examples):
        """Tokenize raw text using the configured tokenizer."""
        # with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        # if "Token indices sequence length is longer than the" in cl.out:
        #     logger.warning(
        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
        #         " before being passed to the model."
        #     )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,  # type:ignore
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,  # type:ignore
                desc="Running tokenizer on dataset",  # type:ignore
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
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024
        logger.warning(
            f"Config {config} does not have 'max_position_embeddings' attribute. "
            f"Using default value of {max_pos_embeddings}."
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        """Concatenate and chunk tokenized text into fixed-size blocks."""
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we
        # exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together,
    # so group_texts throws away a remainder for each of those groups of 1,000
    # texts. You can adjust that batch_size here but a higher value might be
    # slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of
    # the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,  # type:ignore
                load_from_cache_file=not data_args.overwrite_cache,  # type:ignore
                desc=f"Grouping texts in chunks of {block_size}",  # type:ignore
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    train_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]  # type:ignore
        assert isinstance(
            train_dataset, (datasets.Dataset, datasets.IterableDataset)
        )
        if data_args.max_train_samples is not None:
            if isinstance(train_dataset, datasets.IterableDataset):
                train_dataset = train_dataset.take(data_args.max_train_samples)
            elif isinstance(train_dataset, datasets.Dataset):
                max_train_samples = min(
                    train_dataset.num_rows, data_args.max_train_samples
                )
                train_dataset = train_dataset.select(range(max_train_samples))
        # available_splits = train_dataset.info.splits
        # if (train_split := getattr(available_splits, "train")) is not None:
        #     if data_args.max_train_samples is not None:
        #         max_train_samples = min(
        #             train_split.num_examples,
        #             data_args.max_train_samples,
        #         )
        #         train_dataset = train_dataset.select(  # type:ignore
        #             range(max_train_samples)
        #         )
        # if available_splits is not None and "train" not in available_splits:
        #     if isinstance(train_dataset, datasets.IterableDataset):
        #         train_dataset = train_dataset.take(max_eval_samples)
        #     else:
        #         eval_dataset = eval_dataset.select(range(max_eval_samples))

    # if training_args.eval_on_start and evaluate is not None:
    #     metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    #
    #     def compute_metrics(eval_preds):
    #         preds, labels = eval_preds
    #         # preds have the same shape as the labels, after the argmax(-1) has been calculated
    #         # by preprocess_logits_for_metrics but we need to shift the labels
    #         labels = labels[:, 1:].reshape(-1)
    #         preds = preds[:, :-1].reshape(-1)
    #         predictions = decode_predictions(tokenizer, preds)
    #         predictions_df = pd.DataFrame(predictions)
    #         if wandb is not None and getattr(wandb, "run", None) is not None:
    #             records_table = wandb.Table(dataframe=predictions_df)
    #             # log the table to wandb
    #             assert wandb is not None and wandb.run is not None
    #             wandb.log({"sample_predictions": records_table})
    #         return metric.compute(predictions=preds, references=labels)

    eval_dataset = None
    if training_args.do_eval:
        assert evaluate is not None
        if validation_split_name not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[validation_split_name]  # type: ignore
        if data_args.max_eval_samples is not None:
            # max_eval_samples = min(
            #     # len(eval_dataset),
            #     # eval_dataset.info.splits["train"].num_examples,  # type:ignore
            #     data_args.max_eval_samples,
            # )
            if isinstance(eval_dataset, datasets.IterableDataset):
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                eval_dataset = eval_dataset.select(
                    range(data_args.max_eval_samples)
                )  # type:ignore

        def preprocess_logits_for_metrics(logits, labels):
            """Prepare logits for metric computation by argmax over vocabulary."""
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        # if evaluate is not None:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(
            eval_preds: tuple[torch.Tensor, torch.Tensor],
        ) -> dict | None:
            """Compute accuracy on shifted language modeling labels."""
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            # predictions = decode_predictions(tokenizer, preds)
            # predictions_df = pd.DataFrame(predictions)
            # if wandb is not None and getattr(wandb, "run", None) is not None:
            #     records_table = wandb.Table(dataframe=predictions_df)
            #     # log the table to wandb
            #     assert wandb is not None and wandb.run is not None
            #     wandb.log({"sample_predictions": records_table})
            # return metric.compute(predictions=preds, references=labels)
            return metric.compute(predictions=preds, references=labels)

    if rank == 0 and wandb is not None:
        wandb.run.watch(model, log="all")  # type:ignore

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,  # type:ignore
        eval_dataset=eval_dataset if training_args.do_eval else None,  # type:ignore
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=(
            compute_metrics  # type:ignore
            if training_args.do_eval and not is_torch_xla_available()  # type:ignore
            else None
        ),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics  # type:ignore
            if training_args.do_eval and not is_torch_xla_available()
            else None
        ),
    )

    # if wandb is not None and getattr(wandb, "run", None) is not None:
    #     # from transformers.integrations.integration_utils import W
    #     # from wandb.integration import WandbEvalCallback
    #
    #     # evals_callback = WandbEvalCallback(trainer, tokenizer)
    #     from ezpz.integrations import WandbPredictionProgressCallback
    #
    #     callback = WandbPredictionProgressCallback(
    #         trainer=trainer,
    #         tokenizer=tokenizer,
    #         val_dataset=(eval_dataset if training_args.do_eval else None),
    #         num_samples=10,
    #         freq=2,
    #     )
    #     trainer.add_callback(callback)

    assert any([training_args.do_train, training_args.do_eval]), (
        "Nothing to do! Set --do_train or --do_eval."
    )

    # Training
    if training_args.do_train:
        assert isinstance(trainer, Trainer)
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else train_dataset.info.splits["train"].num_examples  # type:ignore
        )
        assert train_dataset is not None
        metrics["train_samples"] = min(
            max_train_samples,
            train_dataset.info.splits["train"].num_examples,  # type:ignore
        )  # type:ignore

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # Evaluation
    if training_args.do_eval:
        assert isinstance(trainer, Trainer) and callable(
            getattr(trainer, "evaluate")
        ), (
            "Trainer must be an instance of `transformers.Trainer` "
            "and have an `evaluate` method."
        )
        assert trainer.evaluate is not None and callable(trainer.evaluate), (
            "Trainer must have an `evaluate` method."
        )
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else eval_dataset.info.splits["train"].num_examples  # type:ignore
        )
        metrics["eval_samples"] = min(
            max_eval_samples,
            eval_dataset.info.splits["train"].num_examples,  # type:ignore
        )  # type:ignore
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if wandb is not None and wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        # model.eval()
        # losses = []
        # assert eval_dataset is not None
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #         loss = outputs.loss
        #         losses.append(loss.item())
        # # Evaluation
        # if training_args.do_eval:
        #     import datasets
        #     import pandas as pd
        #
        #     logger.info("*** Evaluate ***")
        #     assert eval_dataset is not None, (
        #         "eval_dataset must be defined for evaluation."
        #     )
        #
        #     metrics = trainer.evaluate()
        #
        #     max_eval_samples = (
        #         data_args.max_eval_samples
        #         if data_args.max_eval_samples is not None
        #         else len(eval_dataset)  # type:ignore
        #     )
        #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))  # type:ignore
        #     try:
        #         perplexity = math.exp(metrics["eval_loss"])
        #     except OverflowError:
        #         perplexity = float("inf")
        #     metrics["perplexity"] = perplexity
        #     trainer.log_metrics("eval", metrics)
        #     trainer.save_metrics("eval", metrics)
        #
        #     # sample_dataset = eval_dataset.take(10)
        #     # assert sample_dataset is not None
        #     # # generate predictions
        #     # if sample_dataset is not None and isinstance(sample_dataset, datasets.Dataset):
        #     #     predictions = trainer.predict(sample_dataset)
        #     #     # decode predictions and labels
        #     #     predictions = decode_predictions(tokenizer, predictions)
        #     #     # add predictions to a wandb.Table
        #     #     predictions_df = pd.DataFrame(predictions)
        #     #     # predictions_df["epoch"] = state.epoch
        #     #     if wandb is not None and getattr(wandb, "run", None) is not None:
        #     #         records_table = wandb.Table(dataframe=predictions_df)
        #     #         # log the table to wandb
        #     #         assert wandb is not None and wandb.run is not None
        #     #         wandb.log({"sample_predictions": records_table})

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.config.update(kwargs)
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return 0


if __name__ == "__main__":
    # import patdb; patdb.debug()
    # devid = (
    #         os.environ.get(
    #             "PALS_LOCAL_RANKID",
    #             os.environ.get(
    #                 "OMPI_COMM_WORLD_LOCAL_RANK",
    #                 os.environ.get(
    #                     "LOCAL_RANK",
    #                     None
    #                 )
    #             )
    #     )
    # )
    t0 = time.perf_counter()
    main()
    ezpz.dist.cleanup()
    logger.info(f"Took {time.perf_counter() - t0:.2f} seconds")
