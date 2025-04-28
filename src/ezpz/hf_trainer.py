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
from pathlib import Path
import sys
from itertools import chain
from typing import Optional

import ezpz

from dataclasses import dataclass, field


import torch

MODEL_FOR_CAUSAL_LM_MAPPING = None
try:
    import transformers
    from transformers import (
        CONFIG_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        HfArgumentParser,
        Trainer,
        TrainingArguments,
        default_data_collator,
        is_torch_xla_available,
        set_seed,
    )
    from transformers.testing_utils import CaptureLogger
    from transformers.trainer_utils import get_last_checkpoint

    # from transformers.utils import send_example_telemetry
    from transformers.utils.versions import require_version
except (ImportError, ModuleNotFoundError):
    print(
        '"transformers" library is not installed. Please install it using "pip install transformers"'
    )
    sys.exit(1)


logger = ezpz.get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(
    getattr(conf, "model_type", "") for conf in MODEL_CONFIG_CLASSES
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the wandb project to use. If not specified, will use the model name."
            )
        },
    )

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split_str: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split string to use for the train split (via the datasets library)."
        },
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the train split to use (via the datasets library)."
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation split to use (via the datasets library)."
        },
    )
    validation_split_str: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split string to use for the validation split (via the datasets library)."
        },
    )
    test_split_name: Optional[str] = field(
        default="test",
        metadata={
            "help": "The name of the test split to use (via the datasets library)."
        },
    )
    test_split_str: Optional[str] = field(
        default=None,
        metadata={
            "help": "The split string to use for the test split (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(
        default=False, metadata={"help": "Enable streaming mode"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={
            "help": "Whether to keep line breaks when using TXT files or not."
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0",
                "The streaming feature requires `datasets>=2.0.0`",
            )

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, a json or a txt file."


def parse_args() -> dict:
    """Returns dictionary of

    `{"model": model_args, "data": data_args, "training": training_args}`
    """
    # NOTE:
    #   See all possible arguments by passing the --help flag to this script.
    #   We now keep distinct sets of args, for a cleaner separation of concerns.
    rank = ezpz.get_rank()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)  # type:ignore
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
        wandb = None

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
            run.config.update(ezpz.get_dist_info())
            run.config.update(training_args.to_dict())
            # wandb.log({"train_iter": 0}, step=0)
    # NOTE:
    #   Sending telemetry.
    #   Tracking the example usage helps us better allocate resources to
    #   maintain them. The information sent is the one passed as arguments
    #   along with your Python/PyTorch versions.
    # send_example_telemetry("hf_trainer", model_args, data_args)

    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

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

    # log_level = training_args.get_process_log_level()
    # log_level = training_args.get_log_level() if rank == 0 else 50  # "CRITICAL"
    # logger.setLevel(log_level)
    log_level_info = 20  # "INFO"
    log_level_critical = 50  # "CRITICAL"
    log_level = log_level_info if rank == 0 else log_level_critical
    try:
        import datasets

        datasets.utils.logging.set_verbosity(log_level)
    except Exception:
        pass
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


def resolve_optimizer(optimizer, deepspeed_config):
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


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


def main():
    # hfloglevel = "INFO" if rank == 0 else "ERROR"
    # logging.getLogger("datasets").setLevel(hfloglevel)

    rank = ezpz.setup_torch()

    try:
        import wandb
    except (ImportError, ModuleNotFoundError):
        wandb = None

    try:
        import evaluate
    except (ImportError, ModuleNotFoundError):
        evaluate = None
        print(
            '"evaluate" library is not installed. '
            "We will continue without running evaluations. "
            'Please install it using "pip install evaluate" to run evaluations'
        )

    args = parse_args()
    data_args = args["data"]
    model_args = args["model"]
    training_args = args["training"]

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
    set_seed(training_args.seed)

    from datasets import load_dataset

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    train_split_name = data_args.train_split_name
    validation_split_name = data_args.validation_split_name
    test_split_name = data_args.test_split_name
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if (
            validation_split_name not in raw_datasets.keys()
            and training_args.do_eval
        ):  # type:ignore
            try:
                raw_datasets[validation_split_name] = load_dataset(  # type:ignore
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets[train_split_name] = load_dataset(  # type: ignore
                    data_args.dataset_name,
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
                raw_datasets[validation_split_name] = load_dataset(  # type:ignore
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=train_split_name,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                try:
                    raw_datasets[train_split_name] = load_dataset(  # type:ignore
                        data_args.dataset_name,
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
                    raw_datasets[train_split_name] = load_dataset(  # type:ignore
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split=train_split_name,
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        streaming=data_args.streaming,
                        trust_remote_code=model_args.trust_remote_code,
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
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if validation_split_name not in raw_datasets.keys():  # type:ignore
            raw_datasets[validation_split_name] = load_dataset(  # type:ignore
                extension,
                data_files=data_files,
                split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets[train_split_name] = load_dataset(  # type:ignore
                extension,
                data_files=data_files,
                split=f"{train_split_name}[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

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

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
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
        model = AutoModelForCausalLM.from_config(
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
    embedding_size = model.get_input_embeddings().weight.shape[0]
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
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
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
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
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

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
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

    if training_args.eval_on_start and evaluate is not None:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            predictions = decode_predictions(tokenizer, preds)
            predictions_df = pd.DataFrame(predictions)
            if wandb is not None and getattr(wandb, "run", None) is not None:
                records_table = wandb.Table(dataframe=predictions_df)
                # log the table to wandb
                assert wandb is not None and wandb.run is not None
                wandb.log({"sample_predictions": records_table})
            return metric.compute(predictions=preds, references=labels)

    train_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]  # type:ignore
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(  # type:ignore
                range(max_train_samples)
            )  # type:ignore

    eval_dataset = None
    if training_args.do_eval:
        if validation_split_name not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[validation_split_name]  # type: ignore
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))  # type:ignore

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        if evaluate is not None:
            metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                predictions = decode_predictions(tokenizer, preds)
                predictions_df = pd.DataFrame(predictions)
                if (
                    wandb is not None
                    and getattr(wandb, "run", None) is not None
                ):
                    records_table = wandb.Table(dataframe=predictions_df)
                    # log the table to wandb
                    assert wandb is not None and wandb.run is not None
                    wandb.log({"sample_predictions": records_table})
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
            if training_args.do_eval  # and not is_torch_xla_available()  # type:ignore
            else None
        ),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics  # type:ignore
            if training_args.do_eval  # and not is_torch_xla_available()
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
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)  # type:ignore
        )
        assert train_dataset is not None
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))  # type:ignore

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        import datasets
        import pandas as pd

        logger.info("*** Evaluate ***")
        assert eval_dataset is not None, (
            "eval_dataset must be defined for evaluation."
        )

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)  # type:ignore
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))  # type:ignore
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # sample_dataset = eval_dataset.take(10)
        # assert sample_dataset is not None
        # # generate predictions
        # if sample_dataset is not None and isinstance(sample_dataset, datasets.Dataset):
        #     predictions = trainer.predict(sample_dataset)
        #     # decode predictions and labels
        #     predictions = decode_predictions(tokenizer, predictions)
        #     # add predictions to a wandb.Table
        #     predictions_df = pd.DataFrame(predictions)
        #     # predictions_df["epoch"] = state.epoch
        #     if wandb is not None and getattr(wandb, "run", None) is not None:
        #         records_table = wandb.Table(dataframe=predictions_df)
        #         # log the table to wandb
        #         assert wandb is not None and wandb.run is not None
        #         wandb.log({"sample_predictions": records_table})

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
        wandb.config.update(**kwargs)
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
