#!/usr/bin/env python
"""
hf_trainer.py

Complete, self-contained script for fine-tuning a model on a text file or a dataset for causal language modeling.

Modified from:
https://github.com/huggingface/transformers/blob/51ed61e2f05176f81fa7c9decba10cc28e138f61/examples/pytorch/language-modeling/run_clm.py
"""

import math
import os
import sys
import logging
from itertools import chain
from typing import Optional, Tuple
from dataclasses import dataclass, field

# Third-party imports - Grouped together
try:
    import torch
    import datasets
    from datasets import load_dataset
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
        PreTrainedTokenizerBase,
        PreTrainedModel,
        PretrainedConfig,
    )
    from transformers.testing_utils import CaptureLogger
    from transformers.trainer_utils import get_last_checkpoint
    from transformers.utils import send_example_telemetry
    from transformers.utils.versions import require_version

    # Define MODEL_TYPES after ensuring MODEL_FOR_CAUSAL_LM_MAPPING is loaded
    MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
    MODEL_TYPES = tuple(
        getattr(conf, "model_type", "") for conf in MODEL_CONFIG_CLASSES
    )

except (ImportError, ModuleNotFoundError) as e:
    print(
        f"Error importing Hugging Face libraries: {e}\n"
        'Please ensure "transformers", "datasets", and "torch" are installed.'
        ' Use "pip install transformers datasets torch"'
    )
    sys.exit(1)

# Optional imports
try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None  # type: ignore # Ignore type checking if wandb is not installed

try:
    import evaluate
except (ImportError, ModuleNotFoundError):
    evaluate = None # type: ignore # Ignore type checking if evaluate is not installed
    print(
        '"evaluate" library not found. Evaluation metrics will not be computed. '
        'Install with "pip install evaluate"'
    )


# Local application/library specific imports
try:
    import ezpz # Assuming ezpz is a local library
except (ImportError, ModuleNotFoundError):
    print(
        '"ezpz" library not found. Please ensure it is installed and accessible.'
    )
    sys.exit(1)

# --- Constants ---
# Using constants for split names improves readability and reduces errors
TRAIN_SPLIT_DEFAULT = "train"
VALIDATION_SPLIT_DEFAULT = "validation"
TEST_SPLIT_DEFAULT = "test"

# Logging configuration
logger = ezpz.get_logger(__name__) # Use ezpz's logger setup


# --- Argument Dataclasses ---
# (Kept mostly as-is, as they integrate with HfArgumentParser)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

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
    # Using constants for default split names
    train_split_name: Optional[str] = field(
        default=TRAIN_SPLIT_DEFAULT,
        metadata={
            "help": "The name of the train split to use (via the datasets library)."
        },
    )
    validation_split_name: Optional[str] = field(
        default=VALIDATION_SPLIT_DEFAULT,
        metadata={
            "help": "The name of the validation split to use (via the datasets library)."
        },
    )
    test_split_name: Optional[str] = field(
        default=TEST_SPLIT_DEFAULT,
        metadata={
            "help": "The name of the test split to use (via the datasets library)."
        },
    )
    train_split_str: Optional[str] = field( # Kept for potential slicing syntax like 'train[:10%]'
        default=None,
        metadata={
            "help": "The split string to use for the train split (e.g., 'train[:10%]', via the datasets library)."
        },
    )
    validation_split_str: Optional[str] = field( # Kept for potential slicing syntax
        default=None,
        metadata={
            "help": "The split string to use for the validation split (e.g., 'validation[50%:]', via the datasets library)."
        },
    )
    test_split_str: Optional[str] = field( # Kept for potential slicing syntax
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

        supported_extensions = ["csv", "json", "txt"]
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            if extension not in supported_extensions:
                raise ValueError(f"`train_file` should be one of: {supported_extensions}. Got: {extension}")
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            if extension not in supported_extensions:
                 raise ValueError(f"`validation_file` should be one of: {supported_extensions}. Got: {extension}")


# --- Helper Functions ---

def setup_logging(training_args: TrainingArguments, rank: int) -> None:
    """Configures logging for transformers and datasets libraries."""
    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    # Set verbosity based on rank - info for rank 0, warnings/errors otherwise
    log_level = logging.INFO if rank == 0 else logging.WARNING # Use standard logging levels
    transformers.utils.logging.set_verbosity(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # Enable standard handlers and formatting
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log basic process info
    logger.warning(
        f"Process rank: {rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16}"
    )
    if rank == 0:
        logger.info(f"Training/evaluation parameters: {training_args}")


def setup_wandb(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments, rank: int) -> None:
    """Initializes Weights & Biases if available and enabled."""
    if wandb is not None and rank == 0 and not os.environ.get("WANDB_DISABLED", False):
        try:
            run_name = training_args.run_name or f"hf_trainer_{model_args.model_name_or_path or model_args.model_type}"
            run = ezpz.setup_wandb(project_name="ezpz.hf_trainer", name=run_name) # Pass run_name if available
            if run is not None:
                # Combine args into a single config dictionary
                config_dict = {}
                config_dict.update(ezpz.get_dist_info())
                config_dict.update(vars(model_args)) # Use vars() for dataclasses
                config_dict.update(vars(data_args))
                config_dict.update(training_args.to_dict()) # TrainingArguments has a method
                run.config.update(config_dict, allow_val_change=True) # Allow changes if re-running
            else:
                logger.warning("ezpz.setup_wandb returned None, W&B initialization might have failed.")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")


def parse_arguments() -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    """Parses command line arguments into respective dataclasses."""
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse from JSON file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Parse from command line arguments
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


def check_output_dir(training_args: TrainingArguments) -> Optional[str]:
    """Checks for existing checkpoints in the output directory."""
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            # Optionally set resume_from_checkpoint if not explicitly provided
            # training_args.resume_from_checkpoint = last_checkpoint
    return last_checkpoint


def load_model_and_tokenizer(
    model_args: ModelArguments, training_args: TrainingArguments
) -> Tuple[PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel]:
    """Loads the model configuration, tokenizer, and model."""
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # Training from scratch
        config = CONFIG_MAPPING[model_args.model_type]() # type: ignore # Assume model_type is valid if no path/name
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides:
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
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "Cannot train from scratch without specifying a base model/tokenizer. "
            "Instantiating a new tokenizer from scratch is not supported by this script. "
            "Provide --model_name_or_path or --tokenizer_name."
        )

    # --- Load Model ---
    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "low_cpu_mem_usage": model_args.low_cpu_mem_usage,
    }

    # Handle torch_dtype
    if model_args.torch_dtype:
        try:
            torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype != "auto" else "auto"
            model_kwargs["torch_dtype"] = torch_dtype
        except AttributeError:
            logger.warning(f"Invalid torch_dtype specified: {model_args.torch_dtype}. Using default.")


    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            **model_kwargs
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model size: {n_params / 1e6:.2f}M parameters")


    # Resize embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing token embeddings from {embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    return config, tokenizer, model


def load_and_prepare_datasets(
    data_args: DataTrainingArguments,
    model_args: ModelArguments, # Needed for cache_dir, token, etc.
    training_args: TrainingArguments, # Needed for main_process_first
    tokenizer: PreTrainedTokenizerBase,
    config: PretrainedConfig,
) -> Tuple[Optional[datasets.Dataset], Optional[datasets.Dataset]]:
    """Loads raw datasets, preprocesses them (tokenization, grouping), and returns train/eval datasets."""

    # --- Load Raw Datasets ---
    logger.info("Loading raw datasets...")

    train_split_name = data_args.train_split_name or TRAIN_SPLIT_DEFAULT
    validation_split_name = data_args.validation_split_name or VALIDATION_SPLIT_DEFAULT
    test_split_name = data_args.test_split_name or TEST_SPLIT_DEFAULT # Although test isn't used later, keep consistent

    if data_args.dataset_name:
        # Load from Hugging Face Hub
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
            split=None # Load all available splits first
        )

        # Handle split creation/selection more robustly
        # Use specified split strings if provided
        train_split = data_args.train_split_str if data_args.train_split_str else train_split_name
        val_split = data_args.validation_split_str if data_args.validation_split_str else validation_split_name

        # Check if required splits exist
        if train_split_name not in raw_datasets and not data_args.train_split_str:
             raise ValueError(f"Train split '{train_split_name}' not found in dataset '{data_args.dataset_name}'. Available: {list(raw_datasets.keys())}")

        # Handle validation split creation if needed and possible
        if training_args.do_eval and validation_split_name not in raw_datasets and not data_args.validation_split_str:
            logger.info(f"Validation split '{validation_split_name}' not found. Creating from train split.")
            if data_args.validation_split_percentage is None or not (0 < data_args.validation_split_percentage < 100):
                raise ValueError("Cannot create validation split without a valid --validation_split_percentage (1-99).")

            # Try splitting using datasets built-in method if not streaming
            if not data_args.streaming and hasattr(raw_datasets[train_split_name], "train_test_split"):
                split_percentage = data_args.validation_split_percentage / 100.0
                split_dataset = raw_datasets[train_split_name].train_test_split(
                    test_size=split_percentage, shuffle=True, seed=training_args.seed
                )
                raw_datasets[train_split_name] = split_dataset["train"]
                raw_datasets[validation_split_name] = split_dataset["test"]
                logger.info(f"Created validation split ({data_args.validation_split_percentage}%) from '{train_split_name}'.")
            else:
                # Fallback or handle streaming case (manual slicing or using the full train set might be needed)
                logger.warning(
                    f"Cannot automatically create validation split from '{train_split_name}' "
                    f"(streaming={data_args.streaming} or split method unavailable). "
                    f"Using full '{train_split_name}' for validation as fallback."
                )
                raw_datasets[validation_split_name] = raw_datasets[train_split_name] # Use full train as val


        # Select the desired splits based on names or string specifiers
        final_raw_datasets = datasets.DatasetDict()
        if training_args.do_train:
            final_raw_datasets[train_split_name] = raw_datasets[train_split]
        if training_args.do_eval:
            final_raw_datasets[validation_split_name] = raw_datasets[val_split]
        # Add test split if needed for prediction later
        # if training_args.do_predict and test_split_name in raw_datasets:
        #     final_raw_datasets[test_split_name] = raw_datasets[test_split_name]

        raw_datasets = final_raw_datasets # Overwrite with the selected/created splits


    else:
        # Load from local files
        data_files = {}
        if data_args.train_file:
            data_files[train_split_name] = data_args.train_file
        if data_args.validation_file:
            data_files[validation_split_name] = data_args.validation_file

        if not data_files:
             raise ValueError("No train or validation files provided when not using `dataset_name`.")

        # Determine file extension
        first_file = data_args.train_file or data_args.validation_file
        extension = first_file.split(".")[-1] # type: ignore

        if extension == "txt":
            extension = "text"
            dataset_args = {"keep_linebreaks": data_args.keep_linebreaks}
        elif extension in ["csv", "json"]:
            dataset_args = {}
        else:
             raise ValueError(f"Unsupported file extension: {extension}. Use csv, json, or txt.")

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            split=None, # Load all splits defined in data_files
            **dataset_args,
        )

        # Handle validation split creation if only train_file is provided
        if training_args.do_eval and validation_split_name not in raw_datasets:
            logger.info(f"Validation file not provided. Creating validation split from '{train_split_name}'.")
            if train_split_name not in raw_datasets:
                 raise ValueError("Cannot create validation split as train split is missing.")
            if data_args.validation_split_percentage is None or not (0 < data_args.validation_split_percentage < 100):
                 raise ValueError("Cannot create validation split without a valid --validation_split_percentage (1-99).")

            if not data_args.streaming and hasattr(raw_datasets[train_split_name], "train_test_split"):
                split_percentage = data_args.validation_split_percentage / 100.0
                split_dataset = raw_datasets[train_split_name].train_test_split(
                    test_size=split_percentage, shuffle=True, seed=training_args.seed
                )
                raw_datasets[train_split_name] = split_dataset["train"]
                raw_datasets[validation_split_name] = split_dataset["test"]
                logger.info(f"Created validation split ({data_args.validation_split_percentage}%) from '{train_split_name}'.")
            else:
                 logger.warning(
                     f"Cannot automatically create validation split from '{train_split_name}' "
                     f"(streaming={data_args.streaming} or split method unavailable). "
                     f"Proceeding without validation data."
                 )
                 # Decide how to handle this: maybe skip eval or raise error?
                 # For now, let it proceed, eval_dataset will be None later if split doesn't exist


    # --- Preprocessing ---
    logger.info("Preprocessing datasets...")

    # Determine text column
    # Use the first split available (train or validation) to get features
    sample_split = train_split_name if train_split_name in raw_datasets else validation_split_name
    if sample_split not in raw_datasets:
        raise ValueError(f"No usable split ('{train_split_name}' or '{validation_split_name}') found in raw datasets for preprocessing.")

    column_names = list(raw_datasets[sample_split].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    logger.info(f"Using '{text_column_name}' as the text column.")

    # Tokenization function
    # Needs tokenizer and text_column_name from the outer scope
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            # Handle potential None values in the text column gracefully
            texts_to_tokenize = [str(t) if t is not None else "" for t in examples[text_column_name]]
            output = tokenizer(texts_to_tokenize)
        # Log warning suppression only if the specific warning occurred
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "Ignoring tokenizer warnings about long sequences; input will be chunked."
            )
        return output

    # Apply tokenization
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names, # Remove original columns after tokenization
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Determine block size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        # Guard against models with very large or undefined max length
        if block_size > getattr(config, 'max_position_embeddings', 1024) or block_size > 2048: # Added a practical upper limit
            effective_max_len = getattr(config, 'max_position_embeddings', 1024)
            chosen_block_size = min(1024, effective_max_len if effective_max_len > 0 else 1024)
            logger.warning(
                f"Tokenizer `model_max_length` ({block_size}) is large or potentially problematic. "
                f"Using block_size={chosen_block_size} based on config or default (1024). "
                f"Override with --block_size."
            )
            block_size = chosen_block_size
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"Specified block_size ({data_args.block_size}) > tokenizer max length ({tokenizer.model_max_length}). "
                f"Using tokenizer max length: {tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    if block_size <= 0:
        raise ValueError(f"Calculated invalid block_size: {block_size}. Check model config and --block_size.")

    logger.info(f"Using block size: {block_size}")


    # Grouping function
    # Needs block_size from the outer scope
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size we exclude this batch and return an empty dict.
        if total_length < block_size:
            return {k: [] for k in examples.keys()} # Return empty dict correctly

        # Drop the remainder to ensure full blocks
        total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Create labels (shifted inside the model)
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply grouping
    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    # --- Select Final Datasets and Apply Sampling ---
    train_dataset = None
    eval_dataset = None

    if training_args.do_train:
        if train_split_name not in lm_datasets:
            raise ValueError(f"--do_train requires a '{train_split_name}' dataset, but it's not available after processing.")
        train_dataset = lm_datasets[train_split_name]
        if data_args.max_train_samples:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            logger.info(f"Truncating train dataset to {max_train_samples} samples.")
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if validation_split_name not in lm_datasets:
             logger.warning(f"--do_eval requires a '{validation_split_name}' dataset, but it's not available. Skipping evaluation.")
             # Ensure eval_dataset remains None if split is missing
        else:
            eval_dataset = lm_datasets[validation_split_name]
            if data_args.max_eval_samples:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                logger.info(f"Truncating evaluation dataset to {max_eval_samples} samples.")
                eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


def initialize_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingArguments,
    train_dataset: Optional[datasets.Dataset],
    eval_dataset: Optional[datasets.Dataset],
    model_args: ModelArguments, # Needed for metric cache_dir
) -> Trainer:
    """Initializes the Hugging Face Trainer."""

    # --- Compute Metrics Function (if evaluating) ---
    compute_metrics = None
    if training_args.do_eval and eval_dataset is not None:
        if evaluate is None:
            logger.warning("Evaluate library not found. Cannot compute metrics.")
        elif is_torch_xla_available():
             logger.warning("Metrics computation is not supported with TPU currently (is_torch_xla_available=True).")
        else:
            try:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

                def compute_metrics_fn(eval_preds):
                    # eval_preds are predictions and labels after potential slicing by Trainer
                    preds, labels = eval_preds
                    # Preprocess_logits_for_metrics may return tuple, get first element (logits)
                    if isinstance(preds, tuple):
                        preds = preds[0]

                    # Get actual predictions
                    preds = preds.argmax(dim=-1)

                    # Shift labels and preds for accuracy calculation in CLM
                    # Labels are typically shifted inside the model, so align predictions
                    labels = labels[:, 1:].reshape(-1) # Remove first token label (no prediction for it)
                    preds = preds[:, :-1].reshape(-1) # Remove last token prediction (no label for it)

                    # Filter out padding tokens (-100)
                    valid_indices = labels != -100
                    labels = labels[valid_indices]
                    preds = preds[valid_indices]

                    if len(labels) == 0: # Handle case where all labels might be masked
                        return {"accuracy": 0.0}

                    return metric.compute(predictions=preds, references=labels)

                compute_metrics = compute_metrics_fn
                logger.info("Accuracy metric loaded for evaluation.")

            except Exception as e:
                logger.error(f"Failed to load accuracy metric: {e}. Evaluation will proceed without metrics.")

    # Preprocess logits function (only needed if compute_metrics is defined)
    # This helps save memory by only keeping argmax predictions
    preprocess_logits_for_metrics = None
    if compute_metrics:
         def preprocess_logits_for_metrics_fn(logits, labels):
             if isinstance(logits, tuple):
                 # Depending on the model and config, logits may contain extra tensors,
                 # like past_key_values, but logits always come first
                 logits = logits[0]
             return logits.argmax(dim=-1) # Return simple argmax for accuracy
         preprocess_logits_for_metrics = preprocess_logits_for_metrics_fn


    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer, # Pass tokenizer for saving purposes
        data_collator=default_data_collator, # Use default collator which handles labels for CLM
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    return trainer


# --- Main Execution ---

def main():
    # 1. Setup distributed environment (if applicable) and parse args
    rank = ezpz.setup_torch() # Assuming this sets up DDP/device placement
    model_args, data_args, training_args = parse_arguments()

    # 2. Setup logging, W&B, telemetry after parsing args
    setup_logging(training_args, rank)
    setup_wandb(model_args, data_args, training_args, rank) # Needs all args for config
    # Send telemetry (as in original script)
    send_example_telemetry("hf_trainer_refactored", model_args, data_args) # Use modified name potentially

    # 3. Set seed for reproducibility
    set_seed(training_args.seed)

    # 4. Check for existing checkpoints
    last_checkpoint = check_output_dir(training_args)
    # If resuming, HfArgumentParser handles overriding, but `check_output_dir` logs the detection.

    # 5. Load model and tokenizer
    config, tokenizer, model = load_model_and_tokenizer(model_args, training_args)

    # 6. Load and prepare datasets
    train_dataset, eval_dataset = load_and_prepare_datasets(
        data_args, model_args, training_args, tokenizer, config
    )

    # 7. Initialize Trainer
    trainer = initialize_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_args=model_args # Pass for metric cache dir
    )

    # 8. Optional: Watch model with W&B
    if wandb is not None and rank == 0 and hasattr(wandb, 'run') and wandb.run is not None:
        try:
            wandb.watch(model, log="all", log_freq=max(100, training_args.logging_steps or 500)) # Log gradients/parameters
            logger.info("W&B watching model.")
        except Exception as e:
            logger.error(f"Failed to set up W&B model watch: {e}")


    # 9. Training
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save model, tokenizer, and training state
        # trainer.save_model() # Saves the tokenizer too is tokenizer is passed to Trainer init
        # Workaround for potential FSDP issues, save on rank 0 - Trainer handles this internally now usually
        # If you need explicit rank 0 saving, use trainer.save_model() within if rank == 0 block.
        # The trainer handles saving based on training_args.save_strategy etc.
        # Let's ensure the state is saved.
        trainer.save_state() # Saves optimizer, scheduler, RNG state etc.

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset) # type: ignore # Guarded by do_train
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset)) # type: ignore

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # 10. Evaluation
    if training_args.do_eval:
        if eval_dataset is None:
             logger.warning("Evaluation dataset not available, skipping evaluation.")
        else:
            logger.info("*** Starting Evaluation ***")
            metrics = trainer.evaluate()

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset) # type: ignore # Guarded by do_eval
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset)) # type: ignore

            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # 11. Write final arguments to a file (optional but good practice)
    if rank == 0: # Only save args on main process
        output_dir = training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        all_args = {
            "model_args": vars(model_args),
            "data_args": vars(data_args),
            "training_args": training_args.to_dict(),
        }
        import json
        args_save_path = os.path.join(output_dir, "run_args.json")
        try:
            with open(args_save_path, "w") as f:
                json.dump(all_args, f, indent=4)
            logger.info(f"Run arguments saved to {args_save_path}")
        except Exception as e:
             logger.error(f"Failed to save run arguments to {args_save_path}: {e}")


    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
