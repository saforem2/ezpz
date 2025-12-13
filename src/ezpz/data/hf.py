"""
ezpz/datasets/hf.py

HuggingFace Datasets loading and tokenization.
"""

import os
from typing import Optional
import ezpz

from torch.utils.data import (
    IterableDataset,
    DistributedSampler,
    DataLoader,
    Dataset,
)

import datasets
from pathlib import Path
from transformers import AutoTokenizer

from ezpz.configs import (
    HfModelArguments,
    HfDataTrainingArguments,
)

logger = ezpz.get_logger(__name__)


from torch.utils.data import get_worker_info


def get_data_parallel_map_dataset(
    dataset: Dataset,
    rank: int,
    batch_size: int,
    num_workers: int = 0,
):
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=ezpz.get_world_size(),
        rank=rank,
        shuffle=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )


class DataParallelIterableDataset(IterableDataset):
    # def __init__(self, dataset: datasets.IterableDataset):
    #     self.dataset = dataset

    def __len__(self):
        return len(self)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        world_size = ezpz.get_world_size()
        rank = ezpz.get_rank()
        sampler = DistributedSampler(
            dataset=self,
            num_replicas=(num_workers * world_size),
            rank=(rank * num_workers + worker_id),
            shuffle=False,
        )
        for i in iter(sampler):
            yield i


def split_dataset(
    data_args: HfDataTrainingArguments,
    train_split_name: str = "train",
    validation_split_name: Optional[str] = None,
    cache_dir: Optional[str | os.PathLike | Path] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    # model_args: HfModelArguments,
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
    cache_dir = (
        Path("./.cache/hf/datasets") if cache_dir is None else cache_dir
    )
    assert cache_dir is not None and isinstance(cache_dir, (str, os.PathLike))
    cache_dir = Path(cache_dir).as_posix()
    if validation_split_name is not None:
        try:
            dsets[validation_split_name] = datasets.load_dataset(  # type:ignore
                dataset_name,
                data_args.dataset_config_name,
                split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                cache_dir=cache_dir,
                token=token,
                streaming=data_args.streaming,
                trust_remote_code=trust_remote_code,
            )
            dsets[train_split_name] = datasets.load_dataset(  # type: ignore
                dataset_name,
                data_args.dataset_config_name,
                split=f"{train_split_name}[{data_args.validation_split_percentage}%:]",
                cache_dir=cache_dir,
                token=token,
                streaming=data_args.streaming,
                trust_remote_code=trust_remote_code,
            )
        except ValueError:
            # In some cases, the dataset doesn't support slicing.
            # In this case, we just use the full training set as validation set.
            dsets[validation_split_name] = datasets.load_dataset(  # type:ignore
                dataset_name,
                data_args.dataset_config_name,
                split=train_split_name,
                cache_dir=cache_dir,
                token=token,
                streaming=data_args.streaming,
                trust_remote_code=trust_remote_code,
            )
            try:
                dsets[train_split_name] = datasets.load_dataset(  # type:ignore
                    dataset_name,
                    data_args.dataset_config_name,
                    split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                    cache_dir=cache_dir,
                    token=token,
                    streaming=data_args.streaming,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                # In some cases, the dataset doesn't support slicing.
                # In this case, we just use the full training set as validation set.
                dsets[train_split_name] = datasets.load_dataset(  # type:ignore
                    dataset_name,
                    data_args.dataset_config_name,
                    split=train_split_name,
                    cache_dir=cache_dir,
                    token=token,
                    streaming=data_args.streaming,
                    trust_remote_code=trust_remote_code,
                )

    if data_args.streaming:
        return datasets.IterableDatasetDict(dsets)
    return datasets.DatasetDict(dsets)


def get_hf_datasets(
    data_args,
    tokenizer_name: Optional[str],
    model_name_or_path: Optional[str],
    cache_dir: Optional[str | os.PathLike | Path] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    do_train: bool = False,
    use_fast_tokenizer: bool = True,
    revision: str = "main",
):
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
    cache_dir = (
        Path("./.cache/hf/datasets") if cache_dir is None else cache_dir
    )
    assert cache_dir is not None and isinstance(cache_dir, (str, os.PathLike))
    cache_dir = Path(cache_dir).as_posix()
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # dataset = datasets.load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir=cache_dir,
        #     token=model_args.token,
        #     streaming=data_args.streaming,
        #     trust_remote_code=model_args.trust_remote_code,
        # )
        raw_datasets = split_dataset(
            data_args,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code,
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
            cache_dir=cache_dir,
            token=token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if validation_split_name not in raw_datasets.keys():  # type:ignore
            raw_datasets[validation_split_name] = datasets.load_dataset(  # type:ignore
                extension,
                data_files=data_files,
                split=f"{train_split_name}[:{data_args.validation_split_percentage}%]",
                cache_dir=cache_dir,
                token=token,
                **dataset_args,
            )
            raw_datasets[train_split_name] = datasets.load_dataset(  # type:ignore
                extension,
                data_files=data_files,
                split=f"{train_split_name}[{data_args.validation_split_percentage}%:]",
                cache_dir=cache_dir,
                token=token,
                **dataset_args,
            )

    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "use_fast": use_fast_tokenizer,
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_kwargs
        )
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if wandb is not None and getattr(wandb, "run", None) is not None:
    #     wandb.watch(model, log="all")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if do_train:
        column_names = list(raw_datasets[train_split_name].features)  # type:ignore
    else:
        column_names = list(raw_datasets[validation_split_name].features)  # type:ignore
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger(
    #     "transformers.tokenization_utils_base"
    # )

    def tokenize_function(examples):
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
    #     metric = evaluate.load("accuracy", cache_dir=cache_dir)
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
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)
