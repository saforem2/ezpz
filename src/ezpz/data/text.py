"""
ezpz/data/text.py
"""

from itertools import chain

import ezpz
import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_length: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.tokens = torch.randint(
            self.vocab_size,
            size=(len(self), self.seq_length + 1),
            # Set a seed to make this toy dataset the same on each rank
            # Fabric will add a `DistributedSampler` to shard the data
            # correctly
            generator=torch.Generator().manual_seed(
                (ezpz.get_rank() + 1) * 1234
            ),
        )

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, item: int):
        return self.tokens[item]


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m = f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                try:
                    m = f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator)["text"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs["input_ids"]:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)


def get_hf_dataset(
    dataset: str = "eliplutchok/fineweb-small-sample",
    tokenizer_name: str = "meta-llama/llama-2-7b-hf",
):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True
        )

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset(dataset)  # , streaming=True)
    dataset = dataset.with_format("torch")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


# def get_dataloader(batch_size):
#     # tokenized_datasets = get_dataset()
#     train_dataset = tokenized_datasets['train']
#
#     def collate_tokenize(data):
#         text_batch = [element['text'] for element in data]
#         tokenized = tokenizer(
#             text_batch, padding='longest', truncation=True, return_tensors='pt'
#         )
#         return tokenized
#
#     dataloader = DataLoader(
#         train_dataset, batch_size=batch_size, collate_fn=collate_tokenize
#     )
#
#     return dataloader


def get_random_dataset(
    batch_size: int, vocab_size: int, seq_length: int
) -> dict:
    return {
        "dataset": (
            dset := RandomTokenDataset(
                vocab_size=vocab_size, seq_length=seq_length
            )
        ),
        "dataloader": DataLoader(dset, batch_size=batch_size),
    }


def get_hf_data(dataset_repo: str = "eliplutchok/fineweb-small-sample"):
    from ezpz.data.llama import LlamaDataLoader

    llama_data_loader = LlamaDataLoader(dataset_repo=dataset_repo)
    return llama_data_loader


# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples, max_seq_length):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for k, t in concatenated_examples.items()
    }
    return result
