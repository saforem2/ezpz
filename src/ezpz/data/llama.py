"""
ezpz/data/llama.py
"""

from typing import Any

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class LlamaDataLoader:
    def __init__(
        self,
        dataset_repo: str,
        tokenizer_name: str = "hf-internal-testing/llama-tokenizer",
        max_length: int = 512,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 2,
        split: str = "train",
    ):
        """
        Initializes the LlamaDataLoader.

        Args:
            dataset_repo (str): Hugging Face dataset repository path.
            tokenizer_name (str): Name or path of the LLaMA tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): Number of workers for data loading.
            split (str): Dataset split to load (e.g., "train", "validation").
        """
        self.dataset_repo = dataset_repo
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.split = split

        # Load dataset and tokenizer
        self.dataset = self._load_dataset()
        self.tokenizer = self._load_tokenizer()

    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        return load_dataset(self.dataset_repo, split=self.split)

    def _load_tokenizer(self):
        """Load the LLaMA tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

        return tokenizer

    def _tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        """
        Tokenizes the input examples.

        Args:
            examples (Dict[str, Any]): A batch of examples from the dataset.

        Returns:
            Dict[str, Any]: Tokenized examples with input_ids and attention_mask.
        """
        return self.tokenizer(
            examples["text"],  # Replace "text" with the appropriate key in your dataset
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def get_data_loader(self) -> DataLoader:
        """
        Creates and returns a PyTorch DataLoader.

        Returns:
            DataLoader: A PyTorch DataLoader for the tokenized dataset.
        """
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"],  # Remove non-tokenized columns
        )

        # Convert to PyTorch format
        tokenized_dataset.set_format(  # type:ignore
            type="torch", columns=["input_ids", "attention_mask"]
        )

        # Create DataLoader
        data_loader = DataLoader(
            tokenized_dataset,  # type:ignore
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        return data_loader


# Example usage
if __name__ == "__main__":
    # Replace with your Hugging Face dataset repo path
    dataset_repo = "eliplutchok/fineweb-small-sample"
    llama_data_loader = LlamaDataLoader(dataset_repo=dataset_repo)
    # Get the DataLoader
    data_loader = llama_data_loader.get_data_loader()
    # Iterate through the DataLoader
    for batch in data_loader:
        print(batch["input_ids"].shape)  # Example: torch.Size([8, 512])
        print(batch["attention_mask"].shape)  # Example: torch.Size([8, 512])
        break  # Just show the first batch
