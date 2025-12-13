"""
generate.py
"""

import torch
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

import ezpz


def parse_args():
    """
    Parse command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using a model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type to use for the model.",
    )
    return parser.parse_args()


def prompt_model(model, tokenizer, prompt, max_length: int = 64, **kwargs) -> str:
    """
    Generate text using a model and tokenizer.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for encoding and decoding.
        prompt: The input prompt to generate text from.
        max_length: The maximum length of the generated text.
        **kwargs: Additional arguments to pass to the model's generate method.

    Example:

        >>> import ezpz
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer #, Trainer, TrainingArguments
        >>> import torch
        >>> model_name = "argonne-private/AuroraGPT-7B"
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> model = AutoModelForCausalLM.from_pretrained(model_name)
        >>> model.to(ezpz.get_torch_device_type())
        >>> model.to(torch.bfloat16)
        >>> result = tokenizer.batch_decode(model.generate(**tokenizer("Who are you?", return_tensors="pt").to(ezpz.get_torch_device_type()), max_length=128))
    """
    return tokenizer.batch_decode(
        model.generate(
            **tokenizer(prompt, return_tensors="pt").to(ezpz.get_torch_device_type()),
            max_length=max_length,
            **kwargs,
        )
    )


def main():
    args = parse_args()
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # model.resize_token_embeddings(len(tokenizer))
    model.to(ezpz.get_torch_device_type())
    if args.dtype in {"bfloat16", "bf16", "b16"}:
        model.to(torch.bfloat16)
    elif args.dtype in {"float16", "fp16", "f16"}:
        model.to(torch.float16)
    elif args.dtype in {"float32", "fp32", "f32"}:
        model.to(torch.float32)
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    print(f"Model loaded: {model_name}")
    print("Enter a prompt. Type 'exit' to quit.")
    while True:
        try:
            prompt = str(input("Enter a prompt: "))
            if str(prompt.lower().strip("").strip("")) == "exit":
                print("Exiting!")
                break
            max_length = int(input("Enter max length: "))
            print(prompt_model(model, tokenizer, prompt, max_length))
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
