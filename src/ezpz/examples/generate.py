"""Interactive text generation loop for Hugging Face causal language models.

Launch with:

    ezpz launch -m ezpz.examples.generate --model_name <repo/model>

Help output (``python3 -m ezpz.examples.generate --help``):

    usage: generate.py [-h] [--model_name MODEL_NAME]
                       [--dtype {float16,bfloat16,float32}]

    Generate text using a model.

    options:
      -h, --help            show this help message and exit
      --model_name MODEL_NAME
                            Name of the model to use.
      --dtype {float16,bfloat16,float32}
                            Data type to use for the model.
"""

import torch
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

import ezpz
from ezpz.cli.flags import build_generate_parser


def parse_args():
    """Parse CLI arguments for interactive generation."""
    parser = build_generate_parser()
    return parser.parse_args()


def prompt_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 64,
    **kwargs: object,
) -> str:
    """Generate text using a model and tokenizer.

    Args:
        model: Causal LM used for generation.
        tokenizer: Tokenizer that encodes/decodes text.
        prompt: Input prompt to seed generation.
        max_length: Maximum number of tokens to generate.
        **kwargs: Extra parameters forwarded to ``model.generate``.

    Returns:
        Decoded text returned by the model.

    Examples:
        >>> model_name = \"argonne-private/AuroraGPT-7B\"
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> model = AutoModelForCausalLM.from_pretrained(model_name)
        >>> _ = prompt_model(model, tokenizer, \"Who are you?\", max_length=32)
    """
    return tokenizer.batch_decode(
        model.generate(
            **tokenizer(prompt, return_tensors="pt").to(ezpz.get_torch_device_type()),
            max_length=max_length,
            **kwargs,
        )
    )


def main():
    """Load a model and enter an interactive text generation REPL."""
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
