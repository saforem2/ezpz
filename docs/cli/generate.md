# ✍️ `ezpz-generate`

/// details | Warning
    type: warning

Experimental / not well tested. The flow works in simple cases but has not
seen broad coverage—treat as best-effort and be ready to fall back to your
own HF script if needed.

///

Interactive text generation loop for Hugging Face causal language models.

- Loads a model and tokenizer via 🤗 `transformers`
- Moves the model to the device detected by `ezpz.get_torch_device_type()`
- Prompts you for text and a max length, then streams a single completion

## Usage

```bash
# direct console script
ezpz-generate --model_name meta-llama/Llama-3.2-1B --dtype bfloat16

# equivalent module form (useful with ezpz launch)
python -m ezpz.examples.generate --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
ezpz launch -- python -m ezpz.examples.generate --model_name meta-llama/Llama-3.2-1B
```

### Flags

- `--model_name` (default: `meta-llama/Llama-3.2-1B`): Hugging Face repo/model to load.
- `--dtype` (default: `bfloat16`, choices: `float16|bfloat16|float32`): Torch dtype for the model.

At runtime the script will prompt for:

- `prompt`: Text to feed the model.
- `max length`: Token limit passed to `model.generate`.

### Notes

- Expects `torch` and `transformers` to be installed and a compatible accelerator
  available (GPU strongly recommended).
- Tokenizer `pad_token` is set to `eos_token` before generation.
- Type “exit” at the prompt or press `Ctrl+C` to quit.
