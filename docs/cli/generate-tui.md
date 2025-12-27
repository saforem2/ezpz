# 🖥️ `ezpz-generate-tui`

> [!WARNING]
> Experimental / not well tested. Expect rough edges and be ready to fall back
> to a simpler script if things misbehave.

Textual TUI for interactive prompt-based text generation.

- Load a Hugging Face causal LM + tokenizer, choose dtype and max length, and
  generate responses from a simple UI.
- Uses `ezpz.get_torch_device_type()` to pick the accelerator.
- Keyboard: `Ctrl+Enter` to generate, `q` or `Ctrl+C` to quit.

## Usage

```bash
# direct console script
ezpz-generate-tui

# module form (works with ezpz launch)
python -m ezpz.examples.generate_tui
ezpz launch -- python -m ezpz.examples.generate_tui
```

### Defaults and controls

- Default model: `meta-llama/Llama-3.2-1B`
- Default dtype: `bfloat16` (`float16` and `float32` also available)
- Default max length: `128`
- Buttons: Load Model, Generate (⌃⏎), Stop, Clear output

### Notes

- Requires `textual`, `torch`, and `transformers` installed; GPU strongly
  recommended for any non-toy model.
- Generation runs on a worker thread to keep the UI responsive; cancelling uses
  the Stop button.
