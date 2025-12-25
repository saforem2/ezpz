"""Textual TUI for interactive prompt-based text generation.

Launch with:

    ezpz launch -m ezpz.examples.generate_tui

Help output (``python3 -m ezpz.examples.generate_tui --help``):

    (No CLI flags; run the module directly to start the UI.)

Notes:
  - Load a model, then enter a prompt and hit Generate (Ctrl+Enter works too).
  - Press 'q' or Ctrl+C to quit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import torch
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

import ezpz  # uses ezpz.get_torch_device_type()

# --- Textual UI ---
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    RichLog,
)

DTYPE_CHOICES = [
    ("bfloat16", "bfloat16"),
    ("float16", "float16"),
    ("float32", "float32"),
]

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAXLEN = "128"


def prompt_model(
    model, tokenizer, prompt: str, max_length: int = 64, **kwargs
) -> str:
    """Generate text using a model and tokenizer."""
    outputs = model.generate(
        **tokenizer(prompt, return_tensors="pt").to(ezpz.get_torch_device_type()),
        max_length=max_length,
        **kwargs,
    )
    decoded = tokenizer.batch_decode(outputs)
    # return first sequence for simplicity
    return decoded[0]


@dataclass
class LoadedModel:
    """Container for a loaded causal LM and tokenizer."""

    model_name: str
    dtype: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


class GenerateApp(App):
    """Textual application to load a model and generate text via a TUI."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
    }

    #left {
        width: 38%;
        min-width: 36;
        max-width: 72;
        border: solid $accent 10%;
        padding: 1 2;
        height: 1fr;
    }

    #right {
        layout: vertical;
        border: solid $accent 10%;
        padding: 1 2;
        height: 1fr;
    }

    .field {
        margin-bottom: 1;
    }

    .field Label {
        width: 100%;
        color: $text 80%;
    }

    #controls {
        height: auto;
        content-align: center middle;
        padding-top: 1;
    }

    #prompt_input {
        dock: top;
        height: auto;
    }

    #output_log {
        height: 1fr;
        border: solid $accent 5%;
        padding: 1 1;
    }

    #status_bar {
        height: auto;
        color: $text 70%;
    }
    """

    BINDINGS = [
        Binding("ctrl+enter", "generate", "Generate"),
        Binding("q", "quit", "Quit"),
    ]

    # reactive state
    status: reactive[str] = reactive("Ready.")
    is_loading: reactive[bool] = reactive(False)
    is_generating: reactive[bool] = reactive(False)

    # loaded model holder
    lm: Optional[LoadedModel] = None

    def compose(self) -> ComposeResult:
        """Build the Textual layout."""
        yield Header(show_clock=True)
        with Container(id="body"):
            # Left panel: settings
            with VerticalScroll(id="left"):
                yield Label("Model Settings", classes="field")
                yield Label("Model name", classes="field")
                self.model_name = Input(value=DEFAULT_MODEL, placeholder="hf/org-or-user/model", id="model_name")
                yield self.model_name

                yield Label("DType", classes="field")
                self.dtype_select = Select(((label, value) for label, value in DTYPE_CHOICES), value=DEFAULT_DTYPE, id="dtype")
                yield self.dtype_select

                yield Label("Max length", classes="field")
                self.maxlen_input = Input(value=DEFAULT_MAXLEN, placeholder="e.g., 128", id="maxlen")
                yield self.maxlen_input

                with Horizontal(id="controls"):
                    self.load_btn = Button("Load Model", id="load_btn", variant="primary")
                    self.clear_btn = Button("Clear", id="clear_btn")
                    yield self.load_btn
                    yield self.clear_btn

                yield Static("", id="status_bar")

            # Right panel: prompt + output
            with Container(id="right"):
                yield Label("Prompt", id="prompt_label")
                self.prompt_input = Input(placeholder="Type your prompt…", id="prompt_input")
                yield self.prompt_input

                with Horizontal():
                    self.gen_btn = Button("Generate  ⌃⏎", id="gen_btn", variant="success", disabled=True)
                    self.stop_btn = Button("Stop", id="stop_btn", disabled=True)
                    yield self.gen_btn
                    yield self.stop_btn

                yield Label("Output")
                # self.output_log = TextArea(id="output_log", highlight=True, wrap=True)
                # self.output_log = Log(id="output_log", highlight=True, classes="output")
                self.output_log = RichLog(id="output_log", highlight=True)
                yield self.output_log

        yield Footer()

    # --- helpers ---

    def _set_status(self, msg: str) -> None:
        """Update the status bar and reactive state."""
        self.status = msg
        status_bar = self.query_one("#status_bar", Static)
        status_bar.update(Text(f"Status: {msg}"))

    def _validate_maxlen(self) -> int:
        """Parse and validate the max length field."""
        try:
            ml = int(self.maxlen_input.value.strip())
            if ml <= 0:
                raise ValueError
            return ml
        except Exception:
            raise ValueError("max_length must be a positive integer")

    def _dtype_to_torch(self, dtype: str):
        """Map string dtype choices to torch dtypes."""
        d = dtype.lower()
        if d in {"bfloat16", "bf16", "b16"}:
            return torch.bfloat16
        if d in {"float16", "fp16", "f16"}:
            return torch.float16
        if d in {"float32", "fp32", "f32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    # --- events ---

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for load/generate/stop/clear."""
        bid = event.button.id
        if bid == "load_btn":
            await self._handle_load()
        elif bid == "gen_btn":
            await self._handle_generate()
        elif bid == "stop_btn":
            await self._handle_stop()
        elif bid == "clear_btn":
            self.output_log.clear()
            self.prompt_input.value = ""
            self._set_status("Cleared.")

    async def action_generate(self) -> None:
        """Keyboard binding for generate (Ctrl+Enter)."""
        await self._handle_generate()

    # --- actions ---

    async def _handle_load(self) -> None:
        """Load a HF model/tokenizer asynchronously."""
        if self.is_loading:
            return
        model_name = self.model_name.value.strip() or DEFAULT_MODEL
        dtype = self.dtype_select.value or DEFAULT_DTYPE

        self.is_loading = True
        self._set_status(f"Loading model '{model_name}' as {dtype}…")
        self.load_btn.disabled = True
        self.gen_btn.disabled = True

        try:
            # offload heavy HF load to a thread to avoid blocking the UI
            loaded: LoadedModel = await asyncio.to_thread(self._load_model_blocking, model_name, dtype)
            self.lm = loaded
            self._set_status(f"Model loaded: {loaded.model_name} [{dtype}] on {ezpz.get_torch_device_type()}")
            self.output_log.write(Text(f"[bold green]Loaded[/] {loaded.model_name} ({dtype})").plain)
            self.gen_btn.disabled = False
        except Exception as e:
            self.output_log.write(Text(f"[red]Load failed:[/] {e}").plain)
            self._set_status("Load failed. See output.")
        finally:
            self.is_loading = False
            self.load_btn.disabled = False

    def _load_model_blocking(self, model_name: str, dtype: str) -> LoadedModel:
        """Load model/tokenizer on a worker thread."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(ezpz.get_torch_device_type())
        model.to(self._dtype_to_torch(dtype))
        return LoadedModel(model_name=model_name, dtype=dtype, model=model, tokenizer=tokenizer)

    async def _handle_generate(self) -> None:
        """Generate text for the current prompt asynchronously."""
        if self.is_generating:
            return
        if not self.lm:
            self._set_status("Load a model first.")
            self.output_log.write("[yellow]Load a model first.[/]")
            return

        prompt = self.prompt_input.value.strip()
        if not prompt:
            self._set_status("Enter a prompt.")
            return

        try:
            maxlen = self._validate_maxlen()
        except ValueError as ve:
            self._set_status(str(ve))
            self.output_log.write(f"[yellow]{ve}[/]")
            return

        self.is_generating = True
        self.gen_btn.disabled = True
        self.stop_btn.disabled = False
        self._set_status("Generating…")

        # Do generation in a thread; keep a cancellation token
        self._current_task = asyncio.create_task(
            asyncio.to_thread(self._generate_blocking, prompt, maxlen)
        )

        try:
            result = await self._current_task
            self.output_log.write("[bold]Prompt:[/] " + prompt)
            self.output_log.write("[bold cyan]Output:[/]\n" + result)
            self._set_status("Done.")
        except asyncio.CancelledError:
            self.output_log.write("[yellow]Generation cancelled.[/]")
            self._set_status("Cancelled.")
        except Exception as e:
            self.output_log.write(f"[red]Generation failed:[/] {e}")
            self._set_status("Failed. See output.")
        finally:
            self.is_generating = False
            self.gen_btn.disabled = False
            self.stop_btn.disabled = True
            self._current_task = None

    def _generate_blocking(self, prompt: str, maxlen: int) -> str:
        """Run generation on a worker thread to keep UI responsive."""
        assert self.lm is not None
        return prompt_model(self.lm.model, self.lm.tokenizer, prompt, max_length=maxlen)

    async def _handle_stop(self) -> None:
        """Cancel an in-flight generation task if present."""
        task = getattr(self, "_current_task", None)
        if task and not task.done():
            task.cancel()

    # graceful exit on Ctrl+C
    async def on_shutdown_request(self) -> None:
        """Attempt to free model resources on exit."""
        try:
            # free GPU memory
            if self.lm is not None:
                del self.lm.model
                del self.lm.tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            pass


if __name__ == "__main__":
    GenerateApp().run()
