# Stream and mix datasets with Zephon

Zephon lets a training run read multiple datasets, sample them according to
weights, and tokenize and pack examples as they are consumed. In ezpz's
`fsdp_tp` example, the dataset mixture is a TOML recipe selected when the job
starts. Change the recipe to run a different mixture; you do not need to edit
the training code or preprocess a combined dataset.

The underlying [Zephon library](https://github.com/datologyai/zephon) is a
standalone Python data pipeline. Its iterable can be used in other training
loops. ezpz currently provides a ready-made training-loop adapter for
`ezpz.examples.fsdp_tp`; using Zephon in a different ezpz example or another
framework means wiring its batches and checkpoint state into that loop.

## Install

Install the optional extra in the environment used to launch training:

```bash
uv pip install 'ezpz[zephon]'
```

This extra includes Zephon's Hugging Face dataset backend. The base ezpz
dependencies include `transformers`, which Zephon uses for tokenization. Use
public HF datasets to run without credentials; private or gated datasets also
need Hugging Face access configured on every worker.

## Write a runtime recipe

Create a TOML file such as `data/mix.toml`:

```toml
text_field = "text"
tokenizer_id = "gpt2"
seed = 42
chunk_size = 4

[[sources]]
name = "source_a"
path = "hf://org/dataset-a/config/train"
weight = 0.7

[[sources]]
name = "source_b"
path = "hf://org/dataset-b/train"
weight = 0.3
```

Replace the example URIs with the public dataset repository, optional
configuration, and split you want. Zephon's HF URI form is
`hf://org/name[@revision]/[config/]split`; the config segment is omitted for
datasets without named configurations. Each `name` must be unique, and the
mixture weights are relative proportions. `text_field` must name the string
field in each dataset record. The ezpz adapter also accepts `fmt` on a source
when you need to select a supported storage format explicitly.

The recipe is read when the loader first builds its pipeline. A run uses one
fixed recipe; changing the file does not mutate an already-running pipeline.
Keep the recipe alongside the run configuration/checkpoint so that a resume
uses the same sources, weights, tokenizer, seed, and chunking settings.

## Launch with ezpz

Pass the recipe to the FSDP + tensor-parallel example:

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp \
  --model small \
  --batch-size 4 \
  --dataloader zephon \
  --zephon-recipe data/mix.toml
```

Relative local paths in a recipe are resolved relative to the TOML file.
`hf://` paths are passed to Zephon as URIs. The adapter streams, decodes,
tokenizes, packs, and batches the selected records. It saves/restores its
loader state through the example's existing checkpoint flow. Elastic resume
across data-parallel sizes is subject to Zephon's lane and batch-size
constraints; see the [Zephon user guide](https://docs.zephon.io/).

For a no-network smoke run, omit `--zephon-recipe`; `fsdp_tp` uses its bundled
local JSONL recipe by default. The standalone
[`fsdp_tp` example page](../examples/fsdp-tp.md) documents its other loader
modes.

## Use Zephon in another training loop

For a project using Zephon without the ezpz adapter, install the library and
the HF backend (plus Transformers for tokenization):

```bash
uv pip install 'zephon[hf]' transformers
```

Then use its Python API directly:

```python
from zephon import Pipeline
from zephon.io import Dataset
from zephon.work import MixtureSpec, StaticMixtureWorkSource

datasets = [
    Dataset.from_path("source_a", "hf://org/dataset-a/config/train"),
    Dataset.from_path("source_b", "hf://org/dataset-b/train"),
]
work = StaticMixtureWorkSource(
    datasets=datasets,
    mixture=MixtureSpec({"source_a": 0.7, "source_b": 0.3}),
    seed=42,
    chunk_size=4,
)
pipeline = (
    Pipeline(work)
    .decode_text()
    .tokenize(tokenizer_id="gpt2", field="text", padding=True)
    .pack_flat(max_length=2048, algorithm="wrap")
    .batch(4, drop_last=True)
)

for batch in pipeline:
    train_step(batch.to_training())
```

This is ordinary Python iteration, so a project can consume the batches in
its own training loop and manage Zephon's checkpoint state with its model and
optimizer state. See the [Zephon user guide](https://docs.zephon.io/) for
supported dataset formats, operators, storage backends, and integration
details.
