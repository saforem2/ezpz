# Fine-tuning with LoRA and the `ezpz.tinker` API

`ezpz.tinker` gives you two things:

- **LoRA adapters** for the native `Transformer`, composable with FSDP2 and
  the async DCP checkpoint layer.
- **A decoupled training step** — `forward_backward` and `optim_step` as
  separate calls — shaped after
  [Tinker](https://tinker-docs.thinkingmachines.ai/), but running in your own
  allocation rather than as a hosted service.

## Quick start

```bash
ezpz launch --nhosts 4 -- python3 -m ezpz.examples.fsdp_tp \
    --model agpt-2b --lora-rank 16 \
    --ckpt-dir ./ckpts --save-interval 100 --train-iters 2000
```

That freezes the base weights and trains only the adapters — roughly
`rank/dim` of the parameters, with checkpoints shrinking by about the same
factor.

| flag | default | meaning |
|---|---|---|
| `--lora-rank` | `0` (off) | inner dimension `r`; `0` means full fine-tuning |
| `--lora-alpha` | `= rank` | update is scaled by `alpha/rank` |
| `--lora-dropout` | `0.0` | dropout on the adapter branch input |
| `--lora-target` | `attn,mlp` | `attn` (wq/wk/wv/wo), `mlp` (w1/w2/w3), `unembed` |

LoRA composes with FSDP2, HSDP and tensor parallelism. At `tp > 1` the
adapters are sharded alongside the weights they wrap:

```bash
ezpz launch --nhosts 4 -- python3 -m ezpz.examples.fsdp_tp \
    --model agpt-2b --tp 2 --lora-rank 16
```

??? note "How the adapters are sharded, and why it needs a 2-rank test"

    `parallelize_module` dispatches on the module *class*, so a plan entry
    for `attention.wq` fails once `wq` is a `LoRALinear`. `lora_tp_plan`
    rewrites each such key into three — `.base`, `.A`, `.B` — and *derives*
    the two adapter styles from the base's:

    - **`A` mirrors what the base consumes**: same style class and
      `input_layouts`, with `output_layouts=Replicate()` (`B` reads it) and
      `use_local_output=False` (otherwise `B` sees `r/tp` instead of `r`).
    - **`B` mirrors what the base produces**: same class and
      `output_layouts`, so its result can be summed with the base's, with
      `input_layouts=Replicate()` since it is fed by `A`, not by `x`.

    Copying the base's class matters. Under `RowwiseParallel`
    (`attention.wo`) the activation arrives sharded on the feature
    dimension; a `ColwiseParallel` `A` declares it replicated and keeps a
    full-width weight, so the contraction dimensions disagree:

    ```text
    Sharding propagation failed for aten.mm.default(
        Spec(f32[16, 64](R)), Spec(f32[128, 8](S(1))))
    ```

    The plan also needs the model, because `--lora-target` is selective —
    the default leaves the unembedding unwrapped — and `parallelize_module`
    **ignores plan keys that match no module without raising**. Retargeting
    an unwrapped `output` would silently leave it an unsharded `nn.Linear`
    that dies on its first DTensor input.

    Neither bug is visible single-rank: a `world_size=1` mesh satisfies
    every placement trivially, and `ColwiseParallel`'s input hook uses
    `DTensor.from_local(..., run_check=False)`, so torch believes a
    mislabeled layout rather than rejecting it. `tests/test_tinker_lora_tp.py`
    runs a real 2-rank gloo mesh and gates on tp=2 matching tp=1
    numerically — a layout bug can run cleanly and still be wrong.

## Why LoRA starts as a no-op

`B` is zero-initialized, so `y = Wx + (alpha/r)·B(A(x))` equals `Wx` exactly
at step 0. An adapted model's output is bit-identical to the base model's
before any training — so a bad adapter can never silently corrupt your
starting point.

## The decoupled step

The example's loop and the client below run the **same** implementation
(`ezpz.tinker.step`), so they cannot drift.

```python
from ezpz.tinker import AdamParams, LocalTrainingClient

client = LocalTrainingClient(state)          # state built by fsdp_tp.train()
for batch in loader:
    client.forward_backward([batch]).result()
    client.optim_step(AdamParams(learning_rate=3e-4)).result()
```

Splitting the two is what makes these possible:

**Gradient accumulation** — pass several microbatches; each is scaled by
`1/N` for you, and the result matches one batch of the same total size:

```python
client.forward_backward([mb0, mb1, mb2, mb3]).result()   # accumulate
client.optim_step().result()                              # one update
```

The fused loop could not express this: `zero_grad` sat immediately before
`backward()` and discarded every gradient but the last.

**Per-step hyperparameters** — `AdamParams` carries `learning_rate` and
`grad_clip_norm`, so an RL loop can vary them without rebuilding the
optimizer.

**Other losses** — `Datum` carries per-loss inputs alongside the data
(`LossFnType` reserves `ppo`, `dro`, and friends), so adding an algorithm
does not change the client API. Only `cross_entropy` is implemented today;
the others raise `NotImplementedError` rather than silently misbehaving.

Calls return a future (`.result()`). It resolves immediately — the shape
exists so a remote backend could be added without touching call sites.

## Adapter-only checkpoints

```python
client.save_state("./ckpts", adapters_only=True).result()   # default
```

This passes `StateDictOptions(ignore_frozen_params=True)` to the existing
DCP layer, so only the trainable adapters are written.

!!! note "Adapter checkpoints are not standalone"

    They contain no base weights. Keep the base model available, and load
    with the same option. `--save-interval` in `fsdp_tp` still writes full
    checkpoints; `adapters_only` is opt-in through the client.

To fold adapters back into the base weights for export or inference:

```python
from ezpz.tinker import merge_adapters
merge_adapters(model)     # in place; model becomes a plain Transformer
```

Not safe mid-training — it discards the separate adapter parameters the
optimizer holds state for.

## What is not here yet

- **`LocalSamplingClient.sample`** raises `NotImplementedError`. Use
  `ezpz.examples.generate` against a saved checkpoint; the in-process path
  lands with the RL loop, which is the first thing that needs a real
  train→sample handoff.
- **HF models.** `apply_lora` targets the native `Transformer`; the HF branch
  already forces `tp=1` and takes a separate sharding path.
