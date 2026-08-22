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

## HuggingFace models

The same flags work on HF Llama-family models — Llama, Mistral, Qwen2, Gemma:

```bash
python3 -m ezpz.launch --np 12 -- \
  python3 -m ezpz.examples.fsdp_tp \
    --model Qwen/Qwen3-0.6B --lora-rank 16 --dataset random
```

`LoraConfig` gates by *role* (`train_attn` / `train_mlp` / `train_unembed`)
rather than by name, because the roles are stable across model families even
when the spellings are not. `target_names()` emits both vocabularies at once:

| role | native | HuggingFace |
| --- | --- | --- |
| `attn` | `wq` `wk` `wv` `wo` | `q_proj` `k_proj` `v_proj` `o_proj` |
| `mlp` | `w1` `w2` `w3` | `gate_proj` `up_proj` `down_proj` |
| `unembed` | `output` | `lm_head` |

Emitting both is safe rather than sloppy: `apply_lora` wraps a child only
when its attribute name is in the target set **and** it is an `nn.Linear`, so
a spelling the model does not use never fires. No architecture sniffing, and
nothing to mis-detect.

Two things to know:

- **HF runs are FSDP-only.** The HF path forces `--tp 1` (the TP plan is
  written against the native module names), so LoRA composes with FSDP2 and
  HSDP there, not with tensor parallelism.
- **Tied embeddings are fine.** When `tie_word_embeddings=True`,
  `lm_head.weight` *is* `embed_tokens.weight`; the adapter is additive and
  the shared tensor stays frozen, so `--lora-target unembed` does not
  accidentally thaw the input embedding.

If a requested role adapts nothing, the run **fails at setup** rather than
training. That matters more than it sounds: a LoRA request that silently
falls back to full fine-tuning produces a plausible loss curve, a normal-looking
log, and a checkpoint that is quietly 100× larger than intended.

## Why LoRA starts as a no-op

`B` is zero-initialized, so `y = Wx + (alpha/r)·B(A(x))` equals `Wx` exactly
at step 0. An adapted model's output is bit-identical to the base model's
before any training — so a bad adapter can never silently corrupt your
starting point.

That guarantee survives `--meta-init`. Adapters are built on the base
layer's own device and dtype — so a `meta` base gives `meta` adapters
(meta-init is not quietly defeated) and a bf16 base gives bf16 adapters
(no dtype mismatch on the first adapter matmul). `to_empty()` leaves
their storage uninitialized and `Transformer.init_weights` only reaches
the *base* weights through the wrapper, so `parallelize` re-initializes
every adapter afterwards; without that, `B` would not be zero and
training would start from whatever was in memory.

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

**Gradient accumulation** — pass several microbatches; each is scaled
for you by its share of the tokens, so the result matches one batch of
the same total size:

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

`forward_backward` accepts a raw tensor, an `{"input_ids": ...}` dict, a
`ModelInput`, or a `Datum` — the Tinker-shaped form works as advertised:

```python
from ezpz.tinker import Datum, ModelInput
client.forward_backward([Datum(model_input=ModelInput(token_ids=ids),
                               loss_fn_inputs={})]).result()
```

**Ragged microbatches are weighted by tokens**, not by count. Each
microbatch's cross-entropy is already a mean over its own tokens, so
scaling them all by `1/N` would give a 300-token and a 100-token
microbatch equal pull where a single combined batch weights them
0.75/0.25. Microbatches with no valid targets are skipped — their loss
is `0/0`.

Calls return a future (`.result()`). It resolves immediately — the shape
exists so a remote backend could be added without touching call sites.

## Adapter-only checkpoints

```python
client.save_state("./ckpts", adapters_only=True).result()   # default
```

This passes `StateDictOptions(ignore_frozen_params=True)` to the existing
DCP layer, so only the trainable adapters are written. Pass
`overwrite=True` to clear an existing checkpoint at that step first —
without it DCP writes into the existing directory and can leave a mix of
old and new shards. (`ttl_seconds` is accepted for signature
compatibility with Tinker and ignored; local checkpoints never expire.)

Restore it with the matching flag — `load_state` defaults to
`adapters_only=True` for the same reason `save_state` does, so the
round-trip works without passing anything:

```python
client.load_state("./ckpts").result()                      # adapters
client.load_state("./ckpts", adapters_only=False).result() # full ckpt
```

!!! note "Adapter checkpoints are not standalone"

    They contain no base weights, so `adapters_only` must **match** on
    save and load — loading one as a full checkpoint asks DCP for
    parameters the file does not contain. Keep the base model available.
    `--save-interval` in `fsdp_tp` still writes full checkpoints;
    `adapters_only` is opt-in through the client.

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
- **Fused-QKV architectures** (GPT-2, Falcon, GPT-NeoX, MPT). GPT-2's
  projections are `transformers.pytorch_utils.Conv1D`, which is not an
  `nn.Linear` subclass, so `apply_lora` cannot wrap them; Falcon and GPT-NeoX
  pack q/k/v into a single `query_key_value` that would have to be split
  before an adapter could target one projection. These fail with a named
  error rather than adapting nothing.
