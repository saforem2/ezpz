# Fine-Tuning & Training LLMs

A task-oriented guide to training / fine-tuning large language models with
`ezpz`. If you know *what* you want to run (fine-tune a pretrained HF model,
or pretrain a model from scratch) but not *which* example or *how to scale
it*, start here — then follow the per-module example pages for the deep dive.

`ezpz` gives you **three paths**:

| Path | Module (example · API · source) | What it does | Larger-model knob |
|------|--------------------------------|--------------|-------------------|
| **1. HF fine-tune (`Trainer`)** | `hf_trainer` ([example](../examples/hf-trainer/index.md) · [API](../python/Code-Reference/examples/hf_trainer.md) · [src](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py)) | Fine-tune a *pretrained* HF model via HuggingFace `Trainer` + FSDP | `--model_name_or_path` + node count |
| **2. HF fine-tune (custom loop)** | `hf` ([example](../examples/hf.md) · [API](../python/Code-Reference/examples/hf.md) · [src](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf.py)) | Same dataset/model setup, hand-rolled training loop | `--model_name_or_path` + node count |
| **3. From-scratch (FSDP + TP)** | `fsdp_tp` ([example](../examples/fsdp-tp.md) · [API](../python/Code-Reference/examples/fsdp_tp.md) · [src](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)) | Pretrain a Llama-arch model from scratch (FSDP2 + tensor/HSDP parallel) | `--model {s,m,l,xl,xxl,xxxl}` / `agpt-2b` / `agpt-20b` |

!!! tip "Which path?"
    - Fine-tuning an existing checkpoint (Llama, Qwen, Mistral, …)? → **Path 1**
      (`hf_trainer`) is the default; **Path 2** (`hf`) is the same targets in an
      explicit loop, handy for A/B'ing the loop against the `Trainer`.
    - Pretraining from scratch, or you need tensor/sequence parallelism for a
      model too big for FSDP alone? → **Path 3** (`fsdp_tp`).
    - `fsdp_tp` **also** accepts an HF repo id in `--model` (see below), so it
      can FSDP-fine-tune an HF model too (TP off).

---

## Models / configurations

### Paths 1 & 2 — HF fine-tune

Both HF paths take a real HuggingFace repo via `--model_name_or_path` (they do
**not** use the `--model {s,m,l,…}` size presets — those are Path 3 only).
Suggested targets:

| Role | `--model_name_or_path` | ~params | Notes |
|------|------------------------|---------|-------|
| Small | `meta-llama/Llama-3.2-1B` | 1B | single-node |
| **Default** | `meta-llama/Llama-3.1-8B` | 8B | smallest target that meaningfully exercises multi-GPU FSDP for fine-tuning |
| Alt (Qwen) | `Qwen/Qwen2.5-7B` — or a **Qwen3** 8B/14B repo once confirmed | 7–14B | current-gen alternative arch |
| Alt (Mistral) | `mistralai/Mistral-7B-v0.3` | 7B | GQA baseline |
| Larger | `Qwen/Qwen2.5-32B` | 32B | multi-node FSDP |
| **Largest** | `meta-llama/Llama-3.3-70B` | 70B | multi-node |

!!! warning "Confirm exact repo id + revision before running"
    Model repos are gated and get renamed/re-tagged; the Qwen3 line in
    particular is evolving — pick the specific published repo (e.g. an
    `-Instruct` vs base variant) and pin its revision when you record results.
    The tokenizer usually matches the model repo (`--tokenizer_name` = same id).

### Path 3 — from-scratch FSDP+TP, `--model` size ladder

`fsdp_tp` exposes a `--model` size ladder:

| `--model` | dim × layers | ~params | default seq_len | notes |
|-----------|--------------|---------|-----------------|-------|
| `s` (`small`) | 768 × 12 | ~125M | 2048 | |
| `m` (`medium`) | 1024 × 16 | ~246M | 2048 | |
| `l` (`large`) | 1536 × 16 | ~495M | 2048 | |
| `xl` (`xlarge`) | 2048 × 24 | ~1.5B | 2048 | Llama-1.5B-class |
| `xxl` | 4096 × 32 | ~7B | 4096 | Llama-7B-class |
| `xxxl` | 5120 × 40 | ~13B | 4096 | Llama-13B-class |
| `agpt-2b` | 2048 × 12, vocab 256k | ~2B | 8192 | torchtitan AuroraGPT parity |
| `agpt-20b` | 5120 × 64, vocab 256k | ~20B | 2048 | torchtitan AuroraGPT parity |

- **`--model xxl`** (~7B) is the first rung that requires the FSDP2 +
  tensor-parallel machinery (and the memory-bounded loss impls) to fit, so
  it's a representative large-model benchmark.
- **`--model xxxl`** (~13B) and **`agpt-20b`** (~20B) are the largest configs;
  `agpt-*` match a real torchtitan production arch for A/B comparison.
- **Aliases** all resolve: `small→s`, `medium→m`, `large→l`,
  `xlarge/extra-large→xl`, `xxlarge→xxl`, `xxxlarge→xxxl`; `agpt2b`, `agpt_2b`,
  `AGPT-2B` → `agpt-2b`. Use `--model debug` for a laptop-friendly tiny model.
- **Arbitrary HF model.** `--model` also accepts a HuggingFace repo id (any
  value containing a `/`, e.g. `--model meta-llama/Llama-3.1-8B`). `fsdp_tp`
  then loads real weights via `AutoModelForCausalLM.from_pretrained` instead of
  building the from-scratch preset — i.e. it can **fine-tune** an HF model
  under FSDP2, not just pretrain the size ladder. Caveats: this path is
  **FSDP2-only** — `--tp > 1` is forced to `1` (ezpz's TP plan targets its own
  module names, not HF `LlamaDecoderLayer`/`GemmaDecoderLayer`/…); the
  tokenizer defaults to the same repo unless `--tokenizer_name` is set; and
  gated repos honor `HF_TOKEN`.

---

## Expected resource requirements

Rough guidance — validate against a real run before publishing (see
[Follow-up runs](#follow-up-runs-to-publish-results)). "GPUs" assumes
80 GB-class accelerators (A100-80G / H100 / Aurora PVC tile).

### Paths 1 & 2 — HF fine-tune

| Target | Precision / memory levers | Min GPUs (FSDP) | Notes |
|--------|---------------------------|-----------------|-------|
| Llama-3.2-1B (baseline) | bf16 | 1–4 | single node |
| **Llama-3.1-8B / Qwen2.5-7B** (default) | bf16 + `gradient_checkpointing` + `shard_grad_op` | ~4–8 (1–2 nodes) | activation checkpointing needed at `block_size=8192` |
| Qwen2.5-32B | bf16 + `full_shard` + grad-ckpt | ~8–16 (2–4 nodes) | multi-node FSDP |
| **Llama-3.3-70B** (largest) | bf16 + `full_shard` + grad-ckpt | ~16–32 (4–8 nodes) | tune `block_size` down if OOM |

Memory levers (both HF paths): `--fsdp=full_shard` (vs `shard_grad_op`),
`--gradient_checkpointing=true`, and lowering `--block_size` /
`--per_device_train_batch_size`.

### Path 3 — from-scratch FSDP+TP

| `--model` | ~params | Min GPUs | Suggested parallelism | Memory levers |
|-----------|---------|----------|-----------------------|---------------|
| `xl` | ~1.5B | 1–4 | `--tp 1` | `--reshard-after-forward` (default) |
| **`xxl`** (default large) | ~7B | ~8 (1–2 nodes) | `--tp 2` + HSDP | `--loss-impl=loss-parallel`, `--act-mem-budget<1.0`, `--compile` |
| `xxxl` | ~13B | ~16 (2–4 nodes) | `--tp 2`–`4` | same + smaller `--seq-len` |
| `agpt-20b` | ~20B | ~24–48 (4–8 nodes) | `--tp 4` + HSDP | `--loss-impl=loss-parallel` (256k vocab → logits dominate) |

Key large-model levers (all `fsdp_tp` flags):

- `--tp N` — tensor-parallel degree (shard within a node).
- `--dp-replicate` / `--dp-shard` — HSDP (replicate × shard); constraint
  `dp_replicate * dp_shard * tp == WORLD_SIZE`. `--dp-shard -1` = "use all
  remaining ranks".
- `--reshard-after-forward {always,never}` (or `--no-reshard-after-forward`) —
  `always` = ZeRO-3 (lowest memory, default), `never` = ZeRO-2 (more memory,
  skips the backward all-gather).
- `--loss-impl {eager,chunked,chunked-backward,compiled,loss-parallel,fused-linear}`
  — at large vocab (agpt's 256k) the `(B·T, vocab)` logits tensor is the OOM
  driver; `loss-parallel` (needs tp>1) or `fused-linear` bound it.
- `--act-mem-budget <float>` — inductor recompute budget (needs `--compile`);
  `<1.0` recomputes activations in backward to cut peak memory.

---

## Path 1: HF fine-tune with `hf_trainer` (HuggingFace `Trainer`)

> Full walkthrough: [`ezpz.examples.hf_trainer`](../examples/hf-trainer/index.md)
> · [Aurora vs Polaris comparison](../examples/hf-trainer/comparison.md)

If you already have a Python environment with `torch` + `mpi4py`, run without
installing anything via `uv`:

```bash
# Small: Llama-3.2-1B
TMPDIR=$(pwd) uv run \
    --with "git+https://github.com/saforem2/ezpz" \
    --python=$(which python3) \
    ezpz launch python3 -m ezpz.examples.hf_trainer \
        --streaming \
        --dataset_name=eliplutchok/fineweb-small-sample \
        --tokenizer_name meta-llama/Llama-3.2-1B \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --bf16=true \
        --do_train=true \
        --do_eval=true \
        --report-to=wandb \
        --logging-steps=1 \
        --include-tokens-per-second=true \
        --max-steps=50000 \
        --include-num-input-tokens-seen=true \
        --optim=adamw_torch \
        --logging-first-step \
        --include-for-metrics='inputs,loss' \
        --max-eval-samples=50 \
        --per_device_train_batch_size=1 \
        --block_size=8192 \
        --gradient_checkpointing=true \
        --fsdp=shard_grad_op
```

**Larger targets** — swap the model + tokenizer id (and prefer `full_shard`
for 32B/70B; reduce `--block_size` if you OOM):

```bash
# Llama-3.1-8B
        --tokenizer_name meta-llama/Llama-3.1-8B \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --fsdp=shard_grad_op \
        --block_size=8192 \

# Qwen2.5-7B (or a Qwen3 8B/14B repo once confirmed)
        --tokenizer_name Qwen/Qwen2.5-7B \
        --model_name_or_path Qwen/Qwen2.5-7B \
        --fsdp=shard_grad_op \

# Llama-3.3-70B (multi-node)
        --tokenizer_name meta-llama/Llama-3.3-70B \
        --model_name_or_path meta-llama/Llama-3.3-70B \
        --fsdp=full_shard \
        --block_size=4096 \
```

If you need `torch` and/or `mpi4py`, add the extras:

```bash
TMPDIR=$(pwd) uv run \
    --with "git+https://github.com/saforem2/ezpz[torch,mpi,hf]" \
    --python=$(which python3) \
    # ...
```

For automatic module loading + virtual-environment setup:

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
```

before launching.

## Path 2: HF fine-tune with `hf` (hand-rolled training loop)

> Full walkthrough: [`ezpz.examples.hf`](../examples/hf.md)

`hf` mirrors the dataset/model setup of `hf_trainer` but uses an explicit
training loop (like the other `ezpz` examples) instead of the HF `Trainer`.
Same `--model_name_or_path` scaling — useful for comparing the custom loop
against the `Trainer` path on identical targets.

```bash
# Llama-3.1-8B, custom loop.
# hf.py enables FSDP via ACCELERATE_USE_FSDP (it does NOT read --fsdp), and it
# reports tokens/sec on its own (train/tokens_per_sec) — no --include-* flags.
ACCELERATE_USE_FSDP=true TMPDIR=$(pwd) uv run \
    --with "git+https://github.com/saforem2/ezpz" \
    --python=$(which python3) \
    ezpz launch python3 -m ezpz.examples.hf \
        --streaming \
        --dataset_name=eliplutchok/fineweb-small-sample \
        --tokenizer_name meta-llama/Llama-3.1-8B \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --bf16=true \
        --do_train=true \
        --do_eval=true \
        --report-to=wandb \
        --logging-steps=1 \
        --max-steps=100 \
        --include-num-input-tokens-seen=true \
        --optim=adamw_torch \
        --logging-first-step \
        --include-for-metrics='inputs,loss' \
        --max-eval-samples=100 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --block_size=8192 \
        --gradient_checkpointing=true
```

The same larger-target swaps from Path 1 apply here (Llama-3.1-8B /
Qwen2.5-7B / Qwen2.5-32B / Llama-3.3-70B).

!!! note
    The 8B+/70B targets are gated HF repos — configure `HF_HOME` and
    `huggingface-cli login` (or `HF_TOKEN`) before the run.

## Path 3: from-scratch FSDP+TP with `--model`

> Full walkthrough: [`ezpz.examples.fsdp_tp`](../examples/fsdp-tp.md)

A representative large config — `xxl` (~7B), tensor-parallel + HSDP, with the
memory-bounded loss and compile levers:

```bash
TMPDIR=$(pwd) uv run \
    --with "git+https://github.com/saforem2/ezpz" \
    --python=$(which python3) \
    ezpz launch python3 -m ezpz.examples.fsdp_tp \
        --model xxl \
        --tp 2 \
        --dp-shard -1 \
        --dp-replicate 1 \
        --reshard-after-forward always \
        --loss-impl loss-parallel \
        --act-mem-budget 0.5 \
        --compile \
        --seq-len 4096 \
        --batch-size 1
```

> `fsdp_tp` enables wandb automatically (via `ezpz`'s `History` / `setup_wandb`
> and `WANDB_*` env vars) — there is no `--report-to` flag here.

Other size-ladder targets (same flags, just change `--model`):

```bash
--model xl      # ~1.5B, fits at tp=1 on a single node
--model xxxl    # ~13B, prefer --tp 2..4 and a smaller --seq-len
```

Largest config — `agpt-20b` (~20B, 256k vocab), higher TP:

```bash
    ezpz launch python3 -m ezpz.examples.fsdp_tp \
        --model agpt-20b \
        --tp 4 \
        --dp-shard -1 \
        --loss-impl loss-parallel \
        --act-mem-budget 0.5 \
        --compile \
        --seq-len 2048 \
        --batch-size 1
```

!!! note
    `--dp-shard -1` = "use all remaining ranks" =
    `WORLD_SIZE / (dp_replicate * tp)`. For pure HSDP, set `--dp-replicate > 1`
    (e.g. one replica per node, sharded within).

**Fine-tune an arbitrary HF model via `fsdp_tp`** — pass a repo id (with a
`/`) as `--model`. Loads real HF weights under FSDP2 (TP is forced off, so
drop `--tp`); the same memory levers apply:

```bash
    ezpz launch python3 -m ezpz.examples.fsdp_tp \
        --model meta-llama/Llama-3.1-8B \
        --dp-shard -1 \
        --reshard-after-forward always \
        --act-mem-budget 0.5 \
        --compile \
        --seq-len 4096 \
        --batch-size 1
        # --tokenizer_name defaults to --model; set HF_TOKEN for gated repos
        # (wandb auto-enabled; no --report-to flag on fsdp_tp)
```

Any HF repo works here (e.g. `Qwen/Qwen2.5-7B`, `mistralai/Mistral-7B-v0.3`) —
this overlaps with Paths 1 & 2 but keeps you in the FSDP2 training loop rather
than the HF `Trainer` / custom-loop harnesses.

---

## Follow-up runs to publish results

Before the numbers are publishable, complete these on-node runs and record
them (wandb + the resource tables above):

1. **Fit / OOM sweep per target.** Confirm the default configs (Llama-3.1-8B
   on Paths 1 & 2; `--model xxl` on Path 3) actually fit at the stated GPU
   counts; record peak memory and the smallest node count that fits. Adjust
   `block_size` / `seq_len` / `act-mem-budget` and re-record on OOM.
2. **Throughput / MFU.** Capture tokens/sec-per-GPU and MFU at steady state
   (past step ~20, after any `torch.compile` warmup/recompile). Path 1
   (`hf_trainer`): `--include-tokens-per-second`. Path 2 (`hf`): reported
   automatically as `train/tokens_per_sec` (no flag). Path 3 (`fsdp_tp`): the
   `tps_per_gpu` / `mfu` history metrics.
3. **`Trainer` vs custom-loop A/B (Paths 1 vs 2).** Run the same target on
   both HF paths and compare throughput/MFU + loss curves.
4. **Scaling curve.** Run each target at 2–3 node counts (1, 2, 4) for a
   strong/weak-scaling plot.
5. **Model-family sweep (HF).** One run each of the alt targets
   (Qwen2.5-7B / Mistral-7B / Qwen2.5-32B, plus a Qwen3 repo once confirmed)
   to compare arch/tokenizer effects at fixed data.
6. **Largest-config validation.** One end-to-end run each of Llama-3.3-70B
   (HF) and `agpt-20b` (Path 3) to prove they launch, train, and checkpoint at
   multi-node scale.
7. **A/B vs torchtitan (Path 3).** Compare `agpt-2b` / `agpt-20b` MFU against
   a matched torchtitan run (the presets are byte-identical for this) and note
   any gap.

---

## Running on Perlmutter @ NERSC

1. Submit an interactive job:

    ```bash
    NODES=2 ; HRS=02 ; QUEUE=interactive
    salloc --nodes $NODES --qos $QUEUE --time $HRS:30:00 \
        -C 'gpu' --gpus=$(( 4 * NODES )) -A amsc013_g
    ```

    !!! tip
        For the 8B+/70B (HF) or `xxl`/`agpt-20b` (Path 3) targets, bump
        `NODES` per the resource tables above.

1. Load modules:

    ```bash
    module load cudatoolkit/12.9 nccl/2.24.3 pytorch cray-mpich
    ```

1. Navigate to `$SCRATCH` and set environment variables:

    ```bash
    cd $SCRATCH
    export UV_CACHE_DIR="$SCRATCH/.cache/uv"
    export HF_HOME="$SCRATCH/.cache/hf"   # gated 8B+/70B repos need auth here
    ```

1. Create and activate a virtual environment:

    ```bash
    uv venv --python=$(which python3) --system-site-packages
    source .venv/bin/activate
    ```

1. Install `ezpz` (+ `mpi4py`):

    ```bash
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz[mpi]"
    ```

1. Run:

    ```bash
    # sanity check: MLP on MNIST
    ezpz launch python3 -m ezpz.examples.test

    # Path 1: HF Trainer (8B)
    ezpz launch python3 -m ezpz.examples.hf_trainer \
        --dataset_name=eliplutchok/fineweb-small-sample --streaming \
        --tokenizer_name meta-llama/Llama-3.1-8B \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --bf16=true --do_train=true --do_eval=true \
        --report-to=wandb --logging-steps=1 --logging-first-step \
        --include-tokens-per-second=true --include-num-input-tokens-seen=true \
        --optim=adamw_torch --include-for-metrics='inputs,loss' \
        --max-steps=100 --max-eval-samples=100 \
        --per_device_train_batch_size=1 --per_device_eval_batch_size=1 \
        --block_size=8192 --gradient_checkpointing=true \
        --fsdp=shard_grad_op \
        --output_dir=outputs/ezpz.hf_trainer/$(tstamp)

    # Path 2: HF custom loop (same target). FSDP via ACCELERATE_USE_FSDP;
    # no --fsdp / --include-tokens-per-second (see the Path 2 section above).
    ACCELERATE_USE_FSDP=true ezpz launch python3 -m ezpz.examples.hf \
        --dataset_name=eliplutchok/fineweb-small-sample --streaming \
        --tokenizer_name meta-llama/Llama-3.1-8B \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --bf16=true --do_train=true --do_eval=true \
        --report-to=wandb --logging-steps=1 --logging-first-step \
        --include-num-input-tokens-seen=true \
        --optim=adamw_torch --include-for-metrics='inputs,loss' \
        --max-steps=100 --max-eval-samples=100 \
        --per_device_train_batch_size=1 --per_device_eval_batch_size=1 \
        --block_size=8192 --gradient_checkpointing=true \
        --output_dir=outputs/ezpz.hf/$(tstamp)

    # Path 3: from-scratch FSDP+TP (xxl / ~7B)  [wandb auto-enabled]
    ezpz launch python3 -m ezpz.examples.fsdp_tp \
        --model xxl --tp 2 --dp-shard -1 \
        --loss-impl loss-parallel --act-mem-budget 0.5 --compile \
        --seq-len 4096 --batch-size 1
    ```

---

## See also

- [🗺️ Distributed Training](distributed-training.md) — how `ezpz` sets up
  `torch.distributed`, scheduler detection, and DDP/FSDP mechanics.
- Per-module deep dives:
  [`hf_trainer`](../examples/hf-trainer/index.md) ·
  [`hf`](../examples/hf.md) ·
  [`fsdp_tp`](../examples/fsdp-tp.md)
- [🤗 HF Trainer: Aurora vs Polaris comparison](../examples/hf-trainer/comparison.md)
