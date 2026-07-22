# 📝 Ready-to-go Examples

This section contains ready-to-run examples demonstrating various features of
the `ezpz` library.

New to `ezpz`? Start with [`ezpz.examples.test`](./test.md) for a minimal
DDP training loop, then move to [`ezpz.examples.fsdp`](./fsdp.md) when you're
ready to shard model parameters. The remaining examples layer on additional
features — Vision Transformers, tensor parallelism, Hugging Face integration —
so pick whichever matches your workload.

### Prerequisites

- **ezpz** installed: `pip install ezpz` or `pip install git+https://github.com/saforem2/ezpz`
- **PyTorch** (auto-detected at runtime)
- **MPI** for multi-GPU training (`mpi4py` + MPICH or OpenMPI). Single-GPU and CPU work without MPI.

### Running Examples

All examples use the same launch pattern:

```bash
ezpz launch python3 -m ezpz.examples.<name> [args]
```

For example, to run the test example:

```bash
ezpz launch python3 -m ezpz.examples.test
```

On a laptop this uses a local `mpirun`; inside a PBS or SLURM job, the
scheduler is auto-detected.

### Picking an Example

| Links | Example | When to use | Level |
|-------|---------|-------------|-------|
| [:lucide-book:][ex-test] · [:lucide-file-code:][api-test] · [:fontawesome-brands-github:][gh-test] | `test` | Starting point — simplest DDP training loop | Beginner |
| [:lucide-book:][ex-fsdp] · [:lucide-file-code:][api-fsdp] · [:fontawesome-brands-github:][gh-fsdp] | `fsdp` | Model too large for one GPU, or memory-efficient training | Beginner |
| [:lucide-book:][ex-vit] · [:lucide-file-code:][api-vit] · [:fontawesome-brands-github:][gh-vit] | `vit` | Vision Transformer with FSDP + optional `torch.compile` | Intermediate |
| [:lucide-book:][ex-fsdp-tp] · [:lucide-file-code:][api-fsdp-tp] · [:fontawesome-brands-github:][gh-fsdp-tp] | `fsdp_tp` | Very large models needing 2D parallelism (FSDP + TP) | Advanced |
| [:lucide-book:][ex-diffusion] · [:lucide-file-code:][api-diffusion] · [:fontawesome-brands-github:][gh-diffusion] | `diffusion` | Diffusion model training with FSDP | Intermediate |
| [:lucide-book:][ex-hf] · [:lucide-file-code:][api-hf] · [:fontawesome-brands-github:][gh-hf] | `hf` | Fine-tune causal LM with Accelerate + FSDP | Intermediate |
| [:lucide-book:][ex-hf-trainer] · [:lucide-file-code:][api-hf-trainer] · [:fontawesome-brands-github:][gh-hf-trainer] | `hf_trainer` | Using HF Trainer with ezpz's launcher | Beginner |
| [:lucide-book:][ex-inference] · [:lucide-file-code:][api-inference] · [:fontawesome-brands-github:][gh-inference] | `inference` | Distributed HF inference (benchmark / generate / eval modes) | Intermediate |

!!! tip "Fine-tuning or training an LLM?"
    See the [**Fine-Tuning & Training LLMs**](../guides/llm-finetuning.md)
    guide for a task-oriented walkthrough of the three paths (`hf_trainer`,
    `hf`, `fsdp_tp`) — which to pick, larger-model targets, resource
    requirements, and how to scale.

### Model size ladder

All 5 training examples (`test`, `fsdp`, `vit`, `diffusion`, `fsdp_tp`)
expose the same `--model {debug,s,m,l,xl,xxl,xxxl}` preset ladder
targeting consistent parameter counts. Architectures differ; the
ladder positions are aligned.

| Preset | Target | `test` | `fsdp` | `vit` | `diffusion` | `fsdp_tp` |
|--------|-------:|-------:|-------:|------:|------------:|----------:|
| `debug` | smoke | ~110K | ~10K | ~50K | ~50K | ~10K |
| `s` (small) | ~100M | 107M | 76M | 87M | 101M | 125M |
| `m` (medium) | ~250M | 248M | 227M | 204M | 274M | 246M |
| `l` (large) | ~500M | 449M | 605M | 632M | 500M | 495M |
| `xl` | ~1B | 858M | 1.21B | 1.21B | 939M | 1.21B |
| `xxl` | ~5B | 3.43B | 4.84B | 5.44B | 5.50B | 5.93B |
| `xxxl` | ~10B | 9.88B | 9.68B | 9.67B | 11.4B | 11.34B |

Long-form aliases (`small`, `medium`, `large`, `xlarge`, `extra-large`,
`xxlarge`, `extra-extra-large`, `xxxlarge`, `extra-extra-extra-large`)
map onto the same short-name canonical presets.

> **Breaking change** as of this release: `--model small` (and other
> long-form aliases) now resolve to the production-scale ladder above.
> Old toy-scale `small/medium/large` (~250K–1M params) are gone. Use
> `--model debug` for the laptop-runnable smoke-test path.

For Llama-specific configs, `fsdp_tp` additionally exposes the
torchtitan-flavored `agpt-2b` (1.99B params) and `agpt-20b` (20.74B
params) presets, which reproduce the AuroraGPT registry exactly.

[ex-test]: test.md "Example"
[api-test]: ../python/Code-Reference/examples/test.md "API Reference"
[gh-test]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/test.py "GitHub Source"
[ex-fsdp]: fsdp.md "Example"
[api-fsdp]: ../python/Code-Reference/examples/fsdp.md "API Reference"
[gh-fsdp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py "GitHub Source"
[ex-vit]: vit.md "Example"
[api-vit]: ../python/Code-Reference/examples/vit.md "API Reference"
[gh-vit]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py "GitHub Source"
[ex-fsdp-tp]: fsdp-tp.md "Example"
[api-fsdp-tp]: ../python/Code-Reference/examples/fsdp_tp.md "API Reference"
[gh-fsdp-tp]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py "GitHub Source"
[ex-diffusion]: diffusion.md "Example"
[api-diffusion]: ../python/Code-Reference/examples/diffusion.md "API Reference"
[gh-diffusion]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/diffusion.py "GitHub Source"
[ex-hf]: hf.md "Example"
[api-hf]: ../python/Code-Reference/examples/hf.md "API Reference"
[gh-hf]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf.py "GitHub Source"
[ex-hf-trainer]: hf-trainer/index.md "Example"
[api-hf-trainer]: ../python/Code-Reference/examples/hf_trainer.md "API Reference"
[gh-hf-trainer]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/hf_trainer.py "GitHub Source"
[ex-inference]: inference.md "Example"
[api-inference]: ../python/Code-Reference/examples/inference.md "API Reference"
[gh-inference]: https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/inference.py "GitHub Source"
