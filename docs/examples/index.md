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
