# 📝 Ready-to-go Examples

This section contains ready-to-run examples demonstrating various features of
the `ezpz` library.

New to `ezpz`? Start with [`ezpz.examples.test`](./test.md) for a minimal
DDP training loop, then move to [`ezpz.examples.fsdp`](./fsdp.md) when you're
ready to shard model parameters. The remaining examples layer on additional
features — Vision Transformers, tensor parallelism, Hugging Face integration —
so pick whichever matches your workload.

### Picking an Example

| Example | When to use |
|---------|-------------|
| `test` | Starting point — simplest DDP training loop |
| `fsdp` | Model too large for one GPU, or you want memory-efficient training |
| `vit` | Vision Transformer with FSDP + optional `torch.compile` |
| `fsdp_tp` | Very large models needing 2D parallelism (FSDP + Tensor Parallel) |
| `diffusion` | Diffusion model training with FSDP |
| `hf` | Fine-tune a causal LM with an explicit training loop (Accelerate + FSDP) |
| `hf_trainer` | Using Hugging Face Trainer with ezpz's launcher |

--8<-- "../includes/examples-example-table.md"
