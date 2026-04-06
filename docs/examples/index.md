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

| Example | When to use | Level |
|---------|-------------|-------|
| `test` | Starting point — simplest DDP training loop | Beginner |
| `fsdp` | Model too large for one GPU, or you want memory-efficient training | Beginner |
| `vit` | Vision Transformer with FSDP + optional `torch.compile` | Intermediate |
| `fsdp_tp` | Very large models needing 2D parallelism (FSDP + Tensor Parallel) | Advanced |
| `diffusion` | Diffusion model training with FSDP | Intermediate |
| `hf` | Fine-tune a causal LM with an explicit training loop (Accelerate + FSDP) | Intermediate |
| `hf_trainer` | Using Hugging Face Trainer with ezpz's launcher | Beginner |

--8<-- "../includes/examples-example-table.md"
