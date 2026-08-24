# agpt-2b across machines

One config, three systems: `agpt-2b`, `tp=1`, `bs=1`, `seq 2048`, 20 iters.
The point is a like-for-like number, so the config string is identical
everywhere and appears verbatim in each `runs.csv`.

## Results

| system | nodes | GPUs | tok/s | tok/s/GPU | MFU | TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| SunSpot (PVC XPU) | 1 | 12 | 61,217 | 5,101 | — | — |
| Perlmutter (A100-40GB) | 1 | 4 | 40,363 | **10,091** | 30.3% | 94.6 |
| Perlmutter (A100-40GB) | 2 | 8 | 51,244 | 6,406 | 19.2% | 60.1 |

**Per GPU, one A100 does ~2.0x a PVC tile at this config.** Aggregate
favours Sunspot only because that node has 12 tiles to Perlmutter's 4.

Two caveats that matter more than the ratio:

- **The Sunspot row is incomplete.** It predates the current export and
  the run directory is gone, so MFU and TFLOPS cannot be recovered —
  only re-measured. Reproducing it is one `qsub` of the command below.
- **A PVC *tile* is not an A100.** Aurora/Sunspot expose 12 tiles per
  node (2 per GPU, 6 GPUs); "per-GPU" means per-tile there and per-card
  here. Per *node* is the fairer comparison for capacity planning, and
  per-tile-vs-card for efficiency.

There is **no matched Aurora row**. `80b_benchmarks.md` is a different
model entirely (80B, `tp>=2`, 5 steps) and is not comparable.

## Reproducing

Perlmutter:

```bash
sbatch -N1 --export=ALL,BS=1,SEQ=2048,COMPILE=no,VENV=.venv-213 \
    experiments/perlmutter/amsc_bench.sbatch
```

Sunspot/Aurora, same config through PBS:

```bash
python3 -m ezpz.launch --np 12 -- python3 -m ezpz.examples.fsdp_tp \
    --model agpt-2b --tp 1 --dataset random \
    --train-iters 20 --batch-size 1 --seq-len 2048 \
    --outdir "$PWD/outputs/agpt2b-xmachine"

ezpz export-amsc <run-dir> --config agpt-2b/bs1/seq2048/tp1
```

!!! danger "On Perlmutter, `module load nccl/2.24.3` first"

    Without it NCCL finds no `libnccl-net` plugin and falls back to TCP:
    **8.3x slower inter-node, with no error and no warning**. The 2-node
    row above was 6,175 tok/s / 2.31% MFU before the plugin was loaded —
    slower in absolute terms than one node. See
    [Perlmutter](perlmutter.md).
