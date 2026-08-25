# agpt-2b across machines

One config, three systems: `agpt-2b`, `tp=1`, `bs=1`, `seq 2048`, 20 iters.
The point is a like-for-like number, so the config string is identical
everywhere and appears verbatim in each `runs.csv`.

## Results

| system | nodes | devices | tok/s | tok/s/device | MFU | TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| SunSpot (PVC tile) | 1 | 12 | 61,965 | 5,164 | 16.2% | 48.4 |
| Perlmutter (A100-40GB) | 1 | 4 | 40,363 | **10,091** | **30.3%** | 94.6 |
| Perlmutter (A100-40GB) | 2 | 8 | 51,244 | 6,406 | 19.2% | 60.1 |

**Per device, one A100 does ~1.95x a PVC tile, at 1.87x the MFU.**
Aggregate favours Sunspot only because that node has 12 tiles to
Perlmutter's 4.

The Sunspot row was re-measured on 2026-08-25 (job 12473794) because
the original had no MFU or TFLOPS — it predates the current export and
its run directory was gone. Throughput reproduced to within **1.22%**
of the 2026-08-11 figure (61,965 vs 61,217 tok/s) across two weeks and
a different commit, which is the main reason to trust either number.

One caveat the ratio hides: **a PVC *tile* is not an A100.**
Aurora/Sunspot expose 12 tiles per node (2 per GPU, 6 GPUs), so
"per-device" means per-tile there and per-card here. Per *node* is the
fairer comparison for capacity planning; per-device is the efficiency
one.

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
