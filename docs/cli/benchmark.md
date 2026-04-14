# `ezpz benchmark`

Run all ezpz examples sequentially and generate a markdown report with
timing, metrics, and environment information.

## Usage

```bash
# Run all examples
ezpz benchmark

# Run specific examples
ezpz benchmark --examples test fsdp vit

# Override model size
ezpz benchmark --model small

# Custom output directory
ezpz benchmark --outdir outputs/my-benchmark
```

## Options

| Flag | Description |
|------|-------------|
| `--examples` | Space-separated list of examples to run (default: all) |
| `--model` | Model size preset (default: `small`) |
| `--outdir` | Output directory (default: `outputs/benchmarks/{timestamp}`) |

## Available Examples

`test`, `fsdp`, `vit`, `fsdp_tp`, `diffusion`, `hf`, `hf_trainer`

## Output

Each run creates a timestamped directory with:

```
outputs/benchmarks/20260330_091310/
  env.json        # Environment metadata (git, nodes, torch version, ...)
  timings.csv     # Per-example wall time and exit code
  report.md       # Markdown report with results table
  test.log        # Captured stdout/stderr for each example
  fsdp.log
  ...
```

## Example Output

```
Running 7 example(s): test, fsdp, vit, fsdp_tp, diffusion, hf, hf_trainer

════════════════════════════════════════════════════════════════
  [1/7] Running: test
         ETA: ~5m 30s remaining
         cmd: ezpz launch python3 -m ezpz.examples.test --model small
         log: outputs/benchmarks/.../test.log
════════════════════════════════════════════════════════════════
  ✓ test completed in 1m 06s
...
════════════════════════════════════════════════════════════════
  7/7 passed in 13m 51s
────────────────────────────────────────────────────────────────
  ✓ test            1m 06s
  ✓ fsdp               31s
  ✓ vit                42s
  ...
════════════════════════════════════════════════════════════════
```
