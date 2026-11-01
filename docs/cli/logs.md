# `ezpz logs`

Fetch per-example benchmark logs — one file per example, not the MPI-duplicated spool.

```bash
ezpz logs 12481065 --remote sunspot
ezpz logs 12481065 --remote sunspot --example fsdp --lines 100
```

Requires `--remote HOST` + `JOB_ID`. Resolves the job's workdir from `qstat -xf`, then tails `outputs/ezpz-benchmarks/*/EXAMPLE.log` directly.

## Options

| Option | Description |
|--------|-------------|
| `JOB_ID` | Scheduler job ID. |
| `--remote HOST` | Target SSH host. |
| `--example NAME` | Only show this example (e.g. `test`, `fsdp`, `vit`). Default: last-failed example. |
| `--lines N`, `-n N` | Tail lines (default: `200`). |

## Limits

Workdir discovery is PBS-specific (`qstat -xf` / `PBS_O_WORKDIR`). Not portable to Slurm `scontrol` without checking the resolved path. See [`ezpz fetch`](./fetch.md) for spool output and [Remote submission](../guides/remote-submission.md) for the full fetch/logs contract.
