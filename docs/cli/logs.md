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

## Default selection

When `--example` is not given, `logs` shows the most recently modified matching log (`ls -t ... | head -1`). It does **not** know which example failed — a newer successful example can be shown instead of the failure. Pass `--example <name>` when targeting a specific example.

## Limits

Workdir discovery is PBS-specific (`qstat -xf` / `PBS_O_WORKDIR`) — the current `Output_Path` parser handles the common PBS spool layout and a Sunspot fallback for the author's directory, not arbitrary `PBS_O_WORKDIR` exports or SLURM `scontrol` output. For custom workdirs or true Slurm support, verify the resolved path or use [`ezpz fetch`](./fetch.md) for spool output. See [Remote submission](../guides/remote-submission.md) for the full fetch/logs contract.
