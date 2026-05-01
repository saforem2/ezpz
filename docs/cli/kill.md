# Killing Stuck Processes

`ezpz kill` cleans up python processes that ezpz launched, on the
local node or fanned out across the whole job allocation. Useful when
a distributed run hangs, crashes mid-shutdown and leaves stragglers,
or when you want to clear the deck before re-launching.

## Quick Start

```bash
# Kill everything ezpz launched on this node:
ezpz kill

# Kill anything matching a substring (pkill -f style):
ezpz kill train.py

# Same, across every node in the current job:
ezpz kill --all-nodes

# See what would be killed without actually doing it:
ezpz kill --dry-run
```

## How matching works

The default (no positional argument) match looks for processes whose
**environment** contains the `EZPZ_RUN_COMMAND` variable. `ezpz launch`
sets this on every process it starts, so the no-arg form will only
hit jobs that ezpz actually launched — it won't touch unrelated
python processes (notebooks, language servers, etc.).

When you pass a positional argument, the match switches to a
substring search of `/proc/<pid>/cmdline`. This is the same matching
as `pkill -f STRING`:

```bash
ezpz kill train.py        # matches "python -m my.train.py --epochs=10"
ezpz kill ezpz.examples   # matches anything launched via `ezpz launch python -m ezpz.examples...`
```

`ezpz kill` always skips its own PID and its parent shell — the
command itself is safe to run from inside the shell you want to keep.

## CLI Options

```
ezpz kill [STRING] [--all-nodes] [--hostfile PATH]
          [--signal NAME] [--dry-run]
```

| Arg / Flag | Default | Description |
|------|---------|-------------|
| `STRING` (positional) | — | Match any process whose cmdline contains this substring. Without it, match `EZPZ_RUN_COMMAND` env var. |
| `--all-nodes` | local only | SSH into every node in the hostfile and run kill there too |
| `--hostfile` | auto-detect | Hostfile for `--all-nodes` (default: same discovery as `ezpz yeet`) |
| `--signal` | `TERM` | Signal to send (`TERM`, `KILL`, `INT`, `HUP`, `QUIT`) |
| `--dry-run` | — | List matches without sending any signals |

## Signal handling

By default, `ezpz kill` sends `SIGTERM`, waits up to **3 seconds** for
each process to exit, and escalates to `SIGKILL` for any survivor.
This is the standard "graceful then firm" pattern — most well-behaved
torch training scripts will clean up NCCL state and exit cleanly on
SIGTERM, but a hung process won't block forever.

For an immediate hard kill (skip the grace period entirely):

```bash
ezpz kill --signal KILL
```

## Multi-node fan-out

`--all-nodes` discovers nodes the same way `ezpz yeet` does:

1. `--hostfile` if explicitly passed
2. `PBS_NODEFILE` / `HOSTFILE` env vars
3. PBS aux lookup via `PBS_JOBID` (`/var/spool/pbs/aux/<jobid>`)
4. PBS `qstat -fn1wru $USER` fallback for active jobs
5. SLURM `scontrol show hostnames`
6. Localhost fallback (single-node mode)

For each remote node, `ezpz kill` SSHes in and runs `ezpz kill
<STRING>` (without `--all-nodes`, to avoid recursion). SSH connections
are bounded at 16 concurrent — DNS / SSH on a 1000-node allocation
shouldn't fork a thread per node.

Output from each node is prefixed with `[hostname]`:

```
[x1921c0s2b0n0 (local)]
  killed 12345 (TERM): python -m my.train --batch-size 32 ...
  killed 12346 (TERM): python -m my.train --batch-size 32 ...
[x1921c0s4b0n0]
  killed 5678 (TERM): python -m my.train --batch-size 32 ...
[x1921c0s6b0n0]
  killed 9012 (TERM): python -m my.train --batch-size 32 ...
```

## Common workflows

### Recover from a hung run

```bash
ezpz kill --all-nodes              # SIGTERM everywhere
sleep 5
ezpz kill --all-nodes --signal KILL  # firm escalation if anything survived
```

### Clear stragglers before re-launching

```bash
# After a crashed run, often a few rogue python processes are still
# pinned to GPUs:
ezpz kill --all-nodes
ezpz launch python -m my.train ...   # clean re-launch
```

### Targeted: kill only one of several jobs on a node

```bash
ezpz kill --dry-run my.train.run-A    # confirm match scope
ezpz kill my.train.run-A              # only the run-A processes
```

## Platform notes

### macOS (development)

macOS doesn't expose other processes' environment without elevated
privileges, so the no-arg `ezpz kill` form is unsupported there:

```
$ ezpz kill
  ezpz kill (no pattern) is unsupported on macOS — pass a substring
  (`ezpz kill <STR>`) instead. The default EZPZ_RUN_COMMAND-based
  match requires reading /proc/<pid>/environ (Linux only).
```

For local dev on Mac, always pass an explicit substring.

### Linux (production HPC)

Full functionality. PIDs are enumerated from `/proc`, environs read
from `/proc/<pid>/environ`, and cmdlines from `/proc/<pid>/cmdline`.

## Exit codes

- `0` — all matched processes killed (or no matches found)
- `1` — at least one kill failed, or no nodes discovered for `--all-nodes`

## See Also

- [`ezpz launch`](./launch/index.md) — sets the `EZPZ_RUN_COMMAND` env
  var that `ezpz kill` matches on by default
- [`ezpz yeet`](./yeet.md) — uses the same node-discovery logic for
  multi-node fan-out
