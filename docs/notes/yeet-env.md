# Distributing Python Environments

On large HPC clusters, Python environments on shared filesystems
create I/O contention and slow startup times. `ezpz yeet-env` solves
this by rsyncing your environment to node-local `/tmp/` storage on
every worker node in your job.

## Quick Start

```bash
# Inside an interactive job allocation:
ezpz yeet-env
```

That's it. By default, `yeet-env`:

1. Detects the active Python environment (`sys.prefix`)
2. Discovers all nodes from the job's hostfile (PBS/SLURM)
3. Distributes the environment to `/tmp/<env-name>/` on every node
4. Patches the activate scripts so they work from the new location

```
  Source: /path/to/project/.venv (3.2 GB)
  Target: /tmp/.venv/ on 4 node(s)
    local:  node01 (rsync to /tmp/.venv/)
    remote: node02, node03, node04
  Syncing (4 nodes, fanout=16)...
    ✓ node01 (local) — 12.3s
    ✓ node02 — 11.8s
    ✓ node03 — 12.1s
    ✓ node04 — 11.9s
  Done in 24.2s

  To use this environment:
    deactivate 2>/dev/null
    source /tmp/.venv/bin/activate

  Then launch your training (from a shared filesystem path):
    cd /path/to/your/project
    ezpz launch python3 -m your_app.train

  Note: /tmp is node-local. Make sure your working directory
  is on a shared filesystem (e.g. Lustre) before launching,
  so all ranks can access data and outputs.
```

After the transfer, activate the local copy and launch:

```bash
deactivate 2>/dev/null           # leave the current env
source /tmp/.venv/bin/activate   # activate the local copy
cd /path/to/your/project         # shared filesystem for data/outputs
ezpz launch python3 -m your_app.train
```

## CLI Options

```
ezpz yeet-env [--src PATH] [--dst PATH] [--hostfile PATH] [--dry-run]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | Active venv/conda env | Source environment path |
| `--dst` | `/tmp/<env-name>/` | Destination on each node |
| `--hostfile` | Auto-detect from scheduler | Hostfile for node list |
| `--dry-run` | — | Preview without transferring |

## How It Works

### Overview

```mermaid
graph LR
    A["ezpz yeet-env"] --> B["Detect source env"]
    B --> C["Discover nodes"]
    C --> D["Tree-based rsync"]
    D --> E["Patch venv paths"]
    E --> F["Print instructions"]
```

### Tree-based distribution

Instead of syncing from one source to all N nodes (which saturates
the source node's network), `yeet-env` distributes in waves. Nodes
that finish become sources for the next wave:

```mermaid
graph TD
    subgraph "Wave 0"
        S["Source node<br/>(shared filesystem)"]
        S --> A1["node01"]
        S --> A2["node02"]
        S --> A3["..."]
        S --> A16["node16"]
    end

    subgraph "Wave 1"
        A1 --> B1["node17"]
        A1 --> B2["node18"]
        A2 --> B3["node33"]
        A2 --> B4["node34"]
        A16 --> B5["node241"]
        A16 --> B6["node256"]
    end

    subgraph "Wave 2"
        B1 --> C1["node257"]
        B1 --> C2["..."]
        B6 --> C3["node4096"]
    end
```

Each wave multiplies the number of sources by the fanout (16),
so the number of waves scales logarithmically:

| Nodes | Waves | Time (approx) |
|-------|-------|---------------|
| 1–16 | 1 | 1× rsync |
| 17–256 | 2 | 2× rsync |
| 257–4,096 | 3 | 3× rsync |
| 4,097–65,536 | 4 | 4× rsync |

After wave 0, all subsequent waves rsync from `/tmp/` (node-local
SSD) instead of the shared filesystem, which is much faster.

### Node discovery

`yeet-env` uses the same node discovery as `ezpz launch`:

1. Checks `PBS_NODEFILE` or `SLURM_NODELIST` env vars
2. Falls back to scheduler-specific queries (qstat, scontrol)
3. Deduplicates hostnames (PBS nodefiles repeat per-GPU)

### Path patching

After rsync, the copied environment's `bin/activate` script and
Python symlinks still contain hardcoded paths to the original
location. `yeet-env` patches these on each node via SSH:

- `sed` replaces the old `VIRTUAL_ENV` path in activate scripts
- Re-links `python3` symlinks to the system Python
- Updates `pyvenv.cfg` to point to the correct base Python

This makes `source /tmp/<env>/bin/activate` work correctly from
the new location.

### Incremental syncs

Because `yeet-env` uses `rsync -a`, subsequent runs are fast — only
changed files are transferred. This makes it practical to re-run
after installing new packages.

## Examples

### Default: sync the active env

```bash
# Inside an interactive job on Polaris:
ezpz yeet-env
```

### Sync a specific environment

```bash
ezpz yeet-env --src /path/to/my-conda-env
```

### Custom destination

```bash
ezpz yeet-env --dst /local/scratch/myenv
```

### Preview without syncing

```bash
ezpz yeet-env --dry-run
```

### Complete workflow

```bash
# 1. Get an interactive allocation
qsub -A <project> -q debug -l select=2 -l walltime=01:00:00 -I

# 2. Distribute the environment
ezpz yeet-env

# 3. Activate the local copy
deactivate 2>/dev/null
source /tmp/<env-name>/bin/activate

# 4. Launch from a shared filesystem path
cd /path/to/your/project
ezpz launch python3 -m your_app.train
```

## See Also

- [`ezpz launch`](../cli/launch/index.md) — launch distributed training
- [Shell Environment](./shell-environment.md) — legacy shell setup utilities
