# Distributing Python Environments

On large HPC clusters, Python environments on shared filesystems
create I/O contention and slow startup times. `ezpz yeet-env` solves
this by copying your environment to node-local `/tmp/` storage on
every worker node in your job.

## Quick Start

```bash
# Inside an interactive job allocation:
ezpz yeet-env
```

That's it. By default, `yeet-env`:

1. Detects the active Python environment (`sys.prefix`)
2. Discovers all nodes from the job's hostfile (PBS/SLURM)
3. Copies the environment to `/tmp/<env-name>/` on the current node
4. Patches activate scripts, shebangs, and symlinks for the new location
5. Distributes the patched copy to all remote nodes via greedy rsync fan-out

```
  Source: /path/to/project/.venv (3.2 GB)
  Target: /tmp/.venv/ on 4 node(s)
    local:  node01 (rsync to /tmp/.venv/)
    remote: node02, node03, node04
  Syncing (4 nodes)...

    ✓ node01 (local, rsync) — 12.3s
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
ezpz yeet-env [--src PATH] [--dst PATH] [--hostfile PATH]
              [--copy | --compress] [--dry-run]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | Active venv/conda env | Source environment path |
| `--dst` | `/tmp/<env-name>/` | Destination on each node |
| `--hostfile` | Auto-detect from scheduler | Hostfile for node list |
| `--copy` | — | Use `cp -a` for the local copy (faster on Lustre) |
| `--compress` | — | tar.gz → copy → extract (least Lustre metadata I/O) |
| `--dry-run` | — | Preview without transferring |

!!! tip "Choosing a local copy method"

    The default `rsync` is best for **incremental updates** (after
    `pip install`, etc.) but slow for initial copies on Lustre because
    it stats every file individually. For the first transfer, use one
    of the faster methods:

    | Method | Best for | How it works |
    |--------|----------|--------------|
    | `--copy` | Fast initial copy | `cp -a` — sequential dir walk, no checksums |
    | `--compress` | Slowest Lustre / largest envs | tar.gz → copy 1 file → extract locally |
    | *(default)* | Incremental updates | `rsync -rlD` — only transfers changed files |

    ```bash
    # First time: compress for minimal Lustre I/O
    ezpz yeet-env --compress

    # Or: cp for simpler fast copy
    ezpz yeet-env --copy

    # After pip install: rsync only sends diffs
    ezpz yeet-env
    ```

    All three methods only affect the **local** Lustre → `/tmp/` copy.
    Remote node distribution always uses rsync.

## How It Works

### Overview

```mermaid
graph TD
    A["ezpz yeet-env"] --> B["Detect source env"]
    B --> C["Discover nodes<br/>(PBS_NODEFILE / SLURM_NODELIST)"]
    C --> D["Copy to local /tmp/<br/>(rsync, cp -a, or tar.gz)"]
    D --> E["Patch paths + shebangs"]
    E --> F["Greedy rsync fan-out"]
    F --> G["Print instructions"]
```

### Step 1: Local copy + patch

First, `yeet-env` copies the source environment to `/tmp/<env>/` on
the current node using rsync (default), `cp -a` (`--copy`), or
tar.gz (`--compress`). If the local copy fails, distribution is
aborted immediately — no broken environment gets distributed.

After copying, the venv is patched **once** in place:

- Replaces hardcoded `VIRTUAL_ENV` paths in activate scripts
  (`bin/activate`, `bin/activate.csh`, `bin/activate.fish`)
- Re-links `python3` symlinks to the system Python
- Updates `pyvenv.cfg`
- **Rewrites shebangs** in all entry-point scripts (`ezpz`, `pip`,
  `torchrun`, etc.) — pip bakes absolute paths into these at install
  time, so they'd still point to the original Lustre location without
  this step

This patched copy in `/tmp/` becomes the source for all subsequent
rsyncs — no per-node patching or SSH needed.

### Step 2: Greedy fan-out

Instead of syncing from one source to all N nodes (which saturates
the source node's NIC), `yeet-env` uses a **greedy streaming
fan-out**: each node that finishes immediately becomes a source for
others, without waiting for any "wave" to complete.

A single thread pool manages all rsyncs. Each source node is capped
at `MAX_PER_SOURCE=8` concurrent outbound rsyncs to avoid
overwhelming any single NIC. As soon as any rsync completes:

1. That node is registered as a new source
2. New rsyncs are submitted using whichever source has the
   fewest active transfers (load balancing)

The tree grows organically — no synchronized rounds:

```mermaid
graph TD
    subgraph "Local copy + patch"
        S["Source<br/>(shared filesystem)"] -->|"rsync / cp / tar.gz"| L["/tmp/ on node00"]
    end

    subgraph "Fan-out (greedy, up to 8 per source)"
        L --> A1["node01"]
        L --> A2["node02"]
        L --> A3["node03"]
        L --> A4["node04"]
        L --> A5["node05"]
        L --> A6["node06"]
        L --> A7["node07"]
        L --> A8["node08"]

        A1 -->|"immediately<br/>becomes source"| B1["node09"]
        A1 --> B2["node10"]
        A2 --> B3["node11"]
        A2 --> B4["node12"]
        A3 --> B5["node13"]
    end

    subgraph "...continues until all nodes served"
        B1 --> C1["node17"]
        B2 --> C2["node18"]
        B3 --> C3["..."]
    end
```

The key difference from a wave-based approach: if node01 finishes
in 15 seconds but node08 takes 30 seconds, node01 immediately
starts serving new targets — it doesn't wait for node08.

??? info "Scaling behavior"

    The greedy fan-out gives approximately O(log N) wall-clock time:

    - After the local copy, the first 8 rsyncs start from node00
    - As each completes (~15–20s), it starts serving others
    - With 8 initial targets completing, there are 9 sources
    - Those 9 sources can each serve 8 more = 72 concurrent rsyncs
    - After ~2 "generations", 500+ nodes are reachable

    For a 512-node job with a 5 GB venv:

    | Phase | Approx time | Sources |
    |-------|-------------|---------|
    | Local copy (Lustre → `/tmp/`) | ~60s | 1 |
    | First 8 targets complete | ~20s | 9 |
    | Next ~72 targets complete | ~20s | 81 |
    | Remaining ~431 targets | ~20s | 500+ |
    | **Total** | **~2 min** | — |

    Single-source approach for comparison: 512 × 5 GB from one NIC
    at 200 Gbps = **~100s** theoretical minimum, worse in practice
    due to TCP congestion with 512 concurrent connections.

??? info "ASCII diagram: greedy fan-out"

    ```
                       ezpz yeet-env

    Step 0: Local copy + patch
    ══════════════════════════

    Lustre ──rsync/cp/tar──▶ /tmp/.venv (node00)
                                │
                            [patch paths + shebangs]
                                │
    Step 1: Fan-out (greedy, max 8 per source)
    ═══════════════════════════════════════════
                                │
          ┌───┬───┬───┬───┬───┬┴──┬───┬───┐
          ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
         n01 n02 n03 n04 n05 n06 n07 n08
          │   │
          │   └─── (n02 finishes, starts serving) ──▶ n09, n10, ...
          │
          └─── (n01 finishes, starts serving) ──▶ n11, n12, ...

    No waiting for "waves" — each node starts serving
    the moment its rsync completes.

    Key:
      • Each source limited to 8 concurrent outbound rsyncs
      • New sources pick up work immediately (no wave barriers)
      • Load-balanced: new targets assigned to least-busy source
      • All rsyncs from /tmp/ (fast node-local storage)
      • Path patching happens ONCE (step 0), not per-node
    ```

??? info "Detail: how source selection works"

    The thread pool picks the source with the fewest active
    outbound rsyncs. This naturally load-balances across the tree:

    ```
    Sources:          Active rsyncs:
    node00            ████████ (8/8 — at cap, skip)
    node01            ████·· (4/8 — available)      ← picked
    node02            ██████ (6/8 — available)
    node03            ████████ (8/8 — at cap, skip)
    ```

    When node01 is selected, one of its remaining slots is used.
    If all sources are at capacity, the pool waits for any rsync
    to complete before submitting more work.

### Node discovery

`yeet-env` discovers nodes directly from scheduler environment
variables, without importing heavy Python packages (torch, numpy,
etc.) — so the CLI starts in seconds even on slow filesystems:

1. Checks `PBS_NODEFILE` or `HOSTFILE` environment variables
2. For SLURM: expands `SLURM_NODELIST` via `scontrol show hostnames`
3. Deduplicates hostnames (PBS nodefiles repeat per-GPU)

### Path patching

Venv activate scripts, Python symlinks, and **entry-point script
shebangs** contain hardcoded absolute paths. `yeet-env` patches
these **once** on the local `/tmp/` copy before any distribution:

- Replaces the old `VIRTUAL_ENV` path in activate scripts
- Re-links `python3` symlinks to the system Python
- Updates `pyvenv.cfg` to point to the correct base Python
- Rewrites shebangs in all `bin/` scripts (e.g. `#!/old/path/.venv/bin/python3`
  → `#!/tmp/.venv/bin/python3`)

Since patching happens before fan-out, all distributed copies
arrive already patched — no per-node SSH needed.

### Incremental syncs

The default rsync mode uses `-rlD` (recursive, symlinks, devices —
skipping expensive timestamp/permission sync). Subsequent runs only
transfer changed files, making it practical to re-run after
installing new packages.

### Error handling

- **Failed local copy**: distribution is aborted immediately — no
  broken environment gets sent to remote nodes
- **rsync exit 24** (vanished files): treated as success. This
  happens when concurrent rsyncs read from the same `/tmp/` source
  while temporary files (e.g. triton plugin builds) come and go.
- **TTY-aware progress**: spinner and `\r` carriage returns are
  suppressed when stdout is not a terminal (e.g. redirected to a
  file), preventing garbled output in logs.

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

### Real-world example: 64 nodes on Sunspot

??? example "8.3 GB venv → 65 nodes in ~2 minutes"

    ```bash
    $ ezpz yeet-env
      Source: /lus/tegu/.../torchtitan-213/.venv (8.3G)
      Target: /tmp/.venv/ on 65 node(s)
        local:  x1921c0s2b0n0 (rsync to /tmp/.venv/)
        remote: x1921c0s2b0n0-hsn0, x1921c0s3b0n0-hsn0, x1921c0s4b0n0-hsn0, ... (64 nodes)
      Syncing (65 nodes)...

        ✓ x1921c0s2b0n0 (local, rsync) — 49.6s
        ✓ x1921c0s2b0n0-hsn0 — 0.8s
        ✓ x1921c0s6b0n0-hsn0 — 19.6s
        ✓ x1921c1s5b0n0-hsn0 — 20.1s
        ✓ x1921c1s7b0n0-hsn0 — 20.2s
        ...
        ✓ x1921c7s6b0n0-hsn0 — 21.2s
      Done in 91.2s
    ```

    **Timing breakdown:**

    | Phase | Time | Notes |
    |-------|------|-------|
    | Local copy (Lustre → `/tmp/`) | 50s | One-time, includes path patching |
    | Fan-out to 64 remote nodes | ~42s | Greedy, nodes become sources as they finish |
    | **Total** | **~91s** | 8.3 GB to 65 nodes |

    The first 8 nodes complete in ~20s, then immediately start
    serving as sources for the remaining nodes. No node waits
    for others to finish — the tree grows as fast as individual
    rsyncs complete.

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

- [`ezpz launch`](./launch/index.md) — launch distributed training
- [`ezpz.utils.yeet_env`](../python/Code-Reference/utils/yeet_env.md) — Python API reference
- [Shell Environment](../notes/shell-environment.md) — legacy shell setup utilities
