# 🚀 Yeet Env

```bash
ezpz yeet-env [OPTIONS]
```

Broadcast a Python environment tarball to all worker nodes via MPI, then
optionally decompress it on each node's local storage.

This is the second step in the environment distribution workflow (after
[`ezpz tar-env`](./tar-env.md)). It uses MPI collective operations to
efficiently distribute the tarball from rank 0 to every other rank, with
chunked transfers to handle large environments.

## Options

| Flag                   | Default            | Description                                              |
| ---------------------- | ------------------ | -------------------------------------------------------- |
| `--src PATH`           | auto-detected      | Path to the source tarball to broadcast                  |
| `--dst PATH`           | auto-detected      | Destination path on each node for the extracted env      |
| `--decompress`         | `True`             | Decompress the tarball after transfer                    |
| `--chunk-size BYTES`   | `134217728` (128MB) | Transfer chunk size in bytes for MPI broadcast           |
| `--overwrite`          | `False`            | Overwrite existing files at destination                  |

## Example

```bash
# Broadcast with defaults (auto-detect source tarball, decompress on arrival)
ezpz yeet-env

# Specify source and destination explicitly
ezpz yeet-env --src /path/to/env.tar.gz --dst /local/scratch/env

# Use smaller chunks for memory-constrained nodes
ezpz yeet-env --chunk-size 536870912

# Force overwrite of existing destination
ezpz yeet-env --overwrite
```

??? tip "Workflow: tar-env + yeet-env"

    The typical workflow on an HPC cluster looks like:

    ```bash
    # Step 1: Create the tarball (run once, from login node)
    ezpz tar-env

    # Step 2: Broadcast to all compute nodes (run inside job script)
    ezpz yeet-env
    ```

    This avoids the shared-filesystem bottleneck where hundreds of nodes
    simultaneously try to read Python packages from the same NFS/Lustre mount.

## See Also

- [`ezpz tar-env`](./tar-env.md) — create the environment tarball
- [`ezpz.utils.yeet_env`](../python/Code-Reference/utils/yeet_env.md) — Python API reference
