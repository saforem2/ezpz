# 📦 Tar Env

```bash
ezpz tar-env
```

Check for (and create if missing) a tarball of the current Python environment
for distribution to compute nodes.

This is the first step in the two-step environment distribution workflow: ensure
a tarball exists with `ezpz tar-env`, then broadcast it to all worker nodes with
[`ezpz yeet-env`](./yeet-env.md).

## What it does

1. Detects the active Python environment (conda or virtual environment) from
   `sys.executable`
2. Checks whether a `.tar.gz` archive already exists for that environment
3. If no tarball is found, creates one alongside the environment directory
4. Logs the path to the tarball

This is particularly useful on HPC systems where shared filesystems can become
bottlenecks when many nodes simultaneously read Python packages. By packing the
environment into a single tarball and extracting it on each node's local storage,
import times are dramatically reduced.

## Example

```bash
# Check for / create a tarball of the current environment
ezpz tar-env
```

??? tip "Programmatic usage"

    The underlying `check_for_tarball()` function can be called directly from
    Python with additional options:

    ```python
    from ezpz.utils import check_for_tarball

    tarball_path = check_for_tarball(
        env_prefix="/path/to/venv",  # optional, auto-detected from sys.executable
        overwrite=True,              # force re-creation
    )
    ```

## See Also

- [`ezpz yeet-env`](./yeet-env.md) — broadcast the tarball to all worker nodes
- [`ezpz.utils.tar_env`](../python/Code-Reference/utils/tar_env.md) — Python API reference
