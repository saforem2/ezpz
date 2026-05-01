# 📦 Tar Env

```bash
ezpz tar-env
```

Check for (and create if missing) a tarball of the current Python environment
for distribution to compute nodes.

`ezpz tar-env` is a standalone tarball-creation utility. For end-to-end
environment distribution, use [`ezpz yeet`](./yeet.md) instead —
it handles the full Lustre → `/tmp/` copy, path patching, and parallel
fan-out to all worker nodes (with `--compress` providing similar
compression benefits as a separate `tar-env` step).

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

- [`ezpz yeet`](./yeet.md) — distribute files (envs, models, datasets,
  etc.) to all worker nodes via parallel rsync (recommended for
  end-to-end use)
- [`ezpz.utils.tar_env`](../python/Code-Reference/utils/tar_env.md) — Python API reference
