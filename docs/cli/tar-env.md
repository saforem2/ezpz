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
   `sys.executable`.
2. Checks for an existing `.tar.gz` in three locations, in order:
   `<env-parent>/<env-name>.tar.gz` → `/tmp/<env-name>.tar.gz` →
   `<cwd>/<env-name>.tar.gz`. If found (and `overwrite=False`), reuses it.
3. If none exist, creates a new gzipped tarball **next to the
   environment** (e.g. `/path/to/.venv` → `/path/to/.venv.tar.gz`).
4. Returns the absolute path to the tarball.

This is the canonical location for the tarball: subsequent `ezpz yeet`
(no args) invocations will see the same-named `.tar.gz` next to the
detected venv and print a hint suggesting the tarball-broadcast form,
which scales much better than per-file rsync (see the
[scaling section](./yeet.md#scaling-aurora-8--4096-nodes) on the yeet
page).

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
