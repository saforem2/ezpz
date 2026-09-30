# `ezpz update`

Upgrade `ezpz` in the environment you are currently in, and refresh the
cached `utils.sh` that the installer placed.

```bash
ezpz update
```

Two steps, either of which can be skipped:

1. Reinstall the package from git into the **active** environment —
   whichever venv or conda env is on your `PATH` right now.
2. Re-download `utils.sh` into `~/.local/share/ezpz/`.

## Options

| Flag | Description |
|------|-------------|
| `--ref REF` | Install a specific branch, tag or SHA instead of `main` |
| `--utils-only` | Refresh `utils.sh` only; leave the package alone |
| `--package-only` | Upgrade the package only; leave `utils.sh` alone |
| `--dry-run` | Print the commands without running them |

## Examples

```bash
ezpz update                        # package + utils.sh
ezpz update --ref v0.29.3          # pin a release
ezpz update --utils-only           # just the shell helpers
ezpz update --dry-run              # see what would happen
```

## Login nodes only

`ezpz update` **refuses to run inside a PBS or SLURM job**:

```console
$ ezpz update
Error: refusing to update inside a batch job: compute nodes have no
outbound network. Run this on a login node.
```

This is a guard, not a limitation. Compute nodes on Aurora, Polaris and
Sunspot have no outbound route, so the download would hang for ~270 s
and then leave things unconfigured — a failure that surfaces much later
as `CONDA_PREFIX still not set` or as every `ezpz_*` function being
undefined. One line now beats four minutes of confusion later.

`--dry-run` is still allowed inside a job, since it touches nothing.

## A partial download never replaces a good file

`utils.sh` is written to a `.part` file and checked with `bash -n`
before it replaces the installed copy. A truncated download passes a
naive non-empty test and then fails as "every function is undefined",
which is strictly worse than a clean error — so a bad fetch leaves your
working `utils.sh` exactly as it was.

## Why this is a subcommand, not a script

An `ezpz_update` executable that rewrote itself from the network would
be awkward on a shared login node, hard to pin for reproducibility, and
redundant with re-running the installer. As a subcommand it is versioned
with the package, covered by tests, and visible in `git log`.

Re-running the installer does the same job:

```bash
curl -fsSL https://ezpz.cool/install.sh | bash
```

## See also

- [Quickstart](../quickstart.md#one-line-install) — first-time install
- [`ezpz submit`](submit.md)
