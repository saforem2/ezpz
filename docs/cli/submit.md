# `ezpz submit`

Submit jobs to PBS (`qsub`) or SLURM (`sbatch`) schedulers directly from
the command line.

## Two Modes

### 1. Wrap a command

Provide a command after `--` and `ezpz submit` generates a job script
automatically:

```bash
ezpz submit -N 2 -q debug -t 01:00:00 \
    -- python3 -m ezpz.examples.test --model small
```

The generated script includes:

- Scheduler directives (`#PBS` or `#SBATCH`)
- Activation of your current Python environment (venv or conda)
- `ezpz launch` wrapping for distributed execution

### 2. Submit an existing script

Pass a `.sh` file directly:

```bash
ezpz submit job.sh --nodes 4 --time 02:00:00
```

## Options

| Flag | Description |
|------|-------------|
| `-N`, `--nodes` | Number of compute nodes (default: 1) |
| `-t`, `--time` | Walltime in `HH:MM:SS` format (default: `01:00:00`) |
| `-q`, `--queue` | Queue (PBS) or partition (SLURM) (default: `debug`) |
| `-A`, `--account` | Project/account for billing |
| `--filesystems` | PBS filesystems directive (default: `home`) |
| `--job-name` | Job name (auto-derived from command if omitted) |
| `--scheduler` | Force `PBS` or `SLURM` (auto-detected by default) |
| `--dry-run` | Print the script without submitting |
| `--no-launch` | Don't wrap the command with `ezpz launch` |

## Examples

### Dry-run to preview the generated script

```bash
ezpz submit --dry-run -N 2 -q debug -A myproject \
    -- python3 -m ezpz.examples.fsdp --model small
```

### Submit with specific filesystems (PBS/Aurora)

```bash
ezpz submit -N 2 -q debug -t 01:00:00 \
    --filesystems home:eagle:grand \
    -A myproject \
    -- python3 -m ezpz.examples.test
```

### Submit without `ezpz launch` wrapping

```bash
ezpz submit --no-launch -N 1 -q debug \
    -- mpirun -np 4 ./my_binary
```

## Environment Detection

The generated script automatically activates your current environment:

- **venv**: If `VIRTUAL_ENV` is set, adds `source $VIRTUAL_ENV/bin/activate`
- **conda**: If `CONDA_PREFIX` is set, adds `conda activate <env_name>`
- **Custom**: If `EZPZ_SETUP_ENV` points to a file, sources it

## Account Fallback

If `--account` is not provided, `ezpz submit` checks these environment
variables in order:

- `PBS_ACCOUNT` (PBS)
- `SLURM_ACCOUNT` (SLURM)
- `PROJECT`
