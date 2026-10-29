# Remote submission

`ezpz submit` builds a scheduler script locally and submits it on the target
through SSH or the ALCF IRI API. Choose the job's environment before submitting.
`--dry-run` renders the script without making a connection or allocating nodes.

## Choose the environment first

The paths below are **target-side placeholders**. Replace `ACCOUNT` and every
`/shared/PROJECT/USER/...` path with your allocation and real, shared paths on
the target. Pre-create an explicit workdir and install the workload's dependencies
into the selected environment before submitting.

```bash
ezpz submit aurora --dry-run -N 2 -q debug -A ACCOUNT \
    --workdir /shared/PROJECT/USER/runs/example \
    --env-setup-script /shared/PROJECT/USER/site-setup.sh \
    --env /shared/PROJECT/USER/venvs/train \
    -- python3 -m ezpz.examples.test --model debug
```

| Options | Environment setup in the generated job |
|---|---|
| `--env PATH` | Activate an existing venv or conda prefix, or select a Python installation. |
| `--env-setup-script PATH` | Source your shell script in the job shell; it owns the complete setup. |
| Both | Source the script **first**, then activate/select `--env`. |
| Neither | Keep the managed default bootstrap. |

`--env` accepts a prefix such as `/shared/PROJECT/USER/venvs/train`, or a Python
executable under that prefix's `bin/` directory. Executable paths select the
containing installation; they do not rewrite the payload's interpreter name.
Use a command installed in that environment, or its explicit Python path.
Versioned-only installations (for example, `bin/python3.13`) need not provide
unversioned aliases; use the versioned executable in the payload if necessary.
A conda prefix is recognized by `conda-meta`; conda's Bash hook and
`conda activate PATH` run before the command. Initialize conda in the setup
script if it is not already available in the batch shell.

Use `--env-setup-script` for site-specific module initialization, library paths,
cache settings, and other prerequisites. It is **sourced**, not executed in a
separate process, so exports and shell functions remain available. With both
options, load site prerequisites in the script and let `--env` select Python.
An environment directory alone does **not** load a site's accelerator/MPI modules.
A venv's activation must resolve to that same prefix; moved/copied venvs with
stale activation paths fail explicitly rather than silently selecting another
environment. Recreate or repair such an environment before using it.

Either explicit option replaces the managed environment bootstrap: ezpz does
not inject a curl download or install packages into your chosen environment.
Install ezpz there beforehand when using the default `ezpz launch` wrapper.
A missing launcher in the selected environment is an error, not permission to
run ezpz from a different environment on `PATH`.

Relative setup-script names resolve against the job workdir, never `PATH`.
Relative environment paths remain anchored there even if your setup script
changes directory.

Paths are interpreted **on the machine running the job**. A remote setup script
need not exist on your laptop, and ezpz does not upload it. Stage the script,
environment, source, and cached datasets/models on shared storage before the
job. Compute nodes cannot be assumed to have internet access. The managed
remote default still fetches `utils.sh` and may install missing ezpz, so use
explicit setup for offline or reproducible runs.

These options also work for local generated-command submissions. They are
rejected when submitting an existing scheduler script (`ezpz submit job.pbs`):
put setup inside that script instead. The Python API's `env_setup` remains raw
shell. For local default setup, `EZPZ_SETUP_ENV` retains its script/inline-shell
behavior; it is not an environment-prefix option.

!!! warning "Migrating the old `--env`"

    `--env` no longer accepts inline shell or a setup script. Replace
    `--env /path/setup.sh` with `--env-setup-script /path/setup.sh`. Move inline
    commands into a real target-side script and source it with that option.

## Job-unique working directories on PBS and Slurm

Remote generated scripts select and enter the workdir **before** environment
setup, even with either override. Without `--workdir`, both schedulers get a
job-specific directory:

| Scheduler | Submit directory available | Submit directory unavailable |
|---|---|---|
| PBS | `$PBS_O_WORKDIR/ezpz-$PBS_JOBID` | `$HOME/ezpz-$PBS_JOBID` |
| Slurm | `$SLURM_SUBMIT_DIR/ezpz-$SLURM_JOB_ID` | `$HOME/ezpz-$SLURM_JOB_ID` |

The base must be shared across all job nodes. `/tmp` is node-local and is not
the default; using it deliberately requires an explicit distribution/staging
workflow such as `ezpz yeet --compress` and durable output placement.
`--workdir` uses the directory **exactly as given**, with no suffix; pre-create
it on the target. Relative environment/setup paths are relative to the job's
workdir, not your laptop. Prefer absolute target paths. A failed `mkdir` or `cd`
stops the job even with `--no-strict`. Explicit setup runs with fail-fast
shell behavior; `--no-strict` is restored for the workload after setup succeeds.

The submitter's local cwd is never copied into a remote script. Local
submissions without `--remote` retain their current-directory behavior.

## Host shorthand

Positional host sugar: `ezpz submit aurora ...` ≡ `ezpz submit --remote aurora ...`;
`ezpz status aurora` ≡ `ezpz status --remote aurora`. `ezpz status` with no
`JOB_ID` and no `--remote` fans out across known machines and merges results.

## Two backends

| | `ssh` | `iri` |
|---|---|---|
| Machines | anything in `~/.ssh/config` | Aurora, Polaris, Crux |
| Auth | your existing connection | Globus token |
| Job status | `qstat` / `squeue` | structured JSON |
| Needs | nothing extra | [`alcf-tokens`](https://pypi.org/project/alcf-tokens/) |

**`ssh`** runs `qsub` or `sbatch` over an existing connection. The script is
delivered on stdin rather than with a second `scp`, so only one connection
is opened — which matters when each one is gated by MFA.

**`iri`** is the
[ALCF IRI API](https://docs.alcf.anl.gov/services/iri-api/):
`POST /api/v1/compute/job/{resource_id}`. It returns structured JSON
instead of text that has to be scraped out of `qstat`, but it serves only
Aurora, Polaris and Crux — **not Sunspot, and not Perlmutter** (NERSC runs
its own Superfacility API).

Get a token on the machine invoking ezpz (your laptop or login node):

```bash
uvx alcf-tokens login iri
```

Or for a persistent install: `uv tool install alcf-tokens` then `alcf-tokens login iri`.

`ALCF_IRI_TOKEN` is honoured if you would rather supply one directly.

## Backend choice and retry safety

On Aurora, Polaris, and Crux, `auto` tries IRI first. Other targets go directly
to SSH. Choose `--backend ssh` or `--backend iri` to disable automatic fallback.
The successful submission output records which backend returned the job ID.

On the SSH leg, exit `255` denotes a transport failure. Other nonzero exits
are reported as scheduler/remote-command failures rather than retried through
IRI. Bad queues, invalid filesystems, and allocation limits generally need a
corrected request, not a second transport.

!!! warning "Current IRI fallback limitation"

    The current IRI-first implementation catches `RemoteSubmitError` broadly
    before falling back to SSH. It does not classify every IRI failure as
    definitely pre-submission. Do not assume that a timeout means no job was
    created. For controlled mutation workflows, select an explicit backend and
    inspect job state before retrying an ambiguous submission failure.

The intended safety rule is to avoid resubmitting anything the scheduler may
already have accepted. In particular, never blindly retry a lost HTTP response,
a connection dropped after submission, or a scheduler rejection on a different
backend. `--dry-run` avoids all submission side effects while you check the
request and environment script.

## Choosing a backend

Prefer **`ssh`** when:

- the target is Sunspot or Perlmutter (IRI does not serve them);
- you already have a working connection, including MFA;
- you want the scheduler's exact error text on a rejection.

Prefer **`iri`** when:

- you want machine-readable job status without parsing `qstat`;
- you are automating from somewhere with no SSH credentials, such as CI;
- the target is Aurora, Polaris or Crux.

## Worked examples

=== "Aurora (PBS)"

    ```bash
    ezpz submit --remote aurora \
        -N 2 -q debug -A datascience --time 00:30:00 \
        --filesystems home:flare \
        --workdir '/lus/flare/projects/AuroraGPT/me/run' \
        -- python3 -m ezpz.examples.test --model small
    ```

=== "Perlmutter (SLURM)"

    ```bash
    ezpz submit --remote perlmutter \
        -N 2 -q regular -A m4388_g --time 00:30:00 \
        -C gpu --gpus-per-node 4 --ntasks-per-node 4 \
        --workdir /shared/PROJECT/USER/runs/example \
        -- python3 -m ezpz.examples.test --model small
    ```

=== "Force the IRI backend"

    ```bash
    ezpz submit --remote polaris --backend iri \
        -N 2 -q debug -A datascience --time 00:30:00 \
        -- python3 -m ezpz.examples.test
    ```

=== "Preview only"

    ```bash
    ezpz submit --remote aurora --dry-run \
        -N 128 -q prod -A AuroraGPT --time 12:00:00 \
        -- python3 test.py
    ```

## Read job output with `ezpz fetch`

For PBS jobs, fetch a stdout snapshot using your own job ID:

```bash
# Historical example ID; replace it with the ID returned by your submission.
ezpz fetch 12481065 --remote sunspot --lines 80 --no-dedupe
```

This prints the last 80 stdout lines once; it does not follow the file.
`--no-dedupe` preserves repeated lines. The default remote behavior collapses
adjacent identical lines after taking the tail.

IRI can supply the output path, but the file is still read over SSH. Without
usable IRI metadata, fetch discovers PBS `Output_Path` through `qstat -xf`.
You need working noninteractive SSH authentication and a readable output file.
The current fallback does not discover Slurm output paths. See the
[`ezpz fetch` reference](../cli/fetch.md) for current stream limitations.

## Launch wrapping

!!! note "No double `ezpz launch`"

    `ezpz submit` already wraps with `ezpz launch` (CPU binding, hostfile,
    rank counts). If your command redundantly starts with `ezpz launch`,
    the prefix is auto-stripped and a warning is printed. Use `--no-launch`
    only when you need custom launch flags (`ezpz submit --no-launch --
    ezpz launch --timeout 600 -- python train.py`).

For a controller such as `ezpz benchmark` that launches its own distributed
payloads, use `--no-launch` so the controller runs once per job, not once per
rank. Do not submit a benchmark controller under another `ezpz launch`.

## See also

- [`ezpz submit`](../cli/submit.md) — the full flag reference
- [ALCF IRI API](https://docs.alcf.anl.gov/services/iri-api/)
- [Perlmutter](perlmutter.md) — the SLURM specifics
