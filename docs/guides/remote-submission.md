# Remote submission

Queue a job on a machine you are not logged into:

```bash
ezpz submit aurora -N 128 -q prod -A AuroraGPT \
    --time 12:00:00 --workdir '/lus/flare/projects/AuroraGPT/me/test' \
    -- python3 test.py
# also: ezpz submit --remote aurora ...
```

`ezpz submit` builds the scheduler script locally, ships it to the target,
and submits it there. `--dry-run` prints the script without connecting at
all, which is the cheapest way to check what would run.

Positional host sugar: `ezpz submit aurora ...` ≡ `ezpz submit --remote aurora ...`;
`ezpz status aurora` ≡ `ezpz status --remote aurora`. `ezpz status` with no `JOB_ID` and no `--remote` fans out to every reachable machine and merges the results.

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

Get a token once (from the Aurora UAN):

```bash
uvx alcf-tokens login iri
```

Or for a persistent install: `uv tool install alcf-tokens` then `alcf-tokens login iri`.

`ALCF_IRI_TOKEN` is honoured if you would rather supply one directly.

## When `auto` falls back, and when it must not

On IRI-capable hosts (Aurora, Polaris, Crux) `auto` is **IRI-first**: try
the IRI API, and only on auth/network/transport failure fall back to SSH
(`qsub`/`sbatch` over an SSH hop). A scheduler rejection on either path
is never retried — it means "fix the request," and retrying would risk a
double-submit. On non-IRI hosts (Sunspot, Perlmutter, Frontier) `auto`
goes straight to SSH; there is no IRI to fall back to.

The old table (SSH 255 → IRI) still describes the SSH leg, but the IRI leg
is now tried first:

| Outcome | Meaning | What happens |
|---|---|---|
| IRI `0` / SSH `0` | Submitted | Done |
| IRI `401` / network error (auto only) | Token expired / unreachable | Fall back to SSH |
| SSH `255` (transport) | Unreachable / auth rejected | Fall back (only relevant when IRI was skipped) |
| Any scheduler rejection (`rc=1`) | Bad queue / over quota / bad FS | Stop, show the scheduler's message |

The distinction is not cosmetic. `ssh` returns its own status only for
transport failures; for everything else it passes through the remote
command's exit code:

```console
$ ssh unreachable.invalid true          ; echo $?
255
$ ssh aurora 'exit 42'                  ; echo $?
42
$ ssh aurora 'qsub /nonexistent.pbs'    ; echo $?
1
```

**A scheduler rejection is never retried on another backend.** It would
either reproduce the rejection with a worse error message, or — if the
first submission actually landed before the non-zero exit — **submit the
job twice**. These are all `rc=1`, and all of them mean "fix the request",
not "try another road":

```text
qsub: Request rejected. Reason: No active allocation found for project ...
qsub: Invalid filesystem identifiers gila
qsub: would exceed queue generic's per-user limit of jobs in 'Q' state
```

Two further guards:

- **No fallback where IRI cannot help.** On Sunspot or Perlmutter a 255
  is fatal, and the error says so rather than attempting a doomed call.
- **The backend used is always reported**, so you can tell afterwards
  which path a job took:

  ```console
  $ ezpz submit --remote aurora -N 2 -q debug -A myproj -- python3 x.py
  8879044.aurora-pbs-0001  (via ssh)
  ```

Force one backend with `--backend ssh` or `--backend iri` when you do not
want the automatic behaviour.

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
        --workdir "$SCRATCH/run" \
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

## Notes

**Working directory.** Without `--workdir`, the job starts in a per-job
directory under the scheduler's submit dir
(`$PBS_O_WORKDIR/ezpz-$PBS_JOBID` on PBS, `$SLURM_SUBMIT_DIR/…` on SLURM,
`$HOME` fallback) on **shared FS** — not `/tmp` (which is node-local and
would shard output across nodes). Pass `--workdir` explicitly when you want
to control the location.

**Environment setup.** Every remote script is self-bootstrapping — it sets
ALCF proxies, `cd`s into its workdir, then:

```bash
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
# ensure ezpz is present (handles fresh/empty venvs)
uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
```

Overridable with `--env` for site-specific setup. The old target-resolved
`python3 -c 'import ezpz ...'` path still works but is no longer the default.

## See also

- [`ezpz submit`](../cli/submit.md) — the full flag reference
- [ALCF IRI API](https://docs.alcf.anl.gov/services/iri-api/)
- [Perlmutter](perlmutter.md) — the SLURM specifics
