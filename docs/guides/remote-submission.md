# Remote submission

Queue a job on a machine you are not logged into:

```bash
ezpz submit --remote aurora -N 128 -q prod -A AuroraGPT \
    --time 12:00:00 --workdir '/lus/flare/projects/AuroraGPT/me/test' \
    -- python3 test.py
```

`ezpz submit` builds the scheduler script locally, ships it to the target,
and submits it there. `--dry-run` prints the script without connecting at
all, which is the cheapest way to check what would run.

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

Get a token once:

```bash
pip install alcf-tokens
alcf-tokens login
```

`ALCF_IRI_TOKEN` is honoured if you would rather supply one directly.

## When `auto` falls back, and when it must not

`--backend auto` (the default) tries SSH first and falls back to IRI **only
when SSH itself could not connect**. That is exit code 255, and it is the
only code that means the job was never submitted:

| Exit | Meaning | What happens |
|---|---|---|
| `0` | Submitted | Done |
| **`255`** | ssh transport failed — unreachable, auth rejected, connection closed | **Falls back to IRI** |
| anything else | the remote `qsub`/`sbatch` refused the job | **Stops, reports the scheduler's own message** |

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

**Working directory.** Without `--workdir`, the generated script omits its
`cd` entirely and the scheduler starts the job in the remote `$HOME`. The
local working directory is never carried across — it is a path that does
not exist on the target.

**Environment setup.** Remote scripts fetch `utils.sh` over the network,
which is correct here because submission happens from a login node. Inside
a batch script running on a *compute* node the opposite holds — see
[`ezpz submit`](../cli/submit.md) and `SKILL.md`, since compute nodes have
no outbound route.

**Resource ids.** Only Polaris's IRI resource id is published in the ALCF
documentation. The others are looked up at runtime through
`GET /status/resources` rather than hard-coded, because a wrong uuid would
submit to the wrong machine.

## See also

- [`ezpz submit`](../cli/submit.md) — the full flag reference
- [ALCF IRI API](https://docs.alcf.anl.gov/services/iri-api/)
- [Perlmutter](perlmutter.md) — the SLURM specifics
