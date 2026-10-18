# `ezpz cancel`

Positional host sugar: `ezpz cancel aurora 12345` ≡ `ezpz cancel 12345 --remote aurora`.

Cancel a queued or running job, locally or on a remote machine.

```bash
# Local (qdel / scancel)
ezpz cancel 8879044.aurora-pbs-0001

# Remote
ezpz cancel 8879044.aurora-pbs-0001 --remote aurora
```

Locally the scheduler is inferred from the hostname (`PBS` vs `SLURM`)
and the command runs `qdel` or `scancel` accordingly. Remotely the same
inference is applied to the `--remote` host; override it with
`--scheduler` when a host's scheduler cannot be inferred.

Remote cancellation tries SSH first (`qdel`/`scancel` on the login node)
and falls back to the IRI `DELETE /compute/cancel/{resource_id}/{job_id}`
endpoint only when the SSH path fails for a transport-class reason. A
scheduler rejection (e.g. "Unknown Job Id") is never retried on another
backend. See [Remote submission](../guides/remote-submission.md) for the
fallback contract.

## Options

| Flag | Description |
|------|-------------|
| `JOB_ID` (positional, required) | Job id to cancel |
| `--remote HOST` | Host to cancel on (SSH name or ALCF resource). Omit for local `qdel`/`scancel`. |
| `--backend {auto,ssh,iri}` | Remote backend with `--remote` (default: `auto`) |
| `--scheduler {PBS,SLURM}` | Override scheduler auto-detection |
| `--dry-run` | Print what would run without cancelling |

## Examples

### Local cancel

```bash
ezpz cancel 12345.master.cluster.local
```

### Remote cancel via IRI

```bash
ezpz cancel 8879044.aurora-pbs-0001 --remote aurora --backend iri
```

### Preview without cancelling

```bash
ezpz cancel 8879044 --remote aurora --dry-run
# would cancel 8879044 on aurora via auto (scheduler=auto)
```

### Force a scheduler

```bash
ezpz cancel 123 --remote my-alias --scheduler SLURM
```

## Output

On success:

```text
cancelled 8879044.aurora-pbs-0001 (via iri)
```

On failure the process exits non-zero with the backend's error:

```text
error: cancel 8879044 failed via iri (exit 404): ...
```

## Exit codes

- `0` — cancelled (or `--dry-run`).
- Non-zero — `qdel`/`scancel` or IRI `DELETE` failed; the scheduler/API message is printed on stderr.

## See also

- [`ezpz status`](./status.md) — check a job's state before cancelling
- [`ezpz submit`](./submit.md) — queue jobs
- [Remote submission](../guides/remote-submission.md) — backends and auth
