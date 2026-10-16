# `ezpz extend` / `ezpz update-job`

Extend or retarget a **queued** job — bump its walltime, move it to
another queue/partition, or recharge it to a different account — without
resubmitting. Wraps IRI `PUT /compute/job/{resource_id}/{job_id}` (preferred)
and `qalter` / `scontrol update` over SSH as fallback.

```bash
ezpz extend 8879044 --remote aurora --time 02:00:00
ezpz extend 8879044 --remote aurora --time 02:00:00 --queue prod --account AuroraGPT
```

`update-job` is an alias:

```bash
ezpz update-job 8879044 --remote aurora --time 02:00:00
```

!!! warning "Queued jobs only"
    The scheduler and the IRI API both reject updates to running jobs.
    Check `ezpz status --remote <host> <job-id>` first; if the state is
    `active`/`running` the update will fail.

## Options

| Flag | Description |
|------|-------------|
| `JOB_ID` (positional, required) | Job to update |
| `-r`, `--remote HOST` | **Required.** Target host (`aurora`/`polaris`/`crux` or any SSH alias) |
| `-t`, `--time WALLTIME` | New walltime — `HH:MM:SS`, `MM:SS`, or seconds (e.g. `7200`) |
| `-q`, `--queue NAME` | Move to this queue / partition |
| `-A`, `--account NAME` | Charge this account / project |
| `--backend {auto,iri,ssh}` | Backend (default: `auto` — IRI-first on IRI hosts, else SSH) |
| `--dry-run` | Print what would be sent without sending |

You must supply at least one of `--time` / `--queue` / `--account`;
otherwise the command exits with a usage error.

## Backend selection

Same rule as [`ezpz submit --remote`](../guides/remote-submission.md):
`auto` tries IRI first on Aurora/Polaris/Crux and falls back to SSH only
on transport/auth failures. Explicit failures (`400`/`422`, `Unknown Job
Id`, `not served by IRI`) are never retried on the other backend — they
mean "fix the request".

## Examples

### Extend walltime

```bash
ezpz extend 8879044 --remote aurora --time 04:00:00          # HH:MM:SS
ezpz extend 8879044 --remote aurora --time 7200              # seconds
ezpz extend 8879044 --remote aurora --time 30:00             # MM:SS
```

### Retarget queue and account together

```bash
ezpz extend 8879044 --remote aurora --queue prod --account AuroraGPT
```

### Force the SSH path (qalter)

```bash
ezpz extend 8879044 --remote aurora --queue prod --backend ssh
```

### Preview

```bash
ezpz extend 8879044 --remote aurora --time 02:00:00 --dry-run
# would PUT /api/v1/compute/job/{rid}/8879044 {duration:7200, queue:None, account:None} backend=auto
```

## Output

On success the updated job is printed:

```text
8879044.aurora-pbs-0001 queued queue=prod (via iri)
```

On failure the API/scheduler message is printed on stderr and the
process exits non-zero.

## See also

- [`ezpz status`](./status.md) — verify the job is still queued
- [`ezpz cancel`](./cancel.md) — give up and cancel instead
- [Remote submission](../guides/remote-submission.md) — backends and auth
