# `ezpz status`

Inspect job status locally or on a remote ALCF machine.

```bash
# Local (wraps qstat)
ezpz status
ezpz status 8879044.aurora-pbs-0001

# Remote — list recent jobs on Aurora (all equivalent)
ezpz status --remote aurora
ezpz status aurora
ezpz status --remote aurora --limit 20 --states queued,active
```

!!! tip "All-machines status"

    `ezpz status` with no `JOB_ID` and no `--remote` fans out to every
    reachable machine — IRI hosts (Aurora / Polaris / Crux) plus any SSH
    alias that responds — and merges the results into one table (one
    `owner` is your user on IRI, all users on SSH unless you pass
    `--owner`/`--all-users`). Slow or unreachable hosts show a warning
    rather than blocking the rest.

!!! tip "Positional remote sugar"

    `ezpz status aurora` ≡ `ezpz status --remote aurora`; `ezpz cancel
    aurora 123` ≡ `ezpz cancel 123 --remote aurora`. The first positional
    that looks like a known host (`aurora`, `polaris`, `crux`, `sunspot`,
    `sophia`, `sirius`, `perlmutter`, `frontier`) is treated as `--remote`.
    Job IDs are numeric (`8936286` / `8936286.aurora-pbs-...`), host aliases
    are alphabetic, so there is no ambiguity. Custom SSH aliases still
    need `--remote my-alias`.


When `--remote` is omitted the command delegates to local `qstat` (or
`qstat -x` for historical). With `--remote` it prefers the ALCF IRI API
on Aurora / Polaris / Crux — structured JSON, no screen-scraping — and
falls back to `qstat` over SSH when IRI is unavailable or the backend is
forced to `ssh`.

## Backend selection

`--backend auto` (the default) resolves to `iri` on IRI-capable hosts
and `ssh` elsewhere — i.e. the faster path is tried first. Force a
backend with `--backend iri` or `--backend ssh` when you do not want
the automatic choice. See [Remote submission](../guides/remote-submission.md)
for the full fallback contract.

## Options

| Flag | Description |
|------|-------------|
| `JOB_ID` (positional) | A single job id. Without it, `status` lists recent jobs instead of showing one. |
| `--remote HOST` | Host to query (SSH name or ALCF resource like `aurora`, `polaris`). Omit for local `qstat`. |
| `--backend {auto,ssh,iri}` | Remote backend (default: `auto`, which picks `iri` on IRI hosts). |
| `--historical` / `--no-historical` | Include completed/failed/canceled jobs (IRI only; default `--no-historical` for list, single-job lookups query with `historical=true`). |
| `--limit N` | Max jobs when listing (IRI only, default: `20`) |
| `--queue NAME` | Filter listed jobs by queue/partition (IRI list only) |
| `--owner NAME` | Filter listed jobs by owner (IRI list only) |
| `--states LIST` | Filter by IRI states, comma-separated (e.g. `queued,active`). IRI states: `new`, `queued`, `held`, `active`, `completed`, `failed`, `canceled`. |
| `--json` | Emit JSON instead of a table (see [JSON output](#json-output)) |

## Examples

### List queued and running jobs

```bash
ezpz status --remote aurora --states queued,active
```

### List with queue / owner filters

```bash
ezpz status --remote aurora --queue debug --owner foremans --limit 50
```

### Include finished jobs (IRI history)

```bash
ezpz status --remote aurora --historical --limit 50
ezpz status --remote polaris --historical --states completed,failed
```

!!! warning "IRI history can be slow"
    `--historical` with no additional filter asks the API to scan the
    entire history. If the call times out you will see:

    ```text
    error: IRI timed out after 30s — try without --historical or filter by --queue/--states: ...
    ```

    Narrow the query with `--queue`, `--states`, or a smaller `--limit`.

### Machine-readable output

```bash
ezpz status --remote aurora --json > jobs.json
ezpz status 8879044 --remote aurora --json
```

## JSON output

`--json` changes the format without changing which jobs are queried.

* Single job: `{ "id", "state", "exit_code", "queue", "account" }`
* List: an array of `{ "id", "state", "exit_code", "queue" }`

When stdout is a TTY the table is rendered with [`rich`](https://github.com/Textualize/rich) and colour-coded by state; when piped the output is plain text.

## State colours

| State | Colour |
|-------|--------|
| `active`, `running` | green |
| `completed` | cyan |
| `queued` | yellow |
| `held` | magenta |
| `failed`, `canceled` | red |
| `new` | dim |

## Exit codes

- `0` — success.
- Non-zero — local `qstat` / remote API or SSH failure. On IRI errors the message includes the HTTP status (e.g. `IRI status 404: ...`).

## See also

- [`ezpz submit`](./submit.md) — queue jobs (locally or via `--remote`)
- [`ezpz cancel`](./cancel.md) — cancel a queued or running job
- [`ezpz extend`](./extend.md) — change walltime / queue / account of a queued job
- [Remote submission](../guides/remote-submission.md) — backends, auth, and the auto-fallback rule
