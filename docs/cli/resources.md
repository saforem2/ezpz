# `ezpz resources`

List IRI compute resources — the machines the ALCF IRI API can see
(Aurora, Polaris, Crux, …). Uses `GET /status/resources` by default and
understands both response shapes (`list` and `{ resources: [...] }`).

```bash
ezpz resources
ezpz resources --json | jq .
```

The command needs the same `alcf-tokens` / `ALCF_IRI_TOKEN` credential
as the other IRI commands (it hits the same `https://api.alcf.anl.gov`
host), so authenticate once:

```bash
uvx alcf-tokens login iri
# or: uv tool install alcf-tokens && alcf-tokens login iri
# or: export ALCF_IRI_TOKEN=<token>
```

## Options

| Flag | Description |
|------|-------------|
| `--json` | Emit JSON instead of a table |
| `--all` / `--no-all` | `GET /status/resources` (all, default) vs `GET /compute/resources` (compute-only, id `501`-style response) |
| `--limit N` | Max resources to request (default: `100`) |

!!! tip "`--all` vs `--no-all`"
    `--all` (default) hits `/status/resources`, which returns a broader
    inventory including resource health/status. `--no-all` hits
    `/compute/resources`, a narrower endpoint some deployments expose as
    `501 Not Implemented` — use `--all` unless you have a reason not to.

## JSON output

`--json` emits an array of `{ "name", "id", "status" }`. The raw API
objects are included only when `ALCF_IRI_JSON_RAW` is set in the
environment (debug aid).

## Examples

### Pretty table (TTY)

```bash
ezpz resources
```

When stdout is a TTY a colour-coded [`rich`](https://github.com/Textualize/rich) table is used (`up`/`active` green, `degraded` yellow, `down` red).
Full resource ids are printed dimmed below the table for copy-paste.

### Piped / machine-readable

```bash
ezpz resources | cat          # plain text, no rich
# RESOURCE       ID                                       STATUS
# aurora         9c9b...e7a                                up
# polaris        55c1c993-1124-47f9-b823-514ba3849a9a       up

ezpz resources --json | jq '.[] | "\(.name) \(.status)"'
ezpz resources --no-all --json | jq .     # compute-only endpoint
```

### Find the id for a host (for incident filtering)

```bash
ezpz resources --json | jq -r '.[] | "\(.name)\t\(.id)"'
# aurora    9c9b3b2d-...
# polaris   55c1c993-1124-47f9-b823-514ba3849a9a
```

## How Polaris is special

Most callers never need to know resource ids — `ezpz submit --remote`
and `ezpz status --remote` resolve them at runtime via this same
endpoint. Polaris's id (`55c1c993-1124-47f9-b823-514ba3849a9a`) is the
only one published in the ALCF docs; every other id is discovered live.
Hard-coding the wrong UUID would submit to the wrong machine.

## See also

- [`ezpz incidents`](./incidents.md) — maintenance / outage events per resource
- [`ezpz status`](./status.md) — jobs on a resource
- [Remote submission](../guides/remote-submission.md) — IRI auth and backend choice
