# `ezpz incidents`

List IRI incidents — maintenance windows, outages, and other
facility events. Backed by `GET /status/incidents`.

```bash
ezpz incidents
ezpz incidents --resource aurora
ezpz incidents --status down --limit 50
```

Incidents are **public / unauthenticated**: the command works with or
without an IRI token. If a token is available (via `alcf-tokens` or
`ALCF_IRI_TOKEN`) it is sent, otherwise the request is made anonymously.

## Options

| Flag | Description |
|------|-------------|
| `--resource NAME` | Filter to a resource (name like `aurora`/`polaris` or a UUID). |
| `--status {up,down,degraded,unknown}` | Filter by resource status at incident time |
| `--json` | Emit raw JSON instead of a table |
| `--limit N` | Max incidents to request (default: `20`) |

When `--resource` is a friendly name (no `-` in it), ezpz tries to match
it against the `resource_id` of returned incidents. UUIDs are passed
through as-is to the API's `resource_id` query parameter.

## Examples

### All recent incidents

```bash
ezpz incidents
```

On a TTY a colour-coded [`rich`](https://github.com/Textualize/rich) table is
shown (`down` red, `degraded` yellow, `up` green):

```text
Name                     Status    Type          Time
Aurora maintenance       down      maintenance   2026-04-10 02:00
Polaris drain            degraded  outage        2026-04-09 18:30
```

Piped/plain output is space-aligned with a header line:

```bash
ezpz incidents | cat
# NAME                                     STATUS     TYPE         TIME
# Aurora maintenance                       down       maintenance  2026-04-10 02:00
```

### Filter to one resource

```bash
ezpz incidents --resource aurora
ezpz incidents --resource 55c1c993-1124-47f9-b823-514ba3849a9a   # by UUID
```

!!! tip "Find a resource's UUID"
    ```bash
    ezpz resources --json | jq -r '.[] | "\(.name)\t\(.id)"'
    ```

### Only outages

```bash
ezpz incidents --status down
ezpz incidents --status degraded --limit 100
```

### Machine-readable

```bash
ezpz incidents --json | jq .
ezpz incidents --resource aurora --json | jq 'length'
```

### Check before submitting

A quick pre-flight before a large allocation:

```bash
ezpz incidents --resource aurora --status down | head
# if empty, no blocking incidents are recorded
```

## See also

- [`ezpz resources`](./resources.md) — discover resource names and ids
- [`ezpz status`](./status.md) — jobs on a resource
- [Remote submission](../guides/remote-submission.md) — submitting to that resource
