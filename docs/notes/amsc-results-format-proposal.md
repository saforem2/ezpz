---
title: "AmSC results-format proposal (draft, unsent)"
---

!!! note "Draft — not filed"

    Staged for review before sending to
    [amsc-atscale-benchmarks](https://gitlab.com/amsc2/ai-services/at-scale-services/amsc-atscale-benchmarks).
    Filing an issue on a shared team repo is a call for a person to
    make, not an agent. To send it:

    ```bash
    glab issue create -R amsc2/ai-services/at-scale-services/amsc-atscale-benchmarks \
        -t "Proposal: a common results format" \
        -F docs/notes/amsc-results-format-proposal.md
    ```

    Verified against `origin/main` on 2026-08-25.

# Proposal: a common results format

## The problem

There is currently no specification for what a benchmark should publish, and the two benchmarks that publish results disagree with each other:

| benchmark | path | columns |
|---|---|---|
| `training/vit-weather` | `results/Perlmutter/throughput_metrics.csv` | `configuration_name, samples_per_sec, iters_per_sec` |
| `workflow/mldocking` | `results/summary_<system>.txt` | free-form text |
| `training/llm-finetuning` | — | no `results/` at all |

`results/README.md` asks for *"structured subfolders with provenance metadata when possible"*, which is the entire spec. There is no `benchmark.yaml`, no schema, and no CONTRIBUTING.

The practical consequence: a contributor with a finished run cannot tell what to write, and `analysis/` cannot aggregate across benchmarks because no two agree on units, column names, or even file format. "Cross-facility comparison" is the stated goal in the README, and the current layout does not support it.

## Concrete proposal

A per-system CSV, keeping `vit-weather`'s directory convention (which already works and needs no migration):

```
benchmarks/<category>/<name>/results/<System>/runs.csv
```

with a fixed core, one row per run:

| column | meaning |
|---|---|
| `timestamp` | ISO 8601 UTC, run start |
| `system` | `Perlmutter`, `Aurora`, `SunSpot`, `Polaris`, … |
| `config` | free-form config id, e.g. `agpt-2b/bs1/seq2048/tp1` |
| `nodes` | nodes actually used |
| `devices` | accelerators actually used — see note |
| `status` | `pass` / `fail` |
| `wall_time_sec` | instrumented step time |

plus any benchmark-specific metrics as extra columns, empty meaning *not measured* rather than zero. Aggregation keys on the core; extra columns pass through.

Two details that came out of measuring this in practice:

**`devices`, not `gpus`.** Aurora and Sunspot expose 12 PVC *tiles* per node (2 per GPU, 6 GPUs). Calling that column `gpus` makes a per-device ratio read as card-to-card and silently overstates one side of any comparison by 2x.

**Report what ran, not what was allocated.** A job can be allocated 4 nodes and train on 2. Recording the allocation makes throughput-per-device wrong.

## Prior art

This is not a new invention — these columns are what the dashboard's `build_dashboard.py` required, back when that consumer was reachable. `ezpz export-amsc` has been emitting them since. Note that script is not in this repo's history (it lives in the dashboard project), so if the dashboard is still the intended consumer, its requirements should probably be restated here rather than living only there.

## Why now

We have Perlmutter, Sunspot and Aurora results for the same workload and cannot publish them in a way anything can aggregate. Happy to send an MR with:

- `results/SCHEMA.md` documenting the above,
- a `benchmark.yaml` example,
- `vit-weather`'s existing CSV mapped onto it (its three columns fit as benchmark-specific extras),
- our own rows in the new form.

Equally happy to adopt a different shape — the ask is that there **be** one. If the intent is that each benchmark defines its own format, saying that explicitly in `results/README.md` would also resolve it, and we would match `vit-weather` per-benchmark instead.
