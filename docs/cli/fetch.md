# `ezpz fetch`

Read a job's stdout without opening an interactive SSH session. The supported
remote example below is a **PBS stdout snapshot**, not a live follow:

```bash
# Replace this historical job ID with your own submitted job.
ezpz fetch 12481065 --remote sunspot --lines 80 --no-dedupe
```

This prints the last 80 lines of the job's stdout file once. It does not wait
for new output or continuously refresh the terminal. Re-run it for another
snapshot. Output depends on the real job's log; a job ID alone does not imply
the job succeeded.

## How it finds output

1. Where available, query IRI job metadata for `stdout_path`.
2. Otherwise use PBS `qstat -xf JOB_ID` over SSH and parse `Output_Path`,
   including PBS continuation lines.
3. Read the discovered file using `tail` over SSH.

IRI supplies metadata, **not file access**. Even a job submitted over IRI needs
working noninteractive SSH authentication (`BatchMode=yes`) to fetch output.
The output file must exist and be readable on the target. Queued jobs may not
have produced any output, and historical metadata or spool files can expire.

## Options

| Option | Meaning |
|---|---|
| `JOB_ID` | Scheduler job ID; use the one returned by submission. |
| `--remote HOST` | Target SSH host alias; use the explicit option for fetch. |
| `--lines N`, `-n N` | Number of tail lines requested (default: 200). |
| `--dedupe` / `--no-dedupe` | Remote adjacent-line deduplication (default: on). |
| `--out out\|err\|both` | Exposed stream selector; currently incomplete across backends (see below). |

Remote deduplication applies `uniq` **after** taking the tail. It only removes
adjacent identical lines; it does not remove repeated blocks or identify MPI
ranks. The result can contain fewer than `--lines N` lines. Use `--no-dedupe`
when repetitions matter for diagnosis.

## Current limits

- The fallback is PBS `qstat`, not Slurm `scontrol` output-path discovery.
  Do not assume this command can find a Perlmutter or Frontier stdout file.
- `--out err` and `--out both` are not implemented consistently: the SSH PBS
  fallback currently resolves stdout only. The examples intentionally use the
  default stdout stream. PBS jobs generated with `#PBS -j oe` merge stderr into
  stdout, but that is scheduler behavior, not proof of separate-stream support.
- Without `--remote`, local discovery uses PBS metadata or the local working
  directory and its `.OU` files; this is not generic scheduler-independent
  output discovery.
- `ezpz logs` exists for per-example benchmark logs, but its current workdir
  resolver contains site/job-specific assumptions. Do not treat it as a portable
  replacement for fetch without checking the resolved path.

## Submit, inspect, fetch

First inspect a generated command. Replace the account and all shared paths
with real target-side values; pre-create the workdir and install ezpz into the
environment. The setup script is your own validated site initialization:

```bash
ezpz submit sunspot --dry-run -N 2 -q workq -A ACCOUNT \
    --workdir /shared/PROJECT/USER/runs/example \
    --env-setup-script /shared/PROJECT/USER/site-setup.sh \
    --env /shared/PROJECT/USER/venvs/train \
    -- python3 -m ezpz.examples.test --model debug
```

Remove `--dry-run` only after reviewing the script. With the job ID returned
by that real submission:

```bash
# Replace JOB_ID; these commands do not submit another job.
ezpz status JOB_ID --remote sunspot
ezpz fetch JOB_ID --remote sunspot --lines 80 --no-dedupe
```

See [remote submission](../guides/remote-submission.md) for environment ordering,
job-unique working directories, and IRI/SSH authentication.
