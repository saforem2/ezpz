# `ezpz.jobs`

- See [ezpz/`jobs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/jobs.py)

Job metadata persistence for batch schedulers. Tracks job environments, saves
them to disk, and provides a history log of past jobs.

## Overview

When running under a batch scheduler (PBS), `ezpz` saves job metadata to
`~/PBS-jobs/<jobid>/`. This includes environment variables, hostfiles, and a
replay-able launch script.

## Saving and Loading Job Environments

### `savejobenv()`

Collects and persists the current job's environment to disk. Typically called
early in a training script or by the launcher:

```python
from ezpz.jobs import savejobenv

# Basic usage — auto-detects everything
savejobenv()

# With explicit hostfile and verbose output
savejobenv(hostfile="/path/to/hostfile", verbose=True)
```

Optional parameters: `verbose`, `hostfile`, `print_jobenv`,
`verbose_dotenv`, `verbose_get_jobenv`, `verbose_dist_info`,
`verbose_pbs_env`.

This creates a job directory under `~/PBS-jobs/<jobid>/` containing:

- `.jobenv` — key-value file of scheduler environment variables
- `launch.sh` — a shell script to replay the launch command
- Entry in the global jobs log

### `loadjobenv()`

Restore a previously saved job environment:

```python
from ezpz.jobs import loadjobenv

# Load the most recent job
env = loadjobenv()

# Load a specific job directory
env = loadjobenv(jobdir="/home/user/PBS-jobs/12345.pbs-server/")
```

Returns a dictionary of environment variable key-value pairs.

## Launch Script Generation

### `write_launch_shell_script()`

Generates a shell script at `~/.local/bin/launch.sh` that can be used to
re-run the current job's launch command:

```python
from ezpz.jobs import write_launch_shell_script

write_launch_shell_script()
```

## Dotenv Export

### `save_to_dotenv_file()`

Write the job environment to a `.jobenv` file in dotenv format:

```python
from ezpz.jobs import save_to_dotenv_file

path = save_to_dotenv_file(jobenv, hostfile="/path/to/hostfile")
```

## Job History

### `add_to_jobslog()` / `get_jobdirs_from_jobslog()`

Maintain a log of all job directories for easy recall:

```python
from ezpz.jobs import add_to_jobslog, get_jobdirs_from_jobslog

# Record the current job
add_to_jobslog()

# List all past job directories
job_dirs = get_jobdirs_from_jobslog()
for d in job_dirs:
    print(d)
```

::: ezpz.jobs
