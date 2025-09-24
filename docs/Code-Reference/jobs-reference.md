# `ezpz.jobs`

Helpers for persisting and replaying scheduler job environments.

::: ezpz.jobs

## Usage Examples

### Capture the Current Job Environment

```python
import ezpz.jobs as jobs

# Persist metadata to ~/.local/bin/launch.sh and jobenv.{sh,json,yaml}
jobs.savejobenv(verbose=True)
```

### Load the Most Recent Job Environment

```python
import ezpz.jobs as jobs

jobenv = jobs.loadjobenv()
print(jobenv["PBS_JOBID"])
```
