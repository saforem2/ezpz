# `ezpz.launch`

Execution helpers for launching distributed jobs.

::: ezpz.launch

## Usage Examples

### Launch from Python

```python
import ezpz.launch as launch

# Dispatch a command using the scheduler-aware logic
launch.launch(cmd_to_launch="python3 -m ezpz.test_dist", include_python=False)
```

### CLI Fallback Behaviour

```python
import subprocess

# When no scheduler is detected ezpz-launch falls back to mpirun
subprocess.run([
    "ezpz-launch",
    "python3",
    "-m",
    "ezpz.test_dist",
])
```
