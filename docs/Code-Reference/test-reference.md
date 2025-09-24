# `ezpz.test`

Entry-point helpers for the distributed smoke test fallback logic.

::: ezpz.test

## Usage Examples

### Run the Smoke Test via mpirun Fallback

```python
import subprocess

subprocess.run([
    "python3",
    "-m",
    "ezpz.test",
])
```
