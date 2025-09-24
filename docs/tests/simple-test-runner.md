# Simple Test Runner

## Overview

`tests/simple_test_runner.py` is a convenience script that seeds the environment
with safe defaults and invokes pytest against a minimal subset of tests. It is
useful for validating installations where the full suite may require additional
infrastructure.

## Behaviour

- Sets environment variables (`WANDB_MODE`, `EZPZ_LOG_LEVEL`, `RANK`,
  `WORLD_SIZE`, `LOCAL_RANK`) to benign defaults.
- Injects the repository `src/` directory onto `sys.path`.
- Executes `pytest -v tests/test_ezpz.py`.

## Running the Script

```bash
python tests/simple_test_runner.py
```
