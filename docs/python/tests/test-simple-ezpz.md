# Simple ezpz Smoke Tests

## Overview

`tests/test_simple_ezpz.py` acts as a zero-dependency smoke test ensuring the
package can be imported and its most common helpers are available even in a
minimal environment.

## Highlights

- **Import safety** – Guards against accidental regressions that break
  `import ezpz` by skipping when import errors are environment-related.
- **Version exposure** – Checks that `ezpz.__version__` remains a non-empty
  string.
- **Logger bootstrap** – Validates that `ezpz.get_logger` returns a logger with
  the expected logging methods.
- **Environment defaults** – Confirms that important environment variables set
  during initialisation (`WANDB_MODE`, `EZPZ_LOG_LEVEL`) are present.

## Running the Tests

```bash
python -m pytest tests/test_simple_ezpz.py
```
