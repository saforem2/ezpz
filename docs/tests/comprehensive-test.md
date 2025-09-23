# Comprehensive Smoke Tests

## Overview

`tests/comprehensive_test.py` expands on the simple smoke tests by applying
systematic mocking so the `ezpz` modules can be imported and exercised without a
real scheduler or distributed environment. It is ideal for CI or local
validation when external services are unavailable.

## What it Covers

- Validates the main package import, version metadata, and logger availability.
- Confirms the configs, dist, and utils modules expose their critical helpers
  while scheduler detection is mocked to return `UNKNOWN`.
- Ensures environment variables such as `WANDB_MODE`, `EZPZ_LOG_LEVEL`, and
  standard rank variables are respected.

## Running the Test

```bash
python tests/comprehensive_test.py -v
```
