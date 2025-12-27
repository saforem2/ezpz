# Direct Module Smoke Tests

## Overview

`tests/test_modules_directly.py` provides fast smoke coverage for key modules
without importing the entire `ezpz` package. By manipulating `sys.path` and
importing modules individually, the tests ensure that these modules can be
loaded in isolation—a common requirement for downstream integrations.

## Highlights

- **Configs smoke check** – Confirms the configs module exposes path constants
  and helper functions even when imported directly.
- **Utils smoke check** – Exercises timestamp generation, formatting helpers,
  normalisation logic, and tensor grabbing to guarantee top-level helpers work
  without the package initialiser.
- **Lazy import smoke check** – Validates that `ezpz.lazy.lazy_import` returns a
  module proxy with expected attributes.

## Running the Tests

```bash
python -m pytest tests/test_modules_directly.py
```
