# Property-Based Utility Tests

## Overview

`tests/test_property_based.py` uses [Hypothesis](https://hypothesis.readthedocs.io)
for property-based validation of the utility helpers in `ezpz.utils`. The suite
runs only when both `ezpz.utils` and Hypothesis are available and skips
otherwise, making it safe for constrained environments.

## Highlights

- **Idempotent normalisation** – Repeated calls to `utils.normalize` must return
  the same value for any input string.
- **Lowercase guarantee** – Normalised identifiers are forced to lowercase while
  preserving valid characters.
- **Identifier validity** – Ensures the output contains only alphanumeric
  characters or dashes.
- **Formatted pairs** – Confirms `utils.format_pair` produces stable string
  representations for floats with the requested precision.

## Running the Tests

```bash
python -m pytest tests/test_property_based.py
```

> **Tip:** Install Hypothesis via `pip install hypothesis` if it is not already
> present in your development environment.
