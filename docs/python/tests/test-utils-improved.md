# Enhanced Utils Tests

## Overview

`tests/test_utils_improved.py` extends the utilities test coverage by mocking
scheduler detection and hostname lookups, ensuring the helpers behave in stripped
-down environments and when optional dependencies are missing.

## Highlights

- **Timestamp utilities** – Verifies both default and custom-format timestamp
  generation.
- **String normalisation** – Confirms that varying separator and case inputs are
  translated into consistent, kebab-case identifiers.
- **Key-value formatting** – Ensures integers, booleans, and floats (with custom
  precision) are rendered consistently by `utils.format_pair`.

## Running the Tests

```bash
python -m pytest tests/test_utils_improved.py
```
