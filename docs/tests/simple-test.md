# Simple Test Harness

## Overview

`tests/simple_test.py` provides a bare-bones smoke test that verifies the
package imports cleanly and exposes a version number. It is designed to run in
minimal environments where the full test suite might be overkill.

## What it Covers

- **Import sanity** – Confirms `import ezpz` succeeds when `src/` is added to the
  path.
- **Version metadata** – Ensures `ezpz.__version__` exists and is non-empty.

## Running the Test

```bash
python tests/simple_test.py
```
