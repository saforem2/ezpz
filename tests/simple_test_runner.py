#!/usr/bin/env python3
"""Simple test runner that sets up the environment properly."""

import os
import sys
from pathlib import Path

# Set up environment variables to avoid initialization issues
os.environ["WANDB_MODE"] = "disabled"
os.environ["EZPZ_LOG_LEVEL"] = "ERROR"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if __name__ == "__main__":
    import pytest

    # Run tests with minimal configuration
    sys.exit(pytest.main(["-v", str(Path(__file__).parent / "test_ezpz.py")]))
