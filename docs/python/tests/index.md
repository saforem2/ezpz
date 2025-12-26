# Test Suite Documentation

The ezpz test suite provides comprehensive unit tests for all modules in the package. This documentation explains the structure, purpose, and usage of each test file.

## Test Organization

The test suite is organized into individual files, each focusing on a specific module or aspect of the ezpz package:

- [Main Module Tests](test-ezpz.md) - Tests for the core ezpz package
- [Configs Module Tests](test-configs.md) - Tests for configuration management
- [Improved Configs Tests](test-configs-improved.md) - Additional mocked coverage for configs
- [Distributed Computing Tests](test-dist.md) - Tests for distributed training functionality
- [History Module Tests](test-history.md) - Tests for metrics tracking and history management
- [Jobs Module Tests](test-jobs.md) - Tests for job management and scheduling
- [Logging Module Tests](test-log.md) - Tests for logging functionality
- [Profiling Module Tests](test-profile.md) - Tests for performance profiling
- [Tensor Parallel Module Tests](test-tp.md) - Tests for tensor parallel computing
- [Utilities Module Tests](test-utils.md) - Tests for utility functions
- [Enhanced Utilities Tests](test-utils-improved.md) - Mock-heavy tests for utility helpers
- [Launch Module Tests](test-launch.md) - Tests for the launch functionality
- [Terminal Plotting Tests](test-tplot.md) - Tests for terminal-based plotting
- [Lazy Import Tests](test-lazy.md) - Tests for lazy loading functionality
- [Modules-in-Isolation Tests](test-modules-directly.md) - Smoke tests importing modules without `ezpz.__init__`
- [Property-Based Utility Tests](test-property-based.md) - Hypothesis-driven validation for utility helpers
- [Simple ezpz Smoke Tests](test-simple-ezpz.md) - Minimal import/version checks
- [Simple Test Harness](simple-test.md) - Standalone script for import validation
- [Comprehensive Smoke Tests](comprehensive-test.md) - Broad mocked environment checks
- [Simple Test Runner](simple-test-runner.md) - Scripted entry point for quick smoke testing

## Running Tests

To run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_ezpz.py

# Run tests with verbose output
python -m pytest tests/ -v

# Run tests and show coverage
python -m pytest tests/ --cov=src/ezpz
```

## Test Structure

Each test file follows the standard pytest structure:

- Test functions start with `test_`
- Test classes start with `Test`
- Fixtures are used for setup/teardown
- Mocking is used to isolate units under test

## Writing New Tests

When adding new tests, follow these guidelines:

1. Create a new test file following the naming convention `test_*.py`
2. Import the module you want to test
3. Write test functions that verify specific behavior
4. Use pytest fixtures for setup/teardown
5. Use mocking to isolate units under test
6. Add appropriate markers (e.g., `@pytest.mark.slow` for slow tests)
