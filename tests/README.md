# ezpz Test Suite

This directory contains unit tests for the ezpz package.

## Overview

The tests are organized to cover the main modules of the ezpz package:

- `test_basic.py` - Simple tests to verify the test framework works
- `test_ezpz.py` - Tests for the main ezpz module
- `test_configs.py` - Tests for the configs module
- `test_dist.py` - Tests for the distributed computing module
- `test_history.py` - Tests for the history/metrics tracking module
- `test_jobs.py` - Tests for the job management module
- `test_log.py` - Tests for the logging module
- `test_profile.py` - Tests for the profiling module
- `test_tp.py` - Tests for the tensor parallel module
- `test_utils.py` - Tests for the utilities module
- `test_launch.py` - Tests for the launch module
- `test_tplot.py` - Tests for the terminal plotting module
- `test_lazy.py` - Tests for the lazy import module
- `test_configs_improved.py` - Improved tests for the configs module with proper mocking
- `test_utils_improved.py` - Improved tests for the utils module with proper mocking
- `test_property_based.py` - Property-based tests using hypothesis
- `test_integration.py` - Integration tests for core functionality
- `test_config.py` - Test configuration and fixtures

## Running Tests

### Using pytest (recommended)

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

### Using the virtual environment

First, activate the virtual environment:

```bash
source venv/bin/activate
```

Then run the tests as shown above.

## Environment Setup

The tests require several dependencies. To set up the environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install pytest pytest-cov pytest-mock numpy torch mpi4py pyyaml xarray rich sh transformers wandb plotext ambivalent torchinfo ipython omegaconf evaluate tiktoken hypothesis
```

## Test Structure

Each test file follows the standard pytest structure:

- Test functions start with `test_`
- Test classes start with `Test`
- Fixtures are used for setup/teardown
- Mocking is used to isolate units under test

## Running Tests with Coverage

```bash
# Run tests with coverage report
python -m pytest tests/ --cov=src/ezpz

# Run tests with detailed coverage report
python -m pytest tests/ --cov=src/ezpz --cov-report=html

# Run tests with coverage and show missing lines
python -m pytest tests/ --cov=src/ezpz --cov-report=term-missing
```

## Test Categories

### Unit Tests
Focused tests for individual functions and classes with proper mocking to isolate units under test.

### Integration Tests
Tests that verify the interaction between different modules and components.

### Property-Based Tests
Tests that use hypothesis to verify properties of functions with random inputs.

### Mock-Based Tests
Tests that use mocking to avoid dependencies on external systems or complex setup.

## Writing New Tests

### Basic Test Structure

```python
import pytest

def test_example_function():
    """Test example_function with valid input."""
    from ezpz.module import example_function

    result = example_function("test")
    assert result == "expected_output"

def test_example_function_with_invalid_input():
    """Test example_function with invalid input."""
    from ezpz.module import example_function

    with pytest.raises(ValueError):
        example_function(None)
```

### Using Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using a fixture."""
    assert sample_data["key"] == "value"
```

### Mocking External Dependencies

```python
import pytest
from unittest.mock import patch, MagicMock

def test_with_mocking():
    """Test with mocked external dependency."""
    with patch('ezpz.module.external_function', return_value='mocked'):
        from ezpz.module import function_that_uses_external

        result = function_that_uses_external()
        assert result == 'mocked'
```

### Property-Based Testing

```python
import pytest
from hypothesis import given, strategies as st

@given(st.integers())
def test_property_based(value):
    """Test property with random integers."""
    result = some_function(value)
    assert result >= 0  # Example property
```

## Test Markers

- `@pytest.mark.slow` - For tests that take a long time to run
- `@pytest.mark.integration` - For integration tests
- `@pytest.mark.skipif(condition, reason="...")` - To skip tests under certain conditions

## Best Practices

1. **Use descriptive test names** - Test names should clearly describe what is being tested
2. **Test one thing per test** - Each test should focus on a single behavior
3. **Use fixtures for setup/teardown** - Avoid code duplication in test setup
4. **Mock external dependencies** - Isolate the code under test
5. **Test edge cases** - Include tests for boundary conditions and error cases
6. **Use appropriate assertions** - Choose the right assertion for the job
7. **Keep tests fast** - Avoid slow operations in unit tests
8. **Test both positive and negative cases** - Verify correct behavior and error handling

## Troubleshooting

### ImportError issues

If you encounter import errors, make sure:

1. The virtual environment is activated
2. All dependencies are installed
3. The `src` directory is in the Python path

### Environment-specific issues

Some tests may fail due to environment-specific initialization in ezpz. In such cases:

1. Use mocking to bypass problematic initialization
2. Set appropriate environment variables
3. Create isolated test environments

## Continuous Integration

Tests are automatically run in the CI pipeline on every push to the repository. The pipeline includes:

- Unit tests
- Integration tests
- Code coverage analysis
- Security scanning

## Coverage Goals

- **Unit Test Coverage**: 80%+ for core modules
- **Integration Test Coverage**: 70%+ for workflow functions
- **Overall Project Coverage**: 60%+ minimum threshold

Coverage reports are generated automatically and can be viewed in the CI pipeline.

## Coverage Configuration

Coverage is configured using `.coveragerc` file with the following settings:

- Branch coverage enabled
- Source directory set to `src/`
- Common files excluded from coverage (e.g., `__about__.py`)
- HTML reports generated in `htmlcov` directory
