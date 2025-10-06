# Development Workflow

This document outlines the recommended development workflow for the ezpz project.

## Overview

The ezpz project uses a modern Python development workflow with automated testing, 
linting, formatting, and continuous integration.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, or hatch)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saforem2/ezpz.git
   cd ezpz
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Using Hatch (Recommended)

The project is configured to use Hatch for environment management:

1. **Install Hatch**:
   ```bash
   pip install hatch
   ```

2. **Enter development environment**:
   ```bash
   hatch shell
   ```

3. **Run tests**:
   ```bash
   hatch run test
   ```

## Code Quality Tools

### Formatting and Linting

The project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter
- **Black**: Code formatter
- **isort**: Import sorter
- **mypy**: Static type checker

### Running Code Quality Tools

```bash
# Run all linters
hatch run lint

# Run formatting
hatch run format

# Run type checking
hatch run typecheck

# Run security scanning
hatch run security
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality tools before each commit:

1. **Install pre-commit**:
   ```bash
   pip install pre-commit
   ```

2. **Install git hooks**:
   ```bash
   pre-commit install
   ```

The hooks will automatically check and format your code on each commit.

## Testing

### Running Tests

```bash
# Run all tests
hatch run test

# Run tests with coverage
hatch run cov

# Run specific test file
hatch run pytest tests/test_specific.py

# Run tests in verbose mode
hatch run pytest -v
```

### Test Structure

Tests are organized in the `tests/` directory with the following structure:

- `test_basic.py` - Smoke tests to verify the test framework works
- `test_unit_isolated.py` - Isolated unit tests that don't require full imports
- `test_*_improved.py` - Improved tests with proper mocking
- `test_integration.py` - Integration tests for core functionality

### Writing Tests

Follow these guidelines when writing new tests:

1. **Use descriptive test names** that clearly describe what is being tested
2. **Test one thing per test** - Each test should focus on a single behavior
3. **Use fixtures for setup/teardown** - Avoid code duplication in test setup
4. **Mock external dependencies** - Isolate the code under test
5. **Test edge cases** - Include tests for boundary conditions and error cases

### Test Categories

- **Unit Tests**: Focused tests for individual functions and classes
- **Integration Tests**: Tests that verify interaction between components
- **Property-Based Tests**: Tests that use hypothesis to verify properties with random inputs
- **Mock-Based Tests**: Tests that use mocking to avoid external dependencies

## Continuous Integration

### GitHub Actions Workflow

The project uses GitHub Actions for continuous integration with the following jobs:

1. **Test**: Runs tests on multiple Python versions
2. **Lint**: Runs code quality checks (ruff, black, isort)
3. **Security**: Runs security scanning with bandit
4. **Docs**: Builds documentation
5. **Deploy**: Deploys documentation to GitHub Pages

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and add tests

3. **Run tests locally**:
   ```bash
   hatch run test
   ```

4. **Format your code**:
   ```bash
   hatch run format
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

6. **Push to GitHub**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request**

### Code Review Process

All pull requests must be reviewed before merging:

1. **At least one approval** from a maintainer
2. **All CI checks must pass**
3. **Code quality standards must be met**
4. **Tests must pass and provide adequate coverage**

## Release Process

### Versioning

The project follows Semantic Versioning (SemVer):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

### Creating a Release

1. **Update version** in `src/ezpz/__about__.py`

2. **Update CHANGELOG.md** with release notes

3. **Create and push tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

4. **Create GitHub release** with release notes

## Best Practices

### Code Organization

1. **Follow PEP 8** for code style
2. **Use type hints** for all function signatures
3. **Write docstrings** for all public functions and classes
4. **Keep functions small** and focused on a single responsibility
5. **Use meaningful variable names**

### Testing Best Practices

1. **Write tests first** when possible (TDD)
2. **Test both positive and negative cases**
3. **Use appropriate assertions** for the job
4. **Keep tests fast** - avoid slow operations in unit tests
5. **Mock external dependencies** to isolate units under test

### Documentation Best Practices

1. **Keep README.md updated** with current usage examples
2. **Document all public APIs** with docstrings
3. **Include examples** in docstrings when helpful
4. **Update CHANGELOG.md** with all notable changes
5. **Write clear commit messages** that explain the "why"

### Git Best Practices

1. **Make small, focused commits** that address a single issue
2. **Write clear commit messages** following conventional commits
3. **Rebase feature branches** to keep history clean
4. **Use meaningful branch names** like `feature/add-new-functionality`
5. **Delete merged branches** to keep the repository clean