# Development Setup Guide

This guide will help you set up your development environment for ezpz.

## Prerequisites

- Python 3.8 or higher
- Git
- pip or conda for package management

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/saforem2/ezpz.git
   cd ezpz
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Using Hatch (Recommended)

The project uses Hatch for environment management. To set up using Hatch:

1. Install Hatch:
   ```bash
   pip install hatch
   ```

2. Create and enter the development environment:
   ```bash
   hatch shell
   ```

3. Run tests:
   ```bash
   hatch run test
   ```

## Code Formatting and Linting

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
```

### Pre-commit Hooks

To set up pre-commit hooks for automatic code formatting and linting:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install
```

Now, every time you commit, the hooks will automatically check and format your code.

## Testing

The project uses pytest for testing. To run tests:

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

## Documentation

Documentation is built using MkDocs with the Material theme.

To build and serve documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass: `hatch run test`
6. Format your code: `hatch run format`
7. Commit your changes: `git commit -am "Add your feature"`
8. Push to the branch: `git push origin feature/your-feature-name`
9. Create a pull request

## Release Process

1. Update the version in `src/ezpz/__about__.py`
2. Update `CHANGELOG.md`
3. Create a new git tag: `git tag -a v1.2.3 -m "Release version 1.2.3"`
4. Push the tag: `git push origin v1.2.3`
5. Create a release on GitHub