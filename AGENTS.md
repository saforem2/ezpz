# AGENTS.md

## Dev environment tips

- Use `source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env` to
  setup the development environment.
- `ezpz-test` can be used to verify functionality of distributed environment
  when running from a compute node.

## Repo Management

- Use `uv` for dependency management, falling back to `pip` ONLY when `uv`
  breaks (or gives unexpected behavior)

## Development dependency

- Use `ruff` linting/formatting
- Use `mypy` for type checking
- Use `pytest` for testing

