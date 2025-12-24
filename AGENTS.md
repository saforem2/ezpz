# AGENTS.md

## Dev environment tips

- Use `source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env` to
  setup the development environment.
- `ezpz doctor` can be used to verify the functionality of the environment
- `ezpz test` can be used as a distributed PyTorch smoke test to verify:

  1. `ezpz` installed correctly
  2. functionality of distributed PyTorch with MPI

## Repo Management

- Use `uv` for dependency management, falling back to `pip` ONLY when `uv`
  breaks (or gives unexpected behavior)

## Development dependency

- Use `ruff` linting/formatting
- Use `ty` for type checking
- Use `pytest` for testing

