# Repository Guidelines

## Project Structure & Module Organization
- Core package under `src/ezpz/` with launchers, distributed utilities, and CLI entry points; add new modules within matching subpackages such as `src/ezpz/log/`.
- Tests live in `tests/` mirroring package layout with `test_*.py` files and shared fixtures in `tests/conftest.py`.
- Docs are authored in `docs/` with MkDocs configuration in `mkdocs.yml`; avoid duplicating content between README and docs.
- Static figures reside in `assets/`; use lightweight formats and reference them from docs or README as needed.

## Build, Test, and Development Commands
- `python -m pip install -e .[dev]` installs the package editable with linting and test tooling.
- `hatch run test` runs the full pytest suite configured in `pyproject.toml`.
- `hatch run cov` executes pytest with coverage targeting `src/ezpz` and `tests`.
- `pytest tests/test_launch.py -k smoke` focuses on launch smoke tests during iteration.
- `mkdocs serve` builds documentation locally at http://127.0.0.1:8000.

## Coding Style & Naming Conventions
- Follow Ruff style: 4-space indentation, 79-character lines, and double-quoted strings.
- Keep modules and functions snake_case, classes PascalCase, and constants UPPER_SNAKE_CASE.
- Run `ruff check src tests --fix` to format and lint before pushing; keep imports sorted per isort defaults.
- Include type hints on public APIs and prefer explicit imports over globals.

## Testing Guidelines
- Use pytest for all tests; co-locate new cases under `tests/test_<feature>.py` mirroring source layout.
- Name tests descriptively (e.g., `test_launch_handles_missing_hostfile`) and parametrize matrix scenarios.
- Maintain stable coverage; inspect gaps with `hatch run cov` and add regression tests when fixing bugs.

## Commit & Pull Request Guidelines
- Write Conventional Commits (`feat:`, `fix:`, `chore:`) with concise imperative summaries scoped to one logical change.
- Reference related issues and note behavioural shifts or migrations in commit bodies or PR descriptions.
- Ensure CI-critical commands (`hatch run test`, `ruff check src tests --fix`) pass locally before requesting review; attach relevant logs or screenshots for UX or logging changes.
- Summarize user-facing impact and list verification steps in the PR template, including any documentation updates required.
