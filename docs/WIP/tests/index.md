# Tests (WIP Overview)

Purpose: describe test entrypoints and how to run them with `uv`.

## Running
```bash
uv run pytest tests/
uv run pytest tests/ -k dist        # filter
uv run pytest tests/ --cov=src/ezpz --cov=tests
```

## Focus Areas
- `tests/test_dist.py`: distributed setup and launch paths.
- `tests/test_launch.py`: CLI parsing, scheduler detection, command assembly.
- `tests/test_jobs.py`: scheduler-agnostic job discovery.
- `tests/test_utils*.py`: env helpers, tar/yeet utilities.
- `tests/test_history.py`: logging, plotting, wandb offline behavior.

## Tips
- Set `EZPZ_LOG_LEVEL=DEBUG` to inspect launch building during tests.
- Use `WORLD_SIZE=1` for deterministic single-rank runs.
- For torch backends on CI, prefer CPU (`--device cpu`) to avoid GPU deps.
