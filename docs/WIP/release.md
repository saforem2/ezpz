# Release & Deploy (WIP)

Process checklist for cutting releases and publishing docs.

## Versioning
- Bump version in `src/ezpz/__about__.py`.
- Update `CHANGELOG.md` with features/fixes and date.
- Keep extras matrix (cpu/cu128/xpu) consistent with tested torch versions.

## Tests & Quality
```bash
uv run pytest tests/
uv run pytest tests/ --cov=src/ezpz --cov=tests
uv run ruff check .
uv run mypy src/ezpz
```

## Build & Publish (package)
- Build sdist/wheel:
  ```bash
  uv build
  ```
- Publish (if desired):
  ```bash
  uv publish   # requires credentials
  ```

## Docs
- Build locally:
  ```bash
  uv pip install ".[docs]"
  uv run mkdocs build
  ```
- Deploy (GitHub Pages):
  ```bash
  uv run mkdocs gh-deploy --force
  ```
  (Ensure `site_url` in `mkdocs.yml` is correct.)

## Support Matrix (record)
- Python: 3.10–3.12 (tested via CI).
- Torch extras:
  - `cpu`: CPU-only torch.
  - `cu128`: CUDA 12.8 torch/vision.
  - `xpu`: Intel XPU torch/vision (oneAPI).
- Launchers: `mpiexec` (PBS), `srun` (SLURM), `mpirun` (local fallback).

## Pre-release Sanity
- Run `ezpz-test` locally (`WORLD_SIZE=1`, CPU).
- Run `ezpz-launch -m ezpz.test_dist --train-iters 10` in a scheduler allocation (PBS/SLURM) if available.
- Verify `outputs/` and `wandb/` remain writable and clean up stale artifacts.

## Post-release Notes
- Tag and push: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
- Announce changes and link to docs build.
