# AGENTS.md

Guidance for AI coding assistants (and humans) working in this repo.

## Dev environment tips

- Use `source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env` to
  setup the development environment.
- `ezpz doctor` can be used to verify the functionality of the environment.
- `ezpz test` can be used as a distributed PyTorch smoke test to verify:

  1. `ezpz` installed correctly
  2. functionality of distributed PyTorch with MPI

- When running Python directly during development, prefer the project venv:
  `.venv/bin/python` (so you get the editable install + pinned deps).

## Repo management

- Use `uv` for dependency management, falling back to `pip` ONLY when `uv`
  breaks (or gives unexpected behavior).
- Build backend is `hatchling`; the version lives in
  `src/ezpz/__about__.py`.

## Development dependencies

- Use `ruff` for linting/formatting.
- Use `ty` for type checking.
- Use `pytest` for testing — run via `.venv/bin/pytest`.

## Documentation

- Use Google-style docstrings consistently across the codebase.
- **The docs site is built with `zensical`, NOT `mkdocs`.** The config is
  `zensical.toml` (TOML format). `mkdocs.yml` no longer exists — do not
  recreate it. Nav entries use TOML inline-table syntax:
  `{ "Label" = "path.md" }`. Serve locally with `zensical serve`.
- Update docs in the **same change** as the code they describe — don't
  leave doc updates for a follow-up. Each example page has a
  "Common modifications" section; new CLI flags / presets belong there.
- Doc code is often pulled from source via `--8<-- "path:start:end"`
  snippet includes. When you add lines to a `MODEL_PRESETS` dict (or any
  snippet-referenced block), bump the snippet's `end` line so the
  rendered code doesn't truncate.

## Commits & PRs

- Break changes into logical, atomic commits with descriptive messages.
  Never lump unrelated changes into one commit.
- Follow the existing commit style: `type(scope): message`
  (e.g. `fix(history): ...`, `feat(examples): ...`, `chore(todo): ...`).
- **Every PR needs a `release:` label before merge**: one of
  `release:patch`, `release:minor`, `release:major`, or `release:skip`.
  There is no default-to-patch — pick deliberately. Merging triggers
  release-please to auto-cut a tagged release (except `release:skip`).
- Squash-merge is the current convention (commit subject ends with the
  `(#NN)` PR number).

## Module structure

- `src/ezpz/distributed.py` — canonical distributed-training module
  (rank/topology/device/lifecycle/collectives/model-wrapping).
- `src/ezpz/dist.py` — **deprecated** thin re-export shim; new code should
  import from `distributed.py`.
- `src/ezpz/history.py` — metric tracking (`History` class); large + complex.
- `src/ezpz/launch.py` — scheduler-aware launcher (PBS/SLURM/mpirun). Must
  work WITHOUT torch installed (it only orchestrates `mpiexec`); don't add
  torch-importing code to its import path.
- `src/ezpz/cli/` — `ezpz <subcommand>` entry points + argparse (`flags.py`).
- `src/ezpz/examples/` — runnable training examples. Shared helpers live in
  `src/ezpz/examples/_presets.py` (see below).

## Examples: model-size presets

All 5 training examples (`test`, `fsdp`, `vit`, `diffusion`, `fsdp_tp`)
expose a unified `--model` size ladder:

| Preset | Target params | Notes |
|---|---|---|
| `debug` | sub-MB | laptop-runnable smoke test |
| `s` / `small` | ~100M | |
| `m` / `medium` | ~250M | |
| `l` / `large` | ~500M | |
| `xl` | ~1B | |
| `xxl` | ~5B | |
| `xxxl` | ~10B | architectural ceiling for most archs |

- Short names (`s/m/l/xl/xxl/xxxl`) are canonical; long forms
  (`small`/`xlarge`/`extra-large`/etc) are aliases in each module's
  `MODEL_ALIASES`.
- `fsdp_tp` additionally exposes `agpt-2b` / `agpt-20b` (verbatim from
  torchtitan's `agpt_configs`) and accepts HF repo ids via
  `--model owner/repo`.
- When changing a preset, keep the analytic param count within ±40% of
  its target — `tests/test_example_size_targets.py` enforces this.
- Adding/renaming an alias means updating ALL 5 modules' `MODEL_ALIASES`
  in the same change; `tests/test_example_model_sizes.py` pins parity.

## Conventions for cross-file helpers

- If a helper is used by more than one example, put it in a shared leaf
  module (e.g. `ezpz.examples._presets`) and import it everywhere — do NOT
  copy-paste. Copy-pasted helpers drift: a fix in one copy silently leaves
  the others broken. Back this with a parametrized regression test that
  exercises every consumer.
- When a circular import blocks extraction (e.g. `cli/flags.py` is imported
  by `examples/test.py`, so it can't import back), keep the duplicated
  constant but add a drift-pinning test that asserts the two stay in sync
  (see `tests/test_cli_flags_model_choices_in_sync.py`).

## Logging at scale

- Per-rank log lines should be gated to `local_rank == 0` (one line per
  host) — a 96-rank job printing from every rank is unreadable. The
  `setup_torch` summary follows this; new per-rank logs should too.
- Noisy third-party libraries (`transformers`, `datasets`, `huggingface_hub`,
  ...) are quieted via `ezpz.silence_noisy_loggers()`. Note `datasets` and
  `transformers` emit through TWO channels — Python `logging` AND their own
  internal verbosity routers — so both must be silenced.

## Tracked work

- `TODO.md` at the repo root is the tracked-issues list (sectioned,
  numbered, with `Status:` flags). Update it when shipping an item: add a
  `Status: DONE` line + commit/PR ref, then prune in the next audit pass.
