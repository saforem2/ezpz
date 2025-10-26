# TODO

## Product Opportunities

- [x] Ship an `ezpz doctor` CLI subcommand that validates cluster prerequisites before launch.
  1. ✅ Define diagnostic checks (MPI availability, scheduler env vars, WANDB creds, device backends) and add reusable probing helpers under `src/ezpz`.
  2. ✅ Wire a new `doctor` click subcommand in `src/ezpz/cli/__init__.py` that invokes the helpers and surfaces actionable remediation tips.
  3. ✅ Add regression tests with `CliRunner` that cover healthy, warning, and failure scenarios and document the workflow in the CLI guide.

- [x] Introduce a scheduler plug-in registry so site adapters can be added without modifying core code.
  1. ✅ Refactor `get_scheduler()` in `src/ezpz/configs.py` to consult entry points (for example `ezpz.schedulers`) and fall back to existing detection.
  2. ✅ Provide a reference implementation inside the docs and document the plug-in contract in the developer docs.
  3. ✅ Add tests that exercise both built-in and third-party plug-in discovery paths.

- [ ] Provide templated launch configurations for common clusters via a `--template` flag.
  1. Extract the current manual command assembly into reusable templates and store them under `src/ezpz/conf/templates`.
  2. Extend the launch CLI to accept `--template=<name>` and merge template defaults with user overrides.
  3. Publish template usage docs (including Aurora) and cover them with integration-style tests using mocked subprocess calls.

## Reliability & Packaging

- [x] Harden tarball helpers by removing debug breakpoints and shell escapes.
  1. ✅ Replace the lingering `breakpoint(0)` and `os.system` calls in `src/ezpz/utils/__init__.py` with safe `subprocess.run(..., check=True)` calls.
  2. ✅ Ensure temporary artifacts are cleaned up after runs and propagate errors with informative exceptions.
  3. ✅ Add pytest coverage that verifies tarball creation, reuse, overwrite, and failure paths.

- [x] Split optional heavy dependencies into extras and guard their imports.
  1. ✅ Move `wandb`, `pyinstrument`, and `plotext` into dedicated extras in `pyproject.toml`.
  2. ✅ Wrap imports with optional messaging guiding users toward the new extras.
  3. ✅ Document the extras in installation instructions and update the doctor remedies accordingly.

- [x] Automatically generate markdown reports when `ezpz.history.History` renders plots.
  1. ✅ Hook into plot creation to capture figures and emit lightweight Markdown summaries.
  2. ✅ Provide configuration switches to control report destinations and enable/disable behavior.
  3. ✅ Add unit tests verifying report creation and document how to consume the artifacts.

- [x] Add distributed metric reductions (mean/max/min/std) for `ezpz.history.History`.
  1. ✅ Integrate reduction hooks keyed off world size and rank metadata.
  2. ✅ Surface aggregated metrics in both console logging and optional W&B runs.
  3. ✅ Cover edge cases (single rank, mismatched shapes) in the history test suite.

- [x] Introduce JSONL logging support.
  1. ✅ Wire a JSONL handler into the logging configuration with rotation support.
  2. ✅ Expose CLI and config toggles to direct history metrics into JSONL files.
  3. ✅ Document parsing workflows and provide sample notebooks for analysis.

- [x] Allow `--hostfile`, `-n`, `-np` passthrough to the underlying launch command.
  1. ✅ Extend CLI parsing to capture these flags and propagate them into scheduler builders.
  2. ✅ Validate combinations to prevent conflicting user input.
  3. ✅ Update launch documentation and regression tests to cover the new parameters.

- [ ] Replace the linear model with a simple transformer example.
  1. Add a minimal transformer module with clear configuration knobs.
  2. Update tutorials and smoke tests to exercise the new architecture.
  3. Retire or archive the previous linear example to avoid confusion.

- [ ] Swap random data stubs for real datasets while retaining `--random` fallbacks.
  1. Integrate lightweight public datasets suitable for CI and quick experiments.
  2. Provide a flag or config knob to fall back to synthetic data when desired.
  3. Ensure data download/setup instructions are captured in docs.

- [ ] Revisit and fix existing modules under `ezpz.examples`.
  1. Audit each example for runtime errors or outdated APIs.
  2. Standardize argument parsing, logging, and dependency handling.
  3. Add smoke tests (or doctests) to prevent regressions.

- [ ] Enhance `ezpz yeet-env` to infer the active environment and default to uncompressed copies.
  1. Detect the active virtual environment/conda prefix automatically.
  2. Provide a configuration toggle for compression with sensible defaults.
  3. Update docs and tests to reflect the streamlined workflow.

- [ ] Add convenience wrappers for `DDP` and `FSDP`.
  1. Expose high-level helpers that configure standard PyTorch distributed strategies.
  2. Incorporate validation, logging, and profiling hooks centred on these wrappers.
  3. Document usage patterns alongside the new transformer example.

- [ ] Automate creation of distributed data loaders from Hugging Face datasets.
  1. Build adapters that construct samplers/loaders respecting world size and rank.
  2. Provide caching and sharding controls for large-scale training.
  3. Demonstrate usage in the examples directory and add integration coverage.

- [ ] Expand documentation with additional examples and code snippets.
  1. Flesh out API guides with runnable snippets for new utilities.
  2. Add troubleshooting tips tied to the doctor and scheduler plug-in systems.
  3. Publish end-to-end walkthroughs combining new models, data loaders, and wrappers.

- [ ] Split optional heavy dependencies into extras and guard their imports.
  1. Move `wandb`, `pyinstrument`, `plotext`, and similar packages into a new extras section in `pyproject.toml`.
  2. Wrap their imports inside `src/ezpz` with lazy or optional import handling and provide clear error messages when the extras are absent.
  3. Document the new extras in installation instructions and add tests that simulate environments with and without the optional features.

- [ ] Validate launch command assembly with explicit data structures.
  1. Refactor `src/ezpz/launch.py` to encapsulate np/ppn/hostfile validation inside a dataclass or helper class.
  2. Add input validation and descriptive error messages for inconsistent launch parameters.
  3. Cover the new behavior with targeted unit tests that mock `subprocess.Popen` and confirm filter handling still works.

## Documentation & Onboarding

- [ ] Consolidate and streamline the “Getting Started” narrative.
  1. Merge duplicated onboarding steps from `README.md` and `docs/index.md` into a single MkDocs page.
  2. Replace redundant text in the README with a concise pointer to the canonical documentation.
  3. Add a concept map figure under `docs/parallelism.md` to illustrate relationships between CLI, configs, and runtime components.

- [ ] Publish a troubleshooting matrix for common launch failures.
  1. Capture recurring error patterns (e.g., scheduler misconfigurations, WANDB offline warnings) and their resolutions.
  2. Author a dedicated FAQ or troubleshooting page in the docs and link it from relevant sections.
  3. Keep the landing page concise by moving verbose log excerpts into collapsible sections or appendices.

## Developer Experience

- [ ] Add a `pre-commit` workflow mirroring linting and type checking settings.
  1. Create a `.pre-commit-config.yaml` that runs Ruff, isort-compatible sorting, and Pyright.
  2. Update `DEVELOPMENT.md` with setup instructions and integrate the hook into contributor guidance.
  3. Validate hooks in CI and ensure they align with the existing `hatch` tasks.

- [ ] Introduce CLI smoke tests with Click’s testing utilities.
  1. Add a `tests/test_cli_smoke.py` module that exercises `ezpz launch`, `tar-env`, `yeet-env`, and the new `doctor` command via `CliRunner`.
  2. Mock out external side effects (e.g., subprocess calls, filesystem writes) to keep tests fast and deterministic.
  3. Wire the new test file into existing test orchestration scripts and verify coverage remains stable.
