# torchcomms Backend Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ezpz route `torch.distributed` through torchcomms via a single `EZPZ_USE_TORCHCOMMS` environment variable, best-effort with warn-and-fallback.

**Architecture:** torchcomms is a *mode over* the existing `TORCH_BACKEND` transport (nccl/xccl/gloo), not a new backend. A new `use_torchcomms()` helper in `distributed.py` gates on `EZPZ_USE_TORCHCOMMS` + an availability probe (cached at module scope). `_setup_ddp` sets `torch.distributed.config.use_torchcomms = True` before the existing `init_process_group` call (which already passes `device_id=`, torchcomms' one requirement). `get_dist_info()` reports the active state. Mirrors torchtitan's `init_distributed` torchcomms path.

**Tech Stack:** Python, `torch.distributed`, torchcomms (already in the `torch` optional-deps group), pytest + monkeypatch/unittest.mock.

## Global Constraints

- **Backward compatible:** default behavior (env var unset) is byte-identical to today. No raise on missing torchcomms — warn + fall back.
- **Truthy env idiom:** reuse `value.lower() in ("1", "true", "yes", "on")` (matches `distributed.py:456`).
- **Warn on rank 0 only**, exactly once per job (cache the probe result at module scope).
- **Tests run CPU-only**, no GPU/MPI — mock torch as `tests/test_distributed.py` already does (torch is imported real; patch attributes / `sys.modules`).
- **Docs updated in the same pass** as code (project convention).
- **Release:** single `release:patch` PR; merge as a **merge commit, not squash** (project convention).
- Work on branch `feat/torchcomms-backend` (already created; spec committed there as `c1b8aba`).

---

### Task 1: `use_torchcomms()` helper + module-scope cache

**Files:**
- Modify: `src/ezpz/distributed.py` — add helper after `get_torch_backend()` (ends line 430); add `"use_torchcomms"` to `__all__` (near line 64).
- Test: `tests/test_distributed.py` — add `TestUseTorchcomms` class after `TestGetTorchBackend` (ends line 610).

**Interfaces:**
- Consumes: `os.environ`; `torch.distributed.config` (probe).
- Produces:
  - `use_torchcomms() -> bool` — True only when `EZPZ_USE_TORCHCOMMS` is truthy AND torchcomms is usable.
  - `_torchcomms_unavailable_reason() -> str` — reason string when requested-but-unavailable, else `""`.
  - `_reset_torchcomms_cache() -> None` — test hook to clear the module-scope cache.
  - Module global `_TORCHCOMMS_CACHE: tuple[bool, str] | None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_distributed.py` after line 610 (end of `TestGetTorchBackend`):

```python
class TestUseTorchcomms:
    """Tests for ``use_torchcomms`` and its availability probe."""

    def setup_method(self):
        # Each test starts with a clean probe cache.
        dist._reset_torchcomms_cache()

    def teardown_method(self):
        dist._reset_torchcomms_cache()

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("EZPZ_USE_TORCHCOMMS", raising=False)
        assert dist.use_torchcomms() is False
        assert dist._torchcomms_unavailable_reason() == ""

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_and_available(self, monkeypatch, val):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", val)
        fake_tc = MagicMock()
        fake_cfg = MagicMock()
        fake_cfg.use_torchcomms = False  # attr must EXIST
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", fake_cfg, create=True),
        ):
            assert dist.use_torchcomms() is True
            assert dist._torchcomms_unavailable_reason() == ""

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "off"])
    def test_falsy_is_false(self, monkeypatch, val):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", val)
        assert dist.use_torchcomms() is False

    def test_requested_but_package_missing(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        with patch.dict("sys.modules", {"torchcomms": None}):
            # sys.modules[name] = None makes `import torchcomms` raise ImportError
            assert dist.use_torchcomms() is False
            assert "torchcomms" in dist._torchcomms_unavailable_reason().lower()

    def test_requested_but_torch_switch_absent(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        fake_tc = MagicMock()
        cfg_without_switch = MagicMock(spec=[])  # no use_torchcomms attr
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", cfg_without_switch, create=True),
        ):
            assert dist.use_torchcomms() is False
            assert dist._torchcomms_unavailable_reason() != ""

    def test_probe_is_cached(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        fake_tc = MagicMock()
        fake_cfg = MagicMock()
        fake_cfg.use_torchcomms = False
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", fake_cfg, create=True),
        ):
            assert dist.use_torchcomms() is True
        # After the patches exit, torchcomms would look unavailable — but the
        # cached True result must persist (probe ran once).
        assert dist.use_torchcomms() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_distributed.py::TestUseTorchcomms -v`
Expected: FAIL — `AttributeError: module 'ezpz.distributed' has no attribute 'use_torchcomms'` (and `_reset_torchcomms_cache`).

_(If `.venv/bin/python` is absent on the dev host, use the environment's python — real runs execute on ALCF; these tests are CPU-only and torch-mocked.)_

- [ ] **Step 3: Implement the helper**

In `src/ezpz/distributed.py`, insert after line 430 (the `return "gloo"` + blank line ending `get_torch_backend`, before the `# Lifecycle` banner):

```python
# Module-scope cache for the torchcomms availability probe so the import
# check runs once and any warning fires exactly once per process.
_TORCHCOMMS_CACHE: "tuple[bool, str] | None" = None


def _reset_torchcomms_cache() -> None:
    """Clear the cached torchcomms probe result (test hook)."""
    global _TORCHCOMMS_CACHE
    _TORCHCOMMS_CACHE = None


def _probe_torchcomms() -> "tuple[bool, str]":
    """Return (usable, reason). ``usable`` is True only when torchcomms is
    importable AND torch exposes the ``use_torchcomms`` switch. ``reason`` is
    a human-readable explanation when not usable, else ``""``.

    Result is cached at module scope; call ``_reset_torchcomms_cache`` to reprobe.
    """
    global _TORCHCOMMS_CACHE
    if _TORCHCOMMS_CACHE is not None:
        return _TORCHCOMMS_CACHE
    try:
        import torchcomms  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure → unavailable
        _TORCHCOMMS_CACHE = (False, f"torchcomms import failed: {exc}")
        return _TORCHCOMMS_CACHE
    import torch.distributed as _td

    if not hasattr(getattr(_td, "config", None), "use_torchcomms"):
        _TORCHCOMMS_CACHE = (
            False,
            "installed torch lacks torch.distributed.config.use_torchcomms",
        )
        return _TORCHCOMMS_CACHE
    _TORCHCOMMS_CACHE = (True, "")
    return _TORCHCOMMS_CACHE


def _torchcomms_requested() -> bool:
    """Whether EZPZ_USE_TORCHCOMMS is set to a truthy value."""
    return os.environ.get("EZPZ_USE_TORCHCOMMS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def use_torchcomms() -> bool:
    """Whether to route ``torch.distributed`` through torchcomms.

    True only when ``EZPZ_USE_TORCHCOMMS`` is truthy (``1``/``true``/``yes``/
    ``on``) AND torchcomms is usable (package importable + torch exposes
    ``torch.distributed.config.use_torchcomms``). Best-effort: returns False
    when requested-but-unavailable; use :func:`_torchcomms_unavailable_reason`
    to surface why.
    """
    if not _torchcomms_requested():
        return False
    usable, _ = _probe_torchcomms()
    return usable


def _torchcomms_unavailable_reason() -> str:
    """Reason torchcomms is unusable when requested; ``""`` otherwise."""
    if not _torchcomms_requested():
        return ""
    _, reason = _probe_torchcomms()
    return reason
```

Then add `"use_torchcomms"` to the `__all__` list — insert after the `"get_torch_backend",` entry at line 64:

```python
    "get_torch_backend",
    "use_torchcomms",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_distributed.py::TestUseTorchcomms -v`
Expected: PASS (all parametrizations).

- [ ] **Step 5: Commit**

```bash
git add src/ezpz/distributed.py tests/test_distributed.py
git commit -m "feat(dist): add use_torchcomms() env-gated helper

EZPZ_USE_TORCHCOMMS gates a cached availability probe (torchcomms
importable + torch.distributed.config.use_torchcomms present). Returns
False and records a reason when requested-but-unavailable, so callers
can warn and fall back."
```

---

### Task 2: Activate torchcomms in `_setup_ddp` (log + warn-and-fallback)

**Files:**
- Modify: `src/ezpz/distributed.py` — inside `_setup_ddp` (def at line 1691), just before `if not torch.distributed.is_initialized():` (line 1766).
- Test: `tests/test_distributed.py` — extend `TestUseTorchcomms` (or a new `TestSetupDdpTorchcomms`) verifying the flag is set / warning path.

**Interfaces:**
- Consumes: `use_torchcomms()`, `_torchcomms_unavailable_reason()`, `_torchcomms_requested()` (Task 1); `rank`, `backend` (already local in `_setup_ddp`, lines 1706-1709); `logger`.
- Produces: side effect — sets `torch.distributed.config.use_torchcomms = True` when active; rank-0 log/warn.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py` (in `TestUseTorchcomms`, after `test_probe_is_cached`). This tests the small activation block in isolation by exercising the real code path guard — since `_setup_ddp` does full PG init, we test the decision logic via a thin helper. Add the helper as part of Step 3; the test asserts on it:

```python
    def test_activation_sets_flag_when_available(self, monkeypatch):
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        fake_tc = MagicMock()
        fake_cfg = MagicMock()
        fake_cfg.use_torchcomms = False
        with (
            patch.dict("sys.modules", {"torchcomms": fake_tc}),
            patch.object(torch.distributed, "config", fake_cfg, create=True),
        ):
            applied = dist._maybe_enable_torchcomms(rank=0, backend="nccl")
        assert applied is True
        assert fake_cfg.use_torchcomms is True

    def test_activation_warns_when_requested_unavailable(self, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("EZPZ_USE_TORCHCOMMS", "1")
        with patch.dict("sys.modules", {"torchcomms": None}):
            with caplog.at_level(logging.WARNING):
                applied = dist._maybe_enable_torchcomms(rank=0, backend="xccl")
        assert applied is False
        assert any("EZPZ_USE_TORCHCOMMS" in r.message for r in caplog.records)

    def test_activation_noop_when_unset(self, monkeypatch):
        monkeypatch.delenv("EZPZ_USE_TORCHCOMMS", raising=False)
        assert dist._maybe_enable_torchcomms(rank=0, backend="nccl") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_distributed.py::TestUseTorchcomms::test_activation_sets_flag_when_available -v`
Expected: FAIL — `AttributeError: ... has no attribute '_maybe_enable_torchcomms'`.

- [ ] **Step 3: Implement `_maybe_enable_torchcomms` + call it in `_setup_ddp`**

In `src/ezpz/distributed.py`, add this helper immediately after `use_torchcomms()` / `_torchcomms_unavailable_reason()` (from Task 1):

```python
def _maybe_enable_torchcomms(*, rank: int, backend: str) -> bool:
    """Enable torchcomms for the standard PG if requested + available.

    Returns True if torchcomms was activated. When requested but unavailable,
    logs a single rank-0 warning naming the reason and returns False. No-op
    (returns False) when EZPZ_USE_TORCHCOMMS is unset.
    """
    if use_torchcomms():
        import torch.distributed as _td

        _td.config.use_torchcomms = True
        if rank == 0:
            logger.info("Using torchcomms over backend=%s", backend)
        return True
    if _torchcomms_requested() and rank == 0:
        logger.warning(
            "EZPZ_USE_TORCHCOMMS set but torchcomms unavailable (%s); "
            "using standard %s backend.",
            _torchcomms_unavailable_reason(),
            backend,
        )
    return False
```

Then, in `_setup_ddp`, insert the call just before line 1766 (`if not torch.distributed.is_initialized():`), after the rank-0 `logger.info("init_process_group: ...")` block:

```python
    _maybe_enable_torchcomms(rank=rank, backend=backend)

    if not torch.distributed.is_initialized():
```

The existing `init_process_group(**init_kwargs)` (with `device_id=`, line ~1812) is unchanged — it now builds torchcomms-backed PGs when the flag is set.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_distributed.py::TestUseTorchcomms -v`
Expected: PASS (all, including the 3 new activation tests).

- [ ] **Step 5: Commit**

```bash
git add src/ezpz/distributed.py tests/test_distributed.py
git commit -m "feat(dist): enable torchcomms in _setup_ddp before init_process_group

_maybe_enable_torchcomms sets torch.distributed.config.use_torchcomms
when EZPZ_USE_TORCHCOMMS is active, else warns once on rank 0 and falls
back to the standard backend. The existing device_id= on
init_process_group already satisfies torchcomms' device-bound-PG need."
```

---

### Task 3: Report torchcomms state in `get_dist_info()`

**Files:**
- Modify: `src/ezpz/distributed.py` — `info.update({...})` dict at line 1032 (add key next to `DISTRIBUTED_BACKEND`, line 1036).
- Test: `tests/test_distributed.py` — assert the key is present.

**Interfaces:**
- Consumes: `use_torchcomms()` (Task 1).
- Produces: `get_dist_info()` output dict now contains `"TORCHCOMMS": bool`.

- [ ] **Step 1: Write the failing test**

Find the existing `get_dist_info` test (search `get_dist_info` in `tests/test_distributed.py`) and add, or add a new test in the relevant class:

```python
    def test_dist_info_includes_torchcomms(self, monkeypatch):
        monkeypatch.delenv("EZPZ_USE_TORCHCOMMS", raising=False)
        info = dist.get_dist_info(verbose=False)
        assert "TORCHCOMMS" in info
        assert info["TORCHCOMMS"] is False
```

_(Match the existing `get_dist_info` test's fixtures/mocks — it already runs CPU-only in this suite. If `get_dist_info` requires MPI/host mocks, reuse the same `patch`/fixture the neighboring `get_dist_info` test uses.)_

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_distributed.py -k "torchcomms and dist_info" -v`
Expected: FAIL — `KeyError`/assert on missing `"TORCHCOMMS"`.

- [ ] **Step 3: Add the field**

In `src/ezpz/distributed.py`, in the `info.update({...})` block, add after the `"DISTRIBUTED_BACKEND": get_torch_backend(),` line (1036):

```python
            "DISTRIBUTED_BACKEND": get_torch_backend(),
            "TORCHCOMMS": use_torchcomms(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_distributed.py -k "torchcomms" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ezpz/distributed.py tests/test_distributed.py
git commit -m "feat(dist): report TORCHCOMMS state in get_dist_info

Surfaces torchcomms active/inactive in the rank-0 startup banner and any
run metadata / W&B config for benchmark provenance."
```

---

### Task 4: Document `EZPZ_USE_TORCHCOMMS`

**Files:**
- Modify: `docs/configuration.md` — env-var table (row `TORCH_BACKEND` at line 80) + an example near line 56.

**Interfaces:**
- Consumes: nothing (docs).
- Produces: user-facing documentation of the flag.

- [ ] **Step 1: Add the env-var table row**

In `docs/configuration.md`, add a row immediately after the `TORCH_BACKEND` row (line 80):

```markdown
| `EZPZ_USE_TORCHCOMMS` | Route `torch.distributed` through torchcomms over the selected `TORCH_BACKEND` transport. Best-effort: warns and falls back to the standard backend if torchcomms is unavailable. | `1`/`true`/`yes`/`on`. Default: unset (off). |
```

- [ ] **Step 2: Add an example**

Near the existing `TORCH_DEVICE=cpu TORCH_BACKEND=gloo ...` example (line 56), add:

```markdown
# Route distributed comms through torchcomms over nccl
EZPZ_USE_TORCHCOMMS=1 TORCH_BACKEND=nccl ezpz launch -np 2 -- python3 train.py
```

- [ ] **Step 3: Verify docs render / no broken table**

Run: `grep -n "EZPZ_USE_TORCHCOMMS" docs/configuration.md`
Expected: two matches (table row + example). Eyeball the table alignment (pipe count matches neighboring rows).

- [ ] **Step 4: Commit**

```bash
git add docs/configuration.md
git commit -m "docs(config): document EZPZ_USE_TORCHCOMMS env var"
```

---

### Task 5: Full suite green + PR

**Files:** none (verification + PR).

- [ ] **Step 1: Run the distributed test module**

Run: `.venv/bin/python -m pytest tests/test_distributed.py -v`
Expected: PASS (existing tests + all new torchcomms tests). No regressions.

- [ ] **Step 2: Byte-compile check**

Run: `.venv/bin/python -m py_compile src/ezpz/distributed.py`
Expected: no output (success).

- [ ] **Step 3: Confirm default-path unchanged**

Run: `.venv/bin/python -m pytest tests/test_distributed.py -k "not torchcomms" -v`
Expected: PASS — proves the additive change didn't disturb existing behavior.

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin feat/torchcomms-backend
gh pr create --title "feat(dist): wire torchcomms via EZPZ_USE_TORCHCOMMS" \
  --body "$(cat <<'EOF'
Route torch.distributed through torchcomms via a single env var.

## What
- `EZPZ_USE_TORCHCOMMS=1` (+ `TORCH_BACKEND=nccl|xccl`) enables torchcomms as a
  *mode over* the transport — sets `torch.distributed.config.use_torchcomms`
  before the existing `init_process_group` (which already passes `device_id=`).
- Best-effort: warns once on rank 0 and falls back to the standard backend if
  torchcomms is unavailable (package missing or torch too old).
- `get_dist_info()` reports `TORCHCOMMS: true/false`.
- Mirrors torchtitan's `init_distributed` torchcomms path.

## Not yet done (needs on-node validation)
- Aurora (XPU) + CUDA real-run confirmation that torchcomms comes up and trains
  a few steps. Design validated against torchtitan source + torchcomms docs; not
  executed (no torch on dev host).

Spec: docs/superpowers/specs/2026-07-06-torchcomms-backend-design.md
EOF
)"
```

- [ ] **Step 5: Add release label**

```bash
gh pr edit --add-label "release:patch"
```

(Per project convention: every PR needs a `release:` label before merge; merge as a **merge commit, not squash**.)

---

## Self-Review

**Spec coverage:**
- Decision 1 (dedicated `EZPZ_USE_TORCHCOMMS`) → Task 1 (`_torchcomms_requested`, `use_torchcomms`).
- Decision 2 (warn + fallback) → Task 2 (`_maybe_enable_torchcomms` warn path).
- Decision 3 (log line + dist-info field) → Task 2 (log) + Task 3 (dist-info).
- `use_torchcomms()` helper w/ cache → Task 1.
- Touch point A (`_setup_ddp`) → Task 2.
- Touch point B (`get_dist_info`) → Task 3.
- Tests (6 spec cases: unset, truthy variants, package-missing, torch-too-old, dist-info key, caching) → Task 1 (5) + Task 3 (dist-info) + Task 2 (activation/warn). ✓ all covered.
- Docs → Task 4.
- Validation-on-node → captured as PR "Not yet done" + spec Validation section. ✓
- Release (`release:patch`, merge-commit) → Task 5.

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. The one conditional ("if `.venv/bin/python` absent, use env python") is an environment note, not a placeholder — real code paths are complete.

**Type consistency:** `use_torchcomms() -> bool`, `_torchcomms_unavailable_reason() -> str`, `_torchcomms_requested() -> bool`, `_probe_torchcomms() -> tuple[bool,str]`, `_reset_torchcomms_cache() -> None`, `_maybe_enable_torchcomms(*, rank:int, backend:str) -> bool` — names used consistently across Tasks 1-3. `_TORCHCOMMS_CACHE` global referenced only in probe/reset. ✓

**Note on `_probe_torchcomms` caching + tests:** `test_probe_is_cached` relies on the cache persisting a `True` after patches exit; `setup_method`/`teardown_method` call `_reset_torchcomms_cache()` so no cross-test leakage. Verified consistent.
