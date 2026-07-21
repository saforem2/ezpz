"""Tests for the static re-exports added in fix for issue #133.

The lazy `__getattr__` resolution in `ezpz/__init__.py` keeps `import
ezpz` fast (no eager torch / mpi4py imports), but type checkers and
LSP-based editors (Pyright, Pylance, Jedi) can't follow `__getattr__`
indirection. This module verifies that:

1. Every symbol declared in the `TYPE_CHECKING` re-export block ALSO
   resolves at runtime via the lazy walker — so the static view and
   runtime view stay in sync. If someone adds a re-export and the
   underlying function isn't actually on the named submodule, this
   test catches it.

2. `__all__` lists every re-export — so `dir(ezpz)` and `from ezpz
   import *` work.

3. The `py.typed` marker file ships in the package — PEP 561 marker
   that tells type checkers to use the package's inline types.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import ezpz


# Symbols that should be discoverable at the top level. Keep this in
# sync with the TYPE_CHECKING block in ezpz/__init__.py. Each (name,
# module) pair: the name as it appears on `ezpz.<name>`, and the
# submodule it actually lives in (so we can verify both ends).
EXPECTED_STATIC_REEXPORTS = [
    # -- distributed --
    ("all_reduce", "ezpz.distributed"),
    ("barrier", "ezpz.distributed"),
    ("broadcast", "ezpz.distributed"),
    ("cleanup", "ezpz.distributed"),
    ("get_cpus_per_node", "ezpz.distributed"),
    ("get_device_properties", "ezpz.distributed"),
    ("get_dist_info", "ezpz.distributed"),
    ("get_gpus_per_node", "ezpz.distributed"),
    ("get_hostname", "ezpz.distributed"),
    ("get_local_rank", "ezpz.distributed"),
    ("get_machine", "ezpz.distributed"),
    ("get_node_index", "ezpz.distributed"),
    ("get_num_nodes", "ezpz.distributed"),
    ("get_rank", "ezpz.distributed"),
    ("get_torch_backend", "ezpz.distributed"),
    ("get_torch_device", "ezpz.distributed"),
    ("get_torch_device_type", "ezpz.distributed"),
    ("get_world_size", "ezpz.distributed"),
    ("get_world_size_in_use", "ezpz.distributed"),
    ("get_world_size_total", "ezpz.distributed"),
    ("init_device_mesh_safe", "ezpz.distributed"),
    ("flatten_device_mesh_safe", "ezpz.distributed"),
    ("log_dict_as_bulleted_list", "ezpz.distributed"),
    ("print_dist_setup", "ezpz.distributed"),
    ("query_environment", "ezpz.distributed"),
    ("seed_everything", "ezpz.distributed"),
    ("setup_mlflow", "ezpz.distributed"),
    ("setup_torch", "ezpz.distributed"),
    ("setup_wandb", "ezpz.distributed"),
    ("synchronize", "ezpz.distributed"),
    ("timeitlogit", "ezpz.distributed"),
    ("verify_wandb", "ezpz.distributed"),
    ("wrap_model", "ezpz.distributed"),
    ("wrap_model_for_ddp", "ezpz.distributed"),
    ("wrap_model_for_fsdp", "ezpz.distributed"),
    ("wrap_model_for_fsdp2", "ezpz.distributed"),
    # -- utils --
    ("Color", "ezpz.utils"),
    ("DistributedPdb", "ezpz.utils"),
    ("ForkedPdb", "ezpz.utils"),
    ("NoColor", "ezpz.utils"),
    ("check_for_tarball", "ezpz.utils"),
    ("format_compact_summary", "ezpz.utils"),
    ("format_memory_summary", "ezpz.utils"),
    ("format_pair", "ezpz.utils"),
    ("get_current_memory_allocated", "ezpz.utils"),
    ("get_current_memory_reserved", "ezpz.utils"),
    ("get_max_memory_allocated", "ezpz.utils"),
    ("get_max_memory_reserved", "ezpz.utils"),
    ("get_memory_metrics", "ezpz.utils"),
    ("get_timestamp", "ezpz.utils"),
    ("grab_tensor", "ezpz.utils"),
    ("is_memory_metric_key", "ezpz.utils"),
    ("make_tarfile", "ezpz.utils"),
    ("model_summary", "ezpz.utils"),
    ("reset_peak_memory_stats", "ezpz.utils"),
    ("summarize_dict", "ezpz.utils"),
    # -- log --
    ("get_logger", "ezpz.log"),
    ("silence_noisy_loggers", "ezpz.log"),
    # -- history --
    ("History", "ezpz.history"),
    # -- flops --
    ("compute_mfu", "ezpz.flops"),
    ("try_estimate", "ezpz.flops"),
    # -- configs --
    ("get_scheduler", "ezpz.configs"),
    # -- launch --
    ("get_active_jobid", "ezpz.launch"),
    ("get_nodelist_of_active_job", "ezpz.launch"),
]


class TestStaticReexports:
    """Each TYPE_CHECKING re-export should also resolve at runtime."""

    def test_every_reexport_resolves(self):
        """If a symbol is statically re-exported, it must be importable.

        This catches the case where someone adds a name to the
        TYPE_CHECKING block but the underlying submodule doesn't
        define it (or has renamed it). Without this test, type
        checkers would happily resolve `ezpz.foo`, then users would
        hit `AttributeError` at runtime.
        """
        missing = []
        for name, _expected_module in EXPECTED_STATIC_REEXPORTS:
            if not hasattr(ezpz, name):
                missing.append(name)
        assert not missing, (
            f"Statically re-exported but not resolvable at runtime: "
            f"{missing}"
        )

    def test_expected_list_matches_runtime_static_reexports(self):
        """Drift guard: EXPECTED_STATIC_REEXPORTS must cover every name
        in ezpz._STATIC_REEXPORTS.

        Without this, adding a name to _STATIC_REEXPORTS (and the
        TYPE_CHECKING block) without also updating this test file
        would silently pass — but a regression where the name doesn't
        actually resolve at runtime wouldn't be caught for THAT name.
        Sourcery / Copilot flagged this as the main drift risk in the
        PR review.
        """
        expected_names = {name for name, _ in EXPECTED_STATIC_REEXPORTS}
        runtime_names = set(ezpz._STATIC_REEXPORTS)
        missing_from_test = runtime_names - expected_names
        extra_in_test = expected_names - runtime_names
        assert not missing_from_test, (
            f"ezpz._STATIC_REEXPORTS has names not covered by this test "
            f"file's EXPECTED_STATIC_REEXPORTS: {sorted(missing_from_test)}. "
            f"Add them to EXPECTED_STATIC_REEXPORTS so the per-symbol "
            f"resolution test runs against the full set."
        )
        assert not extra_in_test, (
            f"EXPECTED_STATIC_REEXPORTS lists names not in "
            f"ezpz._STATIC_REEXPORTS: {sorted(extra_in_test)}. "
            f"Either add them to _STATIC_REEXPORTS or remove from the "
            f"test file."
        )

    def test_every_reexport_comes_from_advertised_submodule(self):
        """Sanity-check that each symbol lives where the docstring says.

        e.g. `ezpz.setup_torch` should resolve to a callable defined in
        `ezpz.distributed`, not (say) `ezpz.utils`. Protects against
        accidental moves that would silently change the import target.
        """
        mismatches = []
        for name, expected_module in EXPECTED_STATIC_REEXPORTS:
            obj = getattr(ezpz, name)
            actual_module = getattr(obj, "__module__", None)
            if actual_module != expected_module:
                mismatches.append((name, expected_module, actual_module))
        assert not mismatches, (
            "Re-exports defined in unexpected modules: "
            + ", ".join(
                f"{name} expected {exp} got {act}"
                for name, exp, act in mismatches
            )
        )

    def test_reexports_listed_in_dunder_all(self):
        """`from ezpz import *` should give every re-export.

        Also protects `dir(ezpz)` discoverability for editors that
        consume __all__ instead of (or in addition to) walking
        TYPE_CHECKING blocks.
        """
        missing_from_all = [
            name for name, _expected_module in EXPECTED_STATIC_REEXPORTS
            if name not in ezpz.__all__
        ]
        assert not missing_from_all, (
            f"Re-exports not in __all__: {missing_from_all}"
        )

    def test_type_checking_block_matches_static_reexports(self):
        """Drift guard between the TYPE_CHECKING import block and
        ezpz._STATIC_REEXPORTS.

        Parses ezpz/__init__.py and extracts every name imported
        inside `if TYPE_CHECKING:`. That set must equal
        ezpz._STATIC_REEXPORTS — otherwise type checkers and
        `from ezpz import *` consumers see different surfaces.
        """
        import ast

        source = Path(ezpz.__file__).read_text()
        tree = ast.parse(source)

        # Find the `if TYPE_CHECKING:` block at module scope.
        type_checking_imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.If):
                # The guard is `if TYPE_CHECKING:` — match by name.
                test = node.test
                if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                    for stmt in node.body:
                        if isinstance(stmt, ast.ImportFrom):
                            for alias in stmt.names:
                                # Use the alias if present, else the name
                                type_checking_imports.add(
                                    alias.asname or alias.name
                                )

        static_reexports = set(ezpz._STATIC_REEXPORTS)
        only_in_typecheck = type_checking_imports - static_reexports
        only_in_static = static_reexports - type_checking_imports
        assert not only_in_typecheck, (
            f"Imported in TYPE_CHECKING block but missing from "
            f"_STATIC_REEXPORTS (won't appear in __all__): "
            f"{sorted(only_in_typecheck)}"
        )
        assert not only_in_static, (
            f"In _STATIC_REEXPORTS but missing from TYPE_CHECKING "
            f"imports (type checkers won't see them): "
            f"{sorted(only_in_static)}"
        )


class TestPyTypedMarker:
    """PEP 561 marker that tells type checkers the package is typed."""

    def test_py_typed_file_exists(self):
        """`src/ezpz/py.typed` must be present.

        Without this empty file, Pyright / Pylance ignore the package's
        inline annotations and treat everything as `Unknown` — even if
        the TYPE_CHECKING re-exports are correct. This is the marker
        PEP 561 specifies.
        """
        package_dir = Path(ezpz.__file__).parent
        marker = package_dir / "py.typed"
        assert marker.exists(), (
            f"py.typed marker missing at {marker}. Without it, type "
            f"checkers treat the package as untyped."
        )


class TestLazyResolutionStillWorks:
    """The lazy __getattr__ path is unchanged — preserves import speed."""

    def test_submodules_still_resolve_lazily(self):
        """`ezpz.distributed` is a submodule, not a static re-export.

        The lazy walker should still find it (via _LAZY_MODULES) without
        the explicit `from .distributed import distributed` (which would
        be nonsense). Protects against accidentally dropping the lazy
        path while adding static re-exports.
        """
        assert ezpz.distributed is not None
        # The lazy walker caches on first hit; re-accessing should be
        # the same module object.
        assert ezpz.distributed is importlib.import_module("ezpz.distributed")

    def test_top_level_helpers_resolve(self):
        """Symbols defined directly in `__init__.py` (not re-exports)
        also remain accessible."""
        assert callable(ezpz.try_import_wandb)
        assert callable(ezpz.get_torch_version_tuple)
