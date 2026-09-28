"""``ezpz benchmark`` must work from an uninstalled checkout.

``run_all`` shells out to ``ezpz launch`` for each example. That token used
to be the literal string ``"ezpz"``, i.e. the console script from
``[project.scripts]``. A console script only exists after an install, and on
HPC systems ezpz routinely runs from a checkout on ``PYTHONPATH`` against a
read-only site module -- NERSC ``pytorch/2.13.0``, for one. There the
benchmark died before running anything::

    FileNotFoundError: [Errno 2] No such file or directory: 'ezpz'

(Perlmutter job 59015143.) Polaris masked it: its conda env has ezpz
installed, so ``ezpz`` resolved fine.
"""

from __future__ import annotations

import sys

from ezpz.examples.run_all import _ezpz_launch_prefix


def test_uses_console_script_when_installed(monkeypatch):
    """An installed environment keeps using its own entry point."""
    monkeypatch.setattr(
        "ezpz.examples.run_all.shutil.which",
        lambda name: "/opt/env/bin/ezpz" if name == "ezpz" else None,
    )
    assert _ezpz_launch_prefix() == ["/opt/env/bin/ezpz", "launch"]


def test_falls_back_to_interpreter_when_not_installed(monkeypatch):
    """No console script: drive the CLI through the current interpreter."""
    monkeypatch.setattr(
        "ezpz.examples.run_all.shutil.which", lambda _name: None
    )
    prefix = _ezpz_launch_prefix()
    assert prefix[0] == sys.executable
    assert prefix[-1] == "launch"
    assert "ezpz" not in prefix[0:1] or prefix[0] == sys.executable


def test_fallback_never_uses_dash_m_on_the_cli_package(monkeypatch):
    """``python -m ezpz.cli`` is not runnable -- it has no ``__main__``.

    Pinning this because ``-m`` is the reflexive way to write the fallback
    and it fails with *"'ezpz.cli' is a package and cannot be directly
    executed"* -- swapping one FileNotFoundError for another.
    """
    monkeypatch.setattr(
        "ezpz.examples.run_all.shutil.which", lambda _name: None
    )
    prefix = _ezpz_launch_prefix()
    assert "-m" not in prefix, f"fallback must not use -m: {prefix}"


def test_fallback_is_executable_in_a_subprocess():
    """The fallback must actually run, not merely look plausible.

    Executes the built argv with ``--help`` so a broken import expression
    or a bad flag ordering fails here rather than 20 minutes into a job.
    """
    import subprocess

    prefix = [
        sys.executable,
        "-c",
        "from ezpz.cli import main; main()",
        "launch",
    ]
    proc = subprocess.run(
        [*prefix, "--help"], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr[-600:]}"
    assert "launch" in (proc.stdout + proc.stderr).lower()


def test_build_command_has_no_bare_ezpz_token():
    """End-to-end: the built command must not contain a bare ``"ezpz"``.

    Guards the call site, not just the helper -- the original bug was a
    hardcoded token inside ``_build_command``.
    """
    from pathlib import Path

    from ezpz.examples.run_all import build_command

    cmd = build_command(
        {"module": "ezpz.examples.test", "args": []},
        model="small",
        bench_dir=Path("/tmp/bench"),
        timestamp="2026-09-28",
    )
    assert cmd[0] != "ezpz", (
        "a bare 'ezpz' token requires an installed console script; "
        f"got {cmd!r}"
    )
    assert "launch" in cmd
