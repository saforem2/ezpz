"""`ezpz submit` must not generate a curl on the compute-node path.

Compute nodes on Aurora, Polaris and Sunspot have no outbound route.
`source <(curl ...)` hangs for ~270 s there and then leaves the
environment unconfigured, so the job fails much later with
`CONDA_PREFIX still not set` or undefined `ezpz_*` functions. SKILL.md
lists this first under "inside a batch script, three things change" --
and `detect_env_setup()` was emitting exactly that form by default.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from ezpz.submit import detect_env_setup, generate_pbs_script


def _clear(monkeypatch):
    monkeypatch.delenv("EZPZ_SETUP_ENV", raising=False)


def test_default_sources_the_installed_utils_sh(monkeypatch):
    """The default must be a local path, not a network fetch."""
    _clear(monkeypatch)
    out = detect_env_setup()
    assert "curl" not in out, f"compute nodes cannot curl: {out!r}"
    assert out.startswith("source /")
    assert out.endswith("ezpz_setup_env")


def test_the_path_it_emits_actually_exists(monkeypatch):
    """A generated script that sources a missing file is worse than curl."""
    _clear(monkeypatch)
    out = detect_env_setup()
    # shlex, not split(): the path is shell-quoted, so a directory with
    # a space in it would otherwise yield only its first fragment and
    # this check would pass on a path that does not exist.
    path = shlex.split(out)[1]
    assert Path(path).is_file(), f"generated source target missing: {path}"


def test_explicit_env_var_still_wins(monkeypatch, tmp_path):
    """EZPZ_SETUP_ENV is the documented override; keep it working."""
    f = tmp_path / "my_env.sh"
    f.write_text("echo hi\n")
    monkeypatch.setenv("EZPZ_SETUP_ENV", str(f))
    assert detect_env_setup() == f"source {f}"


def test_inline_commands_still_work(monkeypatch):
    """A non-file EZPZ_SETUP_ENV is used verbatim as shell."""
    monkeypatch.setenv("EZPZ_SETUP_ENV", "module load frameworks")
    assert detect_env_setup() == "module load frameworks"


def test_generated_pbs_script_has_no_curl(monkeypatch):
    """End-to-end: the emitted PBS script is compute-node safe."""
    _clear(monkeypatch)
    script = generate_pbs_script(
        "python3 -m ezpz.examples.test", nodes=2, time="00:30:00"
    )
    assert "curl" not in script
    assert script.startswith(
        "#!/bin/bash --login"
    )  # module needs a login shell
    assert "ezpz_setup_env" in script


def test_scheduler_script_extensions_are_recognised(tmp_path, monkeypatch):
    """`.pbs` and `.sbatch` are scripts, not commands.

    Detection accepted only .sh/.bash/.job, so `ezpz submit run.pbs`
    silently treated the filename as a COMMAND to wrap -- generating a
    job that tried to execute "run.pbs" as a shell word. This repo alone
    carries 17 .pbs and 10 .sbatch files.
    """
    from click.testing import CliRunner

    from ezpz.cli import main

    for suffix in (".sh", ".bash", ".job", ".pbs", ".sbatch", ".slurm"):
        f = tmp_path / f"job{suffix}"
        f.write_text("#!/bin/bash\necho hi\n")
        res = CliRunner().invoke(
            main,
            ["submit", str(f), "-N", "1", "--dry-run", "--scheduler", "PBS"],
        )
        assert res.exit_code == 0, f"{suffix}: {res.output[:200]}"
        assert "ezpz launch" not in res.output, (
            f"{suffix} was wrapped as a command instead of submitted as "
            "a script"
        )
