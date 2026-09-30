"""`ezpz update` must not run where the network is unavailable.

Compute nodes on Aurora/Polaris/Sunspot have no outbound route. A
download there hangs ~270 s and then leaves the environment silently
unconfigured -- the failure surfaces much later as `CONDA_PREFIX still
not set` or undefined `ezpz_*` functions. Refusing up front turns four
minutes of confusion into one line.

The other property worth pinning is that a *partial* download never
replaces a good `utils.sh`: a truncated file passes a non-empty check
and then fails as "every function is undefined", which is strictly
worse than a clean error.
"""

from __future__ import annotations


import pytest
from click.testing import CliRunner

from ezpz.cli.update_cmd import _in_batch_job, _pip_install_cmd, update_cmd


@pytest.fixture
def runner():
    return CliRunner()


# ── refusing to run on a compute node ────────────────────────────────────


def test_detects_pbs_job(monkeypatch):
    monkeypatch.setenv("PBS_NODEFILE", "/var/spool/pbs/aux/123")
    assert _in_batch_job() is True


def test_detects_slurm_job(monkeypatch):
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert _in_batch_job() is True


def test_no_batch_vars_means_login_node(monkeypatch):
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert _in_batch_job() is False


def test_refuses_inside_a_job(runner, monkeypatch):
    """The whole point: fail in one line, not after a 270 s hang."""
    monkeypatch.setenv("PBS_NODEFILE", "/var/spool/pbs/aux/123")
    result = runner.invoke(update_cmd, [])
    assert result.exit_code != 0
    assert "compute nodes have no outbound network" in result.output


def test_dry_run_is_allowed_inside_a_job(runner, monkeypatch):
    """--dry-run touches nothing, so the guard should not block it."""
    monkeypatch.setenv("PBS_NODEFILE", "/var/spool/pbs/aux/123")
    result = runner.invoke(update_cmd, ["--dry-run"])
    assert result.exit_code == 0


# ── what it would run ────────────────────────────────────────────────────


def test_dry_run_changes_nothing(runner, monkeypatch, tmp_path):
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.setenv("EZPZ_SHARE_DIR", str(tmp_path))
    result = runner.invoke(update_cmd, ["--dry-run"])
    assert result.exit_code == 0
    assert "dry run" in result.output
    assert not (tmp_path / "utils.sh").exists()


def test_ref_is_passed_to_the_installer(runner, monkeypatch, tmp_path):
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.setenv("EZPZ_SHARE_DIR", str(tmp_path))
    result = runner.invoke(update_cmd, ["--dry-run", "--ref", "v0.29.3"])
    assert "@v0.29.3" in result.output


def test_mutually_exclusive_flags_rejected(runner):
    result = runner.invoke(update_cmd, ["--utils-only", "--package-only"])
    assert result.exit_code != 0


def test_utils_only_skips_the_package(runner, monkeypatch, tmp_path):
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.setenv("EZPZ_SHARE_DIR", str(tmp_path))
    result = runner.invoke(update_cmd, ["--dry-run", "--utils-only"])
    assert "pip install" not in result.output
    assert "utils.sh" in result.output


def test_package_only_skips_utils(runner, monkeypatch, tmp_path):
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.setenv("EZPZ_SHARE_DIR", str(tmp_path))
    result = runner.invoke(update_cmd, ["--dry-run", "--package-only"])
    assert "utils.sh" not in result.output


def test_prefers_uv_when_available(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    assert _pip_install_cmd("pkg")[:3] == ["uv", "pip", "install"]


def test_falls_back_to_pip(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    cmd = _pip_install_cmd("pkg")
    assert cmd[1:4] == ["-m", "pip", "install"]


# ── never replace a good utils.sh with a broken one ──────────────────────


def test_syntax_error_does_not_replace_the_existing_file(
    runner, monkeypatch, tmp_path
):
    """A truncated download must not overwrite a working utils.sh.

    The bad outcome is not a failed update -- it is a *successful* one
    that installs a file which fails later as "every ezpz_* function is
    undefined".
    """
    import sys

    mod = sys.modules["ezpz.cli.update_cmd"]

    good = tmp_path / "utils.sh"
    good.write_text("#!/usr/bin/env bash\necho fine\n")

    class _Resp:
        def read(self):
            return b"#!/usr/bin/env bash\nif then fi done\n"  # invalid

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    with pytest.raises(Exception):
        mod._refresh_utils("https://example.invalid/utils.sh", good)
    assert good.read_text() == "#!/usr/bin/env bash\necho fine\n"


def test_non_script_response_is_rejected(monkeypatch, tmp_path):
    """An HTML error page must not be installed as utils.sh."""
    import sys

    mod = sys.modules["ezpz.cli.update_cmd"]

    class _Resp:
        def read(self):
            return b"<!DOCTYPE html><html>404</html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    dest = tmp_path / "utils.sh"
    with pytest.raises(Exception, match="did not return a shell script"):
        mod._refresh_utils("https://example.invalid/utils.sh", dest)
    assert not dest.exists()


def test_valid_download_is_installed(monkeypatch, tmp_path):
    import sys

    mod = sys.modules["ezpz.cli.update_cmd"]

    # Must define every required entry point: the completeness check
    # exists precisely to reject a file that lacks them.
    body = (
        b"#!/usr/bin/env bash\n"
        b"ezpz_setup_env() { :; }\n"
        b"ezpz_setup_python() { :; }\n"
        b"ezpz_get_machine_name() { :; }\n"
    )

    class _Resp:
        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    dest = tmp_path / "sub" / "utils.sh"
    mod._refresh_utils("https://example.invalid/utils.sh", dest)
    assert dest.read_bytes() == body
    assert not dest.with_suffix(".part").exists()


# ── review follow-ups ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "var",
    ["PBS_NODEFILE", "PBS_JOBID", "SLURM_JOB_ID", "SLURM_JOBID"],
)
def test_all_scheduler_job_vars_are_detected(monkeypatch, var):
    """Every spelling the rest of the codebase recognises.

    `PBS_JOBID` can be set in a compute-node subprocess where
    `PBS_NODEFILE` is not (ezpz/pbs.py), and SLURM uses both
    `SLURM_JOB_ID` and the older `SLURM_JOBID` (ezpz/slurm.py). Missing
    any of them starts the network operation the guard prevents.
    """
    for v in ("PBS_NODEFILE", "PBS_JOBID", "SLURM_JOB_ID", "SLURM_JOBID"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "set")
    assert _in_batch_job() is True


def test_uv_is_pointed_at_the_running_interpreter(monkeypatch):
    """`uv pip install` targets a venv found from the CWD by default.

    Without `--python` it can fail outside a project, or silently
    upgrade an unrelated `.venv` instead of the environment running
    this command.
    """
    import sys

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    cmd = _pip_install_cmd("pkg")
    assert "--python" in cmd
    assert cmd[cmd.index("--python") + 1] == sys.executable


def test_truncated_but_valid_shell_is_rejected(tmp_path, monkeypatch):
    """`bash -n` cannot detect truncation; content must be checked.

    A response cut at a syntactically complete boundary parses fine --
    the first ten lines of the real utils.sh do. Installing that would
    replace a working 140 KB file with a stub whose functions are all
    missing, which is precisely what this command promises not to do.
    """
    import sys

    mod = sys.modules["ezpz.cli.update_cmd"]

    good = tmp_path / "utils.sh"
    good.write_bytes(b"#!/usr/bin/env bash\nezpz_setup_env() { :; }\n")

    # Valid shell, but nothing a caller needs.
    partial = b"#!/usr/bin/env bash\n# header only\nset -o pipefail\n"
    import subprocess

    stub = tmp_path / "stub.sh"
    stub.write_bytes(partial)
    assert subprocess.run(["bash", "-n", str(stub)]).returncode == 0, (
        "precondition: the partial file must PASS bash -n"
    )

    class _Resp:
        def read(self):
            return partial

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    with pytest.raises(Exception, match="truncated"):
        mod._refresh_utils("https://example.invalid/utils.sh", good)
    assert b"ezpz_setup_env" in good.read_bytes(), (
        "a rejected download must leave the existing file untouched"
    )
