"""Execute generated environment setup locally without a scheduler."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

from ezpz.cli import main


def _render(*options: str, command: tuple[str, ...] = ()) -> str:
    result = CliRunner().invoke(
        main,
        [
            "submit",
            "--remote",
            "aurora",
            "--dry-run",
            "--no-launch",
            *options,
            "--",
            *(command or ("true",)),
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output[result.output.index("#!/bin/bash") :]


def _run(
    script: str, cwd: Path, **variables: str
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for key in (
        "PBS_O_WORKDIR",
        "PBS_JOBID",
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_ID",
    ):
        env.pop(key, None)
    env.update(variables)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_explicit_venv_keeps_remote_workdir(tmp_path):
    prefix = tmp_path / "env with spaces"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    workdir = tmp_path / "job output"
    workdir.mkdir()
    script = _render(
        "--env",
        str(prefix),
        "--workdir",
        str(workdir),
        command=(
            "python3",
            "-c",
            "import json, os, sys; "
            "print(json.dumps([sys.prefix, os.getcwd()]))",
        ),
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(prefix), str(workdir.resolve())]
    assert "curl" not in script
    assert "pip install" not in script


def test_setup_script_is_sourced_before_selecting_env(tmp_path):
    prefix = tmp_path / "selected env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    setup = tmp_path / "site's setup.sh"
    setup.write_text("export SITE_VALUE='loaded in job shell'\n")
    script = _render(
        "--env-setup-script",
        str(setup),
        "--env",
        str(prefix),
        "--workdir",
        str(tmp_path),
        command=(
            "python3",
            "-c",
            "import json, os, sys; "
            "print(json.dumps([sys.prefix, os.environ['SITE_VALUE']]))",
        ),
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(prefix), "loaded in job shell"]
    assert "curl" not in script


@pytest.mark.parametrize("strict", [(), ("--no-strict",)])
def test_setup_failure_cannot_be_hidden_by_a_later_success(tmp_path, strict):
    setup = tmp_path / "broken.sh"
    setup.write_text("false\nexport MISLEADING_SUCCESS=yes\n")
    script = _render(
        *strict,
        "--env-setup-script",
        str(setup),
        "--workdir",
        str(tmp_path),
        command=("echo", "payload must not run"),
    )
    result = _run(script, tmp_path)
    assert result.returncode != 0
    assert "payload must not run" not in result.stdout


def test_moved_venv_cannot_silently_activate_its_old_prefix(tmp_path):
    original = tmp_path / "original-env"
    moved = tmp_path / "moved-env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(original)],
        check=True,
        timeout=30,
    )
    original.rename(moved)
    script = _render(
        "--env",
        str(moved),
        "--workdir",
        str(tmp_path),
        command=("echo", "payload must not run"),
    )
    result = _run(script, tmp_path)
    assert result.returncode != 0
    assert "activation" in result.stderr
    assert "payload must not run" not in result.stdout


@pytest.mark.parametrize("scheduler", ["PBS", "SLURM"])
def test_local_generation_runs_script_before_env_activation(
    tmp_path, monkeypatch, scheduler
):
    prefix = tmp_path / "local env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    setup = tmp_path / "setup.sh"
    setup.write_text(
        "export SITE_VALUE=loaded\nexport VIRTUAL_ENV=wrong-prefix\n"
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "submit",
            "--scheduler",
            scheduler,
            "--dry-run",
            "--no-launch",
            "--env",
            str(prefix),
            "--env-setup-script",
            str(setup),
            "--",
            "python3",
            "-c",
            "import json, os, sys; print(json.dumps("
            "[sys.prefix, os.environ['VIRTUAL_ENV'], os.environ['SITE_VALUE']]))",
        ],
    )
    assert result.exit_code == 0, result.output
    script = result.output[result.output.index("#!/bin/bash") :].split(
        "\n" + "-" * 60
    )[0]
    run = _run(script, tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == [str(prefix), str(prefix), "loaded"]
    assert "curl" not in script


def test_no_strict_is_restored_for_payload_after_setup(tmp_path):
    setup = tmp_path / "valid.sh"
    setup.write_text("export SETUP_RAN=yes\n")
    script = _render(
        "--no-strict",
        "--env-setup-script",
        str(setup),
        "--workdir",
        str(tmp_path),
        command=("false",),
    )
    result = _run(script + "\necho payload-continued\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "payload-continued"


def test_python_executable_path_selects_its_environment(tmp_path):
    prefix = tmp_path / "python install"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    # A plain installation has bin/python, but need not have bin/activate.
    (prefix / "bin" / "activate").unlink()
    script = _render(
        "--env",
        str(prefix / "bin" / "python3"),
        "--workdir",
        str(tmp_path),
        command=("python3", "-c", "import sys; print(sys.prefix)"),
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(prefix)


@pytest.mark.parametrize("select_prefix", [False, True])
def test_versioned_only_python_installation_is_accepted(
    tmp_path, select_prefix
):
    prefix = tmp_path / "versioned-only"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    interpreter = (
        prefix
        / "bin"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    interpreter.unlink(missing_ok=True)
    interpreter.symlink_to(sys.executable)
    for name in ("python", "python3"):
        (prefix / "bin" / name).unlink(missing_ok=True)
    command = (str(interpreter), "-c", "import sys; print(sys.prefix)")
    direct = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    assert direct.stdout.strip() == str(prefix)
    script = _render(
        "--env",
        str(prefix if select_prefix else interpreter),
        "--workdir",
        str(tmp_path),
        command=command,
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(prefix)


def test_conda_uses_shell_hook_and_activation(tmp_path):
    prefix = tmp_path / "conda env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    (prefix / "conda-meta").mkdir()
    setup = tmp_path / "conda-init.sh"
    # Exercise conda's shell protocol without requiring conda on the test host.
    setup.write_text(
        "conda() {\n"
        "  if [[ $1 == shell.bash && $2 == hook ]]; then\n"
        "    echo 'export CONDA_HOOK_RAN=yes'\n"
        "  elif [[ $1 == activate ]]; then\n"
        '    export CONDA_PREFIX="$2"\n'
        "    unset VIRTUAL_ENV\n"
        "  else return 1; fi\n"
        "}\n"
    )
    script = _render(
        "--env",
        str(prefix),
        "--env-setup-script",
        str(setup),
        "--workdir",
        str(tmp_path),
        command=(
            "python3",
            "-c",
            "import json, os; print(json.dumps("
            "[os.environ.get('CONDA_PREFIX'), "
            "os.environ.get('CONDA_HOOK_RAN')]))",
        ),
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(prefix), "yes"]


@pytest.mark.parametrize("remote", [(), ("--remote", "aurora")])
@pytest.mark.parametrize("option", ["--env", "--env-setup-script"])
def test_script_mode_rejects_ignored_env_options(tmp_path, remote, option):
    job = tmp_path / "job.sh"
    job.write_text("#!/bin/bash\necho original script\n")
    result = CliRunner().invoke(
        main,
        [
            "submit",
            *remote,
            "--scheduler",
            "PBS",
            "--dry-run",
            option,
            "/target/path",
            str(job),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "generated commands" in result.output


@pytest.mark.parametrize("option", ["--env", "--env-setup-script"])
def test_empty_environment_options_are_usage_errors(option):
    result = CliRunner().invoke(
        main,
        [
            "submit",
            "--remote",
            "aurora",
            "--dry-run",
            option,
            "",
            "--",
            "true",
        ],
    )
    assert result.exit_code == 2
    assert "non-empty" in result.output


def test_auto_workdir_failure_stops_even_without_strict(tmp_path):
    setup = tmp_path / "setup.sh"
    setup.write_text("export SITE_VALUE=ready\n")
    not_directory = tmp_path / "file"
    not_directory.write_text("not a directory")
    script = _render(
        "--no-strict",
        "--env-setup-script",
        str(setup),
        command=("echo", "payload must not run"),
    )
    result = _run(
        script, tmp_path, PBS_O_WORKDIR=str(not_directory), PBS_JOBID="42"
    )
    assert result.returncode != 0
    assert "payload must not run" not in result.stdout


def test_explicit_env_does_not_launch_ezpz_from_another_environment(tmp_path):
    prefix = tmp_path / "empty env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    ambient = tmp_path / "other env" / "bin"
    ambient.mkdir(parents=True)
    executable = ambient / "ezpz"
    executable.write_text("#!/bin/bash\necho wrong environment launched\n")
    executable.chmod(0o755)
    script = _render(
        "--launch",
        "--env",
        str(prefix),
        "--workdir",
        str(tmp_path),
    )
    result = _run(script, tmp_path, PATH=f"{ambient}:{os.environ['PATH']}")
    assert result.returncode != 0
    assert "wrong environment launched" not in result.stdout
    assert "ezpz" in result.stderr and "--env" in result.stderr


@pytest.mark.parametrize(
    "host,base_var,id_var",
    [
        ("aurora", "PBS_O_WORKDIR", "PBS_JOBID"),
        ("perlmutter", "SLURM_SUBMIT_DIR", "SLURM_JOB_ID"),
    ],
)
@pytest.mark.parametrize("use_home", [False, True])
def test_custom_setup_keeps_job_unique_workdirs(
    tmp_path, host, base_var, id_var, use_home
):
    setup = tmp_path / "setup.sh"
    setup.write_text("export SETUP_RAN=yes\n")
    base = tmp_path / "shared submit"
    base.mkdir()
    env = {"HOME": str(base), id_var: "81234"}
    if not use_home:
        env[base_var] = str(base)
    script = _render(
        "--remote",
        host,
        "--env-setup-script",
        str(setup),
        command=("pwd",),
    )
    result = _run(script, tmp_path, **env)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == base / "ezpz-81234"
    assert "curl" not in script


def test_relative_setup_script_cannot_be_shadowed_by_path(tmp_path):
    setup = tmp_path / "setup.sh"
    setup.write_text("export PICKED=job-workdir\n")
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "setup.sh").write_text("export PICKED=ambient-PATH\n")
    script = _render(
        "--env-setup-script",
        "setup.sh",
        "--workdir",
        str(tmp_path),
        command=("bash", "-c", "printf '%s\\n' \"$PICKED\""),
    )
    result = _run(script, tmp_path, PATH=f"{ambient}:{os.environ['PATH']}")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "job-workdir"


def test_relative_env_stays_anchored_when_setup_changes_directory(tmp_path):
    prefix = tmp_path / "env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "setup.sh").write_text(f"cd {shlex.quote(str(elsewhere))}\n")
    script = _render(
        "--env",
        "env",
        "--env-setup-script",
        "setup.sh",
        "--workdir",
        str(tmp_path),
        command=("python3", "-c", "import sys; print(sys.prefix)"),
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(prefix)


def test_perlmutter_does_not_inject_alcf_proxies():
    script = _render(
        "--remote",
        "perlmutter",
        "--env-setup-script",
        "/remote/setup.sh",
    )
    assert "proxy.alcf.anl.gov" not in script


def test_relative_python_prefix_survives_payload_chdir(tmp_path):
    prefix = tmp_path / "python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        timeout=30,
    )
    (prefix / "bin" / "activate").unlink()
    script = _render(
        "--env",
        "python",
        "--workdir",
        str(tmp_path),
        command=(
            "bash",
            "-c",
            "cd ..; python3 -c 'import sys; print(sys.prefix)'",
        ),
    )
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(prefix)
