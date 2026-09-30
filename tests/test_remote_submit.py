"""Remote submission must never turn a scheduler rejection into a retry.

The fallback rule is the whole point of these tests: ``backend="auto"``
may only try IRI when **SSH itself** could not connect (exit 255). Any
other non-zero exit came from the remote ``qsub``/``sbatch``, and
resubmitting it down a second path either reproduces the rejection with
a worse message or -- if the first submission landed before the non-zero
exit -- **submits the job twice**.

Real rejections seen while developing this, all exit 1, none of which
should ever trigger a fallback:

* ``No active allocation found for project ...``
* ``Invalid filesystem identifiers gila``
* ``would exceed queue generic's per-user limit of jobs in 'Q' state``
"""

from __future__ import annotations

import subprocess

import pytest

from ezpz.remote import (
    IRI_MACHINES,
    SSH_TRANSPORT_FAILURE,
    RemoteResult,
    RemoteSubmitError,
    has_iri_backend,
    scheduler_for,
    submit_remote,
)

SCRIPT = "#!/bin/bash\necho hello\n"


class _Proc:
    def __init__(self, rc: int, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = rc, out, err


# ── scheduler / backend detection ────────────────────────────────────────


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("aurora", "PBS"),
        ("aurora-uan-0010", "PBS"),
        ("sunspot", "PBS"),
        ("polaris", "PBS"),
        ("perlmutter", "SLURM"),
        ("frontier", "SLURM"),
    ],
)
def test_scheduler_inferred_from_host(host, expected):
    assert scheduler_for(host) == expected


def test_unknown_host_defaults_to_pbs_with_a_warning(caplog):
    """Guessing is acceptable here only because it is announced."""
    import logging

    with caplog.at_level(logging.WARNING, logger="ezpz.remote"):
        assert scheduler_for("some-new-cluster") == "PBS"
    assert "Unknown remote host" in caplog.text


def test_iri_backend_only_for_supported_machines():
    """Sunspot and Perlmutter are not served by the ALCF IRI API."""
    assert has_iri_backend("aurora")
    assert has_iri_backend("polaris")
    assert not has_iri_backend("sunspot")
    assert not has_iri_backend("perlmutter")
    assert "sunspot" not in IRI_MACHINES
    assert "perlmutter" not in IRI_MACHINES


# ── the fallback rule ────────────────────────────────────────────────────


def test_success_does_not_touch_iri(monkeypatch):
    called = {"iri": 0}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Proc(0, "12345.aurora\n")
    )
    monkeypatch.setattr(
        "ezpz.remote.submit_via_iri",
        lambda *a, **k: called.__setitem__("iri", called["iri"] + 1),
    )
    res = submit_remote("aurora", SCRIPT, backend="auto")
    assert res.job_id == "12345.aurora"
    assert res.backend == "ssh"
    assert called["iri"] == 0


def test_scheduler_rejection_does_NOT_fall_back(monkeypatch):
    """The case that would double-submit. Exit 1 must stop."""
    called = {"iri": 0}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Proc(
            1, "", "qsub: Invalid filesystem identifiers gila"
        ),
    )

    def _boom(*a, **k):
        called["iri"] += 1
        return RemoteResult("should-not-happen", "iri", 0)

    monkeypatch.setattr("ezpz.remote.submit_via_iri", _boom)
    res = submit_remote("aurora", SCRIPT, backend="auto")
    assert called["iri"] == 0, "a qsub rejection must never be retried via IRI"
    assert res.returncode == 1
    assert res.backend == "ssh"
    assert "Invalid filesystem" in res.stderr


def test_ssh_transport_failure_DOES_fall_back(monkeypatch):
    """Exit 255 is SSH's own failure: the job was never submitted."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Proc(SSH_TRANSPORT_FAILURE, "", "Connection closed"),
    )
    monkeypatch.setattr(
        "ezpz.remote.submit_via_iri",
        lambda *a, **k: RemoteResult("iri-999", "iri", 0),
    )
    res = submit_remote("aurora", SCRIPT, backend="auto", iri_kwargs={})
    assert res.backend == "iri"
    assert res.job_id == "iri-999"


def test_no_fallback_when_machine_has_no_iri(monkeypatch, caplog):
    """Sunspot: 255 is fatal, and the message must say why."""
    import logging

    called = {"iri": 0}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Proc(SSH_TRANSPORT_FAILURE, "", "Connection closed"),
    )
    monkeypatch.setattr(
        "ezpz.remote.submit_via_iri",
        lambda *a, **k: called.__setitem__("iri", called["iri"] + 1),
    )
    with caplog.at_level(logging.ERROR, logger="ezpz.remote"):
        res = submit_remote("sunspot", SCRIPT, backend="auto")
    assert called["iri"] == 0
    assert res.returncode == SSH_TRANSPORT_FAILURE
    assert "no ALCF IRI backend" in caplog.text


def test_explicit_ssh_backend_never_falls_back(monkeypatch):
    called = {"iri": 0}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Proc(SSH_TRANSPORT_FAILURE, "", "Connection closed"),
    )
    monkeypatch.setattr(
        "ezpz.remote.submit_via_iri",
        lambda *a, **k: called.__setitem__("iri", called["iri"] + 1),
    )
    res = submit_remote("aurora", SCRIPT, backend="ssh")
    assert called["iri"] == 0
    assert res.backend == "ssh"


def test_unknown_backend_is_rejected():
    with pytest.raises(RemoteSubmitError, match="unknown backend"):
        submit_remote("aurora", SCRIPT, backend="carrier-pigeon")


def test_iri_for_unsupported_machine_raises():
    from ezpz.remote import submit_via_iri

    with pytest.raises(RemoteSubmitError, match="not served by the ALCF IRI"):
        submit_via_iri(
            "sunspot",
            command="true",
            nodes=1,
            duration_seconds=60,
            queue="workq",
            account="datascience",
        )


# ── ssh mechanics ────────────────────────────────────────────────────────


def test_ssh_uses_batchmode_and_pipes_the_script(monkeypatch):
    """No interactive prompt, and the script arrives over stdin.

    stdin delivery avoids a second connection, which matters when each
    one is gated by MFA.
    """
    seen = {}

    def _capture(argv, **kw):
        seen["argv"], seen["input"] = argv, kw.get("input")
        return _Proc(0, "1.host\n")

    monkeypatch.setattr(subprocess, "run", _capture)
    submit_remote("aurora", SCRIPT, backend="ssh")
    assert "BatchMode=yes" in seen["argv"]
    assert seen["input"] == SCRIPT
    assert "qsub" in seen["argv"][-1]


def test_slurm_host_uses_sbatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: (seen.update(argv=argv), _Proc(0, "99\n"))[1],
    )
    submit_remote("perlmutter", SCRIPT, backend="ssh")
    assert "sbatch" in seen["argv"][-1]
    assert "qsub" not in seen["argv"][-1]


def test_missing_ssh_binary_reports_transport_failure(monkeypatch):
    """A missing ssh is a transport problem, so 255 is the right code."""

    def _missing(*a, **k):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(subprocess, "run", _missing)
    res = submit_remote("aurora", SCRIPT, backend="ssh")
    assert res.returncode == SSH_TRANSPORT_FAILURE


# ── what the generated remote script must and must not contain ───────────


def _gen(*extra: str) -> str:
    """Render a remote script via --dry-run."""
    from click.testing import CliRunner

    from ezpz.cli import main

    res = CliRunner().invoke(
        main,
        [
            "submit",
            "--remote",
            "aurora",
            "--dry-run",
            "-N",
            "2",
            "-q",
            "debug",
            "-A",
            "proj",
            *extra,
            "--",
            "python3",
            "t.py",
        ],
    )
    assert res.exit_code == 0, res.output
    return res.output


def test_remote_script_does_not_leak_the_local_cwd():
    """A `cd` into a local path breaks the job on the target."""
    import os

    assert f"cd {os.getcwd()}" not in _gen()


def test_remote_env_setup_resolves_on_the_target():
    """Not a bare curl: compute nodes have no outbound route (#268).

    The script must locate ezpz's own utils.sh ON the target, keeping
    the network fetch only as a fallback.
    """
    out = _gen()
    assert "import ezpz" in out, "should resolve utils.sh on the target"
    assert "ezpz_setup_env" in out


def test_remote_command_is_shell_quoted():
    """Arguments containing spaces must survive the round trip."""
    from click.testing import CliRunner

    from ezpz.cli import main

    res = CliRunner().invoke(
        main,
        [
            "submit",
            "--remote",
            "aurora",
            "--dry-run",
            "-N",
            "1",
            "-q",
            "debug",
            "-A",
            "p",
            "--",
            "python3",
            "t.py",
            "--msg",
            "a b",
        ],
    )
    assert "'a b'" in res.output, (
        "a naive space-join would turn one argument into two"
    )


def test_strict_is_honoured_for_remote_pbs():
    """--no-strict was silently ignored on the PBS path."""
    assert "set -eo pipefail" in _gen()
    assert "set -eo pipefail" not in _gen("--no-strict")


def test_generated_remote_script_is_valid_bash(tmp_path):
    """A script that does not parse fails only once it is on the cluster."""
    import subprocess

    body = "\n".join(
        ln for ln in _gen().splitlines() if not ln.startswith("# would submit")
    )
    f = tmp_path / "gen.sh"
    f.write_text(body)
    proc = subprocess.run(
        ["bash", "-n", str(f)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
