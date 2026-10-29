"""Fixture-based fetch tests: real log reads, simulated scheduler/SSH transport."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from click.testing import CliRunner

from ezpz.cli import main
from ezpz.cli.fetch_cmd import _resolve_qstat_output_path


def test_pbs_output_path_continuations_do_not_merge_stderr():
    metadata = (
        "    Error_Path = login:/shared/ezpz.e42\n"
        "    Output_Path = login:/shared/very-long-directory/\n"
        "\tezpz.o42\n"
        "    Priority = 0\n"
    )
    assert (
        _resolve_qstat_output_path(metadata)
        == "/shared/very-long-directory/ezpz.o42"
    )


def test_documented_pbs_stdout_snapshot_preserves_duplicates(
    tmp_path, monkeypatch
):
    output = tmp_path / "ezpz.o42"
    lines = ["line one", "duplicate", "duplicate", "last line"]
    output.write_text("\n".join(lines) + "\n")
    real_run = subprocess.run
    calls = []

    def unavailable_iri(*args, **kwargs):
        raise RuntimeError("fixture uses PBS SSH fallback")

    def transport(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            assert "qstat -xf 42" in argv[-1]
            return SimpleNamespace(
                returncode=0,
                stderr="",
                stdout=f"    Output_Path = login:{output}\n",
            )
        # Execute the actual generated file-read command against a real fixture.
        # Only SSH is replaced; this is not a live cluster test.
        assert argv[:4] == ["ssh", "-o", "BatchMode=yes", "sunspot"]
        assert "tail -n 80" in argv[-1]
        assert "uniq" not in argv[-1]
        return real_run(["bash", "-c", argv[-1]], **kwargs)

    monkeypatch.setattr("ezpz.remote.get_job_status_via_iri", unavailable_iri)
    monkeypatch.setattr(subprocess, "run", transport)
    result = CliRunner().invoke(
        main,
        ["fetch", "42", "--remote", "sunspot", "--lines", "80", "--no-dedupe"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == lines
    assert len(calls) == 2
