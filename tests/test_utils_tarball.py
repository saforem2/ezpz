"""Tests covering tarball helper utilities."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

import ezpz.utils as utils
from ezpz.utils import check_for_tarball, make_tarfile


def _create_fake_env(env_dir: Path) -> None:
    (env_dir / "bin").mkdir(parents=True, exist_ok=True)
    (env_dir / "bin" / "python").write_text("#!/usr/bin/env python3\n", encoding="utf-8")


def test_make_tarfile_creates_archive(tmp_path, monkeypatch):
    env_dir = tmp_path / "fake-env"
    _create_fake_env(env_dir)
    monkeypatch.chdir(tmp_path)
    artifact = make_tarfile("fake-env.tar.gz", env_dir)
    archive_path = tmp_path / artifact
    assert archive_path.exists()
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    assert "fake-env/bin/python" in names


def test_check_for_tarball_overwrite(tmp_path, monkeypatch):
    env_dir = tmp_path / "fake-env"
    _create_fake_env(env_dir)
    monkeypatch.chdir(tmp_path)
    stale_tar = Path("/tmp") / "fake-env.tar.gz"
    stale_tar.write_bytes(b"stale")
    try:
        result = check_for_tarball(env_prefix=env_dir, overwrite=True)
        archive_path = (Path.cwd() / result).resolve()
        assert archive_path.exists()
        assert archive_path.name == "fake-env.tar.gz"
        assert not stale_tar.exists()
    finally:
        if stale_tar.exists():
            stale_tar.unlink()


def test_check_for_tarball_reuses_existing(tmp_path, monkeypatch):
    env_dir = tmp_path / "fake-env"
    _create_fake_env(env_dir)
    monkeypatch.chdir(tmp_path)
    call_counter = {"invocations": 0}

    def fake_make_tarfile(output_filename, source_dir):
        call_counter["invocations"] += 1
        return original_make_tarfile(output_filename, source_dir)

    original_make_tarfile = make_tarfile
    monkeypatch.setattr(utils, "make_tarfile", fake_make_tarfile)
    first = check_for_tarball(env_prefix=env_dir)
    assert call_counter["invocations"] == 1
    second = check_for_tarball(env_prefix=env_dir)
    assert call_counter["invocations"] == 1
    assert first == second
