"""Tests for ``ezpz.utils.yeet_env``.

Covers argument parsing, chunked broadcasting, file transfer,
tarball creation, and end-to-end execution with mocked distributed ops.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

import ezpz.utils.yeet_env as yeet


# ===================================================================
# parse_args
# ===================================================================


class TestParseArgs:
    """Tests for ``parse_args``."""

    def test_defaults(self):
        """Parses required --src and checks defaults."""
        args = yeet.parse_args(["--src", "/data/model.tar.gz"])
        assert args.src == "/data/model.tar.gz"
        assert args.dst is None
        assert args.decompress is True
        assert args.flags == "xf"
        assert args.chunk_size == yeet.CHUNK_SIZE
        assert args.overwrite is False
        assert args.worker is False

    def test_all_flags(self):
        """Parses all flags explicitly."""
        args = yeet.parse_args([
            "--src", "/data/model.tar.gz",
            "--dst", "/tmp/model.tar.gz",
            "--no-decompress",
            "--flags", "xzf",
            "--chunk-size", "1024",
            "--overwrite",
            "--worker",
        ])
        assert args.src == "/data/model.tar.gz"
        assert args.dst == "/tmp/model.tar.gz"
        assert args.decompress is False
        assert args.flags == "xzf"
        assert args.chunk_size == 1024
        assert args.overwrite is True
        assert args.worker is True

    def test_missing_required_src(self, capsys):
        """Raises SystemExit when --src is missing."""
        with pytest.raises(SystemExit):
            yeet.parse_args([])
        # argparse prints usage to stderr — just verify it was captured
        captured = capsys.readouterr()
        assert "--src" in captured.err


# ===================================================================
# bcast_chunk
# ===================================================================


class TestBcastChunk:
    """Tests for ``bcast_chunk``."""

    @patch("ezpz.distributed.broadcast")
    @patch("ezpz.get_rank", return_value=0)
    @patch("tqdm.trange", side_effect=lambda n, **kw: range(n))
    def test_rank0_broadcasts_data(self, _trange, _rank, mock_bcast):
        """Rank 0 reads data and broadcasts in chunks."""
        data = b"hello world!!"  # 13 bytes
        chunk_size = 5

        mock_bcast.side_effect = lambda obj, root=0: obj
        result = yeet.bcast_chunk(data, chunk_size)
        assert isinstance(result, bytearray)
        assert bytes(result) == data

    @patch("ezpz.distributed.broadcast")
    @patch("ezpz.get_rank", return_value=1)
    @patch("tqdm.trange", side_effect=lambda n, **kw: range(n))
    def test_non_rank0_receives_data(self, _trange, _rank, mock_bcast):
        """Non-rank-0 receives broadcast data (size then chunks)."""
        data = b"abcde"
        chunk_size = 3

        call_count = 0

        def bcast_side_effect(obj, root=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return len(data)
            idx = call_count - 2
            start = idx * chunk_size
            end = min(start + chunk_size, len(data))
            return data[start:end]

        mock_bcast.side_effect = bcast_side_effect
        result = yeet.bcast_chunk(None, chunk_size)
        assert isinstance(result, bytearray)
        assert bytes(result) == data

    @patch("ezpz.distributed.broadcast")
    @patch("ezpz.get_rank", return_value=0)
    @patch("tqdm.trange", side_effect=lambda n, **kw: range(n))
    def test_exact_chunk_boundary(self, _trange, _rank, mock_bcast):
        """Data that exactly divides into chunks."""
        data = b"123456"
        chunk_size = 3

        mock_bcast.side_effect = lambda obj, root=0: obj
        result = yeet.bcast_chunk(data, chunk_size)
        assert bytes(result) == data


# ===================================================================
# transfer
# ===================================================================


class TestTransfer:
    """Tests for ``transfer``."""

    @patch("ezpz.utils.yeet_env.subprocess.run")
    @patch("ezpz.distributed.barrier")
    @patch("ezpz.get_local_rank", return_value=0)
    @patch("ezpz.get_rank", return_value=0)
    @patch("ezpz.get_timestamp", return_value="2024-01-01")
    def test_rank0_reads_broadcasts_writes_decompresses(
        self, _ts, _rank, _local_rank, _barrier, mock_subproc, tmp_path
    ):
        """Rank 0 reads file, broadcasts, writes at dst, decompresses."""
        src = tmp_path / "source.tar.gz"
        dst = tmp_path / "dest.tar.gz"
        src.write_bytes(b"fake tarball data")

        with patch.object(yeet, "bcast_chunk", return_value=bytearray(b"fake tarball data")):
            yeet.transfer(
                src=str(src),
                dst=str(dst),
                decompress=True,
                chunk_size=1024,
                flags="xf",
            )
        assert dst.exists()
        assert dst.read_bytes() == b"fake tarball data"
        # subprocess.run should be called with a list (no shell injection)
        mock_subproc.assert_called_once()
        call_args = mock_subproc.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args[0] == "tar"
        assert str(dst) in call_args

    @patch("ezpz.utils.yeet_env.subprocess.run")
    @patch("ezpz.get_rank", return_value=0)
    @patch("ezpz.get_timestamp", return_value="2024-01-01")
    def test_no_decompress(self, _ts, _rank, mock_subproc, tmp_path):
        """When decompress=False, does not call tar."""
        src = tmp_path / "source.tar.gz"
        dst = tmp_path / "dest.tar.gz"
        src.write_bytes(b"data")

        with patch.object(yeet, "bcast_chunk", return_value=bytearray(b"data")):
            yeet.transfer(
                src=str(src),
                dst=str(dst),
                decompress=False,
                chunk_size=1024,
                flags="xf",
            )
        assert dst.exists()
        mock_subproc.assert_not_called()

    @patch("ezpz.utils.yeet_env.subprocess.run")
    @patch("ezpz.distributed.barrier")
    @patch("ezpz.get_local_rank", return_value=1)
    @patch("ezpz.get_rank", return_value=1)
    @patch("ezpz.get_timestamp", return_value="2024-01-01")
    def test_non_rank0_skips_decompression(
        self, _ts, _rank, _local_rank, _barrier, mock_subproc, tmp_path
    ):
        """Non-local-rank-0 writes file but does not decompress."""
        src = tmp_path / "source.tar.gz"
        dst = tmp_path / "dest.tar.gz"
        src.write_bytes(b"data")

        with patch.object(yeet, "bcast_chunk", return_value=bytearray(b"data")):
            yeet.transfer(
                src=str(src),
                dst=str(dst),
                decompress=True,
                chunk_size=1024,
                flags="xf",
            )
        assert dst.exists()
        mock_subproc.assert_not_called()


# ===================================================================
# _create_tarball_if_needed
# ===================================================================


class TestCreateTarballIfNeeded:
    """Tests for ``_create_tarball_if_needed``."""

    @patch("ezpz.get_rank", return_value=0)
    def test_creates_tarball_from_directory(self, _rank, tmp_path, monkeypatch):
        """Creates a tarball when src is a directory."""
        src_dir = tmp_path / "mydata"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")
        monkeypatch.chdir(tmp_path)

        tarball = tmp_path / "mydata.tar.gz"

        def fake_create_tarball(src_path):
            tarball.write_bytes(b"fake tar")
            return tarball

        with patch("ezpz.utils.create_tarball", side_effect=fake_create_tarball):
            result = yeet._create_tarball_if_needed(str(src_dir), overwrite=False)
        assert result == tarball

    @patch("ezpz.get_rank", return_value=0)
    def test_existing_tarball_raises_without_overwrite(self, _rank, tmp_path, monkeypatch):
        """Raises FileExistsError when tarball exists and overwrite=False."""
        src_dir = tmp_path / "mydata"
        src_dir.mkdir()
        tarball = tmp_path / "mydata.tar.gz"
        tarball.write_bytes(b"existing")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileExistsError, match="already exists"):
            yeet._create_tarball_if_needed(str(src_dir), overwrite=False)

    @patch("ezpz.get_timestamp", return_value="2024-01-01")
    @patch("ezpz.get_rank", return_value=0)
    def test_overwrite_renames_tarball_not_source(self, _rank, _ts, tmp_path, monkeypatch):
        """With overwrite=True, renames the existing tarball, not the source dir."""
        src_dir = tmp_path / "mydata"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")
        tarball = tmp_path / "mydata.tar.gz"
        tarball.write_bytes(b"old tarball")
        monkeypatch.chdir(tmp_path)

        def fake_create_tarball(src_path):
            tarball.write_bytes(b"new tarball")
            return tarball

        with patch("ezpz.utils.create_tarball", side_effect=fake_create_tarball):
            result = yeet._create_tarball_if_needed(str(src_dir), overwrite=True)

        # Source directory should still exist (not renamed)
        assert src_dir.exists()
        # New tarball should be the result
        assert result == tarball


# ===================================================================
# execute_transfer
# ===================================================================


class TestExecuteTransfer:
    """Tests for ``execute_transfer``."""

    @patch.object(yeet, "transfer")
    @patch("ezpz.get_rank", return_value=0)
    def test_file_source(self, _rank, mock_transfer, tmp_path):
        """Transfers a file directly without tarball creation."""
        src = tmp_path / "model.tar.gz"
        src.write_bytes(b"model data")
        dst = tmp_path / "dest.tar.gz"

        args = argparse.Namespace(
            src=str(src),
            dst=str(dst),
            decompress=True,
            chunk_size=1024,
            flags="xf",
            overwrite=False,
        )
        result = yeet.execute_transfer(args)
        assert result == 0
        mock_transfer.assert_called_once()
        call_kwargs = mock_transfer.call_args[1]
        assert call_kwargs["src"] == str(src)
        assert call_kwargs["dst"] == str(dst)

    def test_nonexistent_source_raises(self, tmp_path):
        """Raises AssertionError when source doesn't exist."""
        args = argparse.Namespace(
            src=str(tmp_path / "nonexistent"),
            dst=str(tmp_path / "dest"),
            decompress=True,
            chunk_size=1024,
            flags="xf",
            overwrite=False,
        )
        with pytest.raises(AssertionError, match="does not exist"):
            yeet.execute_transfer(args)

    @patch.object(yeet, "transfer")
    @patch.object(yeet, "_create_tarball_if_needed")
    @patch("ezpz.get_rank", return_value=0)
    def test_directory_source_creates_tarball(
        self, _rank, mock_tarball, mock_transfer, tmp_path
    ):
        """Directory source triggers tarball creation."""
        src_dir = tmp_path / "mydata"
        src_dir.mkdir()
        tarball = tmp_path / "mydata.tar.gz"
        tarball.write_bytes(b"tar content")
        mock_tarball.return_value = tarball

        args = argparse.Namespace(
            src=str(src_dir),
            dst=str(tmp_path / "dest.tar.gz"),
            decompress=True,
            chunk_size=1024,
            flags="xf",
            overwrite=False,
        )
        result = yeet.execute_transfer(args)
        assert result == 0
        mock_tarball.assert_called_once()

    @patch.object(yeet, "transfer")
    @patch("ezpz.get_rank", return_value=0)
    def test_default_dst(self, _rank, mock_transfer, tmp_path):
        """When dst is None, uses /tmp with derived name."""
        src = tmp_path / "model.tar.gz"
        src.write_bytes(b"data")

        args = argparse.Namespace(
            src=str(src),
            dst=None,
            decompress=True,
            chunk_size=1024,
            flags="xf",
            overwrite=False,
        )
        result = yeet.execute_transfer(args)
        assert result == 0
        call_kwargs = mock_transfer.call_args[1]
        assert "/tmp/" in call_kwargs["dst"]
