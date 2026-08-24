"""Recovering the rank-to-host map that every ezpz log already carries."""

from __future__ import annotations

import pytest

from ezpz.failover.topology import (
    hosts_for_ranks,
    parse_rank_host_map,
    parse_rank_host_map_file,
)

# Verbatim from Sunspot job 12473704's attempt-1.log, ANSI stripped.
REAL = (
    "[2026-08-23 17:28:57][I][ezpz/distributed:824:setup_torch] "
    "['x1921c1s0b0n0'][device='xpu'][node=0/1][local_rank=00/11][rank=00/23]\n"
    "[2026-08-23 17:28:57][I][ezpz/distributed:824:setup_torch] "
    "['x1921c1s1b0n0'][device='xpu'][node=1/1][local_rank=00/11][rank=12/23]\n"
)

# The same two lines as they ACTUALLY appear on disk: ezpz colorizes, and
# the escapes land inside the tokens (`[<esc>[94mnode<esc>[0m=`). This
# defeated the first version of the regex, exactly as it defeated
# _PROGRESS_MARKER_RX and the greps the docs used to recommend.
REAL_ANSI = (
    "[\x1b[32m'x1921c1s0b0n0'\x1b[0m][\x1b[94mdevice\x1b[0m=\x1b[32m'xpu'"
    "\x1b[0m][\x1b[94mnode\x1b[0m=\x1b[1;36m0\x1b[0m/\x1b[1;36m1\x1b[0m]"
    "[\x1b[94mlocal_rank\x1b[0m=\x1b[1;36m00\x1b[0m/\x1b[1;36m11\x1b[0m]"
    "[\x1b[94mrank\x1b[0m=\x1b[1;36m00\x1b[0m/\x1b[1;36m23\x1b[0m]\n"
    "[\x1b[32m'x1921c1s1b0n0'\x1b[0m][\x1b[94mdevice\x1b[0m=\x1b[32m'xpu'"
    "\x1b[0m][\x1b[94mnode\x1b[0m=\x1b[1;36m1\x1b[0m/\x1b[1;36m1\x1b[0m]"
    "[\x1b[94mlocal_rank\x1b[0m=\x1b[1;36m00\x1b[0m/\x1b[1;36m11\x1b[0m]"
    "[\x1b[94mrank\x1b[0m=\x1b[1;36m12\x1b[0m/\x1b[1;36m23\x1b[0m]\n"
)


class TestParse:
    def test_recovers_hosts_and_world_size(self):
        m = parse_rank_host_map(REAL)
        assert m is not None
        assert m.hosts == ["x1921c1s0b0n0", "x1921c1s1b0n0"]
        # rank=NN/LAST is inclusive, so 23 means a world of 24.
        assert m.world_size == 24

    def test_colorized_log_parses_identically(self):
        """The form the log actually takes on disk."""
        plain, colored = parse_rank_host_map(REAL), parse_rank_host_map(
            REAL_ANSI
        )
        assert colored is not None and plain is not None
        assert colored.hosts == plain.hosts
        assert colored.world_size == plain.world_size

    def test_spans_are_contiguous_and_bounded(self):
        m = parse_rank_host_map(REAL)
        assert m is not None
        assert m.ranks_for_host("x1921c1s0b0n0") == range(0, 12)
        assert m.ranks_for_host("x1921c1s1b0n0") == range(12, 24)

    @pytest.mark.parametrize(
        "rank,expected",
        [
            (0, "x1921c1s0b0n0"),
            (11, "x1921c1s0b0n0"),
            (12, "x1921c1s1b0n0"),
            (23, "x1921c1s1b0n0"),
        ],
    )
    def test_boundary_ranks(self, rank, expected):
        m = parse_rank_host_map(REAL)
        assert m is not None
        assert m.host_for_rank(rank) == expected

    @pytest.mark.parametrize("rank", [24, 99, -1])
    def test_out_of_range_is_unknown_not_a_guess(self, rank):
        """``None``, never "whichever host sorts last"."""
        m = parse_rank_host_map(REAL)
        assert m is not None
        assert m.host_for_rank(rank) is None

    def test_all_ranks_logging_still_yields_the_span_start(self):
        """``EZPZ_LOG_ALL_RANKS=1`` makes every rank log its own line.

        The default is one line per host (only ``local_rank=0`` logs),
        where lowest-rank and highest-rank per host are the same value
        -- so a mutant keeping the HIGHEST survived the rest of this
        file. Here they differ, and keeping the highest would put the
        span start at 11/23 instead of 0/12, shifting every boundary.
        """
        multi = "".join(
            f"['x1921c1s{node}b0n0'][device='xpu'][node={node}/1]"
            f"[local_rank={lr:02d}/11][rank={node * 12 + lr:02d}/23]\n"
            for node in (0, 1)
            for lr in range(12)
        )
        m = parse_rank_host_map(multi)
        assert m is not None
        assert m.ranks_for_host("x1921c1s0b0n0") == range(0, 12)
        assert m.ranks_for_host("x1921c1s1b0n0") == range(12, 24)
        assert m.host_for_rank(0) == "x1921c1s0b0n0"
        assert m.host_for_rank(12) == "x1921c1s1b0n0"

    def test_a_log_without_topology_lines_is_none(self):
        """``None`` = "this log does not say", distinct from an empty map."""
        assert parse_rank_host_map("iter=1 loss=2.0\niter=2 loss=1.9\n") is None

    def test_a_hostname_elsewhere_is_not_a_topology_line(self):
        """Anchoring, so a path or an error mentioning a host is ignored."""
        noise = (
            "saved checkpoint: /home/x1921c1s0b0n0/ckpt/step-40\n"
            "x1921c1s0b0n0.hsn.cm.sunspot.alcf.anl.gov: rank 7 exited\n"
        )
        assert parse_rank_host_map(noise) is None

    def test_missing_file_is_none(self, tmp_path):
        assert parse_rank_host_map_file(tmp_path / "nope.log") is None


class TestHostsForRanks:
    def test_maps_a_rank_tagged_error_to_its_host(self):
        assert hosts_for_ranks(REAL, [23]) == ["x1921c1s1b0n0"]

    def test_deduplicates_ranks_on_one_host(self):
        assert hosts_for_ranks(REAL, [12, 15, 23]) == ["x1921c1s1b0n0"]

    def test_unknown_ranks_are_dropped_not_guessed(self):
        assert hosts_for_ranks(REAL, [99]) == []

    def test_no_topology_means_no_answer(self):
        assert hosts_for_ranks("iter=1\n", [0]) == []
