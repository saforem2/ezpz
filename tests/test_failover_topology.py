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


class TestHostnameFormMismatch:
    """The scraper's name and PBS's name must resolve to one machine.

    Sunspot job 12473750: PBS wrote
    ``x1921c7s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov`` into the active
    hostfile; the scraper's normalizer returned
    ``x1921c7s1b0n0.hsn.cm.sunspot.alcf.anl.gov``. An exact ``in`` test
    said they were different machines, so the loop logged

        bad nodes: ['x1921c7s1b0n0.hsn...'] -- swapped 0

    and fell through to a blind rotation that retired the HEALTHY host
    and left the dead one running. Correct attribution, discarded on a
    string comparison.
    """

    PBS = [
        f"x1921c7s{i}b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov"
        for i in range(4)
    ]
    SCRAPED_VICTIM = "x1921c7s1b0n0.hsn.cm.sunspot.alcf.anl.gov"

    def _alloc(self, tmp_path):
        from ezpz.launch_autoretry import NodeAllocation

        return NodeAllocation.from_full_nodelist(
            self.PBS, 2, tmp_path / "active.hostfile", tmp_path / "bad.txt"
        )

    def test_scraped_name_swaps_the_pbs_named_host(self, tmp_path):
        alloc = self._alloc(tmp_path)
        swaps = alloc.swap_in([self.SCRAPED_VICTIM], attempt=1)
        assert len(swaps) == 1, (
            "the scraped host must match the -hsn0 form in the hostfile"
        )
        assert not any("s1b0n0" in h for h in alloc.active), (
            "the victim must leave the active set"
        )

    def test_it_is_recorded_as_scraped_not_blind(self, tmp_path):
        """The provenance is the point: this was evidence, not a guess."""
        alloc = self._alloc(tmp_path)
        alloc.swap_in([self.SCRAPED_VICTIM], attempt=1)
        line = (tmp_path / "bad.txt").read_text().strip()
        assert "scraped" in line and "blind" not in line

    def test_the_hostfile_native_name_is_what_gets_recorded(self, tmp_path):
        """Record the name the hostfile uses, not the normalized one.

        Anything reading bad_nodes.txt back against a PBS nodefile
        needs the form PBS uses.
        """
        alloc = self._alloc(tmp_path)
        alloc.swap_in([self.SCRAPED_VICTIM], attempt=1)
        # Compare the recorded line EXACTLY, not by prefix. `startswith`
        # on a hostname is a weak assertion -- it passes for any longer
        # host sharing the prefix, so it could not tell the PBS `-hsn0`
        # form apart from something merely beginning with it. Recording
        # the wrong-but-prefixed name is precisely the failure this
        # class exists to catch.
        # Each line is `<hostname> <provenance> attempt=N`, so compare
        # the hostname FIELD exactly rather than the whole line.
        line = (tmp_path / "bad.txt").read_text().strip()
        # Assert non-empty BEFORE indexing: `split()[0]` on a blank file
        # raises IndexError, which hides the real regression behind an
        # unhelpful error.
        fields = line.split()
        assert fields, f"bad.txt is empty; expected one recorded host, got {line!r}"
        assert fields[0] == (
            "x1921c7s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov"
        )

    def test_a_different_node_still_does_not_match(self, tmp_path):
        """The loosening must not make every host equivalent."""
        alloc = self._alloc(tmp_path)
        swaps = alloc.swap_in(
            ["x1921c7s9b0n0.hsn.cm.sunspot.alcf.anl.gov"], attempt=1
        )
        assert swaps == []
        assert (tmp_path / "bad.txt").read_text() == ""
