"""Tests for :mod:`ezpz.failover.scrape`.

The Aurora pattern set is derived from postmortem analysis of real
production failures; the test fixtures below replay the log lines
that caused each one. Job IDs cited in comments map to the source
material in the original torchtitan failover_lib.sh + scrape script.

The most important test is :func:`test_innocent_rank_signal_11_not_matched`
— this exclusion is the entire reason we don't naively grep for
"died from signal 11/15". A naive matcher would falsely tag the node
whose ranks got the cascading kill, not the node that started it.
"""

from __future__ import annotations

import pytest

from ezpz.failover.patterns import (
    BadNodePattern,
    get_patterns_for_machine,
    register_patterns,
)
from ezpz.failover.scrape import (
    _collect_all_matches_for_debug,
    scrape_bad_nodes,
)


# Helpers --------------------------------------------------------------------

def _make_log(tmp_path, content: str):
    """Write `content` to a fixture log file and return its Path."""
    p = tmp_path / "training.log"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Aurora: PALS shepherd signal-9 kill
# ---------------------------------------------------------------------------

class TestAuroraShepherdSignal9:
    """Pattern: `<host>.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9`.

    Source: jobs 8459818 / 8460301 / 8463659 (PALS shepherd kills on
    Aurora). The shepherd is PALS's per-node daemon; when it dies
    from signal 9, the node went non-responsive and the runtime
    killed it. Almost always a hardware fault.
    """

    def test_single_signal_9_extracted(self, tmp_path):
        log = _make_log(tmp_path, (
            "Some normal training output\n"
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            "More output after\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_multiple_signal_9_deduplicated(self, tmp_path):
        """Same node firing multiple shepherd-9 lines (PALS often
        emits more than one) → one entry in first-seen order."""
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            "more output\n"
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_two_different_nodes_in_first_seen_order(self, tmp_path):
        log = _make_log(tmp_path, (
            "x4001c0s0b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        # Order matters: first-seen wins, both retained.
        assert hosts == [
            "x4001c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
            "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
        ]

    def test_innocent_rank_signal_11_not_matched(self, tmp_path):
        """REGRESSION (job 8466848): rank 1304 died from signal 11 as a
        downstream effect of a std::bad_alloc on a DIFFERENT rank on a
        DIFFERENT node. A naive matcher that greps for "died from
        signal" would tag the wrong node and swap out the *innocent*
        one, leaving the actual bad node in the active set.

        Our scrape MUST NOT match `rank N died from signal {11,15}`
        and MUST NOT match log lines that don't start with the
        hostname-colon prefix.
        """
        log = _make_log(tmp_path, (
            # Real shape from 8466848: the rank-died lines have no
            # hostname prefix and use "rank N died from signal", not
            # "shepherd died from signal".
            "rank 1304 died from signal 11\n"
            "rank 2413 died from signal 11\n"
            "rank 88 died from signal 15\n"
            # And a "shepherd died from signal 11" — also should NOT
            # match (we only match signal 9).
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 11\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == [], (
            f"Innocent ranks/signals must not be tagged, got: {hosts!r}"
        )

    def test_hostmgmt_form_normalized_to_hsn(self, tmp_path):
        """Aurora occasionally emits the .hostmgmtNNNN.cm.aurora form
        instead of the .hsn.cm.aurora form. The scraper's hostname
        normalizer must map both to the HSN form so downstream
        swap_in (which greps the PBS hostfile) finds matches.

        NOTE: the shepherd pattern itself anchors on `.hsn.cm.aurora`,
        so .hostmgmt lines won't match it. The normalizer's job is
        re-canonicalizing names that DID match but came from a
        different code path (e.g. reverse-resolved IPs). Tested more
        directly below via test_gloo_*.
        """
        # Even when shepherd line is already in HSN form, normalizer
        # leaves it alone. This is the easy case.
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]


# ---------------------------------------------------------------------------
# Aurora: gloo TCP peer-closed
# ---------------------------------------------------------------------------

class TestAuroraGlooConnectionClosed:
    """Pattern: `Connection closed by peer [IP]:port` → IP reverse-resolved.

    Source: jobs 8470102 / 8470103 / 8479581. gloo errors typically
    point at a single peer IP across many "Connection closed" lines
    (every rank that was talking to the dead node logs its own copy),
    so deduplication usually collapses to one node.

    Tests stub `reverse_resolve_ip` because we can't do real DNS in
    unit tests — the IPs in real Aurora logs are on the HSN fabric
    and only resolvable through Aurora's name service.
    """

    def test_single_peer_ip_resolved_to_hostname(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov",
        )
        log = _make_log(tmp_path, (
            "RuntimeError: [..gloo..] Connection closed by peer [10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_many_ranks_same_peer_dedup_to_one_node(
        self, tmp_path, monkeypatch
    ):
        """Real production shape: dozens of ranks all log the same
        peer-closed against the same IP. We want ONE entry, not 30."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov",
        )
        lines = [
            f"rank {i}: RuntimeError: [..gloo..] Connection closed by peer "
            f"[10.0.0.42]:{12000 + i}\n"
            for i in range(30)
        ]
        log = _make_log(tmp_path, "".join(lines))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_unresolvable_ip_skipped(self, tmp_path, monkeypatch):
        """If `getent hosts <ip>` fails (binary missing, timeout, etc.)
        the entry is silently dropped — losing one is better than
        tagging a wrong node based on a bogus reverse lookup."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: None,
        )
        log = _make_log(tmp_path, (
            "RuntimeError: [..gloo..] Connection closed by peer [10.0.0.42]:1\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == []

    def test_non_aurora_resolved_name_dropped_by_normalizer(
        self, tmp_path, monkeypatch
    ):
        """Reverse-lookup returning a non-Aurora-looking name (e.g.
        the management interface, or a stale /etc/hosts entry) is
        dropped by the hostname normalizer. Better than tagging a
        nonsense hostname that the active hostfile would never match."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: "some-unrelated-host.example.com",
        )
        log = _make_log(tmp_path, (
            "Connection closed by peer [10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == []

    def test_hostmgmt_form_canonicalized_to_hsn(
        self, tmp_path, monkeypatch
    ):
        """If reverse-lookup returns the .hostmgmtNNNN form, the
        normalizer rewrites it to the .hsn form so downstream swap_in
        finds it in the PBS hostfile (which uses .hsn exclusively)."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: (
                "x4502c1s3b0n0.hostmgmt2042.cm.aurora.alcf.anl.gov"
            ),
        )
        log = _make_log(tmp_path, (
            "Connection closed by peer [10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]


# ---------------------------------------------------------------------------
# Cross-pattern: dedup + ordering
# ---------------------------------------------------------------------------

class TestScraperBehavior:

    def test_both_patterns_fire_same_node_dedup(self, tmp_path, monkeypatch):
        """Same bad node can fire both patterns (shepherd-9 AND gloo
        peer-closed from neighboring ranks). Should still appear once."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov",
        )
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            "Connection closed by peer [10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_both_patterns_fire_different_nodes_both_returned(
        self, tmp_path, monkeypatch
    ):
        """Two separate bad nodes (one shepherd-9, one gloo-closed)
        both end up in the output, shepherd's first because the
        patterns iterate in registration order."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
        )
        log = _make_log(tmp_path, (
            "x4001c0s0b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            "Connection closed by peer [10.0.0.42]:1\n"
        ))
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == [
            "x4001c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
            "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
        ]

    def test_empty_log_returns_empty_list(self, tmp_path):
        log = _make_log(tmp_path, "")
        assert scrape_bad_nodes(log, machine="aurora") == []

    def test_clean_log_returns_empty_list(self, tmp_path):
        log = _make_log(tmp_path, "step=1 loss=2.4\nstep=2 loss=2.3\n")
        assert scrape_bad_nodes(log, machine="aurora") == []

    def test_unknown_machine_returns_empty_list(self, tmp_path):
        """No registered patterns → empty list, not exception. Lets
        the caller's blind-rotation fallback fire."""
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
        ))
        assert scrape_bad_nodes(log, machine="mars-rover-cluster") == []

    def test_missing_log_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scrape_bad_nodes(tmp_path / "does-not-exist.log", machine="aurora")

    def test_binary_garbage_in_log_doesnt_crash(self, tmp_path):
        """Real logs can be partially corrupted by mid-write crashes;
        the scraper must handle that gracefully (errors='replace')."""
        log = tmp_path / "corrupt.log"
        log.write_bytes(
            b"x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            b"\xfe\xff\xfe\xff some garbage \x00\x00\xff\xff\n"
        )
        hosts = scrape_bad_nodes(log, machine="aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]


# ---------------------------------------------------------------------------
# Registry extension
# ---------------------------------------------------------------------------

class TestPatternRegistry:

    def test_can_register_custom_pattern(self, tmp_path):
        """Third parties (or future machine modules) can register
        patterns at import time."""
        def _extract_bad_widget(text: str):
            for line in text.splitlines():
                if "WIDGET_DOWN:" in line:
                    yield line.split("WIDGET_DOWN:")[1].strip()

        register_patterns(
            "fictional-cluster",
            [
                BadNodePattern(
                    name="fictional.widget_down",
                    extractor=_extract_bad_widget,
                    description="The widget went down.",
                )
            ],
            hostname_normalizer=None,
        )
        log = _make_log(tmp_path, "WIDGET_DOWN: node-42\nother stuff\n")
        hosts = scrape_bad_nodes(log, machine="fictional-cluster")
        assert hosts == ["node-42"]

    def test_re_register_overwrites(self, tmp_path):
        """Calling register_patterns twice for the same machine
        replaces the previous registration."""
        first = [BadNodePattern("first", lambda _t: ["nope"], "")]
        second = [BadNodePattern("second", lambda _t: ["yes"], "")]
        register_patterns("test-overwrite", first)
        register_patterns("test-overwrite", second)
        patterns = get_patterns_for_machine("test-overwrite")
        assert len(patterns) == 1 and patterns[0].name == "second"

    def test_explain_mode_breaks_down_per_pattern(
        self, tmp_path, monkeypatch
    ):
        """`_collect_all_matches_for_debug` returns one list per
        pattern, even when only some fired."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.aurora.reverse_resolve_ip",
            lambda ip, **_kw: "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
        )
        log = _make_log(tmp_path, (
            "x4001c0s0b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9\n"
            "Connection closed by peer [10.0.0.42]:1\n"
        ))
        per_pattern = _collect_all_matches_for_debug(log, machine="aurora")
        assert per_pattern == {
            "aurora.shepherd_signal_9": [
                "x4001c0s0b0n0.hsn.cm.aurora.alcf.anl.gov"
            ],
            "aurora.gloo_connection_closed": [
                "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov"
            ],
        }
