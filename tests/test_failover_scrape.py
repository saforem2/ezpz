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
# Sunspot: same failure modes as Aurora, different hostname suffix
# ---------------------------------------------------------------------------

class TestSunspotShepherdSignal9:
    """Pattern: `<host>.hsn.cm.sunspot.alcf.anl.gov: shepherd died from signal 9`.

    Sunspot is Aurora's test-and-dev twin (same PVC + PALS runtime), so the
    shepherd-kill signature is identical apart from the `.sunspot` suffix and
    an optional `-hsnN` on the HSN node token.
    """

    def test_single_signal_9_extracted_hsn_suffix_token(self, tmp_path):
        """HSN node token carries a `-hsn0` suffix (observed on real
        Sunspot allocations)."""
        log = _make_log(tmp_path, (
            "Some normal training output\n"
            "x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
            "shepherd died from signal 9\n"
            "More output after\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        # The `-hsn0` token is canonicalized away by the normalizer so the
        # node has a single canonical name (see normalize_sunspot_hostname).
        assert hosts == ["x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"]

    def test_single_signal_9_extracted_plain_token(self, tmp_path):
        """Plain node token (no `-hsnN`) — the PBS_NODEFILE form may use
        either, so both must match."""
        log = _make_log(tmp_path, (
            "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov: "
            "shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == ["x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"]

    def test_multiple_signal_9_deduplicated(self, tmp_path):
        # Same node emitted BOTH with and without the -hsn0 token — must
        # collapse to ONE canonical entry (the point of stripping -hsnN).
        log = _make_log(tmp_path, (
            "x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
            "shepherd died from signal 9\n"
            "more output\n"
            "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov: "
            "shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == ["x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"]

    def test_innocent_rank_signal_11_not_matched(self, tmp_path):
        """Same critical exclusion as Aurora: cascading `rank N died from
        signal {11,15}` must NOT tag a node, and `shepherd died from
        signal 11` (not 9) must NOT match either."""
        log = _make_log(tmp_path, (
            "rank 1304 died from signal 11\n"
            "rank 88 died from signal 15\n"
            "x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
            "shepherd died from signal 11\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == [], (
            f"Innocent ranks/signals must not be tagged, got: {hosts!r}"
        )

    def test_aurora_suffix_not_matched_on_sunspot(self, tmp_path):
        """A line with the Aurora suffix must not match under the Sunspot
        pattern set (keeps each machine's suffix explicit)."""
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: "
            "shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == []


class TestSunspotGlooConnectionClosed:
    """gloo peer-closed → IP reverse-resolved, then normalized to the
    `.hsn.cm.sunspot` form. `reverse_resolve_ip` is stubbed (no real DNS)."""

    def test_single_peer_ip_resolved_to_hostname(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ezpz.failover.patterns.sunspot.reverse_resolve_ip",
            lambda ip, **_kw: (
                "x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov"
            ),
        )
        log = _make_log(tmp_path, (
            "RuntimeError: [..gloo..] Connection closed by peer "
            "[10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        # reverse-DNS returned the -hsn0 form; normalizer canonicalizes it.
        assert hosts == ["x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"]

    def test_hostmgmt_form_canonicalized_to_hsn(self, tmp_path, monkeypatch):
        """Reverse-lookup returning the .hostmgmtNNNN form is rewritten to
        the .hsn form so downstream swap_in finds it in the PBS hostfile."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.sunspot.reverse_resolve_ip",
            lambda ip, **_kw: (
                "x1922c7s6b0n0.hostmgmt2001.cm.sunspot.alcf.anl.gov"
            ),
        )
        log = _make_log(tmp_path, (
            "Connection closed by peer [10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == ["x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"]

    def test_non_sunspot_resolved_name_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ezpz.failover.patterns.sunspot.reverse_resolve_ip",
            lambda ip, **_kw: "some-unrelated-host.example.com",
        )
        log = _make_log(tmp_path, (
            "Connection closed by peer [10.0.0.42]:12345\n"
        ))
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == []


class TestSunspotHostnameNormalizer:
    """Direct unit tests of `normalize_sunspot_hostname`."""

    def test_hsn_forms_canonicalize_to_one_name(self):
        from ezpz.failover.patterns.sunspot import normalize_sunspot_hostname

        canonical = "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"
        # Both the -hsn0 and suffix-less forms must map to the SAME canonical
        # name so one node never counts as two.
        for h in (
            "x1922c7s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov",
            "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov",
        ):
            assert normalize_sunspot_hostname(h) == canonical

    def test_hostmgmt_rewritten_to_hsn(self):
        from ezpz.failover.patterns.sunspot import normalize_sunspot_hostname

        assert (
            normalize_sunspot_hostname(
                "x1922c7s6b0n0.hostmgmt2001.cm.sunspot.alcf.anl.gov"
            )
            == "x1922c7s6b0n0.hsn.cm.sunspot.alcf.anl.gov"
        )

    def test_junk_dropped(self):
        from ezpz.failover.patterns.sunspot import normalize_sunspot_hostname

        for h in (
            "some-other-host",
            "x1922c7s6b0n0.something-else.example.com",
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov",  # Aurora suffix
        ):
            assert normalize_sunspot_hostname(h) is None


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
            # Registered alongside the shepherd pattern (job 12473749);
            # empty here because this log has no rank-signal-9 line.
            # Asserted explicitly rather than dropped: the exhaustive
            # comparison is what makes this test notice a new pattern.
            "aurora.rank_signal_9": [],
            "aurora.gloo_connection_closed": [
                "x4002c0s0b0n0.hsn.cm.aurora.alcf.anl.gov"
            ],
        }

    def test_re_register_without_normalizer_clears_old_normalizer(
        self, tmp_path
    ):
        """REGRESSION: re-registering a machine with
        ``hostname_normalizer=None`` must clear any previously-
        registered normalizer. Without this, tests (and real plugin
        callers) that re-register the same key with raw hostnames
        would silently keep applying the old normalizer and either
        rewrite or drop the new pattern's outputs.
        """
        from ezpz.failover.patterns import get_hostname_normalizer

        # First registration: WITH a normalizer.
        register_patterns(
            "test-clear-norm",
            [BadNodePattern("p1", lambda _t: ["x"], "")],
            hostname_normalizer=lambda h: f"normalized-{h}",
        )
        assert get_hostname_normalizer("test-clear-norm") is not None

        # Re-register WITHOUT a normalizer.
        register_patterns(
            "test-clear-norm",
            [BadNodePattern("p2", lambda _t: ["y"], "")],
        )
        assert get_hostname_normalizer("test-clear-norm") is None, (
            "Old normalizer must be cleared on re-registration without one"
        )

    def test_import_error_inside_known_module_surfaces(
        self, tmp_path, monkeypatch
    ):
        """REGRESSION: if a per-machine pattern module exists but has
        a real import problem inside it (missing dep, syntax error,
        circular import), the registry should re-raise rather than
        silently behaving as "unknown machine". Otherwise debugging
        why "aurora" suddenly returns [] becomes a nightmare.
        """
        from ezpz.failover.patterns import get_patterns_for_machine

        # Make importlib.import_module raise an ImportError that names
        # a DIFFERENT module than the one we're asking for — that's the
        # signature of "module exists but fails to load some dep".
        def _import_fail(name):
            raise ImportError(
                "No module named 'unrelated_dep'",
                name="unrelated_dep",
            )

        monkeypatch.setattr(
            "ezpz.failover.patterns.importlib.import_module", _import_fail
        )
        # Use an unregistered machine so the lookup falls through to
        # the import path.
        with pytest.raises(ImportError, match="unrelated_dep"):
            get_patterns_for_machine("never-registered-machine")

    def test_import_error_for_unknown_machine_silent(self, monkeypatch):
        """Counterpoint to the above: if the per-machine module simply
        doesn't exist, that's the "unknown machine" case and should
        return [] silently."""
        from ezpz.failover.patterns import get_patterns_for_machine

        def _import_fail(name):
            raise ImportError(
                f"No module named '{name}'",
                name=name,
            )

        monkeypatch.setattr(
            "ezpz.failover.patterns.importlib.import_module", _import_fail
        )
        # Pattern registry lookup for a genuinely unknown machine →
        # silent empty list, no exception.
        assert get_patterns_for_machine("definitely-not-a-machine") == []


# ---------------------------------------------------------------------------
# Auto-detection path (machine=None)
# ---------------------------------------------------------------------------

class TestAutoDetectMachine:
    """Coverage for the ``machine=None`` code path that pulls machine
    name from ``ezpz.get_machine()``. The rest of the suite passes
    an explicit ``machine="aurora"`` and skips this dispatch."""

    def test_auto_detect_uses_lowercased_ezpz_machine(
        self, tmp_path, monkeypatch
    ):
        """``ezpz.get_machine()`` returns title-case (``"Aurora"``);
        registry keys are lowercase. The auto-detect path must
        lowercase the result before lookup."""
        import ezpz
        monkeypatch.setattr(ezpz, "get_machine", lambda: "Aurora")
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: "
            "shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log)  # no machine= arg
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_explicit_machine_arg_also_normalized(self, tmp_path):
        """Explicit override is also lowercased — passing ``"Aurora"``
        (matching the casing of ``ezpz.get_machine()``'s output)
        should find the registered ``"aurora"`` patterns. Pre-fix this
        silently returned []."""
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: "
            "shepherd died from signal 9\n"
        ))
        hosts = scrape_bad_nodes(log, machine="Aurora")
        assert hosts == ["x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"]

    def test_auto_detect_handles_get_machine_raising(
        self, tmp_path, monkeypatch
    ):
        """If ``ezpz.get_machine()`` itself raises (unlikely but
        possible — e.g. called before distributed setup), the scraper
        falls back to an empty-string machine name, which means
        "no patterns" → empty list, NOT a crash."""
        import ezpz

        def _raise():
            raise RuntimeError("get_machine failed")

        monkeypatch.setattr(ezpz, "get_machine", _raise)
        log = _make_log(tmp_path, (
            "x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: "
            "shepherd died from signal 9\n"
        ))
        # No crash; empty list because no patterns registered for "".
        assert scrape_bad_nodes(log) == []


# ---------------------------------------------------------------------------
# Helper: reverse_resolve_ip
# ---------------------------------------------------------------------------

class TestReverseResolveIp:
    """The other suites all monkeypatch this away. Cover the real
    implementation here against mocked ``subprocess.check_output``."""

    def test_success_returns_first_hostname(self, monkeypatch):
        from ezpz.failover.patterns import reverse_resolve_ip
        # `getent hosts <ip>` shape:
        #   "10.0.0.42  some-host.example.com other-host.example.com"
        monkeypatch.setattr(
            "ezpz.failover.patterns.subprocess.check_output",
            lambda *a, **kw: (
                "10.0.0.42  primary.example.com secondary.example.com\n"
            ),
        )
        assert reverse_resolve_ip("10.0.0.42") == "primary.example.com"

    def test_called_process_error_returns_none(self, monkeypatch):
        """Non-zero exit (IP not in any name service) → None."""
        import subprocess
        from ezpz.failover.patterns import reverse_resolve_ip

        def _raise(*a, **kw):
            raise subprocess.CalledProcessError(2, ["getent"])

        monkeypatch.setattr(
            "ezpz.failover.patterns.subprocess.check_output", _raise
        )
        assert reverse_resolve_ip("10.0.0.42") is None

    def test_timeout_returns_none(self, monkeypatch):
        """Slow name service → don't hang failover; return None."""
        import subprocess
        from ezpz.failover.patterns import reverse_resolve_ip

        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired(["getent"], 5)

        monkeypatch.setattr(
            "ezpz.failover.patterns.subprocess.check_output", _raise
        )
        assert reverse_resolve_ip("10.0.0.42", timeout_s=0.1) is None

    def test_getent_missing_returns_none(self, monkeypatch):
        """No `getent` binary on PATH (e.g. macOS dev box, alpine
        container) → None, not crash."""
        from ezpz.failover.patterns import reverse_resolve_ip

        def _raise(*a, **kw):
            raise FileNotFoundError("getent")

        monkeypatch.setattr(
            "ezpz.failover.patterns.subprocess.check_output", _raise
        )
        assert reverse_resolve_ip("10.0.0.42") is None

    def test_empty_output_returns_none(self, monkeypatch):
        """`getent hosts <ip>` exits 0 with empty output on some
        systems when there's no PTR. Treat as miss."""
        from ezpz.failover.patterns import reverse_resolve_ip
        monkeypatch.setattr(
            "ezpz.failover.patterns.subprocess.check_output",
            lambda *a, **kw: "\n",
        )
        assert reverse_resolve_ip("10.0.0.42") is None

    def test_malformed_output_returns_none(self, monkeypatch):
        """Output with only the IP and no hostnames after it (some
        getent implementations) → None."""
        from ezpz.failover.patterns import reverse_resolve_ip
        monkeypatch.setattr(
            "ezpz.failover.patterns.subprocess.check_output",
            lambda *a, **kw: "10.0.0.42\n",
        )
        assert reverse_resolve_ip("10.0.0.42") is None


# ---------------------------------------------------------------------------
# Aurora normalizer (tightened to known suffixes only)
# ---------------------------------------------------------------------------

class TestNormalizeAuroraHostname:
    """The normalizer must reject hostnames whose suffix we haven't
    explicitly seen — even if they start with the `xNc.s.b.n.` prefix.
    Otherwise a stale `/etc/hosts` entry or a cross-cluster name could
    silently get rewritten to a fake Aurora hostname and end up in
    the bad-nodes list."""

    def test_canonical_hsn_passes_through(self):
        from ezpz.failover.patterns.aurora import normalize_aurora_hostname
        host = "x1234c0s0b0n0.hsn.cm.aurora.alcf.anl.gov"
        assert normalize_aurora_hostname(host) == host

    def test_hostmgmt_rewrites_to_hsn(self):
        from ezpz.failover.patterns.aurora import normalize_aurora_hostname
        in_ = "x1234c0s0b0n0.hostmgmt2042.cm.aurora.alcf.anl.gov"
        out = "x1234c0s0b0n0.hsn.cm.aurora.alcf.anl.gov"
        assert normalize_aurora_hostname(in_) == out

    def test_unknown_suffix_with_aurora_like_prefix_dropped(self):
        """REGRESSION: an Aurora-like prefix on a non-Aurora suffix
        must NOT be rewritten — that would tag a wrong node. Before
        the fix, this rewrote ANY ``x...n0.<anything>`` to the HSN
        form, including stale /etc/hosts entries from other clusters.
        """
        from ezpz.failover.patterns.aurora import normalize_aurora_hostname
        # Same xNcNsNbNnN prefix, different/unknown suffix.
        cases = [
            "x1234c0s0b0n0.something-else.example.com",
            "x1234c0s0b0n0.staging-cluster.foo.bar",
            "x1234c0s0b0n0.local",
        ]
        for host in cases:
            assert normalize_aurora_hostname(host) is None, (
                f"Unknown suffix '{host}' should be rejected, got rewrite"
            )

    def test_completely_unrelated_host_dropped(self):
        from ezpz.failover.patterns.aurora import normalize_aurora_hostname
        assert normalize_aurora_hostname("some-other-host") is None
        assert normalize_aurora_hostname("nid001234") is None
        assert normalize_aurora_hostname("polaris-login-1") is None


# ---------------------------------------------------------------------------
# Polaris: CUDA device faults
#
# Polaris differs from Aurora/Sunspot in a way that shapes every test
# below: its dominant production failure is an NVIDIA CUDA-runtime error
# raised inside a rank's Python process, NOT a PALS shepherd kill. A
# shepherd kill arrives pre-labeled with the node that died; a Python
# traceback does not -- mpiexec prints it verbatim on stderr with no host
# prefix unless the launcher passes `--label` (see EZPZ_MPI_LABEL in
# ezpz/pbs.py).
#
# Job 7550301 (2026-08-23, 130 nodes) is the motivating postmortem: two
# ranks raised a CUDA device fault, the tracebacks were unlabeled, the
# scraper found no host, and the caller blind-rotated a HEALTHY node while
# the sick one stayed in the allocation. ~1h of 130 nodes, zero steps.
# ---------------------------------------------------------------------------

_POLARIS_H = "hsn.cm.polaris.alcf.anl.gov"


def test_polaris_unlabeled_cuda_fault_yields_nothing(tmp_path):
    """THE most important Polaris test.

    This is job 7550301's real log shape: a bare CUDA traceback plus a
    watchdog SIGTERM naming an INNOCENT host. The scraper must return []
    so the caller falls back to blind rotation.

    A false positive here is strictly WORSE than blind rotation: it swaps
    a healthy node AND leaves the real culprit in the allocation. The
    only host-attributed line in the entire 1800-line log named
    x3007c0s13b1n0 -- which was not the node that raised the error.
    """
    log = _make_log(
        tmp_path,
        "Traceback (most recent call last):\n"
        '  File ".../ezpz/distributed.py", line 662, in _set_local_device\n'
        "    torch.cuda.set_device(device_index)\n"
        "torch.AcceleratorError: CUDA error: CUDA-capable device(s) "
        "is/are busy or unavailable\n"
        f"x3007c0s13b1n0.{_POLARIS_H}: rank 57 died from signal 15\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == []


def test_polaris_labeled_cuda_fault_is_attributed(tmp_path):
    """The same fault WITH --label: the culprit is named.

    Note the SIGTERM line still names a different (innocent) host and
    must still be ignored.
    """
    log = _make_log(
        tmp_path,
        f"x3006c0s13b1n0.{_POLARIS_H} 57: Traceback (most recent call last):\n"
        f"x3006c0s13b1n0.{_POLARIS_H} 57: torch.AcceleratorError: CUDA "
        "error: CUDA-capable device(s) is/are busy or unavailable\n"
        f"x3007c0s13b1n0.{_POLARIS_H}: rank 57 died from signal 15\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == [
        f"x3006c0s13b1n0.{_POLARIS_H}"
    ]


def test_polaris_cascade_victims_never_tagged(tmp_path):
    """signal 11/15 and nonzero exits are downstream of the primary kill.

    On Polaris the common source of SIGTERM is the idle-output watchdog's
    OWN kill, so those ranks are victims of our teardown.
    """
    log = _make_log(
        tmp_path,
        f"x3001c0s1b0n0.{_POLARIS_H} 5: torch.AcceleratorError: CUDA "
        "error: CUDA-capable device(s) is/are busy or unavailable\n"
        f"x3002c0s1b0n0.{_POLARIS_H}: rank 5 died from signal 15\n"
        f"x3003c0s1b0n0.{_POLARIS_H}: rank 9 died from signal 11\n"
        f"x3004c0s1b0n0.{_POLARIS_H}: rank 3 exited with code 1\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == [
        f"x3001c0s1b0n0.{_POLARIS_H}"
    ]


def test_polaris_cuda_init_variants(tmp_path):
    """A GPU off the PCIe bus surfaces as init failures, not "busy"."""
    log = _make_log(
        tmp_path,
        f"x3010c0s1b0n0.{_POLARIS_H} 2: RuntimeError: CUDA error: "
        "no CUDA-capable device is detected\n"
        f"x3011c0s1b0n0.{_POLARIS_H} 7: RuntimeError: CUDA error: "
        "initialization error\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == [
        f"x3010c0s1b0n0.{_POLARIS_H}",
        f"x3011c0s1b0n0.{_POLARIS_H}",
    ]


def test_polaris_shepherd_sig9_needs_no_label(tmp_path):
    """PALS prefixes shepherd kills itself, so this works unlabeled."""
    log = _make_log(
        tmp_path,
        f"x3013c0s1b0n0.{_POLARIS_H}: shepherd died from signal 9\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == [
        f"x3013c0s1b0n0.{_POLARIS_H}"
    ]


def test_polaris_hsn_suffix_dedupes_to_one_node(tmp_path):
    """`-hsn0` and the plain form name the SAME node.

    Without normalization the same node dedupes as two entries, which
    breaks swap_in's hostfile lookup and burns two spares for one fault.
    """
    log = _make_log(
        tmp_path,
        f"x3014c0s1b0n0-hsn0.{_POLARIS_H} 1: torch.AcceleratorError: CUDA "
        "error: CUDA-capable device(s) is/are busy or unavailable\n"
        f"x3014c0s1b0n0.{_POLARIS_H} 1: RuntimeError: CUDA error: "
        "CUDA-capable device(s) is/are busy or unavailable\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == [
        f"x3014c0s1b0n0.{_POLARIS_H}"
    ]


def test_polaris_clean_log_yields_nothing(tmp_path):
    log = _make_log(tmp_path, "step=1 loss=12.03\nstep=2 loss=11.87\n")
    assert scrape_bad_nodes(log, machine="polaris") == []


@pytest.mark.parametrize(
    "raw,want",
    [
        (f"x3006c0s13b1n0.{_POLARIS_H}", f"x3006c0s13b1n0.{_POLARIS_H}"),
        (f"x3006c0s13b1n0-hsn0.{_POLARIS_H}", f"x3006c0s13b1n0.{_POLARIS_H}"),
        (
            "x3006c0s13b1n0.hostmgmt2042.cm.polaris.alcf.anl.gov",
            f"x3006c0s13b1n0.{_POLARIS_H}",
        ),
        ("x3006c0s13b1n0.something-else.example.com", None),
        ("some-other-host", None),
        # An Aurora host must NOT normalize as a Polaris one.
        ("x1234c0s0b0n0.hsn.cm.aurora.alcf.anl.gov", None),
    ],
)
def test_polaris_hostname_normalizer(raw, want):
    from ezpz.failover.patterns.polaris import normalize_polaris_hostname

    assert normalize_polaris_hostname(raw) == want


def test_polaris_patterns_are_registered():
    """Regression guard for the bug itself.

    Before polaris.py existed this returned [], which made every Polaris
    failover blind. An empty pattern set is indistinguishable from a
    clean log downstream (both yield []), so nothing surfaced the gap --
    hence an explicit test.
    """
    names = {p.name for p in get_patterns_for_machine("polaris")}
    assert "polaris.cuda_device_unavailable" in names
    assert "polaris.shepherd_signal_9" in names


def test_polaris_device_side_assert_not_matched(tmp_path):
    """A device-side assert is an APPLICATION defect, not a node fault.

    It is raised by a failing assertion inside a kernel (out-of-range
    index, bad label, invalid model input). Tagging the host would
    retire a healthy node on every retry while the real bug persists --
    burning the spare pool and never converging.

    The rule for this pattern set: only match conditions where the same
    code would succeed on a different node.
    """
    log = _make_log(
        tmp_path,
        f"x3020c0s1b0n0.{_POLARIS_H} 3: RuntimeError: CUDA error: "
        "device-side assert triggered\n",
    )
    assert scrape_bad_nodes(log, machine="polaris") == []


class TestPolarisGlooPeerClosed:
    """Polaris gloo TCP peer-closed, mirroring the Aurora coverage.

    `reverse_resolve_ip` is stubbed because unit tests cannot do real
    DNS -- the IPs in real logs live on the HSN fabric and only resolve
    through Polaris's name service.

    Unlike the CUDA patterns, this one does NOT depend on `--label`:
    the peer IP is inside the message body.
    """

    def test_single_peer_ip_resolved_to_hostname(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ezpz.failover.patterns.polaris.reverse_resolve_ip",
            lambda ip, **_kw: f"x3030c0s1b0n0.{_POLARIS_H}",
        )
        log = _make_log(
            tmp_path,
            "RuntimeError: [..gloo..] Connection closed by peer "
            "[10.0.0.42]:12345\n",
        )
        assert scrape_bad_nodes(log, machine="polaris") == [
            f"x3030c0s1b0n0.{_POLARIS_H}"
        ]

    def test_many_ranks_same_peer_dedup_to_one_node(
        self, tmp_path, monkeypatch
    ):
        """Dozens of ranks log the same peer-closed. Want ONE entry."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.polaris.reverse_resolve_ip",
            lambda ip, **_kw: f"x3030c0s1b0n0.{_POLARIS_H}",
        )
        lines = [
            f"rank {i}: RuntimeError: [..gloo..] Connection closed by peer "
            f"[10.0.0.42]:{12000 + i}\n"
            for i in range(30)
        ]
        assert scrape_bad_nodes(
            _make_log(tmp_path, "".join(lines)), machine="polaris"
        ) == [f"x3030c0s1b0n0.{_POLARIS_H}"]

    def test_unresolvable_ip_skipped(self, tmp_path, monkeypatch):
        """Losing one entry beats tagging a wrong node on a bogus lookup."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.polaris.reverse_resolve_ip",
            lambda ip, **_kw: None,
        )
        log = _make_log(
            tmp_path,
            "RuntimeError: [..gloo..] Connection closed by peer [10.0.0.42]:1\n",
        )
        assert scrape_bad_nodes(log, machine="polaris") == []

    def test_non_polaris_resolved_name_dropped_by_normalizer(
        self, tmp_path, monkeypatch
    ):
        """A stale /etc/hosts entry or unrelated name must be dropped."""
        monkeypatch.setattr(
            "ezpz.failover.patterns.polaris.reverse_resolve_ip",
            lambda ip, **_kw: "some-unrelated-host.example.com",
        )
        log = _make_log(
            tmp_path, "Connection closed by peer [10.0.0.42]:12345\n"
        )
        assert scrape_bad_nodes(log, machine="polaris") == []

    def test_hostmgmt_form_canonicalized_to_hsn(self, tmp_path, monkeypatch):
        """The management interface maps 1:1 to HSN and is safe to rewrite.

        It must be canonicalized, since swap_in greps the active hostfile
        which only ever carries the `.hsn.` form.
        """
        monkeypatch.setattr(
            "ezpz.failover.patterns.polaris.reverse_resolve_ip",
            lambda ip, **_kw: (
                "x3030c0s1b0n0.hostmgmt2042.cm.polaris.alcf.anl.gov"
            ),
        )
        log = _make_log(
            tmp_path, "Connection closed by peer [10.0.0.42]:12345\n"
        )
        assert scrape_bad_nodes(log, machine="polaris") == [
            f"x3030c0s1b0n0.{_POLARIS_H}"
        ]

class TestRankSignal9:
    """`<host>: rank N died from signal 9` names the host.

    From Sunspot job 12473749, which killed one node's ranks through
    PALS (`palsig -s SIGKILL`). The log ended with exactly two lines:

        x1921c7s1b0n0-hsn0...: rank 12 died from signal 9    <- victim
        x1921c7s0b0n0-hsn0...: rank  1 died from signal 15   <- cascade

    Neither matched anything before this pattern: the set had
    `shepherd died from signal 9`, and the 11/15 exclusion. A rank
    dying of signal 9 fell between them, so the scraper returned
    nothing and the loop blind-rotated the HEALTHY host out while the
    dead one stayed active.
    """

    REAL = (
        "x1921c7s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
        "rank 12 died from signal 9\n"
        "x1921c7s0b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
        "rank 1 died from signal 15\n"
    )

    def test_names_the_killed_host_and_not_the_cascade(self, tmp_path):
        log = tmp_path / "j12473749.log"
        log.write_text(self.REAL)
        hosts = scrape_bad_nodes(log, machine="sunspot")
        assert hosts == ["x1921c7s1b0n0.hsn.cm.sunspot.alcf.anl.gov"], (
            "signal 9 identifies the victim; signal 15 is the innocent "
            "teardown cascade and must not be named"
        )

    def test_aurora_shares_the_pattern(self, tmp_path):
        log = tmp_path / "aurora.log"
        log.write_text(
            "x4610c4s5b0n0.hsn.cm.aurora.alcf.anl.gov: "
            "rank 7 died from signal 9\n"
        )
        assert scrape_bad_nodes(log, machine="aurora") == [
            "x4610c4s5b0n0.hsn.cm.aurora.alcf.anl.gov"
        ]

    def test_a_clean_walltime_cascade_still_names_nobody(self, tmp_path):
        """The regression this pattern must not cause.

        A clean walltime kill SIGTERMs every rank. If signal 9 matching
        were loosened to any signal, every expiring job would retire a
        node and burn a spare.
        """
        log = tmp_path / "walltime.log"
        log.write_text(
            "".join(
                f"x1921c7s{n}b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
                f"rank {n} died from signal 15\n"
                for n in range(4)
            )
        )
        assert scrape_bad_nodes(log, machine="sunspot") == []

    def test_signal_11_cascade_still_names_nobody(self, tmp_path):
        """Job 8466848's case: SIGSEGV cascades name innocent hosts."""
        log = tmp_path / "sig11.log"
        log.write_text(
            "x1921c7s3b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: "
            "rank 1304 died from signal 11\n"
        )
        assert scrape_bad_nodes(log, machine="sunspot") == []

    def test_shepherd_and_rank_forms_both_match(self, tmp_path):
        """Two distinct patterns, deduplicated to one host."""
        log = tmp_path / "both.log"
        h = "x1921c7s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov"
        log.write_text(
            f"{h}: shepherd died from signal 9\n"
            f"{h}: rank 12 died from signal 9\n"
        )
        assert scrape_bad_nodes(log, machine="sunspot") == [
            "x1921c7s1b0n0.hsn.cm.sunspot.alcf.anl.gov"
        ]


class TestUnsupportedMachineWarns:
    """A machine with no pattern set must SAY so, not fail silently.

    `scrape_bad_nodes` returns `[]` both for an unsupported machine and
    for a genuinely clean log, and only the second means "no node to
    blame" -- so a machine with no pattern set fails over blind, with
    no error, no warning and no log line to reveal it (#229).

    These used to use `polaris` as the unsupported example. #230
    registered Polaris patterns, which correctly broke them: they were
    asserting on a machine that is now supported. Using a machine with
    no pattern module pins the BEHAVIOUR instead of a snapshot of which
    machines happen to be covered.
    """

    # Deliberately not a real system. Naming any machine we might
    # support makes these silently vacuous the day its patterns land --
    # exactly what #230 did to the `polaris` version.
    UNSUPPORTED = "no-such-cluster"

    def test_warns_naming_the_machine_and_what_is_available(
        self, tmp_path, caplog
    ):
        import ezpz.failover.scrape as sc

        sc._WARNED_NO_PATTERNS.clear()
        log = tmp_path / "boom.log"
        log.write_text("some failure with no pattern\n")
        # `logger=` is load-bearing: conftest pins every ezpz logger
        # to CRITICAL, so a bare at_level only lowers the ROOT level
        # and the record is dropped before it can propagate.
        with caplog.at_level("WARNING", logger=sc._get_logger().name):
            assert scrape_bad_nodes(log, machine=self.UNSUPPORTED) == []
        msg = caplog.text
        assert self.UNSUPPORTED in msg
        assert "BLIND" in msg, "the consequence must be stated, not implied"
        # Discovered from the package dir, so this cannot go stale.
        assert "aurora" in msg and "sunspot" in msg

    def test_warns_once_per_machine_not_once_per_attempt(
        self, tmp_path, caplog
    ):
        """This runs on every retry; a repeated warning trains people
        to ignore it."""
        import ezpz.failover.scrape as sc

        sc._WARNED_NO_PATTERNS.clear()
        log = tmp_path / "boom.log"
        log.write_text("failure\n")
        with caplog.at_level("WARNING", logger=sc._get_logger().name):
            for _ in range(4):
                scrape_bad_nodes(log, machine=self.UNSUPPORTED)
        assert caplog.text.count("no bad-node patterns registered") == 1

    def test_a_supported_machine_is_silent(self, tmp_path, caplog):
        import ezpz.failover.scrape as sc

        sc._WARNED_NO_PATTERNS.clear()
        log = tmp_path / "clean.log"
        log.write_text("iter=1 loss=2.0\n")
        with caplog.at_level("WARNING", logger=sc._get_logger().name):
            assert scrape_bad_nodes(log, machine="sunspot") == []
        assert "no bad-node patterns registered" not in caplog.text


class TestPerlmutterSrunKilled:
    """SLURM's kill signature (#238).

    Perlmutter runs srun, not PALS, so none of the Aurora/Sunspot
    signatures appear. Worse, srun reports a SIGKILLed rank as rc=143 --
    the same code as a clean walltime expiry -- so a killed node
    collapsed to WALLTIME and the loop did not even retry.

    Captured on job 57540936, which SIGKILLed one node's ranks in a
    2-active/2-spare allocation:

        srun: error: nid001321: tasks 4-7: Killed       <- the victim
        srun: error: nid001320: tasks 0-3: Terminated   <- the cascade
    """

    REAL = (
        "srun: error: nid001321: tasks 4-7: Killed\n"
        "srun: error: nid001320: tasks 0-3: Terminated\n"
    )

    def test_names_the_killed_node_not_the_cascade(self, tmp_path):
        log = _make_log(tmp_path, self.REAL)
        assert scrape_bad_nodes(log, machine="perlmutter") == ["nid001321"]

    def test_singular_task_form_matches(self, tmp_path):
        log = _make_log(
            tmp_path, "srun: error: nid002813: task 4: Killed\n"
        )
        assert scrape_bad_nodes(log, machine="perlmutter") == ["nid002813"]

    def test_comma_list_of_tasks_matches(self, tmp_path):
        log = _make_log(
            tmp_path, "srun: error: nid000042: tasks 1,3,5: Killed\n"
        )
        assert scrape_bad_nodes(log, machine="perlmutter") == ["nid000042"]

    def test_a_clean_teardown_names_nobody(self, tmp_path):
        """The regression this must not cause.

        Every rank gets SIGTERM when a step ends normally. Matching
        Terminated would retire a node on every expiring job.
        """
        log = _make_log(tmp_path, (
            "srun: error: nid001320: tasks 0-3: Terminated\n"
            "srun: error: nid001321: tasks 4-7: Terminated\n"
            "srun: Force Terminated StepId=57540936.0\n"
        ))
        assert scrape_bad_nodes(log, machine="perlmutter") == []

    def test_two_killed_nodes_both_named(self, tmp_path):
        log = _make_log(tmp_path, (
            "srun: error: nid001321: tasks 4-7: Killed\n"
            "srun: error: nid009999: tasks 8-11: Killed\n"
        ))
        assert scrape_bad_nodes(log, machine="perlmutter") == [
            "nid001321", "nid009999"
        ]

    def test_non_nid_hostname_rejected_by_the_pattern(self, tmp_path):
        """The regex itself requires `nidNNNNNN`."""
        log = _make_log(
            tmp_path, "srun: error: login16: tasks 0-3: Killed\n"
        )
        assert scrape_bad_nodes(log, machine="perlmutter") == []

    def test_the_normalizer_also_rejects_non_nid_names(self):
        """Tested DIRECTLY, because the pattern already filters these.

        Going through `scrape_bad_nodes` cannot exercise the
        normalizer's guard -- the regex rejects `login16` first, so a
        mutant that made the normalizer accept anything survived the
        end-to-end version of this test.

        The guard matters on its own: a name not in SLURM's form could
        never match the active hostfile, so admitting one could only
        put an unswappable host in bad_nodes.txt.
        """
        from ezpz.failover.patterns.perlmutter import (
            normalize_perlmutter_hostname as norm,
        )

        assert norm("nid001321") == "nid001321"
        assert norm("nid001321.chn.perlmutter.nersc.gov") == "nid001321"
        for bad in ("login16", "x1921c7s1b0n0", "", "nid", "nidXYZ"):
            assert norm(bad) is None, f"{bad!r} should be rejected"


class TestSrunKilledIsACrashPattern:
    """rc=143 must stop collapsing to WALLTIME on SLURM (#238)."""

    def test_killed_node_is_retried_not_called_walltime(self, tmp_path):
        from ezpz.launch_autoretry import (
            TerminationReason,
            classify_attempt,
        )

        log = _make_log(tmp_path, (
            "iter=10 loss=1.0\n"
            "srun: error: nid001321: tasks 4-7: Killed\n"
            "srun: error: nid001320: tasks 0-3: Terminated\n"
        ))
        r = classify_attempt(
            shell_rc=143, log_path=log, scraped_bad_nodes=[],
            has_spares=True, prior_attempt_had_progress=True,
        )
        assert r.reason is not TerminationReason.WALLTIME, (
            "a SIGKILLed node must not read as a clean walltime expiry"
        )

    def test_a_real_walltime_kill_still_reads_as_walltime(self, tmp_path):
        """The guard that keeps this from burning a spare per job."""
        from ezpz.launch_autoretry import (
            TerminationReason,
            classify_attempt,
        )

        log = _make_log(tmp_path, (
            "iter=99 loss=1.0\n"
            "srun: error: nid001320: tasks 0-3: Terminated\n"
            "srun: Force Terminated StepId=1.0\n"
        ))
        r = classify_attempt(
            shell_rc=143, log_path=log, scraped_bad_nodes=[],
            has_spares=True, prior_attempt_had_progress=True,
        )
        assert r.reason is TerminationReason.WALLTIME
