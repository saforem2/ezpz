"""Tests for `ezpz export-amsc`.

The load-bearing test here is `TestAmscSchemaConformance`: it replicates
the coercion the CONSUMER performs (`build_dashboard.py` in the AmSC
benchmarks repo) rather than asserting our own output shape back at
ourselves. A row that parses cleanly for us but not for the dashboard is
a row that silently never appears.

The throughput fixtures use the real per-step `train/tps` series from a
Sunspot `agpt-2b` run (job 12472885): 1045, 34601, 34686, 34715, 34833,
34650. Step 1 is a 33x outlier, which is what the warmup and reducer
defaults exist for.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from ezpz.export.amsc import (
    AMSC_COLUMNS,
    AMSC_REQUIRED_COLUMNS,
    AmscExportError,
    format_csv,
    load_run_metrics,
    load_run_provenance,
    summarize_run,
    to_amsc_row,
)

# Real series from Sunspot job 12472885 (agpt-2b, tp=1, 12 ranks).
REAL_TPS = [1045.1, 34600.8, 34686.4, 34715.4, 34832.8, 34650.0]


def _write_records(path: Path, tps, *, rank=0, dt=1.0, loss_from=13.0):
    """One JSONL file shaped like History._write_jsonl_entry output."""
    lines = []
    for i, v in enumerate(tps):
        lines.append(
            json.dumps(
                {
                    "timestamp": 1786393250.0 + i,
                    "rank": rank,
                    "metrics": {
                        "train/iter": i + 1,
                        "train/loss": loss_from - i * 0.01,
                        "train/dt": dt,
                        "train/tps": v,
                        "train/tps_per_gpu": v / 12.0,
                        "train/mfu": 5.8,
                        "train/tflops": 17.4,
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    _write_records(tmp_path / "metrics-0.jsonl", REAL_TPS)
    return tmp_path


class TestLoadRunMetrics:
    def test_reads_per_rank_files(self, tmp_path: Path):
        _write_records(tmp_path / "metrics-0.jsonl", REAL_TPS, rank=0)
        _write_records(tmp_path / "metrics-1.jsonl", REAL_TPS, rank=1)
        assert len(load_run_metrics(tmp_path)) == 2 * len(REAL_TPS)

    def test_reads_single_metrics_file(self, tmp_path: Path):
        """vit/test/minimal/hf write one metrics.jsonl, not per-rank."""
        _write_records(tmp_path / "metrics.jsonl", REAL_TPS)
        assert len(load_run_metrics(tmp_path)) == len(REAL_TPS)

    def test_skips_the_precision_record(self, tmp_path: Path):
        """fsdp_tp writes an all-strings `precision/*` record directly
        through History._write_jsonl_entry (fsdp_tp.py:3575). Treating
        it as numeric would raise on float()."""
        p = tmp_path / "metrics-0.jsonl"
        _write_records(p, REAL_TPS)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "timestamp": 1786393260.0,
                        "rank": 0,
                        "metrics": {
                            "precision/mixed_precision": "bf16",
                            "precision/master_weights": "fp32",
                        },
                    }
                )
                + "\n"
            )
        records = load_run_metrics(tmp_path)
        assert len(records) == len(REAL_TPS)
        assert not any(
            k.startswith("precision/")
            for r in records
            for k in r["metrics"]
        )

    def test_survives_a_truncated_final_line(self, tmp_path: Path):
        """A killed job leaves a half-written line; that must not be
        fatal -- the run's earlier steps are still valid data."""
        p = tmp_path / "metrics-0.jsonl"
        _write_records(p, REAL_TPS)
        with p.open("a", encoding="utf-8") as fh:
            fh.write('{"timestamp": 1786393261.0, "rank": 0, "metr')
        assert len(load_run_metrics(tmp_path)) == len(REAL_TPS)

    def test_missing_jsonl_names_what_it_looked_for(self, tmp_path: Path):
        with pytest.raises(AmscExportError, match="metrics"):
            load_run_metrics(tmp_path)

    def test_not_a_directory(self, tmp_path: Path):
        f = tmp_path / "x.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(AmscExportError, match="not a directory"):
            load_run_metrics(f)


class TestSummarize:
    def test_default_drops_the_warmup_outlier(self, run_dir: Path):
        """Step 1 is 1045 vs a ~34.7k steady state. Including it in a
        MEAN understates throughput by ~16%; this is the entire reason
        warmup trimming is on by default."""
        records = load_run_metrics(run_dir)
        trimmed = summarize_run(records, warmup=1, reducer="mean")
        untrimmed = summarize_run(records, warmup=0, reducer="mean")
        assert trimmed["throughput_tokens_per_sec"] > 34000
        assert untrimmed["throughput_tokens_per_sec"] < 30000

    def test_median_is_robust_without_trimming(self, run_dir: Path):
        """A median survives the outlier even at warmup=0 -- which is
        why it is the default reducer, not the mean."""
        records = load_run_metrics(run_dir)
        assert (
            summarize_run(records, warmup=0, reducer="median")[
                "throughput_tokens_per_sec"
            ]
            > 34000
        )

    def test_wall_time_is_sum_of_dt_including_warmup(self, run_dir: Path):
        """Warmup time was really spent; only the throughput reduction
        trims it. And the JSONL timestamp span is NOT usable -- records
        exist only for logged steps."""
        s = summarize_run(load_run_metrics(run_dir), warmup=1)
        assert s["wall_time_sec"] == pytest.approx(len(REAL_TPS) * 1.0)

    def test_final_loss_is_the_last_value(self, run_dir: Path):
        s = summarize_run(load_run_metrics(run_dir))
        assert s["final_loss"] == pytest.approx(13.0 - 0.05)

    def test_absent_metric_is_none_not_zero(self, tmp_path: Path):
        """mfu/tflops are absent when model FLOPs are unknown. Zero
        would read as a measured zero on the dashboard."""
        (tmp_path / "metrics-0.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": 1.0,
                    "rank": 0,
                    "metrics": {"train/iter": 1, "train/tps": 100.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        s = summarize_run(load_run_metrics(tmp_path), warmup=0)
        assert s["mfu"] is None and s["tflops"] is None

    def test_prefers_rank0_over_mixing_ranks(self, tmp_path: Path):
        """Every rank logs near-identical values; mixing them would
        inflate the sample count without adding information."""
        _write_records(tmp_path / "metrics-0.jsonl", REAL_TPS, rank=0)
        _write_records(tmp_path / "metrics-1.jsonl", REAL_TPS, rank=1)
        s = summarize_run(load_run_metrics(tmp_path), warmup=1)
        assert s["steps"] == len(REAL_TPS)

    def test_warmup_larger_than_series_keeps_the_series(self, run_dir: Path):
        """Better a warmup-contaminated number than an empty cell."""
        s = summarize_run(load_run_metrics(run_dir), warmup=999)
        assert s["throughput_tokens_per_sec"] is not None

    def test_rejects_unknown_reducer(self, run_dir: Path):
        with pytest.raises(AmscExportError, match="unknown reducer"):
            summarize_run(load_run_metrics(run_dir), reducer="nope")

    def test_rejects_negative_warmup(self, run_dir: Path):
        with pytest.raises(AmscExportError, match="warmup"):
            summarize_run(load_run_metrics(run_dir), warmup=-1)


class TestProvenance:
    def test_reads_run_info_json(self, tmp_path: Path):
        (tmp_path / "run_info.json").write_text(
            json.dumps(
                {"Distributed": {"MACHINE": "Aurora", "NUM_NODES": "8",
                                 "NGPUS": "96"}}
            ),
            encoding="utf-8",
        )
        p = load_run_provenance(tmp_path)
        assert p["MACHINE"] == "Aurora" and p["NGPUS"] == "96"

    def test_falls_back_to_the_report_markdown(self, tmp_path: Path):
        """run_info.json is new and config.json needs the non-default
        csv backend, but every run has ever written get_dist_info() into
        report-*.md. Without this tier the exporter is useless on every
        run already on disk."""
        (tmp_path / "train").mkdir()
        (tmp_path / "train" / "report-train.md").write_text(
            "# Report\n\n### Distributed\n\n"
            "- **MACHINE**: SunSpot\n"
            "- **NUM_NODES**: 2\n"
            "- **NGPUS**: 24\n\n"
            "## Metric Overview\n",
            encoding="utf-8",
        )
        p = load_run_provenance(tmp_path)
        assert p["MACHINE"] == "SunSpot"
        assert p["NUM_NODES"] == "2"
        assert p["NGPUS"] == "24"
        assert "Metric Overview" not in json.dumps(p)

    def test_run_info_wins_over_report(self, tmp_path: Path):
        (tmp_path / "run_info.json").write_text(
            json.dumps({"Distributed": {"MACHINE": "Aurora",
                                        "NUM_NODES": "8", "NGPUS": "96"}}),
            encoding="utf-8",
        )
        (tmp_path / "report.md").write_text(
            "### Distributed\n\n- **MACHINE**: Stale\n", encoding="utf-8"
        )
        assert load_run_provenance(tmp_path)["MACHINE"] == "Aurora"

    def test_absent_is_empty_not_an_error(self, tmp_path: Path):
        assert load_run_provenance(tmp_path) == {}


class TestRowConstruction:
    def _summary(self, run_dir):
        return summarize_run(load_run_metrics(run_dir), warmup=1)

    def test_explicit_flags_beat_provenance(self, run_dir: Path):
        row = to_amsc_row(
            self._summary(run_dir),
            config="c",
            system="Frontier",
            nodes=4,
            gpus=32,
            provenance={"MACHINE": "SunSpot", "NUM_NODES": "1",
                        "NGPUS": "12"},
        )
        assert (row["system"], row["nodes"], row["gpus"]) == ("Frontier", 4, 32)

    def test_reports_what_the_run_used_not_what_it_was_allocated(
        self, run_dir: Path
    ):
        """REGRESSION (Sunspot job 12473020).

        A scaling sweep ran 1/2/4-node configurations inside one 4-node
        allocation. `NUM_NODES`/`NGPUS` describe the ALLOCATION, so the
        1-node run reported nodes=4, gpus=48 while
        `WORLD_SIZE_IN_USE` was 12 -- which would have published a
        1-node result as a catastrophically slow 4-node one.
        """
        row = to_amsc_row(
            self._summary(run_dir),
            config="c",
            provenance={
                "MACHINE": "SunSpot",
                "NUM_NODES": "4",           # allocation
                "NGPUS": "48",              # allocation
                "GPUS_PER_NODE": "12",
                "WORLD_SIZE_IN_USE": "12",  # what actually ran
            },
        )
        assert (row["nodes"], row["gpus"]) == (1, 12)

    def test_partial_last_node_rounds_up(self, run_dir: Path):
        """18 ranks at 12/node is 2 nodes, not 1."""
        row = to_amsc_row(
            self._summary(run_dir),
            config="c",
            provenance={
                "MACHINE": "S", "GPUS_PER_NODE": "12",
                "WORLD_SIZE_IN_USE": "18",
            },
        )
        assert (row["nodes"], row["gpus"]) == (2, 18)

    def test_falls_back_to_allocation_when_usage_unknown(self, run_dir: Path):
        row = to_amsc_row(
            self._summary(run_dir),
            config="c",
            provenance={"MACHINE": "S", "NUM_NODES": "8", "NGPUS": "96"},
        )
        assert (row["nodes"], row["gpus"]) == (8, 96)

    def test_missing_identity_errors_with_the_flags(self, run_dir: Path):
        """Guessing gpus from the rank-file count would be wrong at any
        ranks-per-GPU other than 1, and a wrong GPU count corrupts every
        per-GPU comparison on the dashboard."""
        with pytest.raises(AmscExportError) as ei:
            to_amsc_row(self._summary(run_dir), config="c")
        msg = str(ei.value)
        assert "--system" in msg and "--nodes" in msg and "--gpus" in msg

    def test_requires_a_config(self, run_dir: Path):
        with pytest.raises(AmscExportError, match="config"):
            to_amsc_row(
                self._summary(run_dir), config="",
                system="S", nodes=1, gpus=12,
            )

    def test_none_becomes_empty_cell(self, tmp_path: Path):
        (tmp_path / "metrics-0.jsonl").write_text(
            json.dumps({"timestamp": 1.0, "rank": 0,
                        "metrics": {"train/iter": 1, "train/tps": 5.0}})
            + "\n",
            encoding="utf-8",
        )
        row = to_amsc_row(
            summarize_run(load_run_metrics(tmp_path), warmup=0),
            config="c", system="S", nodes=1, gpus=1,
        )
        assert row["mfu"] == "" and row["tflops"] == ""


class TestAmscSchemaConformance:
    """Replicate the CONSUMER's parsing, not our own.

    `build_dashboard.py` in the AmSC repo does:

        REQUIRED_COLS = {"timestamp","system","config","nodes","gpus",
                         "status","wall_time_sec"}
        int(nodes); int(gpus); float(wall_time_sec)
        # every other non-required column -> float() when possible

    A row we consider fine but it cannot coerce simply never shows up.
    """

    def _row(self, run_dir):
        return to_amsc_row(
            summarize_run(load_run_metrics(run_dir), warmup=1),
            config="agpt-2b/bs1/seq512/tp1",
            system="SunSpot", nodes=1, gpus=12,
        )

    def test_header_covers_every_required_column(self):
        assert set(AMSC_REQUIRED_COLUMNS) <= set(AMSC_COLUMNS)

    def test_round_trips_through_the_consumers_coercion(self, run_dir: Path):
        text = format_csv([self._row(run_dir)])
        parsed = list(csv.DictReader(io.StringIO(text)))
        assert len(parsed) == 1
        row = parsed[0]
        assert set(AMSC_REQUIRED_COLUMNS) <= set(row)
        int(row["nodes"])
        int(row["gpus"])
        float(row["wall_time_sec"])
        for key, value in row.items():
            if key in AMSC_REQUIRED_COLUMNS or value == "":
                continue
            float(value)  # must not raise

    def test_status_is_a_value_the_dashboard_shows(self, run_dir: Path):
        assert self._row(run_dir)["status"] == "pass"

    def test_timestamp_matches_the_published_format(self, run_dir: Path):
        """Published rows use e.g. 2025-11-13T06:01:20Z."""
        import re

        ts = self._row(run_dir)["timestamp"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts), ts

    def test_header_written_once_when_appending(self, run_dir: Path):
        row = self._row(run_dir)
        text = format_csv([row]) + format_csv([row], header=False)
        assert text.count("timestamp,system,config") == 1
        assert len(list(csv.DictReader(io.StringIO(text)))) == 2


class TestCli:
    def test_exports_a_row(self, run_dir: Path):
        from click.testing import CliRunner

        from ezpz.cli import main

        res = CliRunner().invoke(
            main,
            ["export-amsc", str(run_dir), "--config", "c",
             "--system", "S", "--nodes", "1", "--gpus", "12"],
        )
        assert res.exit_code == 0, res.output
        rows = list(csv.DictReader(io.StringIO(res.output)))
        assert len(rows) == 1
        assert float(rows[0]["throughput_tokens_per_sec"]) > 34000

    def test_append_writes_header_only_once(self, run_dir: Path, tmp_path: Path):
        from click.testing import CliRunner

        from ezpz.cli import main

        dest = tmp_path / "results" / "runs.csv"
        argv = ["export-amsc", str(run_dir), "--config", "c",
                "--system", "S", "--nodes", "1", "--gpus", "12",
                "--append", str(dest)]
        runner = CliRunner()
        assert runner.invoke(main, argv).exit_code == 0
        assert runner.invoke(main, argv).exit_code == 0
        text = dest.read_text(encoding="utf-8")
        assert text.count("timestamp,system,config") == 1
        assert len(list(csv.DictReader(io.StringIO(text)))) == 2

    def test_missing_identity_fails_cleanly(self, run_dir: Path):
        """A stack trace here would bury the actionable message."""
        from click.testing import CliRunner

        from ezpz.cli import main

        res = CliRunner().invoke(
            main, ["export-amsc", str(run_dir), "--config", "c"]
        )
        assert res.exit_code != 0
        assert "--nodes" in res.output
        assert "Traceback" not in res.output
