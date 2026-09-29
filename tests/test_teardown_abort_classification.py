"""A post-completion teardown abort is not a failed run.

#266: on Aurora and Sunspot, ``hf_trainer`` trains all 100 steps, writes
its model shards, prints ``train_runtime``, and *then* rank 4 raises
SIGABRT from a joinable-thread teardown on the XCCL path. The launcher
SIGTERMs the survivors and the aggregate surfaces as exit 143.

Reporting that as ``FAILED`` is wrong twice: it hides that the run
produced valid results, and it buries an upstream teardown bug under a
generic signal.

The classifier requires **both** a teardown marker and a completion
marker. These tests pin that conjunction, because the dangerous failure
mode is the permissive one -- a classifier that swallows genuine crashes
would silently turn real breakage green.
"""

from __future__ import annotations

import pytest

from ezpz.examples.run_all import _classify_nonzero_exit

TEARDOWN = "terminate called without an active exception"
COMPLETION = "{'train_runtime': '73.49', 'train_loss': '2.56'}"


def _log(tmp_path, text: str):
    p = tmp_path / "example.log"
    p.write_text(text, encoding="utf-8")
    return p


def test_completed_then_aborted_is_recognized(tmp_path):
    """The #266 case: real output, then a destructor abort."""
    log = _log(tmp_path, f"step 100/100\n{COMPLETION}\n{TEARDOWN}\n")
    reason = _classify_nonzero_exit(log, 143)
    assert reason is not None
    assert "teardown" in reason
    assert "266" in reason


def test_abort_without_completion_is_still_a_failure(tmp_path):
    """A crash that never finished training must NOT be excused.

    This is the assertion that matters. If the classifier keyed only on
    the abort marker, any early C++ crash would be relabelled a pass.
    """
    log = _log(tmp_path, f"step 3/100\n{TEARDOWN}\n")
    assert _classify_nonzero_exit(log, 143) is None


def test_completion_without_abort_is_not_reclassified(tmp_path):
    """Finished cleanly but exited non-zero for some other reason.

    Without a teardown marker we have no evidence about *why* it exited,
    so it stays a failure.
    """
    log = _log(tmp_path, f"step 100/100\n{COMPLETION}\n")
    assert _classify_nonzero_exit(log, 1) is None


def test_ordinary_failure_is_untouched(tmp_path):
    """An OOM or traceback is a failure, full stop."""
    log = _log(
        tmp_path,
        "step 12/100\nTraceback (most recent call last):\n"
        "torch.OutOfMemoryError: CUDA out of memory\n",
    )
    assert _classify_nonzero_exit(log, 1) is None


def test_shard_write_also_counts_as_completion(tmp_path):
    """FSDP checkpoint write is the other end-of-run marker."""
    log = _log(
        tmp_path,
        f"Writing model shards: 100%|####| 1/1\n{TEARDOWN}\n",
    )
    assert _classify_nonzero_exit(log, 143) is not None


def test_missing_logfile_does_not_raise(tmp_path):
    """A vanished or unreadable log must not crash the summary."""
    assert _classify_nonzero_exit(tmp_path / "nope.log", 143) is None


def test_empty_log_is_a_failure(tmp_path):
    """No output at all is the quota/startup case -- not a pass.

    Seen twice for real: an empty log from ``OSError: [Errno 122] Disk
    quota exceeded`` before any example ran.
    """
    assert _classify_nonzero_exit(_log(tmp_path, ""), 124) is None


@pytest.mark.parametrize("rc", [1, 134, 139, 143])
def test_classification_is_independent_of_exit_code(tmp_path, rc):
    """Evidence in the log decides, not the signal number.

    ``rc=143`` is identical for a crash and a deadlock under mpiexec
    teardown, which is exactly why this classifier reads the log.
    """
    log = _log(tmp_path, f"{COMPLETION}\n{TEARDOWN}\n")
    assert _classify_nonzero_exit(log, rc) is not None
