"""Configuration file for pytest."""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path so we can import ezpz modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set environment variables for testing
os.environ["WANDB_MODE"] = "disabled"
os.environ["EZPZ_LOG_LEVEL"] = "CRITICAL"


@pytest.fixture(autouse=True, scope="session")
def _suppress_ezpz_loggers():
    """Silence all ezpz loggers during tests to keep output clean.

    Tests that need to verify log output should use caplog or
    temporarily lower the level on the specific logger they need.
    """
    # Suppress the root ezpz logger and all children
    for name in ("ezpz", "ezpz.tracker", "ezpz.launch", "ezpz.history"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


# Rendezvous variables are process-global, and a test that writes them
# through raw `os.environ` (rather than monkeypatch) leaks them into every
# test that follows. That is not a cosmetic problem: a later gloo
# rendezvous inherits the stale value, and if it points somewhere
# unreachable -- a fabricated `MASTER_ADDR`, a port owned by nothing --
# the rendezvous does not fail. It blocks until torch's default process
# group timeout, which is *30 minutes*. A handful of those turned this
# suite into a 2-hour run in which every individual file still passed in
# seconds, so nothing looked broken.
#
# Two defenses, because the failure is silent and expensive:
#   * this fixture scrubs the variables between tests, so a leak cannot
#     reach the next test, and
#   * `_no_rendezvous_leak` fails the *leaking* test by name, so the
#     scrub never quietly papers over a real bug.
_RENDEZVOUS_VARS = ("MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _no_rendezvous_leak():
    """Fail a test that leaks MASTER_ADDR/MASTER_PORT, and scrub them.

    Use `monkeypatch.setenv`/`delenv` for these in tests. Note that
    `delenv` alone is not enough when the code under test *writes* the
    variable: monkeypatch records it as absent, restores nothing, and
    the write survives teardown.
    """
    before = {k: os.environ.get(k) for k in _RENDEZVOUS_VARS}
    try:
        yield
    finally:
        after = {k: os.environ.get(k) for k in _RENDEZVOUS_VARS}
        leaked = {
            k: after[k]
            for k in _RENDEZVOUS_VARS
            if after[k] is not None and after[k] != before[k]
        }
        # Scrub first: the next test must not inherit this either way.
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if leaked:
            raise AssertionError(
                f"test leaked rendezvous env {leaked} into the process; "
                "a later gloo rendezvous inherits it and blocks for the "
                "30-minute default PG timeout. Set these via "
                "`monkeypatch.setenv` so teardown restores them."
            )


@pytest.fixture
def mock_dist_env():
    """Mock distributed environment variables."""
    original_env = os.environ.copy()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_pbs_env():
    """Mock PBS environment variables."""
    original_env = os.environ.copy()
    temp_home = tempfile.mkdtemp(prefix="ezpz-home-")
    os.environ["HOME"] = temp_home
    os.environ["PBS_JOBID"] = "12345.test"
    os.environ["PBS_NODEFILE"] = "/tmp/test_nodefile"
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        shutil.rmtree(temp_home, ignore_errors=True)
