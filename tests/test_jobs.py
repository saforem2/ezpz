"""Tests for the ezpz.jobs module."""

import os
import tempfile
from pathlib import Path

import pytest

try:
    import ezpz.jobs as jobs

    JOBS_AVAILABLE = True
except ImportError:
    JOBS_AVAILABLE = False


@pytest.mark.skipif(not JOBS_AVAILABLE, reason="ezpz.jobs not available")
class TestJobs:
    def test_get_jobid(self, mock_pbs_env):
        """Test get_jobid function."""
        jobid = jobs.get_jobid()
        assert isinstance(jobid, str)
        # Should be the job ID without the domain part
        assert jobid == "12345"

    def test_get_jobdir_from_env(self, mock_pbs_env):
        """Test get_jobdir_from_env function."""
        jobdir = jobs.get_jobdir_from_env()
        assert isinstance(jobdir, Path)
        assert jobdir.exists()
        # Should be in the home directory under SCHEDULER-jobs/jobid
        assert str(jobdir).startswith(str(Path.home()))

    def test_jobfile_functions(self, mock_pbs_env):
        """Test jobfile path functions."""
        # Test shell jobfile
        jobfile_sh = jobs.get_jobfile_sh()
        assert isinstance(jobfile_sh, Path)
        assert jobfile_sh.suffix == ".sh"

        # Test yaml jobfile
        jobfile_yaml = jobs.get_jobfile_yaml()
        assert isinstance(jobfile_yaml, Path)
        assert jobfile_yaml.suffix == ".yaml"

        # Test json jobfile
        jobfile_json = jobs.get_jobfile_json()
        assert isinstance(jobfile_json, Path)
        assert jobfile_json.suffix == ".json"

    def test_check_scheduler(self):
        """Test check_scheduler function."""
        # Test with valid scheduler
        assert jobs.check_scheduler("PBS") is True

        # Test with invalid scheduler should raise
        with pytest.raises(AssertionError):
            jobs.check_scheduler("INVALID_SCHEDULER")
