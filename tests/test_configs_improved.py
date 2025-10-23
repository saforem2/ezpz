"""Tests for the ezpz.configs module."""

from pathlib import Path
from unittest.mock import patch

import pytest

# Mock the problematic imports at the module level
with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
    "ezpz.configs.get_scheduler", return_value="UNKNOWN"
), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
    try:
        import ezpz.configs as configs

        CONFIGS_AVAILABLE = True
    except ImportError:
        CONFIGS_AVAILABLE = False


@pytest.mark.skipif(not CONFIGS_AVAILABLE, reason="ezpz.configs not available")
class TestConfigs:
    """Test the configs module."""

    def test_command_exists_with_existing_command(self):
        """Test command_exists function with an existing command."""
        # Test with a command that should exist on most systems
        result = configs.command_exists("python")
        assert isinstance(result, bool)

    def test_command_exists_with_nonexistent_command(self):
        """Test command_exists function with a nonexistent command."""
        result = configs.command_exists("nonexistent_command_xyz")
        assert result is False

    def test_get_logging_config(self):
        """Test get_logging_config function."""
        config = configs.get_logging_config()
        assert isinstance(config, dict)
        assert "version" in config
        assert "handlers" in config
        assert "loggers" in config

    def test_paths_exist(self):
        """Test that important paths exist."""
        assert isinstance(configs.HERE, Path)
        assert isinstance(configs.PROJECT_DIR, Path)
        assert isinstance(configs.CONF_DIR, Path)
        assert isinstance(configs.BIN_DIR, Path)

    def test_config_file_paths(self):
        """Test that config file paths are defined."""
        assert hasattr(configs, "DS_CONFIG_YAML")
        assert hasattr(configs, "DS_CONFIG_JSON")
        assert hasattr(configs, "DS_CONFIG_PATH")

    def test_scheduler_constants(self):
        """Test that scheduler constants are defined."""
        assert hasattr(configs, "SCHEDULERS")
        assert hasattr(configs, "FRAMEWORKS")
        assert hasattr(configs, "BACKENDS")
        assert isinstance(configs.SCHEDULERS, dict)
        assert isinstance(configs.FRAMEWORKS, dict)
        assert isinstance(configs.BACKENDS, dict)
