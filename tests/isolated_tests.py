"""Isolated unit tests for ezpz modules."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set up environment
os.environ["WANDB_MODE"] = "disabled"
os.environ["EZPZ_LOG_LEVEL"] = "ERROR"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLazyModule(unittest.TestCase):
    """Test the lazy module in isolation."""

    def test_lazy_import(self):
        """Test lazy_import function."""
        # We need to import directly from the file to avoid initialization issues
        lazy_path = Path(__file__).parent.parent / "src" / "ezpz" / "lazy.py"
        spec = importlib.util.spec_from_file_location("lazy", lazy_path)
        lazy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lazy)

        # Test importing a standard library module
        os_module = lazy.lazy_import("os")
        self.assertIsNotNone(os_module)
        self.assertTrue(hasattr(os_module, "path"))

        # Test importing a non-existent module (should not raise immediately)
        nonexistent = lazy.lazy_import("nonexistent_module_xyz")
        self.assertIsNotNone(nonexistent)  # Should return a lazy object


class TestConfigsFunctions(unittest.TestCase):
    """Test functions in the configs module in isolation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock environment for configs module
        self.original_env = os.environ.copy()
        os.environ["PBS_O_WORKDIR"] = "/tmp"
        os.environ["SLURM_SUBMIT_DIR"] = "/tmp"

    def tearDown(self):
        """Tear down test fixtures."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_command_exists(self):
        """Test command_exists function."""
        # We'll test this by directly importing the function
        configs_path = Path(__file__).parent.parent / "src" / "ezpz" / "configs.py"
        spec = importlib.util.spec_from_file_location("configs", configs_path)
        configs = importlib.util.module_from_spec(spec)
        # Mock the imports that might cause issues
        configs.sh = MagicMock()
        configs.Path = MagicMock()
        configs.logging = MagicMock()
        configs.OmegaConf = MagicMock()
        configs.Console = MagicMock()
        configs.Text = MagicMock()
        configs.Tree = MagicMock()

        spec.loader.exec_module(configs)

        # Test with a command that should exist
        result = configs.command_exists("python")
        self.assertIsInstance(result, bool)

        # Test with a command that should not exist
        result = configs.command_exists("nonexistent_command_xyz")
        self.assertIsInstance(result, bool)


class TestUtilsFunctions(unittest.TestCase):
    """Test functions in the utils module in isolation."""

    def test_basic_functions(self):
        """Test basic utility functions."""
        # Import utils module directly
        utils_path = (
            Path(__file__).parent.parent / "src" / "ezpz" / "utils" / "__init__.py"
        )
        spec = importlib.util.spec_from_file_location("utils", utils_path)
        utils = importlib.util.module_from_spec(spec)
        # Mock imports that might cause issues
        utils.sys = MagicMock()
        utils.pdb = MagicMock()
        utils.os = MagicMock()
        utils.re = MagicMock()
        utils.logging = MagicMock()
        utils.DummyMPI = MagicMock()
        utils.DummyTorch = MagicMock()
        utils.ModelStatistics = MagicMock()
        utils.xr = MagicMock()
        utils.np = MagicMock()
        utils.asdict = MagicMock()
        utils.ScalarLike = MagicMock()
        utils.PathLike = MagicMock()
        utils.ZeroConfig = MagicMock()
        utils.torch = MagicMock()
        utils.tdist = MagicMock()
        utils.Path = MagicMock()
        spec.loader.exec_module(utils)

        # Test get_timestamp
        import datetime

        utils.datetime = MagicMock()
        utils.datetime.datetime.now.return_value = datetime.datetime(
            2023, 1, 1, 12, 0, 0
        )
        utils.datetime.datetime.now().strftime.return_value = "2023-01-01-120000"

        timestamp = utils.get_timestamp()
        self.assertIsInstance(timestamp, str)
        self.assertGreater(len(timestamp), 0)

        # Test format_pair
        result = utils.format_pair("test", 5)
        self.assertEqual(result, "test=5")

        # Test normalize
        result = utils.normalize("test_name.sub-name")
        self.assertEqual(result, "test-name-sub-name")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
