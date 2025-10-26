"""Test individual modules without importing the main ezpz package."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _run_configs_module_checks() -> bool:
    """Run smoke checks for the configs module; return True on success."""
    import ezpz.configs as configs

    assert configs.HERE is not None
    assert isinstance(configs.command_exists("python"), bool)
    assert isinstance(configs.get_logging_config(), dict)
    return True


def test_configs_module():
    """Test configs module directly."""
    assert _run_configs_module_checks()


def _run_utils_module_checks() -> bool:
    import numpy as np

    import ezpz.utils as utils

    assert isinstance(utils.get_timestamp(), str)
    assert utils.format_pair("test", 5) == "test=5"
    assert utils.normalize("test_name.sub-name") == "test-name-sub-name"
    np_array = np.array([1, 2, 3])
    assert np.array_equal(utils.grab_tensor(np_array), np_array)
    return True


def test_utils_module():
    """Test utils module directly."""
    assert _run_utils_module_checks()


def _run_lazy_module_checks() -> bool:
    import ezpz.lazy as lazy

    os_module = lazy.lazy_import("os")
    assert os_module is not None
    assert hasattr(os_module, "path")
    return True


def test_lazy_module():
    """Test lazy module directly."""
    assert _run_lazy_module_checks()


if __name__ == "__main__":
    print("Running individual module tests...")
    results = [
        _run_configs_module_checks(),
        _run_utils_module_checks(),
        _run_lazy_module_checks(),
    ]

    if all(results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
