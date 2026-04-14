"""Test individual modules can be imported and used directly."""

import numpy as np


def test_configs_module():
    """configs module should provide logging config and path constants."""
    import ezpz.configs as configs

    assert configs.HERE is not None
    assert configs.HERE.exists()
    assert isinstance(configs.command_exists("python"), bool)

    config = configs.get_logging_config()
    assert isinstance(config, dict)
    assert "handlers" in config


def test_utils_module():
    """utils module should provide timestamp, normalize, format_pair, grab_tensor."""
    import ezpz.utils as utils

    # get_timestamp returns a date string
    ts = utils.get_timestamp()
    assert len(ts) >= 10  # at least YYYY-MM-DD

    # format_pair formats key=value
    assert utils.format_pair("lr", 0.001) == "lr=0.001000"

    # normalize replaces special chars
    assert utils.normalize("my_model.v2") == "my-model-v2"

    # grab_tensor passes through numpy
    arr = np.array([1.0, 2.0])
    assert np.array_equal(utils.grab_tensor(arr), arr)


def test_lazy_module():
    """lazy module should defer imports until attribute access."""
    import ezpz.lazy as lazy

    os_mod = lazy.lazy_import("os")
    # Must actually USE the module to verify lazy resolution
    assert os_mod.sep in ("/", "\\")
