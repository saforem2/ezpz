"""Test the normalize function from the actual ezpz.utils module."""

import ezpz.utils as utils


def test_normalize_underscores():
    assert utils.normalize("test_name") == "test-name"


def test_normalize_dots():
    assert utils.normalize("test.name") == "test-name"


def test_normalize_mixed():
    assert utils.normalize("test_name.sub-name") == "test-name-sub-name"


def test_normalize_uppercase():
    assert utils.normalize("TEST_NAME") == "test-name"


def test_normalize_already_clean():
    assert utils.normalize("test-name") == "test-name"
