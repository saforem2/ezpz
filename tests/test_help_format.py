"""The formatter helper must expose stable names and valid HelpFormatter
subclasses whether or not rich-argparse is installed."""
import argparse

from ezpz.cli import help_format


def test_exports_are_helpformatter_subclasses():
    for name in (
        "DefaultsFormatter",
        "ColorFormatter",
        "RawDescAndDefaultsFormatter",
    ):
        cls = getattr(help_format, name)
        assert issubclass(cls, argparse.HelpFormatter), name


def test_have_rich_argparse_is_bool():
    assert isinstance(help_format.HAVE_RICH_ARGPARSE, bool)


def test_formatters_actually_format_help():
    # A parser built with each formatter must produce non-empty help text.
    for name in (
        "DefaultsFormatter",
        "ColorFormatter",
        "RawDescAndDefaultsFormatter",
    ):
        cls = getattr(help_format, name)
        parser = argparse.ArgumentParser(
            prog="t", description="d", formatter_class=cls
        )
        parser.add_argument("--foo", default=1, help="a foo")
        assert "foo" in parser.format_help()


def test_defaults_formatter_appends_defaults(monkeypatch):
    parser = argparse.ArgumentParser(
        prog="t", formatter_class=help_format.DefaultsFormatter
    )
    parser.add_argument("--foo", default=7, help="a foo")
    # NO_COLOR keeps assertion stable regardless of interpreter/TTY.
    # monkeypatch.setenv restores any pre-existing value after the test.
    monkeypatch.setenv("NO_COLOR", "1")
    assert "default: 7" in parser.format_help()
