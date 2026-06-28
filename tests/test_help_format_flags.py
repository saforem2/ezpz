"""flags.py builders must use the shared colorized formatters."""
from ezpz.cli import help_format
from ezpz.cli.flags import (
    build_launch_parser,
    build_test_parser,
    build_doctor_parser,
)


def test_test_parser_uses_defaults_formatter():
    p = build_test_parser(prog="ezpz test")
    assert p.formatter_class is help_format.DefaultsFormatter


def test_launch_parser_uses_rawdesc_defaults_formatter():
    p = build_launch_parser(prog="ezpz launch")
    assert p.formatter_class is help_format.RawDescAndDefaultsFormatter


def test_doctor_parser_uses_defaults_formatter():
    p = build_doctor_parser(prog="ezpz doctor")
    assert p.formatter_class is help_format.DefaultsFormatter
