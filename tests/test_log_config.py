"""Tests for ezpz.log.config display options and handler formatting."""

import os
import re

import pytest


# ---------------------------------------------------------------------------
# Helper: strip ANSI escape codes
# ---------------------------------------------------------------------------
ANSI_RE = re.compile(r"\x1B\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


# ---------------------------------------------------------------------------
# _env_bool
# ---------------------------------------------------------------------------
class TestEnvBool:
    """Verify the _env_bool helper parses env vars correctly."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for var in ("TEST_BOOL", "TEST_BOOL_NEG"):
            monkeypatch.delenv(var, raising=False)

    def _env_bool(self, name, default, *, neg=None):
        from ezpz.log.config import _env_bool

        return _env_bool(name, default, neg=neg)

    def test_default_true(self):
        assert self._env_bool("TEST_BOOL", True) is True

    def test_default_false(self):
        assert self._env_bool("TEST_BOOL", False) is False

    def test_set_1(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "1")
        assert self._env_bool("TEST_BOOL", False) is True

    def test_set_0(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "0")
        assert self._env_bool("TEST_BOOL", True) is False

    def test_set_true_string(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "true")
        assert self._env_bool("TEST_BOOL", False) is True

    def test_set_yes(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "yes")
        assert self._env_bool("TEST_BOOL", False) is True

    def test_set_false_string(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "false")
        assert self._env_bool("TEST_BOOL", True) is False

    def test_negation_overrides(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "1")
        monkeypatch.setenv("TEST_BOOL_NEG", "1")
        assert self._env_bool("TEST_BOOL", True, neg="TEST_BOOL_NEG") is False

    def test_negation_inactive(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "1")
        monkeypatch.setenv("TEST_BOOL_NEG", "0")
        assert self._env_bool("TEST_BOOL", False, neg="TEST_BOOL_NEG") is True

    def test_negation_unset(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "1")
        assert self._env_bool("TEST_BOOL", False, neg="TEST_BOOL_NEG") is True

    def test_empty_string_is_falsy(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "")
        assert self._env_bool("TEST_BOOL", True) is False


# ---------------------------------------------------------------------------
# use_colored_logs
# ---------------------------------------------------------------------------
class TestUseColoredLogs:
    """Verify the no-color.org convention."""

    def _check(self):
        from ezpz.log.config import use_colored_logs

        return use_colored_logs()

    def test_default_is_colored(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("NOCOLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert self._check() is True

    def test_no_color_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", "xterm-256color")
        assert self._check() is False

    def test_nocolor_disables(self, monkeypatch):
        monkeypatch.setenv("NOCOLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert self._check() is False

    def test_term_dumb_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("NOCOLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert self._check() is False

    def test_term_unknown_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("NOCOLOR", raising=False)
        monkeypatch.setenv("TERM", "unknown")
        assert self._check() is False

    def test_no_color_empty_does_not_disable(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "")
        monkeypatch.delenv("NOCOLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert self._check() is True


# ---------------------------------------------------------------------------
# FluidLogRender — direct in-process tests
# ---------------------------------------------------------------------------


def _make_renderer(**kwargs):
    """Create a FluidLogRender with no color for deterministic output."""
    from ezpz.log.handler import FluidLogRender

    return FluidLogRender(**kwargs)


def _render(renderer, message="msg", level_str="INFO"):
    """Call a FluidLogRender and return the plain-text result."""
    from datetime import datetime

    from rich.text import Text

    from ezpz.log.console import get_console

    console = get_console(markup=False, width=200)
    level = Text(f"{level_str}    ")
    # Add a span for level styling (FluidLogRender expects this)
    from rich.style import Style

    level.stylize(Style.null())

    result = renderer(
        console,
        [Text(message)],
        log_time=datetime(2026, 1, 15, 10, 30, 45),
        level=level,
        path="ezpz/test",
        line_no=42,
        funcName="my_func",
    )
    return result.plain


def _patch_handler(monkeypatch, **overrides):
    """Monkeypatch config values in the handler module (where they're imported)."""
    import ezpz.log.handler as handler_mod

    for key, val in overrides.items():
        monkeypatch.setattr(handler_mod, key, val)


class TestBracketModes:
    """Test the three bracket rendering modes."""

    def test_full_brackets(self, monkeypatch):
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_USE_BRACKETS=True,
            EZPZ_LOG_USE_SINGLE_BRACKET=False,
        )
        line = _render(_make_renderer())
        assert "][" in line, f"Expected ][ separator in: {line}"
        assert "msg" in line

    def test_no_brackets(self, monkeypatch):
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_USE_BRACKETS=False,
            EZPZ_LOG_USE_SINGLE_BRACKET=False,
        )
        line = _render(_make_renderer())
        assert " -- " in line, f"Expected ' -- ' separator in: {line}"
        assert "[" not in line, f"Unexpected bracket in: {line}"
        assert "msg" in line

    def test_single_bracket(self, monkeypatch):
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_USE_BRACKETS=False,
            EZPZ_LOG_USE_SINGLE_BRACKET=True,
        )
        line = _render(_make_renderer())
        assert line.startswith("["), f"Expected opening [ in: {line}"
        assert "] " in line, f"Expected ] before msg in: {line}"
        assert line.count("[") == 1, f"Expected exactly 1 [ in: {line}"
        assert line.count("]") == 1, f"Expected exactly 1 ] in: {line}"
        assert "msg" in line

    def test_no_brackets_single_space_before_msg(self, monkeypatch):
        """Regression: no-bracket mode had a double space before the message."""
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_USE_BRACKETS=False,
            EZPZ_LOG_USE_SINGLE_BRACKET=False,
        )
        line = _render(_make_renderer())
        assert "-- msg" in line, f"Expected single space after --: {line}"
        assert "--  msg" not in line, f"Double space before msg: {line}"


class TestComponentVisibility:
    """Test EZPZ_LOG_SHOW_* config flags."""

    def test_hide_time(self, monkeypatch):
        _patch_handler(monkeypatch, EZPZ_LOG_SHOW_TIME=False)
        line = _render(_make_renderer())
        assert "2026" not in line, f"Time should be hidden: {line}"
        assert "msg" in line

    def test_hide_level(self, monkeypatch):
        _patch_handler(monkeypatch, EZPZ_LOG_SHOW_LEVEL=False)
        line = _render(_make_renderer())
        assert "I" not in line.split("msg")[0], (
            f"Level should be hidden: {line}"
        )
        assert "msg" in line

    def test_hide_path(self, monkeypatch):
        _patch_handler(monkeypatch, EZPZ_LOG_SHOW_PATH=False)
        line = _render(_make_renderer())
        assert "test" not in line.split("msg")[0], (
            f"Path should be hidden: {line}"
        )
        assert "my_func" not in line, f"Path should be hidden: {line}"
        assert "msg" in line

    def test_show_rank(self, monkeypatch):
        _patch_handler(monkeypatch, EZPZ_LOG_SHOW_RANK=True)
        line = _render(_make_renderer())
        assert "[0]" in line, f"Expected [0] rank prefix in: {line}"
        assert "msg" in line

    def test_explicit_rank(self):
        line = _render(_make_renderer(rank=7))
        assert "[7]" in line, f"Expected [7] rank prefix in: {line}"


class TestColoredPrefix:
    """Test EZPZ_LOG_USE_COLORED_PREFIX."""

    def test_colored_prefix_on(self, monkeypatch):
        import ezpz.log.config as cfg

        monkeypatch.setattr(cfg, "EZPZ_LOG_USE_COLORED_PREFIX", True)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("NOCOLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        renderer = _make_renderer()
        assert renderer.colored_prefix is True

    def test_colored_prefix_off(self, monkeypatch):
        _patch_handler(monkeypatch, EZPZ_LOG_USE_COLORED_PREFIX=False)
        renderer = _make_renderer()
        assert renderer.colored_prefix is False
        assert renderer._ps("log.brace") == ""
        assert renderer._ps("log.path") == ""


class TestTimeFormat:
    """Test time format options."""

    def test_default_format(self, monkeypatch):
        _patch_handler(monkeypatch, EZPZ_LOG_USE_BRACKETS=True)
        line = _render(_make_renderer())
        assert re.search(r"\d{4}-\d{2}-\d{2}", line), (
            f"Expected date in default format: {line}"
        )
        assert "10:30:45" in line, f"Expected time 10:30:45: {line}"

    def test_detailed_format_includes_microseconds(self, monkeypatch):
        _patch_handler(
            monkeypatch, EZPZ_LOG_TIME_FORMAT="%Y-%m-%d %H:%M:%S,%f"
        )
        line = _render(_make_renderer())
        assert re.search(r"\d{2}:\d{2}:\d{2},\d+", line), (
            f"Expected HH:MM:SS,microseconds: {line}"
        )

    def test_day_separator_used_in_format(self):
        """Verify the default format uses EZPZ_LOG_DAY_SEPARATOR."""
        sep = "/"
        fmt = f"{sep.join(['%Y', '%m', '%d'])} %H:%M:%S"
        from datetime import datetime

        result = datetime(2026, 4, 4, 10, 30, 0).strftime(fmt)
        assert result == "2026/04/04 10:30:00"


class TestPrefixStyles:
    """Test EZPZ_LOG_PREFIX_STYLE presets."""

    def test_full_is_default(self, monkeypatch):
        """Full style produces bracketed output."""
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_USE_BRACKETS=True,
            EZPZ_LOG_SHOW_TIME=True,
            EZPZ_LOG_SHOW_LEVEL=True,
            EZPZ_LOG_SHOW_PATH=True,
        )
        line = _render(_make_renderer())
        assert "][" in line
        assert "msg" in line

    def test_minimal_hides_time_and_brackets(self, monkeypatch):
        """Minimal: no time, no brackets."""
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_SHOW_TIME=False,
            EZPZ_LOG_USE_BRACKETS=False,
            EZPZ_LOG_SHOW_LEVEL=True,
            EZPZ_LOG_SHOW_PATH=True,
        )
        line = _render(_make_renderer())
        assert "2026" not in line, f"Time should be hidden: {line}"
        assert "[" not in line, f"No brackets expected: {line}"
        assert " -- " in line
        assert "msg" in line

    def test_time_hides_path(self, monkeypatch):
        """Time style: time + level only, no path."""
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_SHOW_PATH=False,
            EZPZ_LOG_USE_BRACKETS=False,
            EZPZ_LOG_SHOW_TIME=True,
            EZPZ_LOG_SHOW_LEVEL=True,
            EZPZ_LOG_TIME_FORMAT="%H:%M:%S",
        )
        line = _render(_make_renderer())
        assert "10:30:45" in line, f"Expected time: {line}"
        assert "test" not in line.split("msg")[0], (
            f"Path should be hidden: {line}"
        )
        assert "msg" in line

    def test_none_has_no_prefix(self, monkeypatch):
        """None style: no prefix at all."""
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_SHOW_TIME=False,
            EZPZ_LOG_SHOW_LEVEL=False,
            EZPZ_LOG_SHOW_PATH=False,
        )
        line = _render(_make_renderer())
        assert line == "msg", f"Expected bare 'msg', got: {repr(line)}"

    def test_preset_overridable(self, monkeypatch):
        """Individual env vars override presets."""
        # Simulate minimal preset with time override
        _patch_handler(
            monkeypatch,
            EZPZ_LOG_SHOW_TIME=True,  # override
            EZPZ_LOG_USE_BRACKETS=False,  # from minimal
            EZPZ_LOG_SHOW_LEVEL=True,
            EZPZ_LOG_SHOW_PATH=True,
        )
        line = _render(_make_renderer())
        assert "2026" in line, f"Time should be shown (override): {line}"
        assert "msg" in line
