"""
src/ezpz/logging/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function

import os
import shutil

from rich.default_styles import DEFAULT_STYLES
from rich.style import Style

size = shutil.get_terminal_size()
WIDTH = size.columns
HEIGHT = size.lines
os.environ["COLUMNS"] = f"{WIDTH}"


# -- Configure useful Paths -----------------------
# warnings.filterwarnings('ignore')
# HERE = Path(os.path.abspath(__file__)).parent
# PROJECT_DIR = HERE.parent.parent
# PROJECT_ROOT = PROJECT_DIR
# CONF_DIR = HERE.joinpath('conf')
# BIN_DIR = HERE.joinpath('bin')
# SAVEJOBENV = BIN_DIR.joinpath('savejobenv')
# GETJOBENV = BIN_DIR.joinpath('getjobenv')
# DS_CONFIG_PATH = CONF_DIR.joinpath('ds_config.yaml')
# LOGS_DIR = PROJECT_DIR.joinpath('logs')
# OUTPUTS_DIR = HERE.joinpath('outputs')
# QUARTO_OUTPUTS_DIR = PROJECT_DIR.joinpath('qmd', 'outputs')

# CONF_DIR.mkdir(exist_ok=True, parents=True)
# LOGS_DIR.mkdir(exist_ok=True, parents=True)
# QUARTO_OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
# OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
# OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


def use_colored_logs() -> bool:
    """Return ``False`` when colour should be suppressed.

    Follows the `no-color.org <https://no-color.org>`_ convention: colour is
    **on** by default and only disabled when ``NO_COLOR`` or ``NOCOLOR`` is set
    (to any non-empty value) or ``TERM`` is ``dumb`` / ``unknown``.
    """
    term = os.environ.get("TERM", "")
    if term in ("dumb", "unknown"):
        return False
    for var in ("NO_COLOR", "NOCOLOR"):
        val = os.environ.get(var)
        if val is not None and val != "":
            return False
    return True


DARK = {
    "red": "#FF5252",
    "pink": "#EB53EB",
    "cyan": "#09A979",
    "blue": "#2094F3",
    # "green": "#69DB7C",
    "green": "#50a14f",
    "orange": "#FD971F",
    "magenta": "#FF00FF",
    "blue_grey": "#7D8697",
    "light_pink": "#F06292",
}

RED_ = "#FF5252"
ORANGE_ = "#FD971F"

GREEN_ = "#69DB7C"
CYAN_ = "#09A979"
DARK_GREEN_ = "#14AC3C"
INFO_ = "#00B0FF"
URL_ = "#119EDE"
BLUE_ = "#2094F3"

PURPLE_ = "#AE81FF"
PURPLE_ALT_ = "#9775FA"
LIGHT_PINK_ = "#F06292"
ATTR_ = "#F06292"
PINK_ = "#EB53EB"
MAGENTA_ = "#FF00FF"

WHITE_ = "#F8F8F8"
BLACK_ = "#161616"
LIGHT_GREY_ = "#bdbdbd"
GREY_ = "#838383"

GREEN = Style(color=DARK["green"])
PINK = Style(color=PINK_)
BLUE = Style(color=BLUE_)
RED = Style(color=DARK["red"])
MAGENTA = Style(color=MAGENTA_)
CYAN = Style(color=DARK["cyan"])
GREY_MED = "#838383"

flag_styles = {
    "none": Style(),
    "reset": Style(),
    "dim": Style(dim=True),
    "bold": Style(bold=True),
    "strong": Style(bold=True),
    "italic": Style(italic=True),
    "emphasize": Style(italic=True),
    "underline": Style(underline=True),
    "blink": Style(blink=True),
    "blink2": Style(blink2=True),
    "reverse": Style(reverse=True),
    "strike": Style(strike=True),
    "code": Style(reverse=True),
}

# 1) simple “flag” styles
# _flag_defs = {
#     "none": {},
#     "reset": dict(color="default", bgcolor="default"),
#     "dim": {"dim": True},
#     # "bright": {"bright": True},
#     "bold": {"bold": True},
#     "strong": {"bold": True},
#     "italic": {"italic": True},
#     "emphasize": {"italic": True},
#     "underline": {"underline": True},
#     "blink": {"blink": True},
#     "blink2": {"blink2": True},
#     "reverse": {"reverse": True},
#     "strike": {"strike": True},
#     "code": {"reverse": True, "bold": True},
# }
# flag_styles = {k: Style(**v) for k, v in _flag_defs.items()}

# 2) colors (basic + “color.” prefix)
_colors = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
]
basic_colors = {c: Style(color=c) for c in _colors}
prefixed_colors = {f"color.{c}": Style(color=c) for c in _colors}

# 3) miscellaneous logging + repr overrides
log_styles = {
    "log.day_time_separator": Style(color="blue", dim=True),
    "log.day_color": Style(dim=True),
    # "log.time_color": Style(color="bright_white"),
    "log.time": Style(color="bright_black", dim=True),
    "log.time_color": Style(color="white"),
    "log.colon": Style(dim=True),
    "logging.date": Style(dim=True),
    "logging.time": Style(dim=True),
    # "log.time": Style(color="black", dim=True),
    "log.linenumber": Style(
        color="red",
        bold=False,
        # bold=True,
    ),
    "log.brace": Style(),
    "log.path": Style(color="magenta", bold=False, italic=False),
    "log.parent": Style(
        color="bright_magenta",
        italic=False,
        bold=False,  # , dim=True
    ),
    "logging.keyword": Style(bold=True, color="bright_yellow"),
    "logging.level.notset": Style(dim=True),
    "logging.level.debug": Style(color="bright_blue", bold=True),
    # ----------------- INFO ---------------------------------------
    "log.level.info": Style(color="green"),
    "logging.level.info": Style(color="green"),
    # ----------------- WARN ---------------------------------------
    "log.level.warn": Style(
        color="bright_yellow",
    ),
    "log.level.warning": Style(color="bright_yellow"),
    "logging.level.warn": Style(color="bright_yellow"),
    "logging.level.warning": Style(color="bright_yellow"),
    # ----------------- ERROR --------------------------------------
    "log.level.error": Style(color="bright_red", bold=True),
    "logging.level.error": Style(color="bright_red", bold=True),
    # ----------------- CRITICAL ------------------------------------
    "log.level.critical": Style(
        color="bright_red",
        bold=True,
        reverse=True,
    ),
    "logging.level.critical": Style(
        color="bright_red",
        bold=True,
        reverse=True,
    ),
}
repr_styles = {
    "repr.attr": Style(color="blue"),
    "repr.attrib_equal": Style(color="yellow", bold=True),
    "repr.function": Style(color="bright_green", italic=True),
    "repr.brace": Style(),
    "repr.comma": Style(color="bright_yellow"),
    "repr.colon": Style(color="green"),
    "repr.dash": Style(color="white"),
    "repr.attrib_name": Style(color="bright_blue", bold=False, italic=False),
    # "repr.attrib_value": Style(color="magenta"),
    "repr.ellipsis": Style(color="bright_yellow"),
}

# 4) assemble in one shot
if use_colored_logs():
    STYLES = (
        flag_styles | basic_colors | prefixed_colors | log_styles | repr_styles
    )
else:
    STYLES = {k: Style.null() for k in DEFAULT_STYLES}
DEFAULT_STYLES |= STYLES


# ---------------------------------------------------------------------------
# Log display options (read once at import time)
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool, *, neg: str | None = None) -> bool:
    """Read an env var as a boolean, with optional negation var for compat."""
    from ezpz.log.console import to_bool

    val = os.environ.get(name)
    result = to_bool(val) if val is not None else default
    if neg:
        neg_val = os.environ.get(neg)
        if neg_val is not None and to_bool(neg_val):
            result = False
    return result


# Separators
EZPZ_LOG_DAY_SEPARATOR = os.environ.get("EZPZ_LOG_DAY_SEPARATOR", "-")
EZPZ_LOG_DAY_TIME_SEPARATOR = os.environ.get(
    "EZPZ_LOG_DAY_TIME_SEPARATOR", " "
)

# Time format
EZPZ_LOG_TIME_DETAILED = _env_bool("EZPZ_LOG_TIME_DETAILED", False)
EZPZ_LOG_TIME_FORMAT = os.environ.get(
    "EZPZ_LOG_TIME_FORMAT",
    (
        f"{EZPZ_LOG_DAY_SEPARATOR.join(['%Y', '%m', '%d'])}{EZPZ_LOG_DAY_TIME_SEPARATOR}%H:%M:%S,%f"
        if EZPZ_LOG_TIME_DETAILED
        else f"{EZPZ_LOG_DAY_SEPARATOR.join(['%Y', '%m', '%d'])}{EZPZ_LOG_DAY_TIME_SEPARATOR}%H:%M:%S"
    ),
)

# ---------------------------------------------------------------------------
# Prefix style presets
# ---------------------------------------------------------------------------
# EZPZ_LOG_PREFIX_STYLE sets sensible defaults for the display options below.
# Individual EZPZ_LOG_* env vars still override after the preset is applied.
#
#   full    — [time][level][path] msg           (default)
#   minimal — level path -- msg                 (no time, no brackets)
#   time    — HH:MM:SS level -- msg             (time + level, no path)
#   plain   — time level path -- msg            (all components, no brackets, no color)
#   none    — msg                               (no prefix at all)

_PREFIX_STYLES: dict[str, dict[str, object]] = {
    "full": {},  # all defaults
    "minimal": {
        "EZPZ_LOG_SHOW_TIME": False,
        "EZPZ_LOG_USE_BRACKETS": False,
    },
    "time": {
        "EZPZ_LOG_SHOW_PATH": False,
        "EZPZ_LOG_USE_BRACKETS": False,
        "EZPZ_LOG_TIME_FORMAT": "%H:%M:%S",
    },
    "plain": {
        "EZPZ_LOG_USE_BRACKETS": False,
        "EZPZ_LOG_USE_COLORED_PREFIX": False,
    },
    "none": {
        "EZPZ_LOG_SHOW_TIME": False,
        "EZPZ_LOG_SHOW_LEVEL": False,
        "EZPZ_LOG_SHOW_PATH": False,
    },
}

EZPZ_LOG_PREFIX_STYLE = os.environ.get("EZPZ_LOG_PREFIX_STYLE", "").lower()
_preset = _PREFIX_STYLES.get(EZPZ_LOG_PREFIX_STYLE, {})


def _preset_default(key: str, fallback: object) -> object:
    """Return the preset value for *key* if set, otherwise *fallback*."""
    return _preset.get(key, fallback)


# Component visibility
EZPZ_LOG_SHOW_RANK = _env_bool(
    "EZPZ_LOG_RANK",
    bool(_preset_default("EZPZ_LOG_SHOW_RANK", False)),
    neg="EZPZ_NO_LOG_RANK",
)
EZPZ_LOG_SHOW_TIME = _env_bool(
    "EZPZ_LOG_SHOW_TIME",
    bool(_preset_default("EZPZ_LOG_SHOW_TIME", True)),
    neg="EZPZ_LOG_NO_SHOW_TIME",
)
EZPZ_LOG_SHOW_LEVEL = _env_bool(
    "EZPZ_LOG_SHOW_LEVEL",
    bool(_preset_default("EZPZ_LOG_SHOW_LEVEL", True)),
    neg="EZPZ_LOG_NO_SHOW_LEVEL",
)
EZPZ_LOG_SHOW_PATH = _env_bool(
    "EZPZ_LOG_SHOW_PATH",
    bool(_preset_default("EZPZ_LOG_SHOW_PATH", True)),
    neg="EZPZ_LOG_NO_SHOW_PATH",
)
EZPZ_LOG_ENABLE_LINK_PATH = _env_bool(
    "EZPZ_LOG_ENABLE_LINK_PATH",
    bool(_preset_default("EZPZ_LOG_ENABLE_LINK_PATH", False)),
    neg="EZPZ_LOG_NO_ENABLE_LINK_PATH",
)

# Bracket style
# If EZPZ_LOG_USE_SINGLE_BRACKET is explicitly set, it implies USE_BRACKETS=False
# (unless USE_BRACKETS is also explicitly set).
EZPZ_LOG_USE_SINGLE_BRACKET = _env_bool(
    "EZPZ_LOG_USE_SINGLE_BRACKET",
    bool(_preset_default("EZPZ_LOG_USE_SINGLE_BRACKET", False)),
)
_brackets_default = bool(_preset_default("EZPZ_LOG_USE_BRACKETS", True))
if EZPZ_LOG_USE_SINGLE_BRACKET and "EZPZ_LOG_USE_BRACKETS" not in os.environ:
    _brackets_default = False
EZPZ_LOG_USE_BRACKETS = _env_bool(
    "EZPZ_LOG_USE_BRACKETS",
    _brackets_default,
    neg="EZPZ_LOG_NO_USE_BRACKETS",
)

# Prefix coloring (time, level, path — everything before the message)
EZPZ_LOG_USE_COLORED_PREFIX = _env_bool(
    "EZPZ_LOG_USE_COLORED_PREFIX",
    bool(_preset_default("EZPZ_LOG_USE_COLORED_PREFIX", True)),
    neg="EZPZ_LOG_NO_USE_COLORED_PREFIX",
)

# Override time format from preset (only if EZPZ_LOG_TIME_FORMAT env var not set)
if (
    "EZPZ_LOG_TIME_FORMAT" in _preset
    and "EZPZ_LOG_TIME_FORMAT" not in os.environ
):
    EZPZ_LOG_TIME_FORMAT = str(_preset["EZPZ_LOG_TIME_FORMAT"])
