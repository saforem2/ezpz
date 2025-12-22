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
    term = os.environ.get("TERM", None)
    plain = os.environ.get(
        "NO_COLOR",
        os.environ.get(
            "NOCOLOR",
            os.environ.get(
                "COLOR",
                os.environ.get("COLORS", os.environ.get("DUMB", False)),
            ),
        ),
    )
    return not plain and term not in ["dumb", "unknown"]


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
    "log.colon": Style(color="bright_blue"),
    "logging.date": Style(dim=True),
    "logging.time": Style(dim=True),
    "log.time": Style(color="black", dim=True),
    "log.linenumber": Style(color="magenta"),
    "log.brace": Style(),
    "log.path": Style(color="magenta", bold=False, italic=False),
    "log.parent": Style(color="bright_magenta", italic=False, bold=False),
    "logging.keyword": Style(bold=True, color="bright_yellow"),
    "logging.level.notset": Style(dim=True),
    "logging.level.debug": Style(color="bright_blue", bold=True),
    # ----------------- INFO ---------------------------------------
    "log.level.info": Style(color="green"),
    "logging.level.info": Style(color="green"),
    # ----------------- WARN ---------------------------------------
    "log.level.warn": Style(color="bright_yellow"),
    "log.level.warning": Style(color="bright_yellow"),
    "logging.level.warn": Style(color="bright_yellow"),
    "logging.level.warning": Style(color="bright_yellow"),
    # ----------------- ERROR --------------------------------------
    "logging.level.error": Style(color="bright_red", bold=True),
    "log.level.error": Style(color="bright_red", bold=True),
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
