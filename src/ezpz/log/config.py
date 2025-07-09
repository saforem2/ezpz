"""
src/ezpz/logging/__init__.py
"""

from __future__ import absolute_import, annotations, division, print_function
import shutil
import os
from rich.style import Style
from typing import Dict
from rich.default_styles import DEFAULT_STYLES


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


# 1) simple “flag” styles
_flag_defs = {
    "none": {},
    "reset": dict(color="default", bgcolor="default"),
    "dim": {"dim": True},
    "bright": {"dim": False},
    "bold": {"bold": True},
    "strong": {"bold": True},
    "italic": {"italic": True},
    "emphasize": {"italic": True},
    "underline": {"underline": True},
    "blink": {"blink": True},
    "blink2": {"blink2": True},
    "reverse": {"reverse": True},
    "strike": {"strike": True},
    "code": {"reverse": True, "bold": True},
}
flag_styles = {k: Style(**v) for k, v in _flag_defs.items()}

# 2) colors (basic + “color.” prefix)
_colors = [
    "black","red","green","yellow","blue",
    "magenta","cyan","white","bright_black",
    "bright_red","bright_green","bright_yellow",
    "bright_blue","bright_magenta","bright_cyan",
    "bright_white",
]
basic_colors     = {c: Style(color=c)           for c in _colors}
prefixed_colors  = {f"color.{c}": Style(color=c) for c in _colors}

# 3) miscellaneous logging + repr overrides
_logging_defs = {
    "log.level.warn": {"color": "bright_yellow"},
    "logging.level.info": {"color": "green"},
    "log.colon": {"color": "bright_blue"},
    # …etc…
    "logging.time": {},   # Style.null()
    "logging.date": {"dim": True},
}
log_styles = {k: Style.null() if not v else Style(**v)
              for k, v in _logging_defs.items()}

_repr_defs = {
    "repr.attr": {"color": "blue"},
    "repr.attrib_name": {"color":"bright_blue","italic":True},
    "repr.attrib_equal": {"color":"yellow","bold":True},
    # …etc…
}
repr_styles = {k: Style(**v) for k, v in _repr_defs.items()}

# 4) assemble in one shot
if use_colored_logs():
    STYLES = {
        **flag_styles,
        **basic_colors,
        **prefixed_colors,
        **log_styles,
        **repr_styles,
    }
else:
    STYLES = {k: Style.null() for k in DEFAULT_STYLES}
DEFAULT_STYLES |= STYLES


# TERM = os.environ.get('TERM', None)
# NO__COLOR = use_colorized_logs()
# NO_COLOR = os.environ.get(
#     'NO_COLOR',
#     os.environ.get(
#         'NOCOLOR',
#         os.environ.get(
#             'COLOR', os.environ.get('COLORS', os.environ.get('DUMB', False))
#         ),
#     ),
# )

#
# if not use_colored_logs():
#     STYLES = {f"{k}": Style.null() for k in DEFAULT_STYLES.keys()}
#     DEFAULT_STYLES |= STYLES
#     # STYLES = {f'{k}': Style.null() for k in STYLES.items()}
# else:
#     STYLES: Dict[str, Style] = {
#         "none": Style.null(),
#         "reset": Style(
#             color="default",
#             bgcolor="default",
#             dim=False,
#             bold=False,
#             italic=False,
#             underline=False,
#             blink=False,
#             blink2=False,
#             reverse=False,
#             conceal=False,
#             strike=False,
#         ),
#         "dim": Style(dim=True),
#         "bright": Style(dim=False),
#         "bold": Style(bold=True),
#         "strong": Style(bold=True),
#         "code": Style(reverse=True, bold=True),
#         "italic": Style(italic=True),
#         "emphasize": Style(italic=True),
#         "underline": Style(underline=True),
#         "blink": Style(blink=True),
#         "blink2": Style(blink2=True),
#         "reverse": Style(reverse=True),
#         "strike": Style(strike=True),
#         "black": Style(color="black"),
#         "red": Style(color="red"),
#         "green": Style(color="green"),
#         "yellow": Style(color="yellow"),
#         "magenta": Style(color="magenta"),
#         "cyan": Style(color="cyan"),
#         "white": Style(color="white"),
#         "color.black": Style(color="black"),
#         "color.red": Style(color="red"),
#         "color.green": Style(color="green"),
#         "color.yellow": Style(color="yellow"),
#         "color.blue": Style(color="blue"),
#         "color.magenta": Style(color="magenta"),
#         "color.cyan": Style(color="cyan"),
#         "color.white": Style(color="white"),
#         "color.bright_black": Style(color="bright_black"),
#         "color.bright_red": Style(color="bright_red"),
#         "color.bright_green": Style(color="bright_green"),
#         "color.bright_yellow": Style(color="bright_yellow"),
#         "color.bright_blue": Style(color="bright_blue"),
#         "color.bright_magenta": Style(color="bright_magenta"),
#         "color.bright_cyan": Style(color="bright_cyan"),
#         "color.bright_white": Style(color="bright_white"),
#         "url": Style(conceal=True, underline=True, color="blue"),
#         "num": Style(color="blue"),
#         "repr.brace": Style(color="black", dim=True),
#         "log.brace": Style(color="white", dim=False),
#         "repr.comma": Style(color="bright_yellow"),
#         "repr.colon": Style(color="green"),
#         "repr.function": Style(color="bright_green", italic=True),
#         "repr.dash": Style(color="#838383"),
#         "logging.keyword": Style(bold=True, color="bright_yellow"),
#         "logging.level.notset": Style(dim=True),
#         "logging.level.debug": Style(color="bright_blue", bold=True),
#         "logging.level.error": Style(color="bright_red", bold=True),
#         "log.level.warn": Style(color="bright_yellow"),
#         "logging.level.info": Style(color="green"),
#         "log.level.warning": Style(color="bright_yellow"),
#         "logging.level.warn": Style(color="bright_yellow"),
#         "logging.level.warning": Style(color="bright_yellow"),
#         "log.colon": Style(color="bright_blue"),
#         "log.linenumber": Style(color="white", bold=False, dim=True),
#         "log.parent": Style(color="cyan", italic=True),
#         "log.path": Style(color="blue", bold=False, italic=False),
#         "log.time": Style(color="black"),
#         "logging.time": Style.null(),
#         "logging.date": Style.null(),  # , italic=False),
#         "hidden": Style(color="bright_black", dim=True),
#         "repr.attr": Style(color="blue"),
#         "repr.attrib_name": Style(
#             color="bright_blue", bold=False, italic=True
#         ),
#         "repr.attrib_equal": Style(bold=True, color="yellow"),
#         "repr.attrib_value": Style(color="magenta", italic=False),
#         "repr.ellipsis": Style(color="bright_yellow"),
#         "repr.indent": Style(color="bright_green", dim=True),
#         "repr.error": Style(color="bright_green", bold=True),
#         "repr.str": Style(color="bright_green", italic=True, bold=False),
#         "repr.ipv4": Style(bold=True, color="bright_green"),
#         "repr.ipv6": Style(bold=True, color="bright_green"),
#         "repr.eui48": Style(bold=True, color="bright_green"),
#         "repr.eui64": Style(bold=True, color="bright_green"),
#         "repr.tag_name": Style(color="bright_magenta", bold=True),
#         "repr.number": Style(color="bright_magenta", bold=False, italic=False),
#         "repr.number_complex": Style(
#             color="bright_magenta", bold=True, italic=False
#         ),  # same
#         "repr.bool_true": Style(color="bright_green", italic=True),
#         "repr.bool_false": Style(color="bright_red", italic=True),
#         "repr.none": Style(color="bright_magenta", italic=True),
#         "repr.null": Style(color="bright_magenta", italic=True),
#         "repr.url": Style(
#             underline=True, color="bright_blue", italic=False, bold=False
#         ),
#         "repr.uuid": Style(color="yellow", bold=False),
#         "repr.call": Style(color="magenta", bold=True),
#         "repr.path": Style(color="green"),
#         "repr.filename": Style(color="magenta"),
#         "rule.line": Style(color="bright_green"),
#         "json.brace": Style(bold=True, color="bright_yellow"),
#         "json.bool_true": Style(color="bright_green", italic=True),
#         "json.bool_false": Style(color="bright_red", italic=True),
#         "json.null": Style(color="bright_red", italic=True),
#         "json.number": Style(color="cyan", bold=True, italic=False),
#         "json.str": Style(color="bright_green", italic=False, bold=False),
#         "json.key": Style(color="bright_blue", bold=True),
#     }
#     DEFAULT_STYLES |= STYLES


#     # "num": Style(color='#409CDC', bold=True),
#     # 'repr.number': Style(color='#409CD0', bold=False),
#     # "none": Style.null(),
#     "reset": Style(
#         color="default",
#         bgcolor="default",
#         dim=False,
#         bold=False,
#         italic=False,
#         underline=False,
#         blink=False,
#         blink2=False,
#         reverse=False,
#         conceal=False,
#         strike=False,
#     ),
#     "dim": Style(dim=True),
#     "bright": Style(dim=False),
#     "bold": Style(bold=True),
#     "strong": Style(bold=True),
#     "code": Style(reverse=True, bold=True),
#     "italic": Style(italic=True),
#     "emphasize": Style(italic=True, bold=True),
#     "underline": Style(underline=True),
#     "blink": Style(blink=True),
#     "blink2": Style(blink2=True),
#     "reverse": Style(reverse=True),
#     "encircle": Style(encircle=True),
#     "overline": Style(overline=True),
#     "strike": Style(strike=True),
#     "black": Style(color="black"),
#     "red": Style(color="red"),
#     "green": Style(color="green"),
#     "yellow": Style(color="yellow"),
#     "magenta": Style(color="magenta"),
#     "cyan": Style(color="cyan"),
#     "white": Style(color="white"),
#     "inspect.attr": Style(color="yellow", italic=True),
#     "inspect.attr.dunder": Style(color="yellow", italic=True, dim=True),
#     "inspect.callable": Style(bold=True, color="red"),
#     "inspect.async_def": Style(italic=True, color="bright_cyan"),
#     "inspect.def": Style(italic=True, color="bright_cyan"),
#     "inspect.class": Style(italic=True, color="bright_cyan"),
#     "inspect.error": Style(bold=True, color="red"),
#     "inspect.equals": Style(),
#     "inspect.help": Style(color="cyan"),
#     "inspect.doc": Style(dim=True),
#     "inspect.value.border": GREEN,
#     "live.ellipsis": Style(bold=True, color=DARK["red"]),
#     "layout.tree.row": Style(dim=False, color=DARK["red"]),
#     "layout.tree.column": Style(dim=False, color=BLUE_),
#     "logging.keyword": Style(bold=True, color="yellow"),
#     "logging.level.notset": Style(dim=True),
#     "logging.level.debug": Style(color="bright_green"),
#     # "logging.level.info": Style(color=BLUE_),
#     # "logging.level.warning": Style(color="yellow"),
#     # "log.level.warn": Style(color="yellow"),
#     # "log.level.warning": Style(color="yellow"),
#     "logging.level.error": Style(color="bright_red", bold=True),
#     "logging.level.critical": Style(color="bright_red", bold=True, reverse=True),
#     "log.level": Style.null(),
#     # "log.time": Style(color=DARK["cyan"], dim=True),
#     "log.message": Style.null(),
#     # 'repr.attr': Style(color=DARK['blue_grey']),
#     'repr.attr': Style(color='bright_magenta'),
#     'repr.attrib_name': Style(color="bright_magenta", bold=False, italic=True),
#     "repr.attrib_equal": Style(bold=True, color="yellow"),
#     "repr.attrib_value": Style(color="bright_magenta", italic=False),
#     "repr.ellipsis": Style(color="yellow"),
#     "repr.indent": Style(color="bright_green", dim=True),
#     "repr.error": Style(color="bright_green", bold=True),
#     "repr.str": Style(color="green", italic=False, bold=False),
#     "repr.brace": Style(bold=True),
#     "repr.comma": Style(bold=True),
#     "repr.ipv4": Style(bold=True, color="bright_green"),
#     "repr.ipv6": Style(bold=True, color="bright_green"),
#     "repr.eui48": Style(bold=True, color="bright_green"),
#     "repr.eui64": Style(bold=True, color="bright_green"),
#     "repr.tag_start": Style(bold=True),
#     "repr.tag_end": Style(bold=True),
#     "repr.tag_name": Style(color="bright_magenta", bold=True),
#     "repr.tag_contents": Style(color="default"),
#     'repr.number': Style(color="cyan", bold=True, italic=False),
#     "repr.number_complex": Style(color="bright_cyan", bold=True, italic=False),  # same
#     "repr.bool_true": Style(color="bright_green", italic=True),
#     "repr.bool_false": Style(color="bright_red", italic=True),
#     "repr.none": Style(color=MAGENTA_, italic=True),
#     "repr.url": Style(underline=True, color="bright_blue", italic=False, bold=False),
#     "repr.uuid": Style(color="bright_yellow", bold=False),
#     "repr.call": Style(color="magenta", bold=True),
#     "repr.path": Style(color="green"),
#     "repr.filename": Style(color="magenta"),
#     "rule.line": Style(color="bright_green"),
#     # "rule.text": Style.null(),
#     "json.brace": Style(bold=True, color="bright_yellow"),
#     "json.bool_true": Style(color="bright_green", italic=True),
#     "json.bool_false": Style(color="bright_red", italic=True),
#     "json.null": Style(color="ansired", italic=True),
#     "json.number": Style(color="cyan", bold=True, italic=False),
#     "json.str": Style(color="bright_green", italic=False, bold=False),
#     "json.key": Style(color="bright_blue", bold=True),
#     "prompt": Style.null(),
#     "prompt.choices": Style(color="magenta", bold=True),
#     "prompt.default": Style(color="cyan", bold=True),
#     "prompt.invalid": Style(color="red"),
#     "prompt.invalid.choice": Style(color="red"),
#     # "pretty": Style.null(),
#     "scope.border": Style(color=BLUE_),
#     "scope.key": Style(color="yellow", italic=True),
#     "scope.key.special": Style(color="yellow", italic=True, dim=True),
#     "scope.equals": Style(color=DARK["red"]),
#     "table.header": Style(bold=True),
#     "table.footer": Style(bold=True),
#     # "table.cell": Style.null(),
#     "table.title": Style(italic=True),
#     "table.caption": Style(italic=True, dim=True),
#     "traceback.error": Style(color="red", italic=True),
#     "traceback.border.syntax_error": Style(color="bright_red"),
#     "traceback.border": Style(color="red"),
#     "traceback.text": Style.null(),
#     "traceback.title": Style(color="bright_red", bold=True),
#     "traceback.exc_type": Style(color="bright_red", bold=True),
#     "traceback.exc_value": Style.null(),
#     "traceback.offset": Style(color="bright_red", bold=True),
#     "bar.back": Style(color="grey23"),
#     "bar.complete": Style(color=DARK["green"]),
#     "bar.finished": Style(color=ORANGE_),
#     "bar.pulse": Style(color="rgb(249,38,114)"),
#     "progress.description": Style.null(),
#     "progress.filesize": Style(color=DARK["green"]),
#     "progress.filesize.total": Style(color=DARK["green"]),
#     "progress.download": Style(color=DARK["green"]),
#     "progress.elapsed": Style(color="yellow"),
#     "progress.percentage": Style(color=MAGENTA_),
#     "progress.remaining": Style(color=DARK["cyan"]),
#     "progress.data.speed": Style(color=DARK["red"]),
#     "progress.spinner": Style(color=DARK["green"]),
#     "status.spinner": Style(color=DARK["green"]),
#     # "tree": Style(frame=True),
#     # "tree.line": Style(color='green'),
#     # "markdown.paragraph": Style(),
#     # "markdown.text": Style(),
#     "markdown.em": Style(italic=True),
#     "markdown.emph": Style(italic=True),  # For commonmark backwards compatibility
#     "markdown.strong": Style(bold=True),
#     "markdown.code": Style(bold=True, color=LIGHT_GREY_, bgcolor=BLACK_),
#     "markdown.code_block": Style(color=WHITE_, bgcolor=BLACK_),
#     "markdown.block_quote": Style(color="white", dim=True),
#     "markdown.list": Style(color="cyan", bold=True),
#     # "markdown.item": Style(),
#     "markdown.item.bullet": Style(color="yellow", bold=True),
#     "markdown.item.number": Style(color="yellow", bold=True),
#     "markdown.hr": Style(color="yellow"),
#     "markdown.h1.border": Style(),
#     "markdown.h1": Style(bold=True),
#     "markdown.h2": Style(bold=True, underline=True),
#     "markdown.h3": Style(bold=True),
#     "markdown.h4": Style(bold=True, dim=True),
#     "markdown.h5": Style(underline=True),
#     "markdown.h6": Style(italic=True),
#     "markdown.h7": Style(italic=True, dim=True),
#     "markdown.link": Style(color="bright_blue", underline=True),
#     "markdown.link_url": Style(color="bright_blue", underline=True),
#     "markdown.s": Style(strike=True, color="white", dim=True),
#     "iso8601.date": Style(color=BLUE_),
#     "iso8601.time": Style(color=MAGENTA_),
#     "iso8601.timezone": Style(color="yellow"),
# }
#
