"""Shared chart styling for ezpz plots.

Applies the ``ambivalent`` matplotlib stylesheet and registers Iosevka
(downloaded to ``~/.local/share/fonts/Iosevka/``) so every chart uses
the same monospace face.

Import side-effect: calling ``apply_style()`` is idempotent. The
caller imports this module before creating any figures.

Reproduce font setup once per machine (login node, with proxy):
    URL=https://github.com/be5invis/Iosevka/releases/download/v34.6.1/PkgTTC-Iosevka-34.6.1.zip
    mkdir -p ~/.local/share/fonts/Iosevka
    curl -L -o /tmp/iosevka.zip --proxy http://proxy.alcf.anl.gov:3128 $URL
    unzip -q -o /tmp/iosevka.zip -d ~/.local/share/fonts/Iosevka/
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import matplotlib  # noqa: F401
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

_IOSEVKA_DIR = Path.home() / ".local/share/fonts/Iosevka"

_applied = False


def _ambivalent_stylefile() -> str | None:
    """Locate ambivalent's bundled .mplstyle WITHOUT importing the package.

    `import ambivalent` executes its __init__, which imports IPython -- absent
    from the XPU training .venv, so a plain import raises ModuleNotFoundError
    there. The stylesheet itself is a static file in the package dir, so we
    find the package location via importlib's finder (no execution) and read
    the .mplstyle directly. This lets the in-process LR finder pick up the
    house style on the same env that runs training.
    """
    try:
        spec = importlib.util.find_spec("ambivalent")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for loc in spec.submodule_search_locations:
        cand = Path(loc) / "stylefiles" / "ambivalent.mplstyle"
        if cand.is_file():
            return str(cand)
    return None


def apply_style(font_family: str | list[str] = "Iosevka") -> None:
    """Apply ambivalent style + Iosevka font (idempotent).

    Import-safe in environments without IPython (e.g. the XPU training
    .venv): loads the ambivalent stylesheet from its file rather than
    importing the package. Falls back to matplotlib defaults with a warning
    if the stylesheet can't be located.
    """
    global _applied
    if _applied:
        return
    if _IOSEVKA_DIR.is_dir():
        for f in _IOSEVKA_DIR.iterdir():
            if f.suffix.lower() in (".ttf", ".ttc", ".otf"):
                _fm.fontManager.addfont(str(f))
    stylefile = _ambivalent_stylefile()
    if stylefile is not None:
        plt.style.use(stylefile)
    else:
        warnings.warn(
            "ambivalent stylesheet not found; using matplotlib defaults",
            stacklevel=2,
        )
    if isinstance(font_family, str):
        font_family = [font_family, "DejaVu Sans Mono", "monospace"]
    plt.rcParams["font.family"] = font_family
    _applied = True
