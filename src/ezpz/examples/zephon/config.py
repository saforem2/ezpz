"""Zephon TOML recipe handling for ezpz/examples/*.

Thin shim over the Zephon recipe format: resolves relative paths inside
the TOML w.r.t. that TOML file (same as torchtitan_recipes.zephon) and
exposes a couple of helpers callers use.

"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(recipe: str | Path) -> dict[str, Any]:
    path = Path(recipe).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Zephon recipe not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def recipe_dir(recipe: str | Path) -> Path:
    """Directory this recipe's relative [[sources]] paths are resolved against."""
    return Path(recipe).resolve().parent
