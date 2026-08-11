"""The fine-tuning guide must describe the code that exists.

Docs drift silently: a renamed flag or a removed method leaves prose
that reads fine and is wrong. These check the guide's concrete claims
-- flags, imports, method names -- against the real parser and modules,
so a rename breaks the suite instead of the reader.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

GUIDE = Path(__file__).resolve().parents[1] / "docs/guides/finetuning.md"

pytestmark = pytest.mark.skipif(
    not GUIDE.exists(), reason="docs not present in this checkout"
)


def _text() -> str:
    return GUIDE.read_text()


def _help() -> str:
    from ezpz.examples import fsdp_tp

    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        fsdp_tp.parse_args(["--help"])
    return buf.getvalue()


def test_every_documented_flag_exists():
    help_txt = _help()
    assert len(help_txt) > 1000, "could not capture --help; test is vacuous"
    flags = sorted(set(re.findall(r"`(--[a-z][a-z-]+)`", _text())))
    assert flags, "no flags found in the guide; the regex likely broke"
    missing = [f for f in flags if f not in help_txt]
    assert not missing, f"documented but not in --help: {missing}"


def test_every_documented_import_resolves():
    import ezpz.tinker as T

    names = {
        n.strip()
        for grp in re.findall(r"from ezpz\.tinker import ([\w, ]+)", _text())
        for n in grp.split(",")
        if n.strip()
    }
    assert names, "no ezpz.tinker imports found in the guide"
    missing = [n for n in names if not hasattr(T, n)]
    assert not missing, f"documented but not exported: {missing}"


def test_every_documented_client_method_exists():
    from ezpz.tinker import LocalTrainingClient

    methods = set(re.findall(r"client\.(\w+)\(", _text()))
    assert methods, "no client calls found in the guide"
    missing = [m for m in methods if not hasattr(LocalTrainingClient, m)]
    assert not missing, f"documented but not on the client: {missing}"


def test_lora_target_values_match_the_config():
    """The guide lists the accepted --lora-target roles."""
    from ezpz.tinker.lora import LoraConfig

    doc = _text()
    for role in ("attn", "mlp", "unembed"):
        assert role in doc, f"the guide never mentions the {role!r} target"
    # each role maps to a real LoraConfig field
    for field in ("train_attn", "train_mlp", "train_unembed"):
        assert field in LoraConfig.__dataclass_fields__


def test_tp_support_claim_matches_the_code():
    """The guide says LoRA works at tp>1; the guard must be gone."""
    import inspect

    from ezpz.examples import fsdp_tp

    src = inspect.getsource(fsdp_tp.train)
    claims_support = "tensor parallelism" in _text().lower()
    has_guard = "--lora-rank with --tp > 1 is not supported" in src
    assert not (claims_support and has_guard), (
        "the guide advertises LoRA under tensor parallelism but the code "
        "still refuses it at startup"
    )
