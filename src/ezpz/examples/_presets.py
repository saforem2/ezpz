"""Shared MODEL_PRESETS helpers used across `ezpz.examples.*`.

This module exists because the preset-application logic was previously
copy-pasted into all 5 example modules (vit, fsdp, fsdp_tp, test,
diffusion). A bug fixed in one (the `flag in argv` check rejecting
``--flag=value`` argv tokens) silently regressed in the others — every
preset would clobber an explicit ``--batch-size=32`` user override on
the un-fixed example modules.

Importing from a single source of truth prevents that drift.

The intentionally tiny surface is:

  arg_provided(argv, flags)
      Return True if any of ``flags`` was provided on the command line,
      matching BOTH the space-separated form (``["--seq-len", "8192"]``)
      AND the ``=``-fused form (``"--seq-len=8192"``).

This lives in `_presets.py` rather than `__init__.py` so importing
`ezpz.examples` stays cheap (no implicit pull of preset-application
machinery for callers that just want `get_example_outdir`).
"""

from __future__ import annotations

from typing import Sequence


def arg_provided(argv: Sequence[str], flags: Sequence[str]) -> bool:
    """Return True if any of ``flags`` was provided on the command line.

    Matches both space-separated (``["--batch-size", "32"]``, two
    tokens) and ``=``-fused (``"--batch-size=32"``, one token) forms.
    Without the prefix check the preset-override logic in
    ``apply_model_preset`` would silently clobber any explicit
    ``--flag=value`` user override with the preset's default — every
    preset value would "win" against ``=``-style flags.

    Example::

        >>> arg_provided(["--model", "xxxl", "--batch-size=32"],
        ...              ["--batch-size", "--batch_size"])
        True
        >>> arg_provided(["--model", "xxxl"],
        ...              ["--batch-size", "--batch_size"])
        False

    Args:
        argv: argv-style list of tokens (e.g. ``sys.argv[1:]`` or the
            list passed to ``parser.parse_args``).
        flags: names of the flag(s) to test for. Any of the listed
            spellings counts as "provided" — pass both
            ``["--batch-size", "--batch_size"]`` to accept either
            hyphen or underscore form.
    """
    for flag in flags:
        prefix = flag + "="
        for token in argv:
            if token == flag or token.startswith(prefix):
                return True
    return False
