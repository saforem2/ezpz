"""ezpz shell completion — prototype/draft.

Implements ``ezpz completion [zsh|bash|fish]`` using Click's built-in
``click.shell_completion`` (added in Click 8.0, no shtab needed).

Design
------
A single Click command emits the shell-specific completion script to stdout.
The user (or their dotfiles) redirects it into the shell's completion
directory.  This follows the Click docs pattern::

    _EZPZ_COMPLETE=bash_complete ezpz   # internal dispatch
    _EZPZ_COMPLETE=zsh_complete  ezpz
    _EZPZ_COMPLETE=fish_complete ezpz

and the public wrapper::

    ezpz completion bash   # prints the bash script
    ezpz completion zsh    # prints the zsh script
    ezpz completion fish   # prints the fish script (+ optional install)

Reference: https://click.palletsprojects.com/en/stable/shell-completion/

Current repo state (checked 2026-10-16)
---------------------------------------
* ``pyproject.toml`` pins ``click`` unbounded (effectively ``click>=8``) and
  the venv has ``click==8.5.0`` — shell completion APIs are present
  (``BashComplete``, ``ZshComplete``, ``FishComplete``, ``ShellComplete``,
  ``get_completion_class``).
* CLI entry point is ``src/ezpz/cli/__init__.py:main`` — a ``@click.group`` with
  subcommands registered via ``main.add_command(...)`` (status, cancel, extend,
  resources, incidents, submit, launch, update, test, tar-env, yeet, benchmark,
  doctor, kill, export-amsc). Several use ``ignore_unknown_options=True`` +
  ``UNPROCESSED`` pass-through; Click still completes their *options*/*names*.
* Rich-argparse / rich help formatters do not interfere — completion bypasses
  the formatter and calls ``Command.shell_complete(ctx, incomplete)``.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from typing import Iterable

import click
from click.shell_completion import ShellComplete, get_completion_class

# Keep in sync with ``src/ezpz/cli/__init__.py``'s prog name.
PROG_NAME = "ezpz"
COMPLETE_VAR = "_EZPZ_COMPLETE"
SUPPORTED_SHELLS = ("bash", "zsh", "fish")


# ---------------------------------------------------------------------------
# Pure helper: build the script string for a shell (testable without Click ctx)
# ---------------------------------------------------------------------------

def completion_script(
    shell: str,
    prog_name: str = PROG_NAME,
    complete_var: str = COMPLETE_VAR,
) -> str:
    """Return the shell snippet that enables completion for *shell*.

    Uses the Click-registered :class:`ShellComplete` subclass so the
    generated script stays in sync with the installed Click version
    (template changes between Click releases are picked up without edits).

    Raises:
        click.BadParameter: if *shell* is not one of :data:`SUPPORTED_SHELLS`.
    """
    cls = get_completion_class(shell)
    if cls is None:
        raise click.BadParameter(
            f"unsupported shell {shell!r}; choose from {', '.join(SUPPORTED_SHELLS)}",
            param_hint="shell",
        )
    # ``ShellComplete.__init__`` requires (cli, ctx_args, prog_name, complete_var).
    # Pass the *real* CLI group so ``source_vars`` sees the correct prog_name.
    # Import lazily to avoid importing the full CLI at import-time (some
    # subcommands import heavy deps).
    from ezpz.cli import main as ezpz_main  # local import — intentionally late

    complete: ShellComplete = cls(
        cli=ezpz_main,  # type: ignore[arg-type]
        ctx_args={},
        prog_name=prog_name,
        complete_var=complete_var,
    )
    return complete.source()


# ---------------------------------------------------------------------------
# Click command — what will be wired into ``src/ezpz/cli/__init__.py``
# ---------------------------------------------------------------------------

_COMPLETE_INSTRUCTIONS = {
    "bash": (
        "Bash install (pick one):\n"
        "  # one-off (current shell only)\n"
        "  eval \"$(_EZPZ_COMPLETE=bash_source ezpz)\"\n"
        "  # persistent — depends on bash-completion being installed\n"
        "  _EZPZ_COMPLETE=bash_source ezpz > ~/.local/share/bash-completion/completions/ezpz\n"
        "  # alternative persistent (no bash-completion package needed)\n"
        "  echo 'eval \"$(_EZPZ_COMPLETE=bash_source ezpz)\"' >> ~/.bashrc\n"
        "\n"
        "Notes: Bash completion requires bash >= 4.4 (Click checks at source-time).\n"
        "Homebrew bash on macOS qualifies; Apple /bin/bash 3.2 does NOT."
    ),
    "zsh": (
        "Zsh install:\n"
        "  # persistent (recommended)\n"
        "  mkdir -p ~/.zfunc\n"
        "  _EZPZ_COMPLETE=zsh_source ezpz > ~/.zfunc/_ezpz\n"
        "  # then ensure ~/.zshrc contains (before compinit):\n"
        "  #   fpath+=~/.zfunc\n"
        "  #   autoload -Uz compinit && compinit\n"
        "  # one-off (current shell only):\n"
        "  eval \"$(_EZPZ_COMPLETE=zsh_source ezpz)\"\n"
        "  # oh-my-zsh users: drop _ezpz into $ZSH_CUSTOM/completions/ instead"
    ),
    "fish": (
        "Fish install:\n"
        "  _EZPZ_COMPLETE=fish_source ezpz > ~/.config/fish/completions/ezpz.fish\n"
        "  # one-off is not idiomatic in fish; prefer the file above"
    ),
}


def _install_bash(dest: pathlib.Path, script: str) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(script, encoding="utf-8")
    return dest


def _install_zsh(dest: pathlib.Path, script: str) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(script, encoding="utf-8")
    return dest


def _install_fish(dest: pathlib.Path, script: str) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(script, encoding="utf-8")
    return dest


_INSTALLERS = {
    "bash": (
        _install_bash,
        pathlib.Path.home() / ".local/share/bash-completion/completions/ezpz",
    ),
    "zsh": (_install_zsh, pathlib.Path.home() / ".zfunc/_ezpz"),
    "fish": (_install_fish, pathlib.Path.home() / ".config/fish/completions/ezpz.fish"),
}


@click.command("completion")
@click.argument("shell", type=click.Choice(SUPPORTED_SHELLS), required=False)
@click.option(
    "--install",
    is_flag=True,
    help="Write the completion script to its conventional location (see --help).",
)
@click.option(
    "--dest",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Override install destination (implies --install).",
)
def completion_cmd(
    shell: str | None,
    install: bool,
    dest: pathlib.Path | None,
) -> None:
    """Emit (or install) shell completion for ezpz.

    \b
    Examples:
      ezpz completion bash > /tmp/ezpz.bash          # inspect
      ezpz completion bash --install                 # write to standard path
      ezpz completion zsh  --install --dest ~/.zfunc/_ezpz
      eval \"$(_EZPZ_COMPLETE=bash_source ezpz)\"     # one-liner (Click native)

    With no SHELL argument, prints install instructions for all shells.

    Under the hood this delegates to ``click.shell_completion`` — the script
    is generated by Click's ``{Shell}Complete.source()`` so template fixes
    in future Click releases are inherited without changes here. The env-var
    dispatch (``_EZPZ_COMPLETE=<shell>_source`` / ``<shell>_complete``) is
    handled by ``Command.main(complete_var=...)`` automatically on every
    invocation; no extra wiring is needed beyond registering this command.
    """
    if shell is None:
        # Help mode: show all install instructions.
        click.echo("Usage: ezpz completion [bash|zsh|fish] [--install] [--dest PATH]\n")
        for sh in SUPPORTED_SHELLS:
            click.echo(f"--- {sh} ---")
            click.echo(_COMPLETE_INSTRUCTIONS[sh])
            click.echo()
        click.echo("Tip: Click also supports the one-liner (no subcommand needed):")
        click.echo('  eval "$(_EZPZ_COMPLETE=bash_source ezpz)"   # bash')
        click.echo('  eval "$(_EZPZ_COMPLETE=zsh_source  ezpz)"   # zsh')
        ctx = click.get_current_context()
        ctx.exit(0)

    script = completion_script(shell)

    should_install = install or dest is not None
    if should_install:
        _fn, default_dest = _INSTALLERS[shell]
        target = dest if dest is not None else default_dest
        target = pathlib.Path(target).expanduser()
        _fn(target, script)
        click.echo(f"installed {shell} completion to {target}", err=True)
        # Shell-specific follow-up hint
        if shell == "zsh":
            click.echo(
                "ensure ~/.zshrc has: fpath+=~/.zfunc  and  autoload -Uz compinit && compinit",
                err=True,
            )
        elif shell == "bash":
            click.echo(
                "restart your shell or: source ~/.bashrc  (or eval the script)",
                err=True,
            )
        return

    # Default: emit to stdout so callers can redirect / eval.
    click.echo(script, nl=False)
    # When stdout is a TTY (user ran it bare), also print instructions to stderr
    # so they know what to do with the wall of shell code they just got.
    if sys.stdout.isatty():
        click.echo("\n", err=True)
        click.echo(_COMPLETE_INSTRUCTIONS[shell], err=True)


# ---------------------------------------------------------------------------
# Registration snippet (copy into src/ezpz/cli/__init__.py)
# ---------------------------------------------------------------------------
# from ezpz.cli.completion_cmd import completion_cmd  # new file src/ezpz/cli/completion_cmd.py
# main.add_command(completion_cmd, name="completion")
#
# That's it — Click's Command.main() already handles the _EZPZ_COMPLETE
# env-var dispatch (complete_var="_EZPZ_COMPLETE") before normal parsing.
# To make the one-liner `eval \"$(_EZPZ_COMPLETE=XX_source ezpz)\"` use the
# standard env-var name, set it explicitly where `main` is defined:
#
#   @click.group(context_settings=CONTEXT_SETTINGS)
#   @click.version_option(__version__)
#   def main() -> None: ...
#
#   # In __main__ / entry-point wrapper (or just rely on Click default):
#   # Click defaults complete_var to f"_{prog_name.upper()}_COMPLETE" which
#   # for prog_name="ezpz" is already "_EZPZ_COMPLETE" — so no change needed.
#   # Override only if you rename the binary:
#   #   main(complete_var="_EZPZ_COMPLETE")
#
# Alternatively, emit with Click's newer `_source` suffix (Click >= 8.2):
#   _EZPZ_COMPLETE=bash_source | zsh_source | fish_source  → prints script
#   _EZPZ_COMPLETE=bash_complete | zsh_complete | ...      → prints completions
# The draft above uses `_source` for the one-liners (correct for 8.5.0).

# ---------------------------------------------------------------------------
# Install instructions (also printed by `ezpz completion` with no args)
# ---------------------------------------------------------------------------
INSTALL_INSTRUCTIONS = r"""
Shell completion — install
===========================

Quick one-liners (no subcommand, uses Click native env-var):
  bash:  eval "$(_EZPZ_COMPLETE=bash_source ezpz)"
  zsh:   eval "$(_EZPZ_COMPLETE=zsh_source ezpz)"
  fish:  _EZPZ_COMPLETE=fish_source ezpz > ~/.config/fish/completions/ezpz.fish

Via the `ezpz completion` subcommand (preferred — friendlier errors):
  ezpz completion bash              # print bash script to stdout
  ezpz completion bash --install    # write to ~/.local/share/bash-completion/completions/ezpz
  ezpz completion zsh  --install    # write to ~/.zfunc/_ezpz
  ezpz completion fish --install    # write to ~/.config/fish/completions/ezpz.fish
  ezpz completion zsh --dest /custom/path/_ezpz --install

Detailed per-shell:

bash (requires bash >= 4.4; macOS Homebrew bash qualifies, Apple /bin/bash 3.2 does not)
  persistent (with bash-completion package):
    _EZPZ_COMPLETE=bash_source ezpz > ~/.local/share/bash-completion/completions/ezpz
  persistent (without bash-completion package):
    echo 'eval \"$(_EZPZ_COMPLETE=bash_source ezpz)\"' >> ~/.bashrc
  one-off:
    eval "$(_EZPZ_COMPLETE=bash_source ezpz)"

zsh
  persistent:
    mkdir -p ~/.zfunc
    _EZPZ_COMPLETE=zsh_source ezpz > ~/.zfunc/_ezpz
    # in ~/.zshrc, BEFORE compinit:
    fpath+=~/.zfunc
    autoload -Uz compinit && compinit
  one-off:
    eval "$(_EZPZ_COMPLETE=zsh_source ezpz)"
  oh-my-zsh: drop _ezpz into $ZSH_CUSTOM/completions/ instead of ~/.zfunc

fish
  persistent:
    _EZPZ_COMPLETE=fish_source ezpz > ~/.config/fish/completions/ezpz.fish
  (fish has no eval one-liner; the file is the idiomatic path)

Verification:
  bash:  type _ezpz_completion  # or: complete -p ezpz
  zsh:   which _ezpz            # or: echo $fpath
  fish:  complete -c ezpz
  all:   ezpz <TAB><TAB>  should list subcommands (status, submit, launch, …)
"""

# ---------------------------------------------------------------------------
# Auto-install vs manual — recommendation
# ---------------------------------------------------------------------------
AUTO_INSTALL_NOTE = r"""
Auto-install vs manual — recommendation: MANUAL (do NOT auto-install).

Why not a post-install hook:
  1. No reliable hook. pip/uv/wheel installs do not run arbitrary code
     (PEP 517 forbids setup.py install hooks; hatchling has no post-install
     hook). Workarounds (custom install wrappers, entry-point side effects
     on first run) are fragile and surprising.
  2. Shell detection is unreliable. $SHELL reflects the login shell, not the
     current shell; users often run multiple shells. Writing to the wrong
     dotfile corrupts it.
  3. Dotfile edits are hostile. Appending `eval ...` to ~/.bashrc/~/.zshrc
     without consent breaks idempotency (duplicates on reinstall) and
     violates user expectations. Every prior attempt in the ecosystem
     (pipx, poetry, etc.) moved AWAY from auto-editing dotfiles.
  4. Permissions / managed machines. On HPC login nodes (Aurora, Polaris,
     Perlmutter — ezpz's primary audience) home directories may be shared
     or dotfiles managed by admins; silent writes can conflict.
  5. Click's own docs recommend manual eval/file. The pattern
     `eval \"$(_PROG_COMPLETE=..._source prog)\"` is intentionally explicit.

What to do instead:
  * Ship the `ezpz completion` subcommand (this draft) so the script is
    always one command away.
  * Print a one-time hint from `ezpz doctor` (or first-run) like:
      "Shell completion not installed. Run: ezpz completion --help"
    — advisory only, never writing files.
  * Document the one-liners in README / docs site (zensical) under
    "Shell completion".
  * Optionally offer `ezpz completion <shell> --install` (implemented above)
    for users who WANT an explicit install — opt-in, with --dest override
    and idempotent file write.

If you later want a convenience installer, make it an explicit subcommand
(`ezpz completion --install`) rather than an implicit post-install side
effect. The draft implements this.
"""

# ---------------------------------------------------------------------------
# Self-test helper (optional)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Smoke-test the completion machinery without a real shell.

    Simulates what the shell wrapper does: sets COMP_WORDS/COMP_CWORD and
    invokes the CLI via the _COMPLETE env-var path. Requires `ezpz` importable.
    """
    import os as _os

    try:
        from ezpz.cli import main as _main
    except Exception as exc:  # pragma: no cover
        print(f"skip self-test: cannot import ezpz.cli ({exc})", file=sys.stderr)
        return

    from click.testing import CliRunner

    runner = CliRunner()

    # 1) completion_script() emits non-empty script for each shell (does not
    # require the `completion` subcommand to be registered yet — this is a
    # draft file, so test the helper directly).
    for sh in SUPPORTED_SHELLS:
        script = completion_script(sh)
        assert len(script) > 100, f"completion {sh} output suspiciously short"
        # Each template contains the prog name and complete var
        assert PROG_NAME in script, f"{sh} script missing prog name"
        assert COMPLETE_VAR in script, f"{sh} script missing complete var"
        print(f"ok: completion_script({sh}) ({len(script)} chars)")

    # 1b) the Click command itself — wire it into a throwaway group so we
    # can test its CLI without the real `ezpz` group having it registered.
    @click.group()
    def _probe_main():
        pass

    _probe_main.add_command(completion_cmd, name="completion")
    for sh in SUPPORTED_SHELLS:
        result = runner.invoke(_probe_main, ["completion", sh])
        assert result.exit_code == 0, f"completion {sh} failed: {result.output}"
        assert len(result.output) > 100
        print(f"ok: probe completion {sh} ({len(result.output)} chars)")
    result = runner.invoke(_probe_main, ["completion"])
    assert result.exit_code == 0
    assert "bash" in result.output.lower() and "zsh" in result.output.lower()
    print("ok: probe completion (no args) instructions")

    # 3) native Click env-var dispatch still works (shell_complete path)
    #    Simulate: `ezpz s` with incomplete="s" should suggest status/submit/etc
    #    Click reads COMP_WORDS/COMP_CWORD + _EZPZ_COMPLETE to return completions.
    env = {COMPLETE_VAR: "bash_complete", "COMP_WORDS": "ezpz s", "COMP_CWORD": "1"}
    result = runner.invoke(_main, [], env=env)
    # Click exits 0 after printing completions via the complete path
    # Output lines are "type,value"
    print(f"env dispatch output (truncated): {result.output[:500]!r}")

    print("self-test done.")


if __name__ == "__main__":
    # Allow `python /tmp/ezpz-completion-draft.py --self-test` for quick verify.
    if "--self-test" in sys.argv:
        _self_test()
    elif len(sys.argv) >= 2 and sys.argv[1] in SUPPORTED_SHELLS:
        # Convenience: `python /tmp/ezpz-completion-draft.py bash` prints script
        print(completion_script(sys.argv[1]))
    else:
        print(INSTALL_INSTRUCTIONS)
        print(AUTO_INSTALL_NOTE)
