"""``ezpz update`` — upgrade ezpz and refresh the cached ``utils.sh``.

Deliberately a Python subcommand rather than a shipped ``ezpz_update``
executable. A self-updating script fetched from the network rewrites
itself on a login node you may share, is awkward to pin for
reproducibility, and duplicates what re-running the installer already
does. As a subcommand it is versioned with the package, testable, and
visible in ``git log``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

DEFAULT_UTILS_URL = "https://ezpz.cool/utils.sh"

# Entry points a complete utils.sh must define. Used to detect a
# truncated download, which `bash -n` cannot: a response cut at a
# syntactically complete boundary parses cleanly.
_REQUIRED_FUNCTIONS = (
    "ezpz_setup_env",
    "ezpz_setup_python",
    "ezpz_get_machine_name",
)


def _share_dir() -> Path:
    return Path(
        os.environ.get("EZPZ_SHARE_DIR", Path.home() / ".local/share/ezpz")
    )


def _in_batch_job() -> bool:
    """True when running inside a PBS or SLURM allocation.

    Compute nodes have no outbound route, so a download there hangs for
    ~270 s and then leaves things unconfigured. Better to refuse.
    """
    # All the spellings the rest of the codebase recognises. PBS_JOBID
    # can be set in a compute-node subprocess where PBS_NODEFILE is not
    # (see ezpz/pbs.py), and SLURM uses both SLURM_JOB_ID and the older
    # SLURM_JOBID (see ezpz/slurm.py). Missing any of them starts the
    # very network operation this guard exists to prevent.
    return any(
        os.environ.get(v)
        for v in (
            "PBS_NODEFILE",
            "PBS_JOBID",
            "SLURM_JOB_ID",
            "SLURM_JOBID",
        )
    )


def _pip_install_cmd(spec: str) -> list[str]:
    """Return the install command for the active environment.

    Prefers ``uv pip`` when available, matching the project's own
    tooling, and falls back to ``python -m pip``.
    """
    if shutil.which("uv"):
        # --python is required, not optional: `uv pip install` targets a
        # venv discovered from the CURRENT DIRECTORY, so a bare call can
        # fail outside a project or silently upgrade an unrelated .venv
        # rather than the environment running this command.
        return [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            spec,
        ]
    return [sys.executable, "-m", "pip", "install", "--upgrade", spec]


@click.command(name="update")
@click.option(
    "--ref",
    default=None,
    metavar="REF",
    help="Install a specific git ref (branch, tag, or SHA) instead of main.",
)
@click.option(
    "--utils-only",
    is_flag=True,
    default=False,
    help="Refresh the cached utils.sh only; do not touch the package.",
)
@click.option(
    "--package-only",
    is_flag=True,
    default=False,
    help="Upgrade the package only; do not refresh utils.sh.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would run without doing it.",
)
def update_cmd(
    ref: str | None,
    utils_only: bool,
    package_only: bool,
    dry_run: bool,
) -> None:
    """Upgrade ezpz in the active environment and refresh `utils.sh`.

    Two steps, either of which can be skipped:

    1. Reinstall the package from git into whichever environment is
       active right now.
    2. Re-download the cached ``utils.sh`` that the installer placed.

    Refuses to run inside a batch job: compute nodes have no outbound
    network, so both steps would hang and then fail confusingly.
    """
    if utils_only and package_only:
        raise click.UsageError(
            "--utils-only and --package-only are mutually exclusive"
        )
    if _in_batch_job() and not dry_run:
        raise click.ClickException(
            "refusing to update inside a batch job: compute nodes have no "
            "outbound network. Run this on a login node."
        )

    spec = "git+https://github.com/saforem2/ezpz"
    if ref:
        spec = f"{spec}@{ref}"

    if not utils_only:
        cmd = _pip_install_cmd(spec)
        click.echo(f"$ {' '.join(cmd)}")
        if not dry_run:
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise click.ClickException(
                    f"package upgrade failed (exit {proc.returncode})"
                )

    if not package_only:
        share = _share_dir()
        dest = share / "utils.sh"
        url = os.environ.get("EZPZ_UTILS_URL", DEFAULT_UTILS_URL)
        click.echo(f"$ curl -fsSL {url} -o {dest}")
        if not dry_run:
            _refresh_utils(url, dest)

    if dry_run:
        click.echo("(dry run — nothing was changed)")


def _refresh_utils(url: str, dest: Path) -> None:
    """Download *url* to *dest*, validating before replacing.

    Writes to a ``.part`` file and checks it parses as shell first: a
    truncated ``utils.sh`` passes a non-empty test and then fails much
    later as "every ``ezpz_*`` function is undefined", which is a far
    worse failure than a clean download error.

    Args:
        url: Source of ``utils.sh``.
        dest: Final path.

    Raises:
        click.ClickException: On download failure or a bad file.
    """
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise click.ClickException(f"could not download {url}: {exc}") from exc

    if not data or not data.lstrip().startswith(b"#"):
        raise click.ClickException(
            f"{url} did not return a shell script; refusing to install"
        )
    part.write_bytes(data)

    check = subprocess.run(
        ["bash", "-n", str(part)], capture_output=True, text=True
    )
    if check.returncode != 0:
        part.unlink(missing_ok=True)
        raise click.ClickException(
            f"downloaded utils.sh has a syntax error: {check.stderr.strip()}"
        )

    # `bash -n` alone does NOT catch truncation: a response cut at any
    # syntactically complete boundary parses fine. The first ten lines
    # of the real utils.sh pass it, and installing that would replace a
    # working 140 KB file with a stub whose functions are all missing --
    # exactly the failure this check exists to prevent.
    #
    # So also require the functions a caller actually invokes. These are
    # long-standing entry points; if one is genuinely renamed, this
    # fails loudly at update time rather than silently at job time.
    missing = [
        fn
        for fn in _REQUIRED_FUNCTIONS
        if f"\n{fn}()".encode() not in b"\n" + data
    ]
    if missing:
        part.unlink(missing_ok=True)
        raise click.ClickException(
            f"downloaded utils.sh looks truncated ({len(data)} bytes): "
            f"missing {', '.join(missing)}. Existing file left unchanged."
        )
    part.replace(dest)
    click.echo(f"  utils.sh updated ({len(data)} bytes) -> {dest}")
