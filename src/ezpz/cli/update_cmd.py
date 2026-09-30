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


def _share_dir() -> Path:
    return Path(
        os.environ.get("EZPZ_SHARE_DIR", Path.home() / ".local/share/ezpz")
    )


def _in_batch_job() -> bool:
    """True when running inside a PBS or SLURM allocation.

    Compute nodes have no outbound route, so a download there hangs for
    ~270 s and then leaves things unconfigured. Better to refuse.
    """
    return bool(
        os.environ.get("PBS_NODEFILE") or os.environ.get("SLURM_JOB_ID")
    )


def _pip_install_cmd(spec: str) -> list[str]:
    """Return the install command for the active environment.

    Prefers ``uv pip`` when available, matching the project's own
    tooling, and falls back to ``python -m pip``.
    """
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--upgrade", spec]
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
    2. Re-download the cached ``utils.sh`` that ``ezpz-setup`` sources.

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
    part.replace(dest)
    click.echo(f"  utils.sh updated ({len(data)} bytes) -> {dest}")
