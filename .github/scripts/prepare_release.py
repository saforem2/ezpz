#!/usr/bin/env python3
"""Prepare release artifacts: bump version and update changelog."""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import semver
from filelock import FileLock, Timeout
from git import GitCommandError, Repo
from git.exc import InvalidGitRepositoryError


class ReleaseError(RuntimeError):
    """Raised when the release preparation fails."""


REPO = os.environ.get("GITHUB_REPOSITORY", "")
ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "src" / "ezpz" / "__about__.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
LOCK_FILE = ROOT / ".release.lock"

_VERSION_RE = re.compile(
    r'__version__\s*=\s*"(?P<prefix>v?)(?P<version>\d+\.\d+\.\d+)"'
)
_CONVENTIONAL_RE = re.compile(r"^(?P<type>\w+)(?P<breaking>!?)(?:\([^)]+\))?:")
_REPOSITORY: Repo | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bump",
        choices=["auto", "major", "minor", "patch"],
        default=None,
        help="Semantic version bump to apply. Defaults to env BUMP_TYPE or auto-infer.",
    )
    return parser.parse_args()


def _acquire_lock(timeout: int = 30) -> FileLock:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(LOCK_FILE))
    try:
        lock.acquire(timeout=timeout)
    except Timeout as exc:
        raise ReleaseError(
            "Timed out acquiring release lock; another release may be running."
        ) from exc
    return lock


def _release_lock(lock: FileLock) -> None:
    if lock.is_locked:
        lock.release()


def _get_repository() -> Repo:
    global _REPOSITORY
    if _REPOSITORY is None:
        try:
            _REPOSITORY = Repo(ROOT)
        except InvalidGitRepositoryError as exc:
            raise ReleaseError("Unable to locate Git repository root") from exc
    return _REPOSITORY


def _get_last_tag() -> str | None:
    try:
        return _get_repository().git.describe("--tags", "--abbrev=0")
    except GitCommandError:
        return None


def _collect_commits(since_tag: str | None) -> list[tuple[str, str]]:
    rev = f"{since_tag}..HEAD" if since_tag else "HEAD"
    try:
        commits = list(_get_repository().iter_commits(rev))
    except GitCommandError:
        return []

    return [(commit.hexsha, commit.message.splitlines()[0]) for commit in commits]


def _infer_bump(commits: Iterable[tuple[str, str]], override: str | None = None) -> str:
    bump = (override or os.environ.get("BUMP_TYPE", "")).lower()
    if bump in {"major", "minor", "patch"}:
        return bump

    inferred = "patch"
    for _sha, message in commits:
        lower_msg = message.lower()
        if "breaking change" in lower_msg:
            return "major"
        match = _CONVENTIONAL_RE.match(message)
        if match and match.group("breaking"):
            return "major"
        if match and match.group("type") == "feat" and inferred != "major":
            inferred = "minor"
    return inferred


def _bump_version_file(bump: str) -> tuple[str, str]:
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if not match:
        raise ReleaseError("Unable to locate __version__ assignment in __about__.py")

    prefix = match.group("prefix") or ""
    current_version = match.group("version")
    parsed = semver.VersionInfo.parse(current_version)
    bump_map = {
        "major": parsed.bump_major,
        "minor": parsed.bump_minor,
        "patch": parsed.bump_patch,
    }
    new_version = str(bump_map[bump]())

    new_literal = f'__version__ = "{prefix}{new_version}"'
    updated = _VERSION_RE.sub(new_literal, text, count=1)
    VERSION_FILE.write_text(updated, encoding="utf-8")

    tag = f"{prefix}{new_version}" if prefix else f"v{new_version}"
    return new_version, tag


def _format_release_notes(commits: list[tuple[str, str]], tag: str) -> str:
    if not REPO:
        raise ReleaseError("GITHUB_REPOSITORY environment variable is required")

    date = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    lines = [
        f"#### [{tag}](https://github.com/{REPO}/releases/tag/{tag}) - {date}",
        "",
    ]
    if commits:
        for sha, message in commits:
            lines.append(
                f"- {message} ([`{sha[:7]}`](https://github.com/{REPO}/commit/{sha}))"
            )
    else:
        lines.append("- No changes recorded")

    return "\n".join(lines) + "\n"


def _update_changelog(entry: str) -> None:
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    marker = "#### [Unreleased]"
    if marker not in text:
        new_text = entry + "\n" + text
    else:
        before, after = text.split(marker, 1)
        remainder = after.lstrip("\n")
        new_text = before + marker + "\n\n" + entry + ("\n\n" + remainder if remainder else "\n")
    CHANGELOG_FILE.write_text(new_text.rstrip("\n") + "\n", encoding="utf-8")


def _write_outputs(new_version: str, tag: str, release_entry: str) -> None:
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        try:
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(f"version={new_version}\n")
                fh.write(f"tag={tag}\n")
                fh.write("release_notes<<EOF\n")
                fh.write(release_entry)
                fh.write("EOF\n")
        except OSError as exc:
            print(f"Warning: unable to write GitHub outputs: {exc}", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    lock_handle = _acquire_lock()
    try:
        last_tag = _get_last_tag()
        commits = _collect_commits(last_tag)
        bump_type = _infer_bump(commits, args.bump)
        new_version, tag = _bump_version_file(bump_type)
        release_entry = _format_release_notes(commits, tag)
        _update_changelog(release_entry)
        _write_outputs(new_version, tag, release_entry)
        print(
            f"Prepared {tag} (bump: {bump_type}) with {len(commits)} commit(s).",
            flush=True,
        )
    finally:
        _release_lock(lock_handle)


if __name__ == "__main__":
    main()
