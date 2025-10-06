#!/usr/bin/env python3
"""Prepare release artifacts: bump version and update changelog."""
from __future__ import annotations

import datetime as _dt
import fcntl
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


class ReleaseError(RuntimeError):
    """Raised when the release preparation fails."""


REPO = os.environ.get("GITHUB_REPOSITORY", "")
ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "src" / "ezpz" / "__about__.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
LOCK_FILE = ROOT / ".release.lock"
DELIMITER = "\x01"

_VERSION_RE = re.compile(
    r'__version__\s*=\s*"(?P<prefix>v?)(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"'
)
_CONVENTIONAL_RE = re.compile(r"^(?P<type>\w+)(?P<breaking>!?)(?:\([^)]+\))?:")


def _run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _acquire_lock(timeout: int = 30) -> TextIO:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "w", encoding="utf-8")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise ReleaseError(
                    "Timed out acquiring release lock; another release may be running."
                )
            time.sleep(0.5)


def _release_lock(handle: TextIO) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _get_last_tag() -> str | None:
    try:
        return _run_git("describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        return None


def _collect_commits(since_tag: str | None) -> list[tuple[str, str]]:
    rev = f"{since_tag}..HEAD" if since_tag else "HEAD"
    try:
        log_output = _run_git("log", rev, "--pretty=format:%H%x01%s")
    except subprocess.CalledProcessError:
        return []

    commits: list[tuple[str, str]] = []
    if not log_output:
        return commits

    for line in log_output.splitlines():
        if DELIMITER not in line:
            continue
        try:
            sha, message = line.split(DELIMITER, 1)
        except ValueError:
            continue
        commits.append((sha, message))
    return commits


def _infer_bump(commits: list[tuple[str, str]]) -> str:
    bump = os.environ.get("BUMP_TYPE", "").lower()
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
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    if bump == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    new_version = f"{major}.{minor}.{patch}"
    new_literal = f'__version__ = "{prefix}{new_version}"'
    updated = _VERSION_RE.sub(new_literal, text, count=1)
    VERSION_FILE.write_text(updated, encoding="utf-8")

    tag = f"{prefix}{new_version}" if prefix else f"v{new_version}"
    return new_version, tag


def _format_release_notes(commits: list[tuple[str, str]], tag: str) -> str:
    if not REPO:
        raise ReleaseError("GITHUB_REPOSITORY environment variable is required")

    lines: list[str] = []
    for sha, message in commits:
        url = f"https://github.com/{REPO}/commit/{sha}"
        lines.append(f"- {message} ([`{sha[:7]}`]({url}))")

    if not lines:
        lines.append("- No changes recorded")

    date = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    header = f"#### [{tag}](https://github.com/{REPO}/releases/tag/{tag}) - {date}"
    body = "\n".join(lines)
    return f"{header}\n\n{body}\n"


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
    lock_handle = _acquire_lock()
    try:
        last_tag = _get_last_tag()
        commits = _collect_commits(last_tag)
        bump_type = _infer_bump(commits)
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
