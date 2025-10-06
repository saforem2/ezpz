#!/usr/bin/env python3
"""Prepare release artifacts: bump version and update changelog."""
from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "")
ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "src" / "ezpz" / "__about__.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def _run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _get_last_tag() -> str | None:
    try:
        return _run_git("describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        return None


def _parse_version() -> tuple[str, str]:
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"(v?)(\d+)\.(\d+)\.(\d+)"', content)
    if not match:
        raise ValueError("Unable to locate __version__ assignment in __about__.py")

    prefix, major, minor, patch = match.groups()
    current = f"{major}.{minor}.{patch}"
    return prefix, current


def _bump_version(current: str) -> str:
    major, minor, _patch = (int(part) for part in current.split("."))
    minor += 1
    patch = 0
    return f"{major}.{minor}.{patch}"


def _update_version_file(prefix: str, new_version: str) -> None:
    content = VERSION_FILE.read_text(encoding="utf-8")
    new_literal = f'__version__ = "{prefix}{new_version}"'
    content = re.sub(r'__version__\s*=\s*"v?\d+\.\d+\.\d+"', new_literal, content, count=1)
    VERSION_FILE.write_text(content, encoding="utf-8")


def _collect_commits(since_tag: str | None) -> list[tuple[str, str]]:
    rev_range = f"{since_tag}..HEAD" if since_tag else "HEAD"
    try:
        log_output = _run_git("log", rev_range, "--pretty=format:%H%x01%s")
    except subprocess.CalledProcessError:
        return []

    commits: list[tuple[str, str]] = []
    if not log_output:
        return commits

    for line in log_output.splitlines():
        sha, message = line.split("\u0001", 1)
        commits.append((sha, message))
    return commits


def _format_release_notes(commits: list[tuple[str, str]], tag: str) -> str:
    if not REPO:
        raise RuntimeError("GITHUB_REPOSITORY environment variable is required")

    lines: list[str] = []
    for sha, message in commits:
        url = f"https://github.com/{REPO}/commit/{sha}"
        lines.append(f"- {message} ([`{sha[:7]}`]({url}))")

    if not lines:
        lines.append("- No changes recorded")

    date = _dt.datetime.now(_dt.UTC).date().isoformat()
    header = f"#### [{tag}](https://github.com/{REPO}/releases/tag/{tag}) - {date}"
    body = "\n".join(lines)
    return f"{header}\n\n{body}\n"


def _update_changelog(entry: str) -> None:
    lines = CHANGELOG_FILE.read_text(encoding="utf-8").splitlines()
    try:
        unreleased_index = next(i for i, line in enumerate(lines) if line.startswith("#### [Unreleased"))
    except StopIteration:
        new_content = entry + "\n" + "\n".join(lines) + "\n"
        CHANGELOG_FILE.write_text(new_content, encoding="utf-8")
        return

    insert_index = None
    for i in range(unreleased_index + 1, len(lines)):
        if lines[i].startswith("#### ["):
            insert_index = i
            break
    if insert_index is None:
        insert_index = len(lines)

    entry_lines = ["", *entry.rstrip().splitlines(), ""]
    new_lines = lines[:insert_index] + entry_lines + lines[insert_index:]
    CHANGELOG_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main() -> None:
    prefix, current = _parse_version()
    new_version = _bump_version(current)
    tag = f"{prefix}{new_version}" if prefix else f"v{new_version}"
    _update_version_file(prefix or "", new_version)

    last_tag = _get_last_tag()
    commits = _collect_commits(last_tag)
    release_entry = _format_release_notes(commits, tag)
    _update_changelog(release_entry)

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"version={new_version}\n")
            fh.write(f"tag={tag}\n")
            fh.write("release_notes<<EOF\n")
            fh.write(release_entry)
            fh.write("EOF\n")


if __name__ == "__main__":
    main()
