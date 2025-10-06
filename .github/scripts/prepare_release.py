#!/usr/bin/env python3
"""Prepare release artifacts: bump version and update changelog."""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from dataclasses import dataclass
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
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REPOSITORY: Repo | None = None


@dataclass
class ReleasePlan:
    bump_type: str
    version: str
    plain_version: str
    tag: str
    release_notes: str
    commits: list[tuple[str, str]]
    version_file_contents: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bump",
        choices=["auto", "major", "minor", "patch"],
        default=None,
        help="Semantic version bump to apply. Defaults to env BUMP_TYPE or auto-infer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the release without modifying files or writing GitHub outputs.",
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

    return [(commit.hexsha, commit.message.rstrip()) for commit in commits]


_ALLOWED_BUMPS = {"major", "minor", "patch"}


def _infer_bump(commits: Iterable[tuple[str, str]], override: str | None = None) -> str:
    bump_env = os.environ.get("BUMP_TYPE", "")
    bump = (override or bump_env).lower()
    if bump in _ALLOWED_BUMPS:
        return bump
    if bump_env:
        print(
            f"Warning: Environment variable BUMP_TYPE='{bump_env}' is not recognized. "
            "Falling back to auto-infer bump type.",
            file=sys.stderr,
        )

    inferred = "patch"
    for _sha, message in commits:
        lines = message.splitlines()
        for line in lines:
            if "breaking change" in line.lower():
                return "major"
        first_line = lines[0] if lines else ""
        match = _CONVENTIONAL_RE.match(first_line)
        if match and match.group("breaking"):
            return "major"
        if match and match.group("type") == "feat" and inferred != "major":
            inferred = "minor"
    return inferred


def _calculate_version_update(bump: str) -> tuple[str, str, str, str]:
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if not match:
        raise ReleaseError("Unable to locate __version__ assignment in __about__.py")

    prefix = match.group("prefix") or ""
    current_version = match.group("version")
    try:
        parsed = semver.VersionInfo.parse(current_version)
    except ValueError as exc:
        raise ReleaseError(
            f"Invalid version string '{current_version}' in __about__.py"
        ) from exc

    bump_map = {
        "major": parsed.bump_major,
        "minor": parsed.bump_minor,
        "patch": parsed.bump_patch,
    }
    if bump not in bump_map:
        allowed_bumps = ", ".join(sorted(bump_map.keys()))
        raise ReleaseError(
            f"Unsupported bump type '{bump}'. Allowed types are: {allowed_bumps}"
        )

    new_version_info = bump_map[bump]()
    plain_version = str(new_version_info)
    version_with_prefix = f"{prefix}{plain_version}"
    tag = f"{prefix}{plain_version}"
    new_literal = f'__version__ = "{version_with_prefix}"'
    updated = _VERSION_RE.sub(new_literal, text, count=1)

    return plain_version, version_with_prefix, tag, updated


def _build_release_plan(override: str | None) -> ReleasePlan:
    last_tag = _get_last_tag()
    commits = _collect_commits(last_tag)
    bump_type = _infer_bump(commits, override)
    plain_version, version_with_prefix, tag, updated_text = _calculate_version_update(
        bump_type
    )
    release_notes = _format_release_notes(commits, tag)
    return ReleasePlan(
        bump_type=bump_type,
        version=version_with_prefix,
        plain_version=plain_version,
        tag=tag,
        release_notes=release_notes,
        commits=commits,
        version_file_contents=updated_text,
    )


def _validate_github_repository(repo: str) -> None:
    if not isinstance(repo, str) or not _REPO_RE.match(repo):
        raise ReleaseError(
            f"GITHUB_REPOSITORY must be in the format 'owner/repo', got: {repo!r}"
        )


def _format_release_notes(commits: list[tuple[str, str]], tag: str) -> str:
    if not REPO:
        raise ReleaseError("GITHUB_REPOSITORY environment variable is required")
    _validate_github_repository(REPO)

    date = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    lines = [
        f"#### [{tag}](https://github.com/{REPO}/releases/tag/{tag}) - {date}",
        "",
    ]
    if commits:
        lines.extend(
            f"- {(message.splitlines() or [''])[0]} ([`{sha[:7]}`](https://github.com/{REPO}/commit/{sha}))"
            for sha, message in commits
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


def _append_outputs(path: str, plan: ReleasePlan) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"version={plan.version}\n")
        fh.write(f"plain_version={plan.plain_version}\n")
        fh.write(f"tag={plan.tag}\n")
        fh.write("release_notes<<EOF\n")
        fh.write(plan.release_notes)
        fh.write("EOF\n")


def _write_outputs(plan: ReleasePlan) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        _append_outputs(output_path, plan)
    except OSError as exc:
        message = f"Unable to write GitHub outputs: {exc}"
        print(f"Error: {message}", file=sys.stderr)
        raise ReleaseError(message) from exc


def main() -> None:
    args = _parse_args()
    lock_handle = _acquire_lock()
    try:
        plan = _build_release_plan(args.bump)
        if args.dry_run:
            print(plan.release_notes, end="")
            print(
                f"Dry run: prepared {plan.tag} (bump: {plan.bump_type}) with {len(plan.commits)} commit(s).",
                flush=True,
            )
            print("Dry run: skipping file updates and GitHub outputs.", flush=True)
        else:
            VERSION_FILE.write_text(plan.version_file_contents, encoding="utf-8")
            _update_changelog(plan.release_notes)
            _write_outputs(plan)
            print(
                f"Prepared {plan.tag} (bump: {plan.bump_type}) with {len(plan.commits)} commit(s).",
                flush=True,
            )
    finally:
        _release_lock(lock_handle)


if __name__ == "__main__":
    main()
