#!/usr/bin/env bash
# install-skill.sh — install the /ezpz skill into the current project
# Usage:
#   curl -fsSL https://ezpz.cool/install-skill.sh | bash
#   curl -fsSL https://ezpz.cool/install-skill.sh | bash -s -- --global   # also: user-level
set -euo pipefail

REPO="saforem2/ezpz"
RAW_BASE="https://raw.githubusercontent.com/${REPO}/main"

usage() {
    echo "Usage: install-skill.sh [--global] [--dest DIR]"
    echo "  --global   install to user-level (discoverable from any project)"
    echo "  --dest DIR explicit destination directory"
    exit 0
}

DEST=""
GLOBAL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --global) GLOBAL=1; shift ;;
        --dest) DEST="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

# Pick destination
if [[ -n "$DEST" ]]; then
    : # use explicit DEST
elif [[ "$GLOBAL" -eq 1 ]]; then
    # Ask the runtime where it looks for user-level skills, fall back to Claude default.
    if command -v claude &>/dev/null && claude --help 2>&1 | grep -q "plugin"; then
        # Claude Code user skills live under ~/.claude/skills/<name>/  (also via `~/.agent/skills/` symlink)
        DEST="${HOME}/.claude/skills/ezpz"
    else
        DEST="${HOME}/.claude/skills/ezpz"
    fi
else
    # Project-local: .agents/skills/ezpz is the universal (skills.sh) location.
    # Claude Code also follows .agents/skills via its symlink.
    if git rev-parse --show-toplevel &>/dev/null 2>&1; then
        DEST="$(git rev-parse --show-toplevel)/.agents/skills/ezpz"
    else
        DEST=".agents/skills/ezpz"
    fi
fi

mkdir -p "$DEST"

_fetch() {
    local url="$1" out="$2"
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$out"
    elif command -v wget &>/dev/null; then
        wget -qO "$out" "$url"
    else
        echo "Need curl or wget" >&2; exit 1
    fi
}

echo "→ installing /ezpz skill to ${DEST}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

_fetch "${RAW_BASE}/SKILL.md" "${tmp}/SKILL.md"
_fetch "${RAW_BASE}/.agents/skills/alcf-preflight/SKILL.md" "${tmp}/alcf-preflight-SKILL.md"
_fetch "${RAW_BASE}/.agents/skills/alcf-preflight/preflight.sh" "${tmp}/preflight.sh"

# Validate: a truncated skill is worse than no skill.
test -s "${tmp}/SKILL.md" || { echo "SKILL.md empty after fetch" >&2; exit 1; }
head -1 "${tmp}/SKILL.md" | grep -q "^---" || { echo "SKILL.md missing frontmatter" >&2; exit 1; }

cp "${tmp}/SKILL.md" "${DEST}/SKILL.md"
mkdir -p "${DEST}/.agents/skills/alcf-preflight"
cp "${tmp}/alcf-preflight-SKILL.md" "${DEST}/.agents/skills/alcf-preflight/SKILL.md" 2>/dev/null || true
cp "${tmp}/preflight.sh" "${DEST}/.agents/skills/alcf-preflight/preflight.sh" 2>/dev/null || true
chmod +x "${DEST}/.agents/skills/alcf-preflight/preflight.sh" 2>/dev/null || true

echo "✓ installed /ezpz → ${DEST}/SKILL.md"
echo "  Invoke with /ezpz in your agent."
if [[ "$GLOBAL" -eq 0 ]]; then
    echo "  Tip: add .agents/skills/ezpz to your repo so teammates get it on clone,"
    echo "       or run with --global to install once for your user."
fi
