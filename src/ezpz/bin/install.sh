#!/usr/bin/env bash
# ezpz bootstrap installer.
#
#   curl -fsSL https://ezpz.cool/install.sh | bash
#
# Solves exactly one problem: getting a validated `utils.sh` onto a
# machine. Everything after that -- modules, venv, installing ezpz --
# is what `ezpz_setup_env` already does, so this does not reimplement it.
#
# No PATH shim: `ezpz_*` are shell FUNCTIONS, not executables, so they
# cannot be put on PATH at all -- they have to be sourced into the
# current shell. A wrapper to shorten that line was not worth a second
# artifact to keep in sync, nor the source-vs-execute confusion it
# needed its own guard for.
#
# Design notes, each the result of something that bit us:
#
#   * LOGIN NODES ONLY. Compute nodes on Aurora/Polaris/Sunspot have no
#     outbound route; a curl there hangs ~270 s and then leaves the
#     environment silently unconfigured. This script checks and says so.
#   * NEVER edits your shell rc, and installs nothing on PATH.
#   * Idempotent. Re-running is the supported way to update the cached
#     utils.sh; there is no self-updating executable to go stale.
#
# NOT set -u: this is sourced-adjacent code that touches ${SHELL} and
# friends, and an unset variable should not abort an installer.
set -eo pipefail

EZPZ_URL="${EZPZ_UTILS_URL:-https://ezpz.cool/utils.sh}"
SHARE_DIR="${EZPZ_SHARE_DIR:-${HOME}/.local/share/ezpz}"

_say() { printf '%s\n' "$*"; }
_err() { printf 'error: %s\n' "$*" >&2; }

# ── Refuse to run somewhere the download cannot work ─────────────────────
# A compute node is the one place this reliably fails, and it fails
# slowly and confusingly (270 s, then an unconfigured environment).
if [[ -n "${PBS_NODEFILE:-}${SLURM_JOB_ID:-}" ]]; then
    _err "this looks like a batch job (PBS_NODEFILE/SLURM_JOB_ID is set)."
    _err "Compute nodes have no outbound network -- run this on a login"
    _err "node, then source the installed utils.sh inside your job."
    exit 1
fi

_say "Installing ezpz bootstrap..."
mkdir -p "${SHARE_DIR}"

# Download to .part and validate BEFORE moving into place: a truncated
# utils.sh passes a non-empty check and fails much later as "every
# ezpz_* function is undefined".
tmp="${SHARE_DIR}/utils.sh.part"
if ! curl -fsSL --max-time 120 "${EZPZ_URL}" -o "${tmp}"; then
    _err "could not download ${EZPZ_URL}"
    _err "If you are behind a proxy, export http_proxy/https_proxy first."
    rm -f "${tmp}"
    exit 1
fi
if [[ ! -s "${tmp}" ]] || ! head -1 "${tmp}" | grep -q '^#'; then
    _err "downloaded file does not look like a shell script; refusing to install"
    rm -f "${tmp}"
    exit 1
fi
if ! bash -n "${tmp}" 2>/dev/null; then
    _err "downloaded utils.sh has a syntax error; refusing to install"
    rm -f "${tmp}"
    exit 1
fi
mv "${tmp}" "${SHARE_DIR}/utils.sh"
_say "  utils.sh -> ${SHARE_DIR}/utils.sh ($(wc -c <"${SHARE_DIR}/utils.sh" | tr -d ' ') bytes)"

_say ""
_say "Source it from your shell or a job script:"
_say ""
_say "    source ${SHARE_DIR}/utils.sh && ezpz_setup_env"
_say ""
_say "To update later:  ezpz update   (or re-run this installer)"
