#!/usr/bin/env bash
# ezpz bootstrap installer.
#
#   curl -fsSL https://ezpz.cool/install.sh | bash
#
# Solves exactly one problem: getting `utils.sh` onto a machine and onto
# your PATH. Everything after that -- modules, venv, installing ezpz --
# is what `ezpz_setup_env` already does, so this does not reimplement it.
#
# Design notes, each the result of something that bit us:
#
#   * LOGIN NODES ONLY. Compute nodes on Aurora/Polaris/Sunspot have no
#     outbound route; a curl there hangs ~270 s and then leaves the
#     environment silently unconfigured. This script checks and says so.
#   * NEVER edits your shell rc. It prints the line for you to paste.
#     Silently rewriting .bashrc on a shared account is not ours to do.
#   * Idempotent. Re-running is the supported way to update the cached
#     utils.sh; there is no self-updating executable to go stale.
#
# NOT set -u: this is sourced-adjacent code that touches ${SHELL} and
# friends, and an unset variable should not abort an installer.
set -eo pipefail

EZPZ_URL="${EZPZ_UTILS_URL:-https://ezpz.cool/utils.sh}"
SHARE_DIR="${EZPZ_SHARE_DIR:-${HOME}/.local/share/ezpz}"
BIN_DIR="${EZPZ_BIN_DIR:-${HOME}/.local/bin}"
SHIM="${BIN_DIR}/ezpz-setup"

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
mkdir -p "${SHARE_DIR}" "${BIN_DIR}"

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

# ── The shim ─────────────────────────────────────────────────────────────
# `ezpz-setup` is meant to be SOURCED, not executed: it has to modify the
# calling shell. Running it directly would set up an environment and then
# throw it away, so say so rather than silently doing nothing useful.
cat >"${SHIM}" <<'SHIMEOF'
#!/usr/bin/env bash
# Source this to set up an ezpz environment:
#
#     source ezpz-setup          # or: source ~/.local/bin/ezpz-setup
#
# Executing it instead of sourcing cannot work -- environment changes
# would die with the subshell -- so that case is reported, not ignored.
_ezpz_share="${EZPZ_SHARE_DIR:-${HOME}/.local/share/ezpz}"
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf 'ezpz-setup must be SOURCED, not executed:\n\n' >&2
    printf '    source %s\n\n' "${0}" >&2
    printf 'Executing it would configure a subshell and then discard it.\n' >&2
    exit 1
fi
if [[ ! -f "${_ezpz_share}/utils.sh" ]]; then
    printf 'ezpz: %s/utils.sh not found; re-run the installer:\n' "${_ezpz_share}" >&2
    printf '    curl -fsSL https://ezpz.cool/install.sh | bash\n' >&2
    return 1
fi
source "${_ezpz_share}/utils.sh"
ezpz_setup_env "$@"
SHIMEOF
chmod +x "${SHIM}"
_say "  shim     -> ${SHIM}"

# ── PATH advice, not PATH surgery ────────────────────────────────────────
case ":${PATH}:" in
    *":${BIN_DIR}:"*)
        _say ""
        _say "${BIN_DIR} is already on your PATH."
        ;;
    *)
        case "$(basename "${SHELL:-/bin/bash}")" in
            zsh)  rc="${HOME}/.zshrc" ;;
            bash) rc="${HOME}/.bashrc" ;;
            *)    rc="your shell's startup file" ;;
        esac
        _say ""
        _say "${BIN_DIR} is NOT on your PATH. Add it by appending this to ${rc}:"
        _say ""
        _say "    export PATH=\"${BIN_DIR}:\${PATH}\""
        _say ""
        _say "(Not done automatically -- editing your shell config is your call.)"
        ;;
esac

_say ""
_say "Then, on a login node or inside a job:"
_say ""
_say "    source ezpz-setup"
_say ""
_say "To update later:  ezpz update   (or re-run this installer)"
