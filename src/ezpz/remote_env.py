"""Remote env setup templates for ezpz submit --remote.

Two modes:
- If --workdir is given, use it directly (user controls).
- If not given, use shared FS under PBS_O_WORKDIR/SLURM_SUBMIT_DIR/$HOME
  (NOT /tmp — node-local would shard output and require yeet).
"""

import shlex


REMOTE_PROLOG = """\
# ── ALCF proxies (login + compute nodes) ──
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"
export NO_PROXY="$no_proxy"
"""

# When --workdir is not given, use a shared-fs path (NOT /tmp — node-local, would shard output).
# Require shared FS: flare/eagle/grand on ALCF, $SCRATCH on NERSC, /home fallback.
REMOTE_WORKDIR_AUTO = (
    'WORKDIR="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$HOME}}/'
    'ezpz-${PBS_JOBID:-${SLURM_JOB_ID:-$$}}"; '
    'mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit 1'
)

REMOTE_ENV_SETUP = """\
# ── ezpz environment (self-bootstrapping) ──
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
# Ensure ezpz is present in the active env (handles fresh/empty venvs)
if ! python3 -c "import ezpz" 2>/dev/null; then
  echo "[ezpz] not found in active env — installing..." >&2
  uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz" || \\
  python3 -m pip install --no-cache "git+https://github.com/saforem2/ezpz"
fi
python3 -c "import ezpz; print(f'[ezpz] {ezpz.__version__} ready')" >&2
# Note: WORKDIR is on shared FS — no yeet needed. Only use /tmp + yeet for
# scale-sensitive runs where shared-FS I/O is the bottleneck (explicit opt-in).
"""


def _strict_setup(commands: list[str]) -> str:
    """Fail on setup errors while preserving the payload's errexit policy."""
    return "\n".join(
        [
            "_EZPZ_RESTORE_ERREXIT=0",
            "case $- in *e*) _EZPZ_RESTORE_ERREXIT=1 ;; esac",
            "set -e",
            *commands,
            "if [[ $_EZPZ_RESTORE_ERREXIT == 0 ]]; then set +e; fi",
        ]
    )


def build_env_setup(
    env_path: str | None = None,
    setup_script: str | None = None,
    *,
    require_ezpz: bool = False,
) -> str | None:
    """Return activation commands for an explicit target-side environment.

    Args:
        env_path: Environment directory on the machine running the job.
        setup_script: Target-side shell script, sourced before activation.
        require_ezpz: Require ezpz in the selected environment for launch.

    Returns:
        Shell commands, or ``None`` to retain automatic setup.
    """
    parts = []
    if env_path is not None:
        # Bind relative paths before a setup script can change directory.
        parts.append(f"""\
_EZPZ_ENV={shlex.quote(env_path)}
if [[ "$_EZPZ_ENV" != /* ]]; then _EZPZ_ENV="$PWD/$_EZPZ_ENV"; fi
""")
    if setup_script is not None:
        # A bare `source setup.sh` searches PATH before the current directory.
        # Force target-workdir-relative lookup, even for a bare filename.
        script_path = (
            setup_script
            if setup_script.startswith("/")
            else f"./{setup_script}"
        )
        parts.append(f"source {shlex.quote(script_path)}")
    if env_path is None:
        return _strict_setup(parts) if parts else None
    parts.append("""\
_EZPZ_PYTHON=""
if [[ -f "$_EZPZ_ENV" && -x "$_EZPZ_ENV" && "${_EZPZ_ENV##*/}" =~ ^python([0-9]+([.][0-9]+)*)?[td]?$ ]]; then
    _EZPZ_PYTHON="$_EZPZ_ENV"
    _EZPZ_ENV="$(dirname "$(dirname "$_EZPZ_ENV")")"
else
    for _EZPZ_CANDIDATE in "$_EZPZ_ENV/bin/python3" "$_EZPZ_ENV/bin/python" "$_EZPZ_ENV"/bin/python[0-9]*; do
        if [[ -f "$_EZPZ_CANDIDATE" && -x "$_EZPZ_CANDIDATE" && "${_EZPZ_CANDIDATE##*/}" =~ ^python([0-9]+([.][0-9]+)*)?[td]?$ ]]; then
            _EZPZ_PYTHON="$_EZPZ_CANDIDATE"
            break
        fi
    done
fi
if [[ -z "$_EZPZ_PYTHON" ]]; then
    echo "[ezpz] invalid --env: $_EZPZ_ENV (expected a Python prefix or executable; use --env-setup-script for shell scripts)" >&2
    exit 1
fi
_EZPZ_ENV="$(cd "$_EZPZ_ENV" && pwd -P)" || exit 1
if [[ -d "$_EZPZ_ENV/conda-meta" ]]; then
    if command -v conda >/dev/null 2>&1; then
        _EZPZ_CONDA_HOOK="$(conda shell.bash hook)" || exit 1
    elif [[ -x "${CONDA_EXE:-}" ]]; then
        _EZPZ_CONDA_HOOK="$("$CONDA_EXE" shell.bash hook)" || exit 1
    else
        echo "[ezpz] conda is unavailable; initialize it in --env-setup-script" >&2
        exit 1
    fi
    eval "$_EZPZ_CONDA_HOOK"
    conda activate "$_EZPZ_ENV"
elif [[ -f "$_EZPZ_ENV/bin/activate" ]]; then
    source "$_EZPZ_ENV/bin/activate"
    if [[ ! "${VIRTUAL_ENV:-}" -ef "$_EZPZ_ENV" ]]; then
        echo "[ezpz] --env activation selected another prefix; recreate or repair the moved environment" >&2
        exit 1
    fi
else
    unset VIRTUAL_ENV CONDA_PREFIX
fi
export PATH="$_EZPZ_ENV/bin:$PATH"
hash -r
""")
    if require_ezpz:
        parts.append("""\
if [[ ! -x "$_EZPZ_ENV/bin/ezpz" ]]; then
    echo "[ezpz] ezpz is missing from --env $_EZPZ_ENV; install it there before submitting" >&2
    exit 1
fi
""")
    return _strict_setup(parts)


def build_remote_env_setup(
    workdir: str | None = None,
    *,
    env_setup: str | None = None,
    alcf_proxies: bool = True,
) -> str:
    """Return setup commands, retaining the workdir for custom environments.

    Args:
        workdir: Explicit shared directory, or a scheduler-job-specific default.
        env_setup: Custom activation commands; ``None`` uses managed setup.
        alcf_proxies: Include the ALCF proxy settings only for ALCF targets.

    Returns:
        Shell commands for the generated remote script.
    """
    parts = [REMOTE_PROLOG] if alcf_proxies else []
    if workdir:
        parts.append(
            f'WORKDIR={shlex.quote(workdir)}; cd "$WORKDIR" || exit 1'
        )
    else:
        parts.append(REMOTE_WORKDIR_AUTO)
    parts.append(REMOTE_ENV_SETUP if env_setup is None else env_setup)
    return "\n".join(parts)
