"""Remote env setup templates for ezpz submit --remote.

Two modes:
- If --workdir is given, use it directly (user controls).
- If not given, use shared FS under PBS_O_WORKDIR/SLURM_SUBMIT_DIR/$HOME
  (NOT /tmp — node-local would shard output and require yeet).
"""

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
REMOTE_WORKDIR_AUTO = 'WORKDIR="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$HOME}}/ezpz-${PBS_JOBID:-${SLURM_JOB_ID:-$$}}"; mkdir -p "$WORKDIR"; cd "$WORKDIR"'

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


def build_remote_env_setup(workdir: str | None = None) -> str:
    """Return the shell snippet for remote env setup.

    When workdir is given, cd there first; otherwise create/use a per-job tmpdir.
    """
    parts = [REMOTE_PROLOG]
    if workdir:
        parts.append(f'cd "{workdir}"')
    else:
        parts.append(REMOTE_WORKDIR_AUTO)
    parts.append(REMOTE_ENV_SETUP)
    return "\n".join(parts)
