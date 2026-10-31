"""Submit jobs to a machine you are not logged into.

Two backends, and the choice between them is the whole design problem.

``ssh``
    Runs ``qsub`` / ``sbatch`` over an existing SSH connection. Works for
    every machine in ``~/.ssh/config`` -- including Sunspot and
    Perlmutter, which the ALCF API does not serve -- and reuses whatever
    authentication (ControlMaster, agent, MFA) is already live.

``iri``
    The `ALCF IRI API <https://docs.alcf.anl.gov/services/iri-api/>`_:
    ``POST /api/v1/compute/job/{resource_id}``. Structured JSON in and
    out, no screen-scraping of ``qstat``. Covers **Aurora, Polaris and
    Crux only**, and needs a Globus token from ``alcf-tokens``.

Automatic fallback, and why it is narrow
----------------------------------------
``backend="auto"`` tries SSH and falls back to IRI **only on exit code
255**, which is SSH's own "I could not establish the session" status.
Every other non-zero code is the *remote command's* exit status, and
retrying those would be actively harmful:

.. code-block:: text

    rc=255   unreachable / auth failed / connection closed  -> fall back
    rc=1     qsub rejected the job (bad queue, over quota)  -> STOP
    rc=0     submitted                                      -> done

A ``qsub`` rejection means the scheduler refused for a reason. Re-sending
it through a different path either reproduces the rejection with a worse
error message, or -- if the first submission actually landed before the
non-zero exit -- **submits the job twice**. Wrong queue, wrong project,
wrong filesystem identifier and per-user queue limits all surface as
``rc=1``; each deserves the scheduler's own message, not a silent retry.

The backend that ran is always logged, so a later "which path did this
job take?" has an answer.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# SSH's own failure status. Anything else came from the remote command.
SSH_TRANSPORT_FAILURE = 255

# The IRI API serves these machines (docs.alcf.anl.gov/services/iri-api/,
# "Currently Supported Compute Resources"). Sunspot and Perlmutter are
# absent, so `auto` must fail on them rather than pretend.
IRI_MACHINES: frozenset[str] = frozenset({"aurora", "polaris", "crux"})

# Only Polaris's resource id is published in the docs; the rest are
# looked up at runtime via GET /status/resources rather than guessed.
# Hard-coding an id we cannot verify would submit to the wrong machine.
IRI_RESOURCE_IDS: dict[str, str] = {
    "polaris": "55c1c993-1124-47f9-b823-514ba3849a9a",
}

# Scheduler per machine, for building the right submit command remotely.
REMOTE_SCHEDULERS: dict[str, str] = {
    "aurora": "PBS",
    "sunspot": "PBS",
    "polaris": "PBS",
    "sophia": "PBS",
    "sirius": "PBS",
    "crux": "PBS",
    "perlmutter": "SLURM",
    "frontier": "SLURM",
}

# For positional sugar: is this token a host alias (aurora) or a job arg (script path)?
_KNOWN_HOSTS = frozenset(REMOTE_SCHEDULERS) | IRI_MACHINES


def looks_like_host(s: str) -> bool:
    """Heuristic: does bare token look like a remote host alias?"""
    low = s.lower().split(".")[0].split("-")[0]
    return (
        low in _KNOWN_HOSTS
        or s.lower() in REMOTE_SCHEDULERS
        or s.lower() in IRI_MACHINES
    )


def looks_like_job_id(s: str) -> bool:
    """Heuristic: does token look like a PBS/SLURM job ID (numeric prefix)?"""
    # 8936801 or 8936801.aurora-pbs-... or 8936286.aurora
    base = s.split(".")[0]
    return base.isdigit()


class RemoteSubmitError(RuntimeError):
    """Raised when a remote submission cannot be completed."""


_JOB_ID_RE = __import__("re").compile(r"^[A-Za-z0-9._@\[\]\-:]+$")
_HOST_RE = __import__("re").compile(r"^[A-Za-z0-9._\-]+$")


def _validate_ssh_host(host: str) -> None:
    """Fail fast when *host* contains characters SSH would interpret."""
    if not host or not _HOST_RE.match(host):
        raise ValueError(f"refusing suspicious SSH host {host!r}")


def _validate_job_id(job_id: str) -> None:
    """Fail fast for job IDs that could not have come from PBS."""
    if not job_id or not _JOB_ID_RE.match(job_id):
        raise ValueError(f"refusing suspicious job id {job_id!r}")


@dataclass
class RemoteJobStatus:
    """Status of a job as returned by IRI."""

    job_id: str
    state: str  # new/queued/held/active/completed/failed/canceled
    exit_code: int | None = None
    queue: str | None = None
    account: str | None = None
    owner: str | None = None
    raw: dict | None = None
    name: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    nodes: int | None = None
    walltime: str | None = None
    elapsed: str | None = None
    note: str | None = None
    exec_hosts: list[str] | None = None


@dataclass
class RemoteResult:
    """Outcome of a remote submission.

    Attributes:
        job_id: Scheduler job id, when submission succeeded.
        backend: Which backend produced this result (``"ssh"`` or
            ``"iri"``), so the path taken is recoverable later.
        returncode: Exit status of the submitting command.
        stderr: Captured error output, for surfacing the scheduler's own
            message rather than a paraphrase.
    """

    job_id: str | None
    backend: str
    returncode: int
    stderr: str = ""


def scheduler_for(host: str) -> str:
    """Return ``"PBS"`` or ``"SLURM"`` for *host*.

    Matches on a prefix so ``aurora``, ``aurora-uan-0010`` and an
    ``~/.ssh/config`` alias like ``aurora-gpu`` all resolve.

    Args:
        host: Machine name or SSH alias.

    Returns:
        The scheduler name; defaults to ``"PBS"`` when unrecognised,
        which is the ALCF majority.
    """
    h = host.lower()
    for name, sched in REMOTE_SCHEDULERS.items():
        if h.startswith(name):
            return sched
    logger.warning(
        "Unknown remote host %r; assuming PBS. Pass --scheduler to override.",
        host,
    )
    return "PBS"


def has_iri_backend(host: str) -> bool:
    """True when *host* can be reached through the ALCF IRI API."""
    h = host.lower()
    return any(h.startswith(name) for name in IRI_MACHINES)


def submit_via_ssh(
    host: str,
    script_text: str,
    *,
    remote_path: str | None = None,
    scheduler: str | None = None,
    workdir: str | None = None,
) -> RemoteResult:
    """Write *script_text* on *host* and submit it there.

    The script is delivered over stdin rather than ``scp`` so this needs
    no second connection and no local temp file.

    Args:
        host: SSH destination (any name your ``~/.ssh/config`` resolves).
        script_text: Complete job script.
        remote_path: Where to write it remotely. Defaults to a name under
            ``$HOME`` derived from the job.
        scheduler: ``"PBS"`` or ``"SLURM"``; inferred from *host* if omitted.
        workdir: Directory to submit from, if not the remote ``$HOME``.

    Returns:
        A :class:`RemoteResult`. ``returncode == 255`` means SSH itself
        failed, which is the only case a caller should treat as
        "try another backend".
    """
    sched = (scheduler or scheduler_for(host)).upper()
    submit_cmd = (
        "/opt/pbs/bin/qsub" if sched == "PBS" else "/opt/pbs/bin/sbatch"
    )
    path = remote_path or "~/.ezpz-remote-job.sh"

    # Write via stdin, then submit. `cat >` rather than scp keeps this to
    # one connection, which matters when MFA gates each new one.
    cd = f"cd {shlex.quote(workdir)} && " if workdir else ""
    remote = (
        f"mkdir -p $(dirname {path}) && cat > {path} && chmod +x {path} && "
        f"{cd}{submit_cmd} {path}"
    )
    argv = ["ssh", "-o", "BatchMode=yes", host, remote]

    logger.info("Submitting to %s via ssh (%s)", host, submit_cmd)
    try:
        proc = subprocess.run(
            argv,
            input=script_text,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        return RemoteResult(
            None, "ssh", SSH_TRANSPORT_FAILURE, "ssh not found"
        )
    except subprocess.TimeoutExpired:
        return RemoteResult(
            None, "ssh", SSH_TRANSPORT_FAILURE, "ssh timed out after 180s"
        )

    if proc.returncode == 0:
        return RemoteResult(proc.stdout.strip(), "ssh", 0, proc.stderr)
    return RemoteResult(None, "ssh", proc.returncode, proc.stderr.strip())


def submit_via_iri(
    host: str,
    *,
    command: str,
    nodes: int,
    duration_seconds: int,
    queue: str,
    account: str,
    workdir: str | None = None,
    filesystems: str | None = None,
) -> RemoteResult:
    """Submit through the ALCF IRI API.

    Requires a token from ``alcf-tokens get-token iri``. Only the
    machines in :data:`IRI_RESOURCE_IDS` are reachable this way.

    Args:
        host: Machine name; must have an IRI resource id.
        command: Shell command line to run.
        nodes: Node count.
        duration_seconds: Walltime in seconds (IRI takes seconds, not
            ``HH:MM:SS``).
        queue: Scheduler queue name.
        account: Project to charge.
        workdir: Remote working directory.
        filesystems: PBS filesystem identifiers; Aurora needs ``flare``.

    Returns:
        A :class:`RemoteResult` with ``backend="iri"``.

    Raises:
        RemoteSubmitError: If *host* has no IRI resource id, or no token
            is available.
    """
    h = host.lower()
    if not has_iri_backend(host):
        raise RemoteSubmitError(
            f"{host!r} is not served by the ALCF IRI API "
            f"(supported: {', '.join(sorted(IRI_MACHINES))})"
        )
    token = _iri_token()
    resource_id = next(
        (rid for name, rid in IRI_RESOURCE_IDS.items() if h.startswith(name)),
        None,
    ) or _resolve_resource_id(host, token)
    # IRI schema (docs.alcf.anl.gov/services/iri-api/#2.1): filesystems
    # lives at attributes.custom_attributes.filesystems, NOT at top-level
    # custom_attributes and NOT at attributes.filesystems. stdout/stderr
    # are mandatory top-level fields.
    payload: dict[str, object] = {
        "executable": "/bin/bash",
        "arguments": ["-lc", command],
        "name": "ezpz",
        "resources": {"node_count": nodes},
        "attributes": {
            "duration": duration_seconds,
            "queue_name": queue,
            "account": account,
            "custom_attributes": {
                "filesystems": filesystems or "home",
            },
        },
        "stdout_path": f"{workdir}/ezpz-iri.out"
        if workdir
        else "/tmp/ezpz-iri.out",
        "stderr_path": f"{workdir}/ezpz-iri.err"
        if workdir
        else "/tmp/ezpz-iri.err",
    }

    import json
    import urllib.error
    import urllib.request

    url = f"https://api.alcf.anl.gov/api/v1/compute/job/{resource_id}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    logger.info("Submitting to %s via IRI API", host)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        return RemoteResult(None, "iri", exc.code, detail)
    except urllib.error.URLError as exc:
        return RemoteResult(None, "iri", 1, str(exc.reason))

    job_id = str(body.get("id") or body.get("job_id") or "").strip()
    return RemoteResult(job_id or None, "iri", 0, "")


def _resolve_resource_id(host: str, token: str) -> str:
    """Look up *host*'s IRI resource id at runtime.

    Only Polaris's id appears in the ALCF docs. Rather than hard-code
    ids we cannot verify -- which risks submitting to the wrong machine
    -- ask the documented ``GET /status/resources`` endpoint and match
    on name.

    Args:
        host: Machine name.
        token: IRI bearer token.

    Returns:
        The resource id.

    Raises:
        RemoteSubmitError: If the lookup fails or no resource matches.
    """
    import json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "https://api.alcf.anl.gov/api/v1/status/resources",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RemoteSubmitError(
            f"could not list IRI resources to find {host!r}: {exc}"
        ) from exc

    items = data if isinstance(data, list) else data.get("resources", [])
    h = host.lower()
    for item in items:
        name = str(item.get("name", "")).lower()
        if name and (h.startswith(name) or name.startswith(h)):
            rid = item.get("id") or item.get("resource_id")
            if rid:
                return str(rid)
    raise RemoteSubmitError(
        f"no IRI resource matching {host!r}; saw: "
        f"{[i.get('name') for i in items]}"
    )


def _iri_token() -> str:
    """Fetch an IRI access token via ``alcf-tokens``.

    Returns:
        The bearer token.

    Raises:
        RemoteSubmitError: If ``alcf-tokens`` is missing or fails. The
            message names the install command, because the usual cause
            is simply not having it.
    """
    env_token = os.environ.get("ALCF_IRI_TOKEN")
    if env_token:
        return env_token
    # Try uvx first (ephemeral, no install), then plain alcf-tokens
    proc = None
    last_exc = None
    for cmd in (
        ["uvx", "alcf-tokens", "get-token", "iri"],
        ["alcf-tokens", "get-token", "iri"],
    ):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            break
        except FileNotFoundError as exc:
            last_exc = exc
            continue
    if proc is None:
        raise RemoteSubmitError(
            "alcf-tokens not found; `uvx alcf-tokens login iri` (or "
            "`uv tool install alcf-tokens` then `alcf-tokens login iri`), "
            "or set ALCF_IRI_TOKEN"
        ) from last_exc
    if proc.returncode != 0:
        raise RemoteSubmitError(
            f"alcf-tokens failed: {proc.stderr.strip() or proc.returncode}"
        )
    return proc.stdout.strip()


def submit_remote(
    host: str,
    script_text: str,
    *,
    backend: str = "auto",
    scheduler: str | None = None,
    workdir: str | None = None,
    iri_kwargs: dict | None = None,
) -> RemoteResult:
    """Submit to host, choosing a backend.

    With backend=auto this tries IRI first on IRI-capable hosts
    (aurora/polaris/crux) and falls back to SSH on auth/network failure.
    On non-IRI hosts (sunspot/perlmutter) it goes straight to SSH.
    A non-zero exit from the remote qsub / sbatch is reported as-is:
    the scheduler refused for a reason, and resubmitting down another path
    risks a duplicate job.

    Args:
        host: Machine name or SSH alias.
        script_text: Complete job script (used by the SSH backend).
        backend: ``"ssh"``, ``"iri"`` or ``"auto"``.
        scheduler: Override the inferred scheduler.
        workdir: Remote working directory.
        iri_kwargs: Arguments for :func:`submit_via_iri`, needed when the
            IRI backend is used.

    Returns:
        A :class:`RemoteResult` whose ``backend`` field records the path
        actually taken.

    Raises:
        RemoteSubmitError: For an unknown *backend*, or when IRI is
            requested for a machine it does not serve.
    """
    backend = backend.lower()
    if backend not in {"ssh", "iri", "auto"}:
        raise RemoteSubmitError(
            f"unknown backend {backend!r} (expected ssh, iri or auto)"
        )

    if backend == "iri":
        return submit_via_iri(host, **(iri_kwargs or {}))

    # auto = IRI-first on IRI hosts, SSH-first otherwise
    if backend == "auto" and has_iri_backend(host):
        try:
            return submit_via_iri(host, **(iri_kwargs or {}))
        except RemoteSubmitError as exc:
            # Auth/token failure -> fall back to SSH; explicit "not served" shouldn't happen here
            msg = str(exc)
            if "not served" in msg:
                raise
            logger.warning("IRI submit failed (%s), falling back to SSH", msg)
            # fall through to SSH path below

    result = submit_via_ssh(
        host, script_text, scheduler=scheduler, workdir=workdir
    )
    if result.returncode == 0 or backend == "ssh":
        return result

    if result.returncode != SSH_TRANSPORT_FAILURE:
        logger.error(
            "Remote %s failed (exit %d): %s",
            "qsub/sbatch",
            result.returncode,
            result.stderr,
        )
        return result

    if not has_iri_backend(host):
        logger.error(
            "ssh to %s failed (exit 255) and %s has no ALCF IRI backend "
            "(IRI serves: %s). Original error: %s",
            host,
            host,
            ", ".join(sorted(IRI_MACHINES)),
            result.stderr,
        )
        return result

    logger.warning(
        "ssh to %s failed (exit 255: %s) — falling back to the IRI API",
        host,
        result.stderr or "no detail",
    )
    return submit_via_iri(host, **(iri_kwargs or {}))


# ── IRI status / cancel (verified against live OpenAPI at /openapi.json) ──


def get_job_status_via_iri(
    host: str, job_id: str, *, historical: bool = True
) -> RemoteJobStatus:
    """Fetch one job's status via IRI GET /compute/status/{rid}/{job_id}."""
    token = _iri_token()
    h = host.lower()
    resource_id = next(
        (rid for n, rid in IRI_RESOURCE_IDS.items() if h.startswith(n)), None
    ) or _resolve_resource_id(host, token)
    import json as _json
    import urllib.error as _ue
    import urllib.request as _ur

    # Accept short numeric IDs by not quoting dots; still encode safely
    safe_jid = urllib.parse.quote(job_id, safe=".-")
    url = f"https://api.alcf.anl.gov/api/v1/compute/status/{resource_id}/{safe_jid}"
    if historical:
        url += "?historical=true"
    req = _ur.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with _ur.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read().decode())
    except _ue.HTTPError as exc:
        detail = (
            exc.read().decode(errors="replace")[:500]
            if hasattr(exc, "read")
            else str(exc)
        )
        raise RemoteSubmitError(f"IRI status {exc.code}: {detail}") from exc
    except (TimeoutError, OSError) as exc:
        raise RemoteSubmitError(f"IRI status timeout: {exc}") from exc
    st = body.get("status") or {}
    jspec = body.get("job_spec") or {}
    return RemoteJobStatus(
        job_id=body.get("id", job_id),
        state=st.get("state", "unknown"),
        exit_code=st.get("exit_code"),
        queue=(jspec.get("attributes") or {}).get("queue_name"),
        account=(jspec.get("attributes") or {}).get("account"),
        raw=body,
        name=body.get("name") or jspec.get("name") or "",
        stdout_path=jspec.get("stdout_path") or body.get("stdout_path") or "",
        stderr_path=jspec.get("stderr_path") or body.get("stderr_path") or "",
    )


def list_jobs_via_iri(
    host: str,
    *,
    historical: bool = True,
    limit: int = 20,
    queue: str | None = None,
    owner: str | None = None,
    states: list[str] | None = None,
) -> list[RemoteJobStatus]:
    """List jobs via IRI POST /compute/status/{rid}."""
    token = _iri_token()
    h = host.lower()
    resource_id = next(
        (rid for n, rid in IRI_RESOURCE_IDS.items() if h.startswith(n)), None
    ) or _resolve_resource_id(host, token)
    import json as _json
    import urllib.error as _ue
    import urllib.request as _ur

    qs = f"?historical={'true' if historical else 'false'}&limit={limit}"
    url = f"https://api.alcf.anl.gov/api/v1/compute/status/{resource_id}{qs}"
    payload: dict = {}
    if states:
        payload["states"] = states
    if queue:
        payload["queue"] = queue
    if owner:
        payload["owner"] = owner
    req = _ur.Request(
        url,
        data=_json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _ur.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read().decode())
    except _ue.HTTPError as exc:
        detail = (
            exc.read().decode(errors="replace")[:500]
            if hasattr(exc, "read")
            else str(exc)
        )
        raise RemoteSubmitError(f"IRI list {exc.code}: {detail}") from exc
    except (TimeoutError, OSError) as exc:
        raise RemoteSubmitError(f"IRI list timeout: {exc}") from exc
    jobs = (
        body
        if isinstance(body, list)
        else body.get("jobs") or body.get("data") or []
    )
    out = []
    for j in jobs:
        st = j.get("status") or {}
        jspec = j.get("job_spec") or {}
        jattr = jspec.get("attributes") or {}
        out.append(
            RemoteJobStatus(
                job_id=j.get("id", ""),
                state=st.get("state", "unknown"),
                exit_code=st.get("exit_code"),
                queue=jattr.get("queue_name"),
                account=jattr.get("account"),
                raw=j,
                name=j.get("name") or jspec.get("name") or "",
                stdout_path=jspec.get("stdout_path")
                or j.get("stdout_path")
                or "",
                stderr_path=jspec.get("stderr_path")
                or j.get("stderr_path")
                or "",
            )
        )
    return out


def _pbs_json_to_status(
    jid: str, info: dict, *, host: str = ""
) -> RemoteJobStatus:
    """Convert one PBS ``qstat -f -F json`` ``Jobs[jid]`` dict to :class:`RemoteJobStatus`.

    Mirrors the field extraction in ``qmert`` (``/home/foremans/.local/bin/qmert``):
    queue, walltime, nodes, elapsed, comment/note, exec_host.
    """
    state_raw: str = str(
        info.get("job_state") or info.get("state") or "unknown"
    ).strip()
    # Normalise PBS single-letter → IRI-style long form so existing renderers work.
    _pbs_state_map = {
        "R": "active",
        "Q": "queued",
        "H": "held",
        "E": "active",  # exiting — still on nodes, treat as active
        "F": "completed",
        "U": "held",  # suspended
        "X": "completed",  # transit; rare
        "B": "active",  # Array job begining
    }
    state = _pbs_state_map.get(state_raw, state_raw).lower()

    rl = info.get("Resource_List", {}) or {}
    name = str(info.get("Job_Name") or "").strip()
    queue = str(info.get("queue") or "").strip() or None
    account = (
        str(info.get("Account_Name") or info.get("project") or "").strip()
        or None
    )
    walltime = str(rl.get("walltime") or "").strip() or None
    nodes_val = rl.get("nodect")
    if nodes_val is None:
        nodes_val = rl.get("nodes")
        # nodes may be a hostname string (e.g. "x4002c2s0b0n0") for interactive jobs
        if isinstance(nodes_val, str) and not nodes_val.isdigit():
            nodes_val = None
    try:
        nodes = int(nodes_val) if nodes_val is not None else None
    except (TypeError, ValueError):
        nodes = None

    ru = info.get("resources_used", {}) or {}
    elapsed = str(ru.get("walltime") or "").strip() or None

    note = str(info.get("comment") or "").strip() or None
    if note and note.lower().startswith("not running:"):
        note = note[len("Not Running:") :].strip()

    raw_host = str(info.get("exec_host") or info.get("exec_vnode") or "")
    exec_hosts: list[str] | None = None
    if raw_host:
        hosts: list[str] = []
        for chunk in raw_host.replace("+", " ").split():
            h = chunk.split("/")[0].split(".")[0].split(":")[0].strip("()")
            if h and h not in hosts:
                hosts.append(h)
        exec_hosts = hosts if hosts else None

    owner_field = str(info.get("Job_Owner") or info.get("euser") or "")
    owner = (
        owner_field.split("@")[0]
        if "@" in owner_field
        else (owner_field or None)
    )

    return RemoteJobStatus(
        job_id=jid,
        state=state,
        exit_code=None,
        queue=queue,
        account=account,
        owner=owner,
        raw={"_pbs": info, "_fanout_host": host} if host else {"_pbs": info},
        name=name,
        nodes=nodes,
        walltime=walltime,
        elapsed=elapsed,
        note=note,
        exec_hosts=exec_hosts,
    )


def _ssh_remote_user(host: str) -> str | None:
    """Resolve the SSH user for *host* from ``ssh -G`` (respects ``~/.ssh/config``)."""
    import subprocess as _spu

    try:
        proc = _spu.run(
            ["ssh", "-G", host], capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.lower().startswith("user "):
                    val = line.split(None, 1)[1].strip() if " " in line else ""
                    return val or None
    except Exception:
        pass
    return None


def resolve_remote_owner(host: str) -> str | None:
    """Owner to filter remote jobs on *host*.

    Prefers the ``User`` from ``~/.ssh/config`` (``ssh -G HOST``) — e.g.
    ``foremans`` on ``aurora`` even when the local macOS login is ``sam``.
    Falls back to :func:`getpass.getuser` only when no SSH entry exists
    (custom hostname) or ``ssh`` is unavailable.
    """
    user = _ssh_remote_user(host)
    if user:
        return user
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return None


def list_jobs_via_ssh(
    host: str,
    *,
    historical: bool = False,
    limit: int = 20,
    owner: str | None = None,
) -> list[RemoteJobStatus]:
    """List jobs via SSH bulk ``qselect`` + ``qstat -f -F json`` (qmert path).

    One SSH call for IDs, one for the JSON dump — no per-job ``qstat -xf``.
    Identical fidelity to ``qmert``: full ``NODES``/``WALL``/``JOBNAME``/``NOTE``
    columns and running-job nodelists, all in the bulk payload.

    Falls back to ``qstat -x`` text-table parsing when JSON is unavailable,
    preserving existing ``limit``/``owner`` semantics.

    Args:
        host: SSH destination.
        historical: Include completed/exiting history (``qstat -x``).
        limit: Max jobs when text-table fallback is used (ignored for JSON bulk).
        owner: Filter to this local user (``qselect -u``). ``None`` → all jobs.

    Returns:
        A list of :class:`RemoteJobStatus` sorted ``R`` → ``Q`` → ``H``.
        Empty list when the user has no matching jobs (same as ``qmert``).

    Raises:
        RemoteSubmitError: When SSH itself fails (exit 255) or the JSON dump
            cannot be parsed and the text fallback also fails.
    """
    import json as _js
    import shlex as _sx
    import subprocess as _sp

    # ── 1. Resolve IDs with qselect (scoped to owner, avoids whole-cluster JSON) ──
    _validate_ssh_host(host)
    if owner is not None:
        if not owner or any(c in owner for c in (";", "&", "|", "`", "$", "\n", "\r")):
            raise ValueError(f"refusing suspicious owner {owner!r}")
    qselect_arg = f" -u {_sx.quote(owner)}" if owner else ""
    xflag = " -x" if historical else ""
    argv_sel = [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        f"/opt/pbs/bin/qselect{qselect_arg}{xflag} 2>/dev/null",
    ]
    proc_sel = _sp.run(argv_sel, capture_output=True, text=True, timeout=15)  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if proc_sel.returncode == 255:
        raise RemoteSubmitError(
            f"ssh transport 255 to {host}: {proc_sel.stderr.strip()}"
        )
    ids = [
        ln.strip() for ln in (proc_sel.stdout or "").splitlines() if ln.strip()
    ]
    # Fallback: if qselect returned nothing, try bare qstat JSON (covers edge where
    # qselect filters out jobs we own cross-site, and matches qmert's pre-filter)
    if not ids:
        # Empty means no jobs — don't fall through to full-cluster dump.
        return []

    # ── 2. Bulk JSON dump: one SSH hop carries every column + note + nodelist ──
    # Pass IDs as separate qstat args (qselect → qstat positional). This avoids
    # a whole-cluster dump where a single malformed other-user record can poison
    # json.load() and silently drop us to the text fallback — qmert's first fix.
    # --all-users on aurora is 500+ jobs: bulk JSON with that many IDs times
    # out (server-side serialization). Fall back to text table for that case,
    # which streams in one `qstat` without per-job JSON expansion.
    if owner is None and len(ids) > 60:
        # Don't attempt bulk JSON — jump straight to text-table path below.
        # Keep ids=None sentinel so the fallback knows this was intentional.
        ids = []  # empty → skip JSON attempt, fall through to text path
        _skip_json = True
    else:
        _skip_json = False
        _MAX_BULK = 120
        truncated = len(ids) > _MAX_BULK
        if truncated:
            ids = ids[:_MAX_BULK]
        for _jid in ids:
            _validate_job_id(_jid)
        qids = " ".join(_sx.quote(j) for j in ids)
    if not _skip_json:
        argv_js = [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"/opt/pbs/bin/qstat -f -F json {qids} 2>/dev/null",
        ]
        proc_js = _sp.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            argv_js,
            capture_output=True,
            text=True,
            timeout=max(25, len(ids) * 10),
        )
        if (
            proc_js.returncode != 255
            and proc_js.stdout
            and proc_js.stdout.strip().startswith("{")
        ):
            try:
                data = _js.loads(proc_js.stdout)
                jobs_raw: dict = data.get("Jobs") or {}
                out: list[RemoteJobStatus] = []
                for jid, info in jobs_raw.items():
                    try:
                        out.append(_pbs_json_to_status(jid, info, host=host))
                    except Exception:
                        continue
                # Apply qmert ordering: R before E before Q before H, then by id.
                order = {"active": 0, "running": 0, "queued": 2, "held": 3}
                out.sort(
                    key=lambda j: (order.get(j.state.lower(), 4), j.job_id)
                )
                if truncated and out:
                    for _j in out:
                        _j.raw = _j.raw or {}
                        _j.raw["_truncated"] = True
                    import sys as _sys
                    print(f"[warn] showing {len(out)}/{_MAX_BULK}+ jobs (PBS bulk cap); use --limit or narrower filters", file=_sys.stderr)
                if out:
                    return out
                # JSON succeeded but yielded nothing — don't silently return empty;
                # fall through to text path so we still show something.
            except Exception:
                pass  # fall through to text fallback

    # ── 3. Text-table fallback (older PBS, or json load failure) ──
    _validate_ssh_host(host)
    qstat_cmd = "/opt/pbs/bin/qstat -x" if historical else "/opt/pbs/bin/qstat"
    argv_txt = ["ssh", "-o", "BatchMode=yes", host, qstat_cmd]
    proc_txt = _sp.run(argv_txt, capture_output=True, text=True, timeout=15)  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if proc_txt.returncode == 255:
        raise RemoteSubmitError(
            f"ssh transport 255 to {host}: {proc_txt.stderr.strip()}"
        )
    if proc_txt.returncode != 0 or not proc_txt.stdout.strip():
        return []
    out_txt: list[RemoteJobStatus] = []
    for line in proc_txt.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("Job id") or line.startswith("----"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        if owner and len(parts) >= 3 and parts[2] != owner:
            continue
        jid = parts[0].rstrip("*")
        if "." in jid:
            jid = jid.split(".")[0]
        qname = parts[-1] if len(parts) >= 2 else ""
        st = parts[-2] if len(parts) >= 3 else ""
        nm = parts[1] if len(parts) >= 2 else ""
        usr = parts[2] if len(parts) >= 3 else ""
        if owner and usr != owner:
            continue
        out_txt.append(
            RemoteJobStatus(
                job_id=jid,
                state=st,
                queue=qname,
                raw={"name": nm, "_fanout_host": host},
                owner=usr,
                name=nm,
            )
        )
        if limit and len(out_txt) >= limit:
            break
    return out_txt


def get_job_status_via_ssh(
    host: str, job_id: str, *, historical: bool = True
) -> RemoteJobStatus:
    """Fetch one job via SSH ``qstat -f -F json`` — rich fields, no bulk indirection."""
    _validate_ssh_host(host)
    _validate_job_id(job_id)
    import json as _js2
    import shlex as _sx2
    import subprocess as _sp2

    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        f"/opt/pbs/bin/qstat -f -F json {_sx2.quote(job_id)} 2>/dev/null",
    ]
    proc = _sp2.run(argv, capture_output=True, text=True, timeout=15)  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if proc.returncode == 255:
        raise RemoteSubmitError(
            f"ssh transport 255 to {host}: {proc.stderr.strip()}"
        )
    if proc.stdout and proc.stdout.strip().startswith("{"):
        try:
            data = _js2.loads(proc.stdout)
            jobs = data.get("Jobs") or {}
            if jobs:
                # PBS keys include FQDN suffix; match the first entry.
                jid, info = next(iter(jobs.items()))
                return _pbs_json_to_status(jid, info, host=host)
        except Exception:
            pass
    # Fallback: plain qstat text for this job (covers historical where -f json
    # may not include F/E records on older PBS).
    cmd = (
        f"/opt/pbs/bin/qstat -x {_sx2.quote(job_id)}"
        if historical
        else f"/opt/pbs/bin/qstat {_sx2.quote(job_id)}"
    )
    argv2 = ["ssh", "-o", "BatchMode=yes", host, cmd]
    proc2 = _sp2.run(argv2, capture_output=True, text=True, timeout=15)  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    if proc2.returncode != 0 or not proc2.stdout.strip():
        err = (proc2.stderr or proc2.stdout or "job not found").strip()
        raise RemoteSubmitError(err[:400])
    # Parse single-line summary
    for line in proc2.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("Job id") or line.startswith("----"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        jid = parts[0].rstrip("*")
        if "." in jid:
            jid = jid.split(".")[0]
        qname = parts[-1] if len(parts) >= 2 else ""
        st = parts[-2] if len(parts) >= 3 else ""
        nm = parts[1] if len(parts) >= 2 else ""
        usr = parts[2] if len(parts) >= 3 else ""
        return RemoteJobStatus(
            job_id=jid,
            state=st,
            queue=qname,
            raw={"name": nm, "_fanout_host": host},
            owner=usr,
            name=nm,
        )
    raise RemoteSubmitError(f"could not parse status for {job_id}")


def cancel_via_iri(host: str, job_id: str) -> RemoteJobStatus:
    """Cancel via IRI DELETE /compute/cancel/{rid}/{job_id} (204 on success)."""
    token = _iri_token()
    h = host.lower()
    resource_id = next(
        (rid for n, rid in IRI_RESOURCE_IDS.items() if h.startswith(n)), None
    ) or _resolve_resource_id(host, token)
    import urllib.error as _ue
    import urllib.request as _ur

    url = f"https://api.alcf.anl.gov/api/v1/compute/cancel/{resource_id}/{urllib.parse.quote(job_id, safe='')}"
    req = _ur.Request(
        url, headers={"Authorization": f"Bearer {token}"}, method="DELETE"
    )
    try:
        with _ur.urlopen(req, timeout=30) as resp:
            # 204 has no body
            if resp.status in (200, 204):
                return RemoteJobStatus(
                    job_id=job_id, state="canceled", raw={"via": "iri"}
                )
    except _ue.HTTPError as exc:
        detail = (
            exc.read().decode(errors="replace")[:500]
            if hasattr(exc, "read")
            else str(exc)
        )
        raise RemoteSubmitError(f"IRI cancel {exc.code}: {detail}") from exc
    raise RemoteSubmitError("IRI cancel: unexpected response")


def cancel_via_ssh(
    host: str, job_id: str, *, scheduler: str | None = None
) -> RemoteJobStatus:
    """Cancel via SSH qdel/scancel."""
    import shlex as _shlex
    import subprocess as _sp

    sched = (scheduler or scheduler_for(host)).upper()
    cmd = (
        f"/opt/pbs/bin/qdel {_shlex.quote(job_id)}"
        if sched == "PBS"
        else f"scancel {_shlex.quote(job_id)}"
    )
    argv = ["ssh", "-o", "BatchMode=yes", host, cmd]
    proc = _sp.run(argv, capture_output=True, text=True, timeout=30)
    if proc.returncode == 0:
        return RemoteJobStatus(
            job_id=job_id,
            state="canceled",
            raw={"via": "ssh", "stdout": proc.stdout},
        )
    raise RemoteSubmitError(
        f"cancel failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
    )


def update_via_iri(
    host: str,
    job_id: str,
    *,
    duration_seconds: int | None = None,
    queue: str | None = None,
    account: str | None = None,
    name: str | None = None,
    node_count: int | None = None,
    filesystems: str | None = None,
) -> RemoteJobStatus:
    """PUT /api/v1/compute/job/{rid}/{jid} — body is JobSpec-Input."""
    if not any(
        v is not None
        for v in (
            duration_seconds,
            queue,
            account,
            name,
            node_count,
            filesystems,
        )
    ):
        raise ValueError(
            "nothing to update: pass duration_seconds/queue/account/..."
        )
    if not has_iri_backend(host):
        raise RemoteSubmitError(
            f"{host!r} not served by IRI (supports {', '.join(sorted(IRI_MACHINES))})"
        )
    if duration_seconds is not None and duration_seconds < 1:
        raise ValueError("duration_seconds must be >=1")
    token = _iri_token()
    rid = next(
        (v for k, v in IRI_RESOURCE_IDS.items() if host.lower().startswith(k)),
        None,
    ) or _resolve_resource_id(host, token)
    import json as _json2
    import urllib.error as _ue2
    import urllib.request as _ur2

    attrs: dict = {}
    if duration_seconds is not None:
        attrs["duration"] = duration_seconds
    if queue is not None:
        attrs["queue_name"] = queue
    if account is not None:
        attrs["account"] = account
    if filesystems is not None:
        attrs["custom_attributes"] = {"filesystems": filesystems}
    payload: dict = {}
    if name is not None:
        payload["name"] = name
    if node_count is not None:
        payload["resources"] = {"node_count": node_count}
    if attrs:
        payload["attributes"] = attrs
    url = f"https://api.alcf.anl.gov/api/v1/compute/job/{rid}/{urllib.parse.quote(job_id, safe='')}"
    req = _ur2.Request(
        url,
        data=_json2.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    logger.info("IRI updateJob %s on %s: %s", job_id, host, payload)
    try:
        with _ur2.urlopen(req, timeout=60) as resp:
            body = (
                _json2.loads(resp.read().decode())
                if getattr(resp, "length", 1) != 0
                else {}
            )
    except _ue2.HTTPError as exc:
        detail = (
            exc.read().decode(errors="replace")[:600]
            if hasattr(exc, "read")
            else str(exc)
        )
        if exc.code == 401:
            raise RemoteSubmitError(
                f"IRI update 401 (token expired?): {detail}"
            ) from exc
        raise RemoteSubmitError(f"IRI update {exc.code}: {detail}") from exc
    except _ue2.URLError as exc:
        raise RemoteSubmitError(f"IRI update network: {exc.reason}") from exc
    st = body.get("status") or {}
    jspec = body.get("job_spec") or {}
    jattr = jspec.get("attributes") or {}
    return RemoteJobStatus(
        job_id=body.get("id", job_id),
        state=st.get("state", "unknown"),
        exit_code=st.get("exit_code"),
        queue=jattr.get("queue_name", queue),
        account=jattr.get("account", account),
        raw=body,
    )


def update_via_ssh(
    host: str,
    job_id: str,
    *,
    duration_seconds: int | None = None,
    queue: str | None = None,
    account: str | None = None,
    scheduler: str | None = None,
) -> RemoteJobStatus:
    import shlex as _shlex2
    import subprocess as _sp2

    sched = (scheduler or scheduler_for(host)).upper()
    parts = []
    if duration_seconds is not None:
        h, m, s = (
            duration_seconds // 3600,
            (duration_seconds % 3600) // 60,
            duration_seconds % 60,
        )
        wl = f"{h:02d}:{m:02d}:{s:02d}"
        parts.append(
            f"-l walltime={_shlex2.quote(wl)}"
            if sched == "PBS"
            else f"--time={_shlex2.quote(wl)}"
        )
    if queue is not None:
        parts.append(
            f"-q {_shlex2.quote(queue)}"
            if sched == "PBS"
            else f"-p {_shlex2.quote(queue)}"
        )
    if account is not None:
        parts.append(f"-A {_shlex2.quote(account)}")
    if not parts:
        raise ValueError("nothing to update")
    cmd = (
        ("qalter" if sched == "PBS" else "scontrol update JobId=")
        + " "
        + " ".join(parts)
        + f" {_shlex2.quote(job_id)}"
        if sched == "PBS"
        else f"scontrol update JobId={_shlex2.quote(job_id)} "
        + " ".join(parts)
    )
    argv = ["ssh", "-o", "BatchMode=yes", host, cmd]
    proc = _sp2.run(argv, capture_output=True, text=True, timeout=30)
    if proc.returncode == 0:
        return RemoteJobStatus(
            job_id=job_id,
            state="queued",
            raw={"via": "ssh", "stdout": proc.stdout},
        )
    if proc.returncode == 255:
        raise RemoteSubmitError(f"ssh transport 255: {proc.stderr.strip()}")
    raise RemoteSubmitError(
        f"update failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
    )


def update_remote(
    host: str,
    job_id: str,
    *,
    backend: str = "auto",
    duration_seconds: int | None = None,
    queue: str | None = None,
    account: str | None = None,
    **kw,
) -> RemoteJobStatus:
    """Auto: IRI-first on IRI hosts, else SSH. 401/network falls back; 400/422 stop."""
    backend = backend.lower()
    if backend not in {"ssh", "iri", "auto"}:
        raise RemoteSubmitError(
            f"unknown backend {backend!r} (expected ssh, iri or auto)"
        )
    if backend == "iri":
        return update_via_iri(
            host,
            job_id,
            duration_seconds=duration_seconds,
            queue=queue,
            account=account,
            **kw,
        )
    if backend == "ssh":
        return update_via_ssh(
            host,
            job_id,
            duration_seconds=duration_seconds,
            queue=queue,
            account=account,
        )
    if has_iri_backend(host):
        try:
            return update_via_iri(
                host,
                job_id,
                duration_seconds=duration_seconds,
                queue=queue,
                account=account,
                **kw,
            )
        except RemoteSubmitError as exc:
            msg = str(exc)
            if "not served" in msg:
                raise
            if "Unknown Job Id" in msg or "Invalid job" in msg:
                raise
            if " 400" in msg or " 422" in msg:
                raise
            logger.warning("IRI update failed (%s), falling back to SSH", msg)
    return update_via_ssh(
        host,
        job_id,
        duration_seconds=duration_seconds,
        queue=queue,
        account=account,
    )


def _iri_get(path: str, token: str, timeout: int = 30):
    import json as _jg
    import urllib.error as _ueg
    import urllib.request as _urg

    req = _urg.Request(
        f"https://api.alcf.anl.gov{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with _urg.urlopen(req, timeout=timeout) as r:
            return _jg.loads(r.read().decode())
    except _ueg.HTTPError as e:
        detail = (
            e.read().decode(errors="replace")[:600]
            if hasattr(e, "read")
            else str(e)
        )
        raise RemoteSubmitError(f"IRI GET {path} {e.code}: {detail}") from e
    except (TimeoutError, OSError) as e:
        raise RemoteSubmitError(f"IRI GET {path} unreachable: {e}") from e


def get_compute_resources(token: str | None = None):
    tok = token or _iri_token()
    return _iri_get("/api/v1/compute/resources", tok)


def get_all_resources(token: str | None = None, limit: int = 100):
    tok = token or _iri_token()
    return _iri_get(f"/api/v1/status/resources?limit={limit}", tok)


def list_incidents(
    resource_id: str | None = None, status: str | None = None, limit: int = 20
):
    tok = None
    try:
        tok = _iri_token()
    except RemoteSubmitError:
        pass  # incidents can be public
    params = {"limit": str(limit)}
    if resource_id:
        params["resource_id"] = resource_id
    if status:
        params["status"] = status
    qs = urllib.parse.urlencode(params)
    path = f"/api/v1/status/incidents?{qs}"
    headers = {"Accept": "application/json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    import json as _ji
    import urllib.request as _uri

    req = _uri.Request(f"https://api.alcf.anl.gov{path}", headers=headers)
    try:
        with _uri.urlopen(req, timeout=15) as r:
            body = _ji.loads(r.read().decode())
    except Exception as e:
        raise RemoteSubmitError(f"incidents fetch failed: {e}") from e
    if isinstance(body, list):
        return body
    return body.get("incidents", body.get("data", body.get("events", [])))


def cancel_remote(
    host: str,
    job_id: str,
    *,
    backend: str = "auto",
    scheduler: str | None = None,
) -> RemoteJobStatus:
    """Cancel with same auto-fallback rule as submit: 255 → IRI, else stop."""
    backend = backend.lower()
    if backend not in {"ssh", "iri", "auto"}:
        raise RemoteSubmitError(
            f"unknown backend {backend!r} (expected ssh, iri or auto)"
        )
    if backend == "iri":
        return cancel_via_iri(host, job_id)
    # try SSH first
    try:
        return cancel_via_ssh(host, job_id, scheduler=scheduler)
    except RemoteSubmitError as exc:
        if backend == "ssh":
            raise
        # auto: only fall back on transport failure (255) and IRI-capable host
        msg = str(exc)
        # Heuristic: check if error came from SSH transport vs scheduler
        # cancel_via_ssh raises with scheduler stderr for scheduler rejections — do NOT fall back
        if "Unknown Job Id" in msg or "Invalid job" in msg:
            raise
        if not has_iri_backend(host):
            raise
        logger.warning("SSH cancel failed, falling back to IRI: %s", msg)
        return cancel_via_iri(host, job_id)
