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


class RemoteSubmitError(RuntimeError):
    """Raised when a remote submission cannot be completed."""


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
    submit_cmd = "qsub" if sched == "PBS" else "sbatch"
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
    payload: dict[str, object] = {
        "executable": "/bin/bash",
        "arguments": ["-lc", command],
        "name": "ezpz",
        "resources": {"node_count": nodes},
        "attributes": {
            "duration": duration_seconds,
            "queue_name": queue,
            "account": account,
        },
    }
    if workdir:
        payload["stdout_path"] = workdir
        payload["stderr_path"] = workdir
    if filesystems:
        payload["custom_attributes"] = {"filesystems": filesystems}

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
    try:
        proc = subprocess.run(
            ["alcf-tokens", "get-token", "iri"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RemoteSubmitError(
            "alcf-tokens not found; `pip install alcf-tokens` and run "
            "`alcf-tokens login`, or set ALCF_IRI_TOKEN"
        ) from exc
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
    """Submit to *host*, choosing a backend.

    With ``backend="auto"`` this tries SSH first and falls back to IRI
    **only when SSH itself could not connect** (exit 255). A non-zero
    exit from the remote ``qsub`` / ``sbatch`` is reported as-is: the
    scheduler refused for a reason, and resubmitting down another path
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

    result = submit_via_ssh(
        host, script_text, scheduler=scheduler, workdir=workdir
    )
    if result.returncode == 0 or backend == "ssh":
        return result

    if result.returncode != SSH_TRANSPORT_FAILURE:
        # The remote scheduler ran and refused. Do NOT retry elsewhere.
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
