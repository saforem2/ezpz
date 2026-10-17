"""Fetch stdout/stderr for a job (local or remote)."""

from __future__ import annotations

import subprocess
import sys

import click


@click.command(name="fetch", context_settings={"ignore_unknown_options": True})
@click.argument("job_id", required=False)
@click.option("--remote", default=None, metavar="HOST", help="Host to fetch from (SSH name or IRI resource). Omit for local $PBS_O_WORKDIR / SLURM_SUBMIT_DIR.")
@click.option("--out", "which", type=click.Choice(["out", "err", "both"]), default="out", help="Which stream to fetch (default: out).")
@click.option("--lines", "-n", type=int, default=200, help="Tail lines to show (default: 200).")
@click.pass_context
def fetch_cmd(ctx, job_id, remote, which, lines):
    """Fetch stdout/stderr for a job.

    Without --remote, cats the PBS/spool output file locally.
    With --remote, resolves stdout_path via IRI or qstat -xf over SSH.
    """
    if not remote:
        _fetch_local(job_id, which, lines)
        return
    # Try IRI stdout_path first (when job_id given), then SSH fallback
    if job_id:
        try:
            from ezpz.remote import get_job_status_via_iri

            st = get_job_status_via_iri(remote, job_id, historical=True)
            path = (st.stdout_path or "").strip() if which in ("out", "both") else ""
            err_path = (st.stderr_path or "").strip() if which == "both" else ""
            if path:
                # stdout_path may be a dir (old bug) or file — handle both
                if path.endswith("/") or "." not in path.rsplit("/", 1)[-1]:
                    path = path.rstrip("/") + "/ezpz-iri.out"
                if err_path and (err_path.endswith("/") or "." not in err_path.rsplit("/", 1)[-1]):
                    err_path = err_path.rstrip("/") + "/ezpz-iri.err"
                _fetch_remote_cat(remote, path, which, lines, err_path if which == "both" else None)
                return
        except Exception:
            pass  # fall through to SSH
    _fetch_remote_qstat(remote, job_id, which, lines)


def _fetch_local(job_id, which, lines):
    import os

    out_path = None
    if job_id:
        proc = subprocess.run(["qstat", "-xf", job_id], capture_output=True, text=True)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if "Output_Path" in line or "stdout_path" in line.lower():
                    if ":" in line:
                        out_path = line.split(":", 1)[-1].strip()
                        break
    if not out_path:
        out_path = os.environ.get("PBS_O_WORKDIR") or os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
    _cat_tail(out_path, lines, which)


def _fetch_remote_cat(host, path, which, lines, err_path=None):
    import shlex

    for p in ([path] if which == "out" else [err_path, path] if which == "both" and err_path else [path]):
        if not p:
            continue
        label = f"=== {host}:{p} ===" if which == "both" else ""
        if label:
            click.echo(label, err=True)
        qpath = shlex.quote(p)
        # Use -n +N for head compatibility; tail -n for most systems
        argv = ["ssh", "-o", "BatchMode=yes", host, f"tail -n {lines} {qpath} 2>&1 || cat {qpath} 2>&1 | tail -n {lines}"]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0 and not proc.stdout.strip():
            click.echo(f"error: {proc.stderr.strip() or proc.stdout.strip() or 'no output'}", err=True)
            sys.exit(proc.returncode or 1)


def _fetch_remote_qstat(host, job_id, which, lines):
    import shlex

    qcmd = f"/opt/pbs/bin/qstat -xf {shlex.quote(job_id)}" if job_id else "/opt/pbs/bin/qstat -xf"
    argv = ["ssh", "-o", "BatchMode=yes", host, qcmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    out_path = None
    for line in proc.stdout.splitlines():
        if "Output_Path" in line and ":" in line:
            # Output_Path = host:/path — split on first ":"
            rhs = line.split(":", 1)[-1].strip()
            out_path = rhs.split()[0] if rhs else None
            break
    if out_path and ":" in out_path and not out_path.startswith("/"):
        # FQDN leftover like aurora-uan-0011.hostmgmt...:/home/... — re-split
        out_path = out_path.split(":", 1)[-1]
    if not out_path:
        click.echo("error: could not discover output path (qstat has no Output_Path) — job may not have started", err=True)
        sys.exit(1)
    qpath = shlex.quote(out_path)
    argv2 = ["ssh", "-o", "BatchMode=yes", host, f"tail -n {lines} {qpath} 2>&1 || cat {qpath} 2>&1 | tail -n {lines}"]
    proc2 = subprocess.run(argv2, capture_output=True, text=True, timeout=30)
    sys.stdout.write(proc2.stdout)
    if proc2.returncode != 0 and not proc2.stdout.strip():
        click.echo(f"error: fetch failed: {proc2.stderr.strip() or 'no output'}", err=True)
        sys.exit(proc2.returncode or 1)


def _cat_tail(path, lines, which):
    import pathlib

    p = pathlib.Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("*.OU"), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            p = candidates[0]
    if not p.exists():
        click.echo(f"error: no output at {p}", err=True)
        sys.exit(1)
    text = p.read_text(errors="replace")
    for ln in text.splitlines()[-lines:]:
        print(ln)
