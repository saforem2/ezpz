"""Fetch stdout/stderr for a job (local or remote) + per-example benchmark logs."""

from __future__ import annotations

import subprocess
import sys

import click


@click.command(name="fetch", context_settings={"ignore_unknown_options": True})
@click.argument("job_id", required=False)
@click.option("--remote", default=None, metavar="HOST", help="Host to fetch from (SSH name or IRI resource). Omit for local $PBS_O_WORKDIR / SLURM_SUBMIT_DIR.")
@click.option("--out", "which", type=click.Choice(["out", "err", "both"]), default="out", help="Which stream to fetch (default: out).")
@click.option("--lines", "-n", type=int, default=200, help="Tail lines to show (default: 200).")
@click.option("--dedupe/--no-dedupe", default=True, help="Deduplicate MPI rank spam (default: on).")
@click.pass_context
def fetch_cmd(ctx, job_id, remote, which, lines, dedupe):
    """Fetch stdout/stderr for a job.

    Without --remote, cats the PBS/spool output file locally.
    With --remote, resolves stdout_path via IRI or qstat -xf over SSH.

    For benchmark jobs, use `ezpz logs` to fetch per-example logs directly.
    """
    if not remote:
        _fetch_local(job_id, which, lines)
        return
    if job_id:
        try:
            from ezpz.remote import get_job_status_via_iri

            st = get_job_status_via_iri(remote, job_id, historical=True)
            path = (st.stdout_path or "").strip() if which in ("out", "both") else ""
            err_path = (st.stderr_path or "").strip() if which == "both" else ""
            if path:
                if path.endswith("/") or "." not in path.rsplit("/", 1)[-1]:
                    path = path.rstrip("/") + "/ezpz-iri.out"
                if err_path and (err_path.endswith("/") or "." not in err_path.rsplit("/", 1)[-1]):
                    err_path = err_path.rstrip("/") + "/ezpz-iri.err"
                _fetch_remote_cat(remote, path, which, lines, err_path if which == "both" else None, dedupe=dedupe)
                return
        except Exception:
            pass
    _fetch_remote_qstat(remote, job_id, which, lines, dedupe=dedupe)


@click.command(name="logs", context_settings={"ignore_unknown_options": True})
@click.argument("job_id", required=False)
@click.option("--remote", default=None, metavar="HOST", help="Host to fetch from (SSH name or IRI resource).")
@click.option("--example", default=None, help="Only show this example (e.g. test, fsdp, vit). Without it, shows the last-failed example's log.")
@click.option("--lines", "-n", type=int, default=200, help="Tail lines (default: 200).")
@click.pass_context
def logs_cmd(ctx, job_id, remote, example, lines):
    """Fetch per-example benchmark logs (not the spool).

    Resolves the job's WORKDIR from qstat, then tails
    $WORKDIR/outputs/ezpz-benchmarks/*/EXAMPLE.log directly.
    Much less spam than the spool (which duplicates per MPI rank).
    """
    import shlex

    if not remote and not job_id:
        click.echo("error: need --remote HOST and JOB_ID for remote logs (e.g. ezpz logs 12481065 --remote sunspot)", err=True)
        sys.exit(2)
    host = remote or "localhost"
    # Resolve WORKDIR from qstat -xf (the per-job dir like /home/foremans/ezpz-12481065...)
    # For PBS: Error_Path often has the WORKDIR prefix; also check env
    qcmd = f"/opt/pbs/bin/qstat -xf {shlex.quote(job_id)}" if job_id else "/opt/pbs/bin/qstat -xf"
    argv = ["ssh", "-o", "BatchMode=yes", host, qcmd] if remote else ["bash", "-lc", qcmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    workdir = None
    raw = proc.stdout.replace("\n\t", "")
    for line in raw.splitlines():
        if "PBS_O_WORKDIR" in line or "Error_Path" in line or "Output_Path" in line:
            # Extract path after :/ or =
            if ":/" in line:
                cand = line[line.find(":/") + 1:].strip().split()[0]
                if cand.startswith("/"):
                    workdir = cand.rsplit("/", 1)[0] if cand.endswith(".o12481065") or cand.endswith(".e") else None
                    if workdir and "ezpz" in cand:
                        workdir = cand  # already has filename
                    elif cand.endswith("/") or "ezpz-12481065" in cand:
                        # This is the WORKDIR itself or contains it
                        if "ezpz-12481065" in cand:
                            workdir = cand[: cand.find("ezpz-12481065") + len("ezpz-12481065.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov")]
                        else:
                            workdir = cand
                    break
    if not workdir:
        # Fallback: conventional location
        if job_id:
            base = job_id.split(".")[0]
            workdir = f"/home/foremans/ezpz-{base}.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov"
        else:
            click.echo("error: could not resolve workdir from qstat", err=True)
            sys.exit(1)
    # Benchmark logs live at $WORKDIR/outputs/ezpz-benchmarks/*/EXAMPLE.log
    if example:
        pattern = f"{workdir}/outputs/ezpz-benchmarks/*/{example}.log"
    else:
        pattern = f"{workdir}/outputs/ezpz-benchmarks/*/*.log"
    if remote:
        argv2 = ["ssh", "-o", "BatchMode=yes", host, f"ls -t {pattern} 2>&1 | head -5; echo ---LOG---; tail -n {lines} $(ls -t {pattern} 2>/dev/null | head -1) 2>&1 | tail -n {lines}"]
    else:
        argv2 = ["bash", "-lc", f"tail -n {lines} $(ls -t {pattern} 2>/dev/null | head -1) 2>&1 | tail -n {lines}"]
    proc2 = subprocess.run(argv2, capture_output=True, text=True, timeout=30)
    sys.stdout.write(proc2.stdout)
    if proc2.returncode != 0 and not proc2.stdout.strip():
        click.echo(f"error: {proc2.stderr.strip() or 'no output'}", err=True)
        sys.exit(proc2.returncode or 1)


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


def _fetch_remote_cat(host, path, which, lines, err_path=None, dedupe=True):
    import shlex

    for p in ([path] if which == "out" else [err_path, path] if which == "both" and err_path else [path]):
        if not p:
            continue
        label = f"=== {host}:{p} ===" if which == "both" else ""
        if label:
            click.echo(label, err=True)
        qpath = shlex.quote(p)
        cmd = f"tail -n {lines} {qpath} 2>&1 | uniq" if dedupe else f"tail -n {lines} {qpath} 2>&1"
        argv = ["ssh", "-o", "BatchMode=yes", host, cmd]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0 and not proc.stdout.strip():
            click.echo(f"error: {proc.stderr.strip() or proc.stdout.strip() or 'no output'}", err=True)
            sys.exit(proc.returncode or 1)


def _resolve_qstat_output_path(qstat_output: str) -> str | None:
    """Parse Output_Path from qstat -xf output. Handles PBS line wrapping."""
    raw = qstat_output.replace("\n\t", "")
    for line in raw.splitlines():
        if "Output_Path" in line and ":" in line:
            idx = line.find(":/")
            if idx != -1:
                return line[idx + 1:].strip().split()[0]
            rhs = line.split(":", 1)[-1].strip()
            return rhs.split()[0] if rhs else None
    return None


def _resolve_qstat_output_path_legacy(qstat_output: str) -> str | None:
    return _resolve_qstat_output_path(qstat_output)


def _fetch_remote_qstat(host, job_id, which, lines, dedupe=True):
    import shlex

    qcmd = f"/opt/pbs/bin/qstat -xf {shlex.quote(job_id)}" if job_id else "/opt/pbs/bin/qstat -xf"
    argv = ["ssh", "-o", "BatchMode=yes", host, qcmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    out_path = _resolve_qstat_output_path(proc.stdout)
    if not out_path:
        click.echo("error: could not discover output path from qstat — job may not have started", err=True)
        sys.exit(1)
    if out_path.endswith("/"):
        base = job_id.split(".")[0] if job_id else ""
        if base:
            out_path = out_path + f"ezpz.o{base}"
    qpath = shlex.quote(out_path)
    cmd = f"tail -n {lines} {qpath} 2>&1 | uniq" if dedupe else f"tail -n {lines} {qpath} 2>&1"
    argv2 = ["ssh", "-o", "BatchMode=yes", host, cmd]
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
