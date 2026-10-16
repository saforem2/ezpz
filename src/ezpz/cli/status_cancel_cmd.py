"""Click commands for `ezpz status` and `ezpz cancel` over SSH/IRI."""

from __future__ import annotations

import json
import sys

import click

_STATE_COLORS = {
    "active": "green",
    "running": "green",
    "completed": "cyan",
    "queued": "yellow",
    "held": "magenta",
    "failed": "red",
    "canceled": "red",
    "new": "dim",
}

def _rich_table(jobs, *, title=None):
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Job ID", style="dim", no_wrap=True)
    table.add_column("Name", overflow="fold", max_width=18)
    table.add_column("State", justify="center")
    table.add_column("Queue")
    table.add_column("Output", style="dim", overflow="fold", max_width=32)
    table.add_column("Exit", justify="center")
    for j in jobs:
        color = _STATE_COLORS.get(j.state.lower(), "")
        state_styled = f"[{color}]{j.state}[/{color}]" if color else j.state
        ec = str(j.exit_code) if j.exit_code is not None else "—"
        short = j.job_id.split(".")[0]
        nm = getattr(j, "name", None) or ""
        # stdout path -> show basename + parent dir
        out = getattr(j, "stdout_path", None) or ""
        if "/" in out:
            out = out.rsplit("/", 1)[-1]  # basename only
            if out == "ezpz-iri.out":
                out = "ezpz-iri.out"
        table.add_row(short, nm or "—", state_styled, j.queue or "—", out or "—", ec)
    Console().print(table)
    from rich.console import Console as _C

    c = _C()
    for j in jobs:
        out = getattr(j, "stdout_path", "") or ""
        hint = f"  → {out}" if out else ""
        c.print(f"  [dim]{j.job_id}{hint}[/dim]")

def _rich_single(st):
    from rich.console import Console

    c = Console()
    color = _STATE_COLORS.get(st.state.lower(), "")
    state = f"[{color}]{st.state}[/{color}]" if color else st.state
    ec = st.exit_code if st.exit_code is not None else "—"
    out = getattr(st, "stdout_path", "") or ""
    err = getattr(st, "stderr_path", "") or ""
    extra = f"\n  [dim]stdout: {out}[/dim]" if out else ""
    if err and err != out:
        extra += f"\n  [dim]stderr: {err}[/dim]"
    c.print(f"[bold]{st.job_id}[/bold]  {state}  [dim]queue={st.queue or '—'}  exit={ec}  (via iri)[/dim]{extra}")


@click.command()
@click.argument("job_id", required=False)
@click.option("--remote", default=None, metavar="HOST", help="Host to query (SSH name or ALCF resource). Omit for local qstat.")
@click.option("--backend", default="auto", type=click.Choice(["auto", "ssh", "iri"]), help="Remote backend with --remote (default: auto).")
@click.option("--historical/--no-historical", default=False, help="Include completed/failed/canceled jobs (IRI only).")
@click.option("--limit", type=int, default=20, help="Max jobs when listing (IRI only, default 20).")
@click.option("--queue", default=None, help="Filter by queue (IRI list).")
@click.option("--owner", default=None, help="Filter by owner (IRI list).")
@click.option("--states", default=None, help="Filter by IRI states, comma-separated (e.g. queued,active).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON instead of table.")
def status_cmd(job_id, remote, backend, historical, limit, queue, owner, states, as_json):
    """Show job status (local or remote)."""
    if not remote:
        _status_local(job_id)
        return
    # Default auto to iri for IRI-capable hosts (faster, no SSH hop)
    from ezpz.remote import RemoteSubmitError, has_iri_backend

    if backend == "auto" and has_iri_backend(remote):
        backend = "iri"
    try:
        if job_id:
            _status_single_remote(remote, job_id, backend=backend, historical=historical, as_json=as_json)
        else:
            _status_list_remote(
                remote, backend=backend, historical=historical, limit=limit, queue=queue, owner=owner, states=states, as_json=as_json
            )
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except (TimeoutError, OSError) as exc:
        click.echo(f"error: IRI timed out after 30s — try without --historical or filter by --queue/--states: {exc}", err=True)
        sys.exit(1)


def _status_local(job_id):
    import subprocess

    cmd = ["qstat", "-x", job_id] if job_id else ["qstat"]
    # Try -x (historical) first, fall back to plain qstat
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and job_id:
        proc2 = subprocess.run(["qstat", job_id], capture_output=True, text=True)
        if proc2.returncode == 0:
            proc = proc2
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)


def _status_single_remote(host, job_id, backend, historical, as_json):
    from ezpz.remote import get_job_status_via_iri, has_iri_backend

    if backend in ("iri", "auto") and has_iri_backend(host):
        if backend == "iri" or backend == "auto":
            # Try IRI first for auto when available; on failure, fall through to SSH if auto
            try:
                st = get_job_status_via_iri(host, job_id, historical=historical)
            except Exception as exc:
                if backend == "iri":
                    raise
                # auto fallback: try SSH
                click.echo(f"note: IRI status failed ({exc}), trying SSH", err=True)
                _status_via_ssh(host, job_id)
                return
            if as_json:
                click.echo(json.dumps({"id": st.job_id, "state": st.state, "exit_code": st.exit_code, "queue": st.queue, "account": st.account}))
            elif sys.stdout.isatty():
                _rich_single(st)
            else:
                ec = st.exit_code if st.exit_code is not None else "-"
                click.echo(f"{st.job_id}  {st.state}  queue={st.queue or '-'}  exit={ec}  (via iri)")
            return
    _status_via_ssh(host, job_id)


def _status_list_remote(host, backend, historical, limit, queue, owner, states, as_json):
    from ezpz.remote import has_iri_backend, list_jobs_via_iri

    if backend in ("iri", "auto") and has_iri_backend(host):
        try:
            st_list = state_list(states) if states else None
            jobs = list_jobs_via_iri(host, historical=historical, limit=limit, queue=queue, owner=owner, states=st_list)
            if as_json:
                click.echo(json.dumps([{"id": j.job_id, "state": j.state, "exit_code": j.exit_code, "queue": j.queue} for j in jobs], indent=2))
            elif sys.stdout.isatty():
                _rich_table(jobs)
            else:
                if not jobs:
                    click.echo("(no jobs)")
                    return
                click.echo(f"{'JOB ID':<50} {'STATE':<10} {'QUEUE':<12} {'EXIT'}")
                for j in jobs:
                    ec = j.exit_code if j.exit_code is not None else "-"
                    click.echo(f"{j.job_id:<50} {j.state:<10} {j.queue or '-':<12} {ec}")
            return
        except Exception as exc:
            if backend == "iri":
                raise
            click.echo(f"note: IRI list failed ({exc}), trying SSH", err=True)
    _status_via_ssh(host, None)


def _status_via_ssh(host, job_id):
    import re
    import subprocess

    cmd = f"/opt/pbs/bin/qstat -x {job_id}" if job_id else "/opt/pbs/bin/qstat"
    argv = ["ssh", "-o", "BatchMode=yes", host, cmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    # Trim FQDN job IDs to short numeric for readability
    out = proc.stdout
    if out and proc.returncode == 0:
        lines = out.splitlines()
        trimmed = []
        for line in lines:
            m = re.match(r"^(\s*)(\d+)\.[^\s]+(.*)", line)
            if m and line.strip() and not line.strip().startswith("Job id"):
                trimmed.append(f"{m.group(1)}{m.group(2)}{m.group(3)}")
            else:
                trimmed.append(line)
        out = "\n".join(trimmed)
        if out and not out.endswith("\n"):
            out += "\n"
    # Hint when falling back to SSH on an IRI-capable host
    from ezpz.remote import has_iri_backend

    if has_iri_backend(host) and sys.stderr.isatty():
        import click as _click

        _click.echo(
            "  [dim]tip: IRI is available for this host — run `uvx alcf-tokens login iri` on the target UAN "
            "for faster, no-SSH status (see `ezpz status --help`)[/dim]",
            err=True,
        )
    sys.stdout.write(out)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)


def state_list(s):
    return [x.strip().lower() for x in s.split(",") if x.strip()]


@click.command()
@click.argument("job_id", required=True)
@click.option("--remote", default=None, metavar="HOST", help="Host to cancel on (SSH name or ALCF resource). Omit for local qdel.")
@click.option("--backend", default="auto", type=click.Choice(["auto", "ssh", "iri"]), help="Remote backend with --remote (default: auto).")
@click.option("--scheduler", default=None, type=click.Choice(["PBS", "SLURM"]), help="Override scheduler auto-detection.")
@click.option("--dry-run", is_flag=True, default=False, help="Print what would run without cancelling.")
def cancel_cmd(job_id, remote, backend, scheduler, dry_run):
    """Cancel a queued or running job."""
    if not remote:
        _cancel_local(job_id, scheduler)
        return
    from ezpz.remote import RemoteSubmitError, cancel_remote

    if dry_run:
        click.echo(f"would cancel {job_id} on {remote} via {backend} (scheduler={scheduler or 'auto'})")
        return
    try:
        result = cancel_remote(remote, job_id, backend=backend, scheduler=scheduler)
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    if result.returncode == 0:
        click.echo(f"cancelled {job_id} (via {result.backend})")
        return
    click.echo(f"error: cancel {job_id} failed via {result.backend} (exit {result.returncode}): {result.stderr}", err=True)
    sys.exit(result.returncode or 1)


def _cancel_local(job_id, scheduler):
    import subprocess

    from ezpz.remote import scheduler_for
    import socket

    sched = (scheduler or scheduler_for(socket.gethostname())).upper()
    cmd = ["qdel", job_id] if sched == "PBS" else ["scancel", job_id]
    # Prefer absolute path for PBS like submit does
    if sched == "PBS":
        for p in ["/opt/pbs/bin/qdel", "qdel"]:
            import shutil

            if shutil.which(p) or p.startswith("/"):
                cmd[0] = p
                break
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)
