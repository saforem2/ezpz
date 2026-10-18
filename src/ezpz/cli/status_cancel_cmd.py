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

def _short_wall(s: str | None) -> str:
    if not s or s == "--":
        return "--"
    # Normalize HH:MM:SS or seconds to short
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, _ = parts
            return f"{int(h):02d}:{int(m):02d}"
        return s
    try:
        secs = int(float(s))
        return f"{secs//3600:02d}:{(secs%3600)//60:02d}"
    except Exception:
        return s


def _rich_table(jobs, *, title=None):
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("JOBID", style="dim", no_wrap=True)
    table.add_column("S", justify="center")
    table.add_column("QUEUE", no_wrap=True)
    table.add_column("NODES", justify="right")
    table.add_column("WALL", justify="right")
    table.add_column("ELAP", justify="right")
    table.add_column("JOBNAME", overflow="fold", max_width=32)
    table.add_column("NOTE", overflow="fold", max_width=40)
    for j in jobs:
        color = _STATE_COLORS.get(j.state.lower(), "")
        st = j.state.upper()[0] if j.state else "?"
        # Map long states to single letter like qmert: R/Q/H/F
        smap = {"active": "R", "running": "R", "queued": "Q", "held": "H", "completed": "F", "failed": "F", "canceled": "F", "new": "Q"}
        s_disp = smap.get(j.state.lower(), st)
        s_styled = f"[{color}]{s_disp}[/{color}]" if color else s_disp
        short = j.job_id.split(".")[0]
        nodes = str(getattr(j, "nodes", None) or "—")
        wall = _short_wall(getattr(j, "walltime", None))
        elap = _short_wall(getattr(j, "elapsed", None)) if getattr(j, "elapsed", None) else ("--" if j.state.lower() in ("queued", "held", "new") else "—")
        note = (getattr(j, "note", None) or "").strip()
        # Trim note noise: drop "Insufficient amount of resource:" prefix shortening
        if note.lower().startswith("insufficient"):
            # Keep short: "at_queue" / "queue_tags" suffix if present
            note = note.split(":")[-1].strip() if ":" in note else note
        table.add_row(short, s_styled, j.queue or "—", nodes, wall, elap, getattr(j, "name", None) or "—", note or "")
    # Use rounded box to match qmert ascii borders
    table.box = __import__("rich.box", fromlist=["ROUNDED"]).ROUNDED
    Console().print(table)
    # Running job nodes block (like qmert)
    running = [j for j in jobs if j.state.lower() in ("active", "running", "r")]
    if running:
        c2 = Console()
        c2.print("")
        c2.print("[bold]running job nodes:[/bold]")
        for j in running:
            hosts = getattr(j, "exec_hosts", None) or []
            if not hosts:
                continue
            shown = " ".join(hosts[:6])
            extra = f" ...(+{len(hosts)-6} more)" if len(hosts) > 6 else ""
            c2.print(f"  {j.job_id.split('.')[0]} ({len(hosts)}): {shown}{extra}")
        c2.print("[dim]Shared connection to {} closed.[/dim]".format((jobs[0].raw or {}).get("_fanout_host", "") if jobs and isinstance(jobs[0].raw, dict) else ""))

def _rich_single(st):
    from rich.console import Console

    c = Console()
    color = _STATE_COLORS.get(st.state.lower(), "")
    state = f"[{color}]{st.state}[/{color}]" if color else st.state
    ec = st.exit_code if st.exit_code is not None else "—"
    out = getattr(st, "stdout_path", "") or ""
    err = getattr(st, "stderr_path", "") or ""
    host = getattr(st, "queue", "") or ""  # placeholder — real host from job_id suffix
    # Try to extract host hint from FQDN
    host_hint = ""
    if "." in st.job_id:
        host_hint = st.job_id.split(".", 1)[-1].split(".")[0]  # aurora-pbs-0001
    extra = ""
    if out:
        extra += f"\n  [dim]stdout: {out}[/dim]"
        # Suggest fetch
        short = st.job_id.split(".")[0]
        extra += f"\n  [dim]→ ezpz fetch {short} --remote {host_hint or 'aurora'}[/dim]"
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
@click.option("--owner", default=None, help="Filter by owner (IRI list). Use --all-users to see everyone's jobs.")
@click.option("--all-users", is_flag=True, default=False, help="Include all users' jobs (default: only yours).")
@click.option("--states", default=None, help="Filter by IRI states, comma-separated (e.g. queued,active).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON instead of table.")
def status_cmd(job_id, remote, backend, historical, limit, queue, owner, states, all_users, as_json):
    """Show job status (local or remote)."""
    # Positional sugar: ezpz status aurora  ≡  --remote aurora
    if not remote and job_id:
        from ezpz.remote import looks_like_host as _looks_host
        if _looks_host(job_id):
            remote = job_id
            job_id = None
    if not remote:
        # Fan-out: no job_id and no remote → query every reachable machine
        if job_id is None:
            _status_fanout(backend=backend, historical=historical, limit=limit, queue=queue, owner=owner, states=states, all_users=all_users, as_json=as_json)
            return
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
            if owner is None and not all_users:
                import getpass
                try:
                    owner = getpass.getuser()
                except Exception:
                    pass
            _status_list_remote(
                remote, backend=backend, historical=historical, limit=limit, queue=queue, owner=owner, states=states, as_json=as_json
            )
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except (TimeoutError, OSError) as exc:
        click.echo(f"error: IRI timed out after 30s — try without --historical or filter by --queue/--states: {exc}", err=True)
        sys.exit(1)


def _status_fanout(*, backend, historical, limit, queue, owner, states, all_users, as_json):
    """Fan out status to every reachable machine and merge results."""
    import concurrent.futures
    import getpass
    from ezpz.remote import REMOTE_SCHEDULERS, IRI_MACHINES

    if owner is None and not all_users:
        try:
            owner = getpass.getuser()
        except Exception:
            pass

    # All known hosts: IRI machines + every REMOTE_SCHEDULERS entry (dedup)
    hosts = sorted(set(REMOTE_SCHEDULERS) | set(IRI_MACHINES))

    def _fetch_one(h: str):
        # Historical only makes sense for IRI; for SSH we just use qstat -x
        # For fan-out keep it simple: non-historical, per-host limit
        try:
            from ezpz.remote import has_iri_backend, list_jobs_via_iri
            if has_iri_backend(h):
                jobs = list_jobs_via_iri(h, historical=historical, limit=limit, queue=queue, owner=owner, states=state_list(states) if states else None)
                for j in jobs:
                    j.raw = j.raw or {}
                    j.raw["_fanout_host"] = h
                return (h, jobs, None)
        except Exception as e:
            # Fall through to SSH attempt
            pass
        try:
            import subprocess, shlex
            qcmd = "/opt/pbs/bin/qstat -x" if historical else "/opt/pbs/bin/qstat"
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", h, qcmd]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
            if proc.returncode != 0 or not proc.stdout.strip():
                return (h, [], f"unreachable ({proc.returncode})" if proc.returncode not in (0, None) else "no jobs")
            # Reuse existing SSH parser path: split into RemoteJobStatus
            from ezpz.remote import RemoteJobStatus as _RJS
            jobs = []
            for line in proc.stdout.splitlines():
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
                q = parts[-1] if len(parts) >= 2 else ""
                st = parts[-2] if len(parts) >= 3 else ""
                nm = parts[1] if len(parts) >= 2 else ""
                usr = parts[2] if len(parts) >= 3 else ""
                if owner and usr != owner:
                    continue
                jobs.append(_RJS(job_id=jid, state=st, queue=q, owner=usr, raw={"name": nm, "_fanout_host": h}))
            return (h, jobs, None)
        except Exception as e:
            return (h, [], str(e)[:80])

    # Fan out concurrently (short timeout per host; unreachable hosts are skipped, not fatal)
    all_jobs: list = []
    warnings: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(hosts))) as pool:
        futs = {pool.submit(_fetch_one, h): h for h in hosts}
        for fut in concurrent.futures.as_completed(futs, timeout=35):
            h, jobs, err = fut.result()
            if err:
                warnings.append(f"{h}: {err}")
            elif jobs:
                all_jobs.extend(jobs)

    if not all_jobs:
        if warnings:
            click.echo("(no jobs — " + "; ".join(warnings[:3]) + ")", err=True)
        else:
            click.echo("(no jobs)")
        return

    # Sort: active/queued first, then by id
    order = {"active": 0, "running": 0, "queued": 1, "held": 2, "new": 3, "completed": 4, "failed": 5, "canceled": 6}
    all_jobs.sort(key=lambda j: (order.get(j.state.lower(), 9), j.job_id))

    if as_json:
        click.echo(json.dumps([{"id": j.job_id, "state": j.state, "queue": j.queue, "owner": getattr(j, "owner", None), "exit_code": j.exit_code, "host": (j.raw or {}).get("_fanout_host", "")} for j in all_jobs], indent=2))
        return
    if sys.stdout.isatty():
        # Fan-out rich table: add Host column
        from rich.console import Console
        from rich.table import Table
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Host", style="bold cyan", no_wrap=True)
        table.add_column("Job ID", style="dim", no_wrap=True)
        table.add_column("User", style="dim")
        table.add_column("State", justify="center")
        table.add_column("Queue")
        table.add_column("Exit", justify="center")
        for j in all_jobs:
            color = _STATE_COLORS.get(j.state.lower(), "")
            st_disp = f"[{color}]{j.state}[/{color}]" if color else j.state
            ec = str(j.exit_code) if j.exit_code is not None else "—"
            host = (j.raw or {}).get("_fanout_host", "") if isinstance(j.raw, dict) else ""
            table.add_row(host, j.job_id.split(".")[0], getattr(j, "owner", "") or "—", st_disp, j.queue or "—", ec)
        Console().print(table)
        if warnings and not as_json:
            for w in warnings[:3]:
                click.echo(f"  [dim]⚠ {w}[/dim]", err=True)
    else:
        click.echo(f"{'HOST':<12} {'JOB ID':<14} {'USER':<10} {'STATE':<10} {'QUEUE':<12} {'EXIT'}")
        for j in all_jobs:
            ec = j.exit_code if j.exit_code is not None else "-"
            host = (j.raw or {}).get("_fanout_host", "") if isinstance(j.raw, dict) else ""
            click.echo(f"{host:<12} {j.job_id.split('.')[0]:<14} {getattr(j, 'owner', '') or '-':<10} {j.state:<10} {j.queue or '-':<12} {ec}")


def _status_local(job_id):
    import shutil
    import subprocess

    if not shutil.which("qstat"):
        click.echo(
            "error: qstat not found — no local scheduler. "
            "Did you mean: ezpz status --remote aurora  or  ezpz status aurora ?",
            err=True,
        )
        sys.exit(127)
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
                click.echo(json.dumps({"id": st.job_id, "state": st.state, "exit_code": st.exit_code, "queue": st.queue, "account": st.account, "stdout_path": st.stdout_path, "stderr_path": st.stderr_path}))
            elif sys.stdout.isatty():
                _rich_single(st)
            else:
                ec = st.exit_code if st.exit_code is not None else "-"
                line = f"{st.job_id}  {st.state}  queue={st.queue or '-'}  exit={ec}  (via iri)"
                if st.stdout_path:
                    line += f"\n  stdout: {st.stdout_path}"
                if st.stderr_path and st.stderr_path != st.stdout_path:
                    line += f"\n  stderr: {st.stderr_path}"
                click.echo(line)
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
    # SSH fallback: carry owner filter via function attribute for client-side filtering
    _status_via_ssh._owner_filter = owner  # type: ignore[attr-defined]
    _status_via_ssh(host, None)


def _status_via_ssh(host, job_id):
    import re
    import subprocess

    cmd = f"/opt/pbs/bin/qstat -x {job_id}" if job_id else "/opt/pbs/bin/qstat -x"
    argv = ["ssh", "-o", "BatchMode=yes", host, cmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    # If not TTY (piped), keep raw qstat for grep/awk compat; rich only on TTY
    if not sys.stdout.isatty() or proc.returncode != 0 or not proc.stdout.strip():
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    # TTY: parse qstat -x rows and render rich table (includes E/F/history)
    from ezpz.remote import RemoteJobStatus as _RJS

    raw = proc.stdout
    # Check if --owner filtering was requested via _owner_holder (set by _status_list_remote)
    _owner_filter = getattr(_status_via_ssh, "_owner_filter", None)
    jobs: list[_RJS] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("Job id") or line.startswith("----"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        # Filter by owner if requested (SSH qstat has no server-side filter)
        if _owner_filter and len(parts) >= 3 and parts[2] != _owner_filter:
            continue
        jid = parts[0].rstrip("*")
        if "." in jid:
            jid = jid.split(".")[0]
        queue = parts[-1] if len(parts) >= 2 else ""
        state = parts[-2] if len(parts) >= 3 else ""
        name = parts[1] if len(parts) >= 2 else ""
        user = parts[2] if len(parts) >= 3 else ""
        jobs.append(_RJS(job_id=jid, state=state, queue=queue, raw={"name": name}, owner=user))
    if not jobs:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    # Enrich with qstat -xf per job for qmert columns (walltime, nodes, elap, note, exec_hosts)
    # Only for small lists to keep it fast; best-effort (failures leave —)
    if len(jobs) <= 30:
        _enrich_from_qstat_xf(host, jobs)
    _rich_table(jobs)


def _enrich_from_qstat_xf(host: str, jobs) -> None:
    """Best-effort: fill walltime/nodes/elap/note/exec_hosts from qstat -xf per job."""
    import re
    import subprocess

    for j in jobs:
        try:
            argv = ["ssh", "-o", "BatchMode=yes", host, f"/opt/pbs/bin/qstat -xf {j.job_id}"]
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=8)
            if proc.returncode != 0 or not proc.stdout:
                continue
            txt = proc.stdout.replace("\n\t", "")
            m = re.search(r"Job_Name\s*=\s*(\S+)", txt)
            if m:
                j.name = m.group(1).strip()
            m = re.search(r"Resource_List\.walltime\s*=\s*(\S+)", txt)
            if m:
                j.walltime = m.group(1).strip()
            m = re.search(r"Resource_List\.select\s*=\s*(\d+)", txt)
            if m:
                j.nodes = int(m.group(1))
            else:
                m2 = re.search(r"Resource_List\.nodect\s*=\s*(\d+)", txt)
                if m2:
                    j.nodes = int(m2.group(1))
            m = re.search(r"resources_used\.walltime\s*=\s*(\S+)", txt)
            if m:
                j.elapsed = m.group(1).strip()
            # comment is the scheduling note (blocking reason)
            m = re.search(r"comment\s*=\s*(.+)", txt)
            if m:
                note = m.group(1).strip()
                # PBS comment may be multi-token like "Not Running: Insufficient amount of resource: at_queue ..."
                if note.lower().startswith("not running:"):
                    note = note[len("Not Running:"):].strip()
                j.note = note
            m = re.search(r"exec_host\s*=\s*(\S+)", txt)
            if m:
                raw_hosts = m.group(1).strip()
                # exec_host = x4219c2s3b0n0/0*208+x4219c2s4b0n0/1*208+... -> hosts
                hosts = []
                for chunk in raw_hosts.split("+"):
                    h = chunk.split("/")[0].strip()
                    if h and h not in hosts:
                        hosts.append(h)
                j.exec_hosts = hosts
        except Exception:
            continue


def state_list(s):
    return [x.strip().lower() for x in s.split(",") if x.strip()]


@click.command()
@click.argument("job_id", required=True)
@click.argument("maybe_host", required=False)
@click.option("--remote", default=None, metavar="HOST", help="Host to cancel on (SSH name or ALCF resource). Omit for local qdel. Also positional: ezpz cancel aurora JOBID.")
@click.option("--backend", default="auto", type=click.Choice(["auto", "ssh", "iri"]), help="Remote backend with --remote (default: auto).")
@click.option("--scheduler", default=None, type=click.Choice(["PBS", "SLURM"]), help="Override scheduler auto-detection.")
@click.option("--dry-run", is_flag=True, default=False, help="Print what would run without cancelling.")
def cancel_cmd(job_id, maybe_host, remote, backend, scheduler, dry_run):
    """Cancel a queued or running job."""
    # Positional sugar: ezpz cancel aurora 12345  ≡  --remote aurora 12345
    if maybe_host and not remote:
        from ezpz.remote import looks_like_host as _lh
        # job_id is actually the host when two positionals given
        if _lh(job_id) and maybe_host:
            remote = job_id
            job_id = maybe_host
            maybe_host = None
    # Click gives us first positional as job_id; check if it is a host
    if not remote:
        from ezpz.remote import looks_like_host as _looks_host2
        if _looks_host2(job_id):
            remote = job_id
            # Need second positional — not available in current signature,
            # so we treat as error asking for --remote form for cancel
            # Actually: cancel requires JOB_ID, so `cancel aurora 123` needs 2 args.
            # Click only has one arg, so this path is for `cancel aurora` meaning
            # cancel on aurora without job_id — not valid. Fall through to error.
            pass
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
    import shutil
    import subprocess

    from ezpz.remote import scheduler_for
    import socket

    sched = (scheduler or scheduler_for(socket.gethostname())).upper()
    cmd = ["qdel", job_id] if sched == "PBS" else ["scancel", job_id]
    if not shutil.which(cmd[0]):
        click.echo(
            f"error: {cmd[0]} not found — no local scheduler. "
            f"Did you mean: ezpz cancel --remote aurora {job_id} ?",
            err=True,
        )
        sys.exit(127)
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
