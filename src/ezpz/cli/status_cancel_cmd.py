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
        return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"
    except Exception:
        return s


def _rich_table(jobs, *, host: str | None = None):
    """Render the qmert-style table — columns, ordering, and adornments match.

    Column set: JOBID S QUEUE NODES WALL ELAP JOBNAME NOTE (qmert order).
    Includes width-aware NOTE collapse to sub-lines and a running-job nodelist
    block, matching qmert's terminal-width negotiation.
    """
    from rich.console import Console
    from rich.table import Table
    import shutil

    # ── compute widths with the same truncation/collapse rules as qmert ──
    HEAD = ["JOBID", "S", "QUEUE", "NODES", "WALL", "ELAP", "JOBNAME", "NOTE"]
    NAME_I, NOTE_I = 6, 7
    has_note = any((getattr(j, "note", None) or "").strip() for j in jobs)
    ncol = 8 if has_note else 7

    # Materialize display rows (pre-style) so widths can be measured.
    smap = {
        "active": "R",
        "running": "R",
        "queued": "Q",
        "held": "H",
        "completed": "F",
        "failed": "F",
        "canceled": "F",
        "new": "Q",
    }
    rows_disp: list[list[str]] = []
    notes_by_id: dict[str, str] = {}
    for j in jobs:
        st_raw = j.state.upper()[0] if j.state else "?"
        s_disp = smap.get(j.state.lower(), st_raw)
        short = j.job_id.split(".")[0]
        wall = _short_wall(getattr(j, "walltime", None))
        # -- means queued/not-started, — means not applicable (completed).
        elap_val = getattr(j, "elapsed", None)
        if elap_val:
            elap = _short_wall(elap_val)
        elif j.state.lower() in ("queued", "held", "new"):
            elap = "--"
        else:
            elap = "—"
        nodes = str(getattr(j, "nodes", None) or "—")
        note = (getattr(j, "note", None) or "").strip()
        # Shorten verbose scheduler phrasing; qmert strips "Not Running: " — that is
        # already done in _pbs_json_to_status. Here handle secondary noise.
        if note.lower().startswith("insufficient"):
            note = note.split(":")[-1].strip() if ":" in note else note
        # Remember note even if we collapse the column — shown as sub-line.
        if note:
            notes_by_id[short] = note
        row = [
            short,
            s_disp,
            j.queue or "—",
            nodes,
            wall,
            elap,
            getattr(j, "name", None) or "—",
            note,
        ]
        rows_disp.append(row[:ncol])

    # Terminal width like qmert: COLUMNS env → tput → stty → 0 (unknown = no truncation).
    TW = (
        shutil.get_terminal_size(fallback=(0, 24)).columns
        if hasattr(shutil, "get_terminal_size")
        else 0
    )
    try:
        import os as _osw

        if _osw.environ.get("COLUMNS", "").isdigit():
            TW = int(_osw.environ["COLUMNS"])
    except Exception:
        pass

    def table_w(widths: list[int]) -> int:
        return sum(widths) + 3 * len(widths) + 1

    def widths_for(rws: list[list[str]], ncol_: int) -> list[int]:
        hdrs = HEAD[:ncol_]
        return [
            max(len(str(r[i])) for r in ([hdrs] + rws)) for i in range(ncol_)
        ]

    widths = widths_for(rows_disp, ncol)
    # Too wide: collapse NOTE column into sub-lines (nothing lost).
    dropped_notes: dict[str, str] = {}
    if TW and ncol == 8 and table_w(widths) > TW:
        for r in rows_disp:
            if r[NOTE_I]:
                dropped_notes[r[0]] = r[NOTE_I]
        ncol = 7
        rows_disp = [r[:7] for r in rows_disp]
        widths = widths_for(rows_disp, ncol)
    # Still too wide: ellipsize JOBNAME to fit.
    if TW and table_w(widths) > TW:
        over = table_w(widths) - TW
        neww = max(8, widths[NAME_I] - over)
        if neww < widths[NAME_I]:
            widths[NAME_I] = neww
            for r in rows_disp:
                if len(r[NAME_I]) > neww:
                    r[NAME_I] = r[NAME_I][: neww - 1] + "…"

    # ── build Rich table with precomputed max_width to enforce wrapping ──
    table = Table(
        show_header=True, header_style="bold", box=None, padding=(0, 1)
    )
    table.add_column("JOBID", style="dim", no_wrap=True)
    table.add_column("S", justify="center")
    table.add_column("QUEUE", no_wrap=True)
    table.add_column("NODES", justify="right")
    table.add_column("WALL", justify="right")
    table.add_column("ELAP", justify="right")
    table.add_column(
        "JOBNAME", overflow="fold", max_width=widths[NAME_I] if TW else 32
    )
    if ncol == 8:
        table.add_column(
            "NOTE", overflow="fold", max_width=widths[NOTE_I] if TW else 40
        )

    for j, rd in zip(jobs, rows_disp):
        color = _STATE_COLORS.get(j.state.lower(), "")
        s_styled = f"[{color}]{rd[1]}[/{color}]" if color else rd[1]
        styled = [rd[0], s_styled, rd[2], rd[3], rd[4], rd[5], rd[6]]
        if ncol == 8:
            styled.append(rd[7])
        table.add_row(*styled)
    table.box = __import__("rich.box", fromlist=["ROUNDED"]).ROUNDED
    Console().print(table)

    # ── sub-lines: dropped notes, then running-job node lists (width-aware, like qmert) ──
    c2 = Console()

    def fit_items(items: list[str], prefix_len: int, suffix_tmpl: str):
        if TW <= 0:
            return items[:8], max(0, len(items) - 8)
        avail = TW - prefix_len
        shown, used = [], 0
        for it in items:
            add = len(it) + (1 if shown else 0)
            if used + add > avail - len(suffix_tmpl.format(0)):
                break
            shown.append(it)
            used += add
        return shown, len(items) - len(shown)

    if dropped_notes:
        c2.print("")
        c2.print("[dim]notes:[/dim]")
        for rd in rows_disp:
            jid = rd[0]
            if jid in dropped_notes:
                # Find original state color for the jid
                j0 = next(
                    (jj for jj in jobs if jj.job_id.split(".")[0] == jid), None
                )
                col2 = _STATE_COLORS.get(j0.state.lower(), "") if j0 else ""
                tag = f"[{col2}]{jid}[/{col2}]" if col2 else jid
                c2.print(f"  {tag} [dim]{dropped_notes[jid]}[/dim]")

    running = [
        j for j in jobs if j.state.lower() in ("active", "running", "r")
    ]
    if running:
        c2.print("")
        c2.print("[dim]running job nodes:[/dim]")
        rc2 = _STATE_COLORS.get("active") or _STATE_COLORS.get("running") or ""
        for j in running:
            hosts = getattr(j, "exec_hosts", None) or []
            if not hosts:
                continue
            jid = j.job_id.split(".")[0]
            plain_prefix = f"  {jid} ({len(hosts)}): "
            shown, remaining = fit_items(
                hosts, len(plain_prefix), " ...(+{} more)"
            )
            line = " ".join(shown)
            if remaining:
                line += f" [dim]...(+{remaining} more)[/dim]"
            rc_tag = f"[{rc2}]{jid}[/{rc2}]" if rc2 else jid
            c2.print(f"  {rc_tag} [dim]({len(hosts)})[/dim]: {line}")
        # Match qmert's trailing "Shared connection ..." line only for single-host views.
        if host and jobs and isinstance((jobs[0].raw or {}), dict):
            c2.print(f"[dim]Shared connection to {host} closed.[/dim]")


def _rich_single(st, *, via: str = ""):
    from rich.console import Console

    c = Console()
    color = _STATE_COLORS.get(st.state.lower(), "")
    state = f"[{color}]{st.state}[/{color}]" if color else st.state
    ec = st.exit_code if st.exit_code is not None else "—"
    via_tag = f"  (via {via})" if via else ""
    # Single-job view: queue/nodes/wall/elap/name/note mirror the list columns.
    wall = _short_wall(getattr(st, "walltime", None))
    elap = (
        _short_wall(getattr(st, "elapsed", None))
        if getattr(st, "elapsed", None)
        else None
    )
    nodes = getattr(st, "nodes", None)
    name = getattr(st, "name", None) or ""
    note = (getattr(st, "note", None) or "").strip()
    out = getattr(st, "stdout_path", "") or ""
    err = getattr(st, "stderr_path", "") or ""
    # Try to extract host hint from FQDN
    host_hint = ""
    if "." in st.job_id:
        host_hint = st.job_id.split(".", 1)[-1].split(".")[
            0
        ]  # aurora-pbs-0001
    queue = st.queue or "—"
    summary = f"[bold]{st.job_id}[/bold]  {state}  [dim]queue={queue}  exit={ec}{via_tag}[/dim]"
    if name:
        summary += f"\n  [dim]name:[/dim]  {name}"
    if nodes is not None:
        summary += f"  [dim]nodes:[/dim] {nodes}"
    if getattr(st, "walltime", None):
        summary += f"  [dim]wall:[/dim] {wall}"
    if elap:
        summary += f"  [dim]elapsed:[/dim] {elap}"
    if note:
        summary += f"\n  [dim]note:[/dim] {note}"
    hosts = getattr(st, "exec_hosts", None) or []
    if hosts:
        shown = " ".join(hosts[:8])
        extra_n = f" ...(+{len(hosts) - 8} more)" if len(hosts) > 8 else ""
        summary += f"\n  [dim]nodes:[/dim] {shown}{extra_n}"
    if out:
        summary += f"\n  [dim]stdout: {out}[/dim]"
        short = st.job_id.split(".")[0]
        summary += f"\n  [dim]→ ezpz fetch {short} --remote {host_hint or 'aurora'}[/dim]"
    if err and err != out:
        summary += f"\n  [dim]stderr: {err}[/dim]"
    c.print(summary)


@click.command()
@click.argument("job_id", required=False)
@click.option(
    "--remote",
    default=None,
    metavar="HOST",
    help="Host to query (SSH name or ALCF resource). Omit for local qstat.",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "ssh", "iri"]),
    help="Remote backend with --remote (default: auto).",
)
@click.option(
    "--historical/--no-historical",
    default=False,
    help="Include completed/failed/canceled jobs (IRI only).",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Max jobs when listing (IRI only, default 20).",
)
@click.option("--queue", default=None, help="Filter by queue (IRI list).")
@click.option(
    "--owner",
    default=None,
    help="Filter by owner (IRI list). Use --all-users to see everyone's jobs.",
)
@click.option(
    "--all-users",
    is_flag=True,
    default=False,
    help="Include all users' jobs (default: only yours).",
)
@click.option(
    "--states",
    default=None,
    help="Filter by IRI states, comma-separated (e.g. queued,active).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit JSON instead of table.",
)
@click.option(
    "--watch",
    is_flag=True,
    default=False,
    help="Poll continuously and notify on Q->R / R->F (requires --remote).",
)
@click.option(
    "--interval",
    type=float,
    default=10,
    show_default=True,
    help="Poll interval in seconds (with --watch).",
)
@click.option(
    "--once",
    is_flag=True,
    default=False,
    help="Single poll and exit (alias for non-watch).",
)
@click.option(
    "--no-notify",
    is_flag=True,
    default=False,
    help="Disable desktop notifications.",
)
def status_cmd(
    job_id,
    remote,
    backend,
    historical,
    limit,
    queue,
    owner,
    states,
    all_users,
    as_json,
    watch,
    interval,
    once,
    no_notify,
):
    """Show job status (local or remote)."""
    if once and not watch:
        # --once without --watch: one-shot alias (still requires --remote like --watch)
        eff_remote = remote
        eff_jid = job_id
        if not eff_remote and eff_jid:
            from ezpz.remote import looks_like_host as _lh_once

            if _lh_once(eff_jid):
                eff_remote, eff_jid = eff_jid, None
        if eff_remote:
            if owner is None and not all_users:
                from ezpz.remote import resolve_remote_owner as _rro2

                owner = _rro2(eff_remote) or owner
            from ezpz.cli.watch_cmd import _poll as _once_poll

            sys.exit(
                _once_poll(
                    {
                        "host": eff_remote,
                        "historical": historical,
                        "limit": limit,
                        "owner": owner,
                        "no_notify": True,
                    },
                    interval,
                    once=True,
                )
            )
        # no remote → fall through to normal status (local/fanout) single-shot
    if watch:
        if not remote:
            # resolve positional host sugar first (same as below)
            if job_id:
                from ezpz.remote import looks_like_host as _looks_host2

                if _looks_host2(job_id):
                    remote = job_id
                    job_id = None
        if not remote:
            click.echo(
                "status --watch requires --remote HOST (or positional host)",
                err=True,
            )
            sys.exit(2)
        if owner is None and not all_users:
            from ezpz.remote import resolve_remote_owner as _rro

            owner = _rro(remote) or owner
        from ezpz.cli.watch_cmd import _poll as _watch_poll

        sys.exit(
            _watch_poll(
                {
                    "host": remote,
                    "historical": historical,
                    "limit": limit,
                    "owner": owner,
                    "no_notify": no_notify,
                },
                interval,
                once=once or False,
            )
        )
    # Positional sugar: ezpz status aurora  ≡  --remote aurora
    if not remote and job_id:
        from ezpz.remote import looks_like_host as _looks_host

        if _looks_host(job_id):
            remote = job_id
            job_id = None
    if not remote:
        # Fan-out: no job_id and no remote → query every reachable machine
        if job_id is None:
            _status_fanout(
                backend=backend,
                historical=historical,
                limit=limit,
                queue=queue,
                owner=owner,
                states=states,
                all_users=all_users,
                as_json=as_json,
            )
            return
        _status_local(job_id)
        return
    # backend=auto: prefer SSH bulk JSON (full qmert fidelity) with IRI fallback
    from ezpz.remote import RemoteSubmitError

    try:
        if job_id:
            _status_single_remote(
                remote,
                job_id,
                backend=backend,
                historical=historical,
                as_json=as_json,
            )
        else:
            if owner is None and not all_users:
                # Prefer the SSH-config User for this host (e.g. foremans on aurora
                # even when the local mbph login is sam). Same default qmert uses:
                # $USER on the remote — resolved via ssh -G.
                from ezpz.remote import resolve_remote_owner

                owner = resolve_remote_owner(remote) or owner
            _status_list_remote(
                remote,
                backend=backend,
                historical=historical,
                limit=limit,
                queue=queue,
                owner=owner,
                states=states,
                as_json=as_json,
            )
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except (TimeoutError, OSError) as exc:
        click.echo(
            f"error: timed out after 30s — try without --historical or filter by --queue/--states: {exc}",
            err=True,
        )
        sys.exit(1)


def _status_fanout(
    *, backend, historical, limit, queue, owner, states, all_users, as_json
):
    """Fan out status to every reachable machine and merge results."""
    import concurrent.futures
    from ezpz.remote import IRI_MACHINES, REMOTE_SCHEDULERS

    # All known hosts: IRI machines + every REMOTE_SCHEDULERS entry (dedup)
    hosts = sorted(set(REMOTE_SCHEDULERS) | set(IRI_MACHINES))

    # When no explicit --owner, resolve per-host via ssh -G so
    # "ezpz status" (no --remote) doesn't hide aurora jobs owned by
    # "foremans" when local login is "sam". IRI + SSH text table both
    # filter by owner server/client-side, so the wrong name = no rows.
    _explicit_owner = owner
    _per_host_owner: dict[str, str | None] = {}
    if _explicit_owner is None and not all_users:
        from ezpz.remote import resolve_remote_owner

        for h in hosts:
            _per_host_owner[h] = resolve_remote_owner(h)

    def _fetch_one(h: str):
        eff_owner = (
            _explicit_owner
            if _explicit_owner is not None
            else _per_host_owner.get(h)
        )
        # Try SSH bulk JSON first (qmert path) to get full PBS fields
        # (nodes/walltime/note/nodelist) — same as single-host. Falls to
        # IRI only for non-PBS hosts like perlmutter/sirius.
        try:
            from ezpz.remote import list_jobs_via_ssh

            jobs = list_jobs_via_ssh(
                h, historical=historical, limit=limit, owner=eff_owner
            )
            # Apply --queue/--states post-fetch (PBS qselect only does -u/-x).
            if queue or states:
                _want_q = queue.lower() if queue else None
                _want_s = (
                    {s.strip().lower() for s in states.split(",") if s.strip()}
                    if states
                    else None
                )
                if _want_q:
                    jobs = [
                        j for j in jobs if (j.queue or "").lower() == _want_q
                    ]
                if _want_s:
                    jobs = [j for j in jobs if j.state.lower() in _want_s]
            if limit:
                jobs = jobs[:limit]
            for j in jobs:
                j.raw = j.raw or {}
                j.raw["_fanout_host"] = h
            if jobs:
                return (h, jobs, None)
            # No jobs via SSH (e.g. host uses different PBS path) — fall through
            # to IRI / text fallback below.
        except Exception:
            pass
        try:
            from ezpz.remote import has_iri_backend, list_jobs_via_iri

            if has_iri_backend(h):
                jobs = list_jobs_via_iri(
                    h,
                    historical=historical,
                    limit=limit,
                    queue=queue,
                    owner=eff_owner,
                    states=state_list(states) if states else None,
                )
                for j in jobs:
                    j.raw = j.raw or {}
                    j.raw["_fanout_host"] = h
                return (h, jobs, None)
        except Exception:
            # Fall through to SSH text fallback
            pass
        try:
            import subprocess

            qcmd = (
                "/opt/pbs/bin/qstat -x" if historical else "/opt/pbs/bin/qstat"
            )
            argv = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                h,
                qcmd,
            ]
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=15
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return (
                    h,
                    [],
                    f"unreachable ({proc.returncode})"
                    if proc.returncode not in (0, None)
                    else "no jobs",
                )
            from ezpz.remote import RemoteJobStatus as _RJS

            jobs = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if (
                    not line
                    or line.startswith("Job id")
                    or line.startswith("----")
                ):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                if eff_owner and len(parts) >= 3 and parts[2] != eff_owner:
                    continue
                jid = parts[0].rstrip("*")
                if "." in jid:
                    jid = jid.split(".")[0]
                q = parts[-1] if len(parts) >= 2 else ""
                st = parts[-2] if len(parts) >= 3 else ""
                nm = parts[1] if len(parts) >= 2 else ""
                usr = parts[2] if len(parts) >= 3 else ""
                if eff_owner and usr != eff_owner:
                    continue
                jobs.append(
                    _RJS(
                        job_id=jid,
                        state=st,
                        queue=q,
                        owner=usr,
                        raw={"name": nm, "_fanout_host": h},
                    )
                )
            return (h, jobs, None)
        except Exception as e:
            return (h, [], str(e)[:80])

    # Fan out concurrently (short timeout per host; unreachable hosts are skipped, not fatal)
    all_jobs: list = []
    warnings: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(hosts))
    ) as pool:
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
    order = {
        "active": 0,
        "running": 0,
        "queued": 1,
        "held": 2,
        "new": 3,
        "completed": 4,
        "failed": 5,
        "canceled": 6,
    }
    all_jobs.sort(key=lambda j: (order.get(j.state.lower(), 9), j.job_id))

    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": j.job_id,
                        "state": j.state,
                        "queue": j.queue,
                        "owner": getattr(j, "owner", None),
                        "exit_code": j.exit_code,
                        "host": (j.raw or {}).get("_fanout_host", ""),
                    }
                    for j in all_jobs
                ],
                indent=2,
            )
        )
        return
    if sys.stdout.isatty():
        # Fan-out rich table: reuse the same qmert columns as single-host,
        # plus a leading Host column so cross-machine output is disambiguated.
        from rich.console import Console
        from rich.table import Table as _FanTable

        # Build a rich table mirroring _rich_table's 8-column layout, prepending Host.
        # Use _rich_table's helpers: _short_wall + state coloring.
        _HEAD8 = [
            "JOBID",
            "S",
            "QUEUE",
            "NODES",
            "WALL",
            "ELAP",
            "JOBNAME",
            "NOTE",
        ]
        # Re-materialize display rows exactly like _rich_table, then add Host.
        smap_fan = {
            "active": "R",
            "running": "R",
            "queued": "Q",
            "held": "H",
            "completed": "F",
            "failed": "F",
            "canceled": "F",
            "new": "Q",
        }
        import shutil as _shutil

        TWf = (
            _shutil.get_terminal_size(fallback=(0, 24)).columns
            if hasattr(_shutil, "get_terminal_size")
            else 0
        )
        try:
            import os as _osf

            if _osf.environ.get("COLUMNS", "").isdigit():
                TWf = int(_osf.environ["COLUMNS"])
        except Exception:
            pass
        has_note_fan = any(
            (getattr(j, "note", None) or "").strip() for j in all_jobs
        )
        ncols_fan = 8 if has_note_fan else 7
        rows_fan: list[list[str]] = []
        for j in all_jobs:
            st_raw = j.state.upper()[0] if j.state else "?"
            s_disp = smap_fan.get(j.state.lower(), st_raw)
            short = j.job_id.split(".")[0]
            wall = _short_wall(getattr(j, "walltime", None))
            elap_val = getattr(j, "elapsed", None)
            if elap_val:
                elap = _short_wall(elap_val)
            elif j.state.lower() in ("queued", "held", "new"):
                elap = "--"
            else:
                elap = "—"
            nodes = str(getattr(j, "nodes", None) or "—")
            note = (getattr(j, "note", None) or "").strip()
            if note.lower().startswith("insufficient"):
                note = note.split(":")[-1].strip() if ":" in note else note
            row = [
                short,
                s_disp,
                j.queue or "—",
                nodes,
                wall,
                elap,
                getattr(j, "name", None) or "—",
                note,
            ]
            rows_fan.append(row[:ncols_fan])

        # Width-aware NOTE collapse (same as _rich_table).
        def _tbl_w(ws: list[int]) -> int:
            return sum(ws) + 3 * len(ws) + 1

        def _widths_for_fan(rws: list[list[str]], n: int) -> list[int]:
            hdrs = _HEAD8[:n]
            return [
                max(len(str(r[i])) for r in ([hdrs] + rws)) for i in range(n)
            ]

        widths_fan = _widths_for_fan(rows_fan, ncols_fan)
        dropped_fan: dict[str, str] = {}
        if TWf and ncols_fan == 8 and _tbl_w(widths_fan) > TWf:
            for r in rows_fan:
                if r[7]:
                    dropped_fan[r[0]] = r[7]
            ncols_fan = 7
            rows_fan = [r[:7] for r in rows_fan]
            widths_fan = _widths_for_fan(rows_fan, ncols_fan)
        if TWf and _tbl_w(widths_fan) > TWf:
            over = _tbl_w(widths_fan) - TWf
            neww = max(8, widths_fan[6] - over)
            if neww < widths_fan[6]:
                widths_fan[6] = neww
                for r in rows_fan:
                    if len(r[6]) > neww:
                        r[6] = r[6][: neww - 1] + "…"
        # Adjust for Host column width in TWf check.
        # Add 1 column + padding: re-check and re-collapse if needed.
        # Use actual max host width from data.
        max_host_w = (
            max(
                len((j.raw or {}).get("_fanout_host", "") or "")
                for j in all_jobs
            )
            if all_jobs
            else 4
        )
        max_host_w = max(max_host_w, 4)  # "Host"
        # If still over, the Host column is non-negotiable — truncate NOTE further.
        # For fanout we keep it simple: render table with Host prepended and
        # let Rich handle overflow; width tuning is best-effort.

        table = _FanTable(
            show_header=True, header_style="bold", box=None, padding=(0, 1)
        )
        table.add_column("Host", style="bold cyan", no_wrap=True)
        table.add_column("JOBID", style="dim", no_wrap=True)
        table.add_column("S", justify="center")
        table.add_column("QUEUE", no_wrap=True)
        table.add_column("NODES", justify="right")
        table.add_column("WALL", justify="right")
        table.add_column("ELAP", justify="right")
        table.add_column(
            "JOBNAME", overflow="fold", max_width=widths_fan[6] if TWf else 32
        )
        if ncols_fan == 8:
            table.add_column(
                "NOTE", overflow="fold", max_width=widths_fan[7] if TWf else 40
            )
        for j, rd in zip(all_jobs, rows_fan):
            color = _STATE_COLORS.get(j.state.lower(), "")
            s_styled = f"[{color}]{rd[1]}[/{color}]" if color else rd[1]
            host = (
                (j.raw or {}).get("_fanout_host", "")
                if isinstance(j.raw, dict)
                else ""
            )
            styled = [host, rd[0], s_styled, rd[2], rd[3], rd[4], rd[5], rd[6]]
            if ncols_fan == 8:
                styled.append(rd[7])
            table.add_row(*styled)
        table.box = __import__("rich.box", fromlist=["ROUNDED"]).ROUNDED
        Console().print(table)
        # Sub-lines: dropped notes + running nodelist (mirrors _rich_table / qmert).
        c2 = Console()
        if dropped_fan:
            c2.print("")
            c2.print("[dim]notes:[/dim]")
            for rd in rows_fan:
                jid = rd[0]
                if jid in dropped_fan:
                    j0 = next(
                        (
                            jj
                            for jj in all_jobs
                            if jj.job_id.split(".")[0] == jid
                        ),
                        None,
                    )
                    col2 = (
                        _STATE_COLORS.get(j0.state.lower(), "") if j0 else ""
                    )
                    tag = f"[{col2}]{jid}[/{col2}]" if col2 else jid
                    c2.print(f"  {tag} [dim]{dropped_fan[jid]}[/dim]")
        # Running nodelist: group by job, respecting TWf.
        running_fan = [
            j
            for j in all_jobs
            if j.state.lower() in ("active", "running", "r")
        ]
        if running_fan:
            c2.print("")
            c2.print("[dim]running job nodes:[/dim]")

            def _fit(items: list[str], prefix_len: int, suffix_tmpl: str):
                if TWf <= 0:
                    return items[:8], max(0, len(items) - 8)
                avail = TWf - prefix_len
                shown, used = [], 0
                for it in items:
                    add = len(it) + (1 if shown else 0)
                    if used + add > avail - len(suffix_tmpl.format(0)):
                        break
                    shown.append(it)
                    used += add
                return shown, len(items) - len(shown)

            rc_fan = (
                _STATE_COLORS.get("active")
                or _STATE_COLORS.get("running")
                or ""
            )
            for j in running_fan:
                hosts = getattr(j, "exec_hosts", None) or []
                if not hosts:
                    continue
                jid = j.job_id.split(".")[0]
                plain_prefix = f"  {jid} ({len(hosts)}): "
                shown, remaining = _fit(
                    hosts, len(plain_prefix), " ...(+{} more)"
                )
                line = " ".join(shown)
                if remaining:
                    line += f" [dim]...(+{remaining} more)[/dim]"
                rc_tag = f"[{rc_fan}]{jid}[/{rc_fan}]" if rc_fan else jid
                c2.print(f"  {rc_tag} [dim]({len(hosts)})[/dim]: {line}")
        if warnings and not as_json:
            for w in warnings[:3]:
                if sys.stderr.isatty():
                    from rich.console import Console

                    Console(stderr=True).print(
                        f"[[yellow]⚠️ warn[/yellow]] {w}"
                    )
                else:
                    click.echo(f"[warn] {w}", err=True)
    else:
        # Piped fanout: same full qmert columns as single-host, plus Host prefix.
        smap_p = {
            "active": "R",
            "running": "R",
            "queued": "Q",
            "held": "H",
            "failed": "F",
            "completed": "F",
            "canceled": "F",
            "new": "Q",
        }
        click.echo(
            f"{'HOST':<10} {'JOBID':<12} {'S':<1} {'QUEUE':<14} {'NODES':>5} {'WALL':>5} {'ELAP':>5} {'JOBNAME':<28} {'NOTE'}"
        )
        for j in all_jobs:
            s = smap_p.get(
                j.state.lower(), j.state[:1].upper() if j.state else "?"
            )
            nodes = str(getattr(j, "nodes", None) or "—")
            wall = _short_wall(getattr(j, "walltime", None))
            elap = (
                _short_wall(getattr(j, "elapsed", None))
                if getattr(j, "elapsed", None)
                else (
                    "--"
                    if j.state.lower() in ("queued", "held", "new")
                    else "—"
                )
            )
            host = (
                (j.raw or {}).get("_fanout_host", "")
                if isinstance(j.raw, dict)
                else ""
            )
            click.echo(
                f"{host:<10} {j.job_id.split('.')[0]:<12} {s:<1} {(j.queue or '—'):<14} {nodes:>5} {wall:>5} {elap:>5} {(getattr(j, 'name', None) or '—'):<28} {(getattr(j, 'note', None) or '')}"
            )


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
        proc2 = subprocess.run(
            ["qstat", job_id], capture_output=True, text=True
        )
        if proc2.returncode == 0:
            proc = proc2
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)


def _status_single_remote(host, job_id, backend, historical, as_json):
    if backend == "iri":
        from ezpz.remote import get_job_status_via_iri

        st = get_job_status_via_iri(host, job_id, historical=historical)
        _render_single(st, as_json=as_json, via="iri")
        return
    if backend == "ssh":
        _status_via_ssh(host, job_id)
        return
    # auto: SSH-json first (full job name/nodes/wall/node-list), IRI on transport failure only
    try:
        _status_via_ssh_json(
            host, job_id, historical=historical, as_json=as_json
        )
        return
    except Exception as exc:
        if "255" in str(exc):
            pass  # SSH unreachable → IRI fallback (mirrors submit fallback rule)
        else:
            # Real scheduler error (e.g. unknown job id) — surface it, don't mask with IRI.
            raise
    from ezpz.remote import get_job_status_via_iri

    st = get_job_status_via_iri(host, job_id, historical=historical)
    _render_single(st, as_json=as_json, via="iri")


def _render_single(st, *, as_json: bool, via: str = ""):
    if as_json:
        click.echo(
            json.dumps(
                {
                    "id": st.job_id,
                    "state": st.state,
                    "exit_code": st.exit_code,
                    "queue": st.queue,
                    "account": st.account,
                    "stdout_path": st.stdout_path,
                    "stderr_path": st.stderr_path,
                    "nodes": getattr(st, "nodes", None),
                    "walltime": getattr(st, "walltime", None),
                    "elapsed": getattr(st, "elapsed", None),
                    "note": getattr(st, "note", None),
                    "name": getattr(st, "name", None),
                    "exec_hosts": getattr(st, "exec_hosts", None),
                }
            )
        )
    elif sys.stdout.isatty():
        _rich_single(st, via=via)
    else:
        ec = st.exit_code if st.exit_code is not None else "-"
        line = f"{st.job_id}  {st.state}  queue={st.queue or '-'}  exit={ec}"
        if via:
            line += f"  (via {via})"
        # Also emit rich fields in plain mode so grep/awk still sees them.
        if getattr(st, "name", None):
            line += f"\n  name: {st.name}"
        if getattr(st, "nodes", None) is not None:
            line += f"  nodes: {st.nodes}"
        if getattr(st, "walltime", None):
            line += f"  wall: {st.walltime}"
        if getattr(st, "elapsed", None):
            line += f"  elapsed: {st.elapsed}"
        if (getattr(st, "note", None) or "").strip():
            line += f"\n  note: {st.note.strip()}"
        _hosts = getattr(st, "exec_hosts", None) or []
        if _hosts:
            _shown = " ".join(_hosts[:6])
            _suf = f" ...(+{len(_hosts) - 6} more)" if len(_hosts) > 6 else ""
            line += f"\n  exec_hosts ({len(_hosts)}): {_shown}{_suf}"
        if st.stdout_path:
            line += f"\n  stdout: {st.stdout_path}"
        if st.stderr_path and st.stderr_path != st.stdout_path:
            line += f"\n  stderr: {st.stderr_path}"
        click.echo(line)


def _status_list_remote(
    host, backend, historical, limit, queue, owner, states, as_json
):
    # Explicit backends honour the caller's choice.
    if backend == "iri":
        from ezpz.remote import list_jobs_via_iri

        st_list = state_list(states) if states else None
        jobs = list_jobs_via_iri(
            host,
            historical=historical,
            limit=limit,
            queue=queue,
            owner=owner,
            states=st_list,
        )
        _render_list(jobs, host=host, as_json=as_json)
        return
    if backend == "ssh":
        _status_via_ssh(host, None, owner=owner)
        return
    # auto: SSH bulk JSON first (qmert path — full columns in one hop, no per-job -xf fanout).
    # Falls back to IRI only on SSH transport failure (255).
    ssh_exc: Exception | None = None
    try:
        jobs = _status_via_ssh_json_list(
            host,
            historical=historical,
            limit=limit,
            owner=owner,
            as_json=None,
        )
        jobs = _apply_list_filters(
            jobs, queue=queue, states=states, limit=limit
        )
        _render_list(jobs, host=host, as_json=as_json)
        return
    except Exception as e:
        if "255" not in str(e):
            # Non-transport error shouldn't silently drop to IRI on auto.
            raise
        ssh_exc = e
    # SSH unreachable → try IRI (the narrow fallback from submit semantics).
    assert ssh_exc is not None
    from ezpz.remote import has_iri_backend, list_jobs_via_iri

    if has_iri_backend(host):
        try:
            st_list = state_list(states) if states else None
            jobs = list_jobs_via_iri(
                host,
                historical=historical,
                limit=limit,
                queue=queue,
                owner=owner,
                states=st_list,
            )
            _render_list(jobs, host=host, as_json=as_json)
            return
        except Exception:
            # IRI also failed — surface the original SSH transport note with context.
            click.echo(
                f"note: IRI list failed ({ssh_exc}), giving up", err=True
            )
            raise
    # No IRI backend for this host — propagate the transport error.
    raise ssh_exc


def _render_list(jobs, *, host: str | None = None, as_json: bool = False):
    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": j.job_id,
                        "state": j.state,
                        "exit_code": j.exit_code,
                        "queue": j.queue,
                        "nodes": getattr(j, "nodes", None),
                        "walltime": getattr(j, "walltime", None),
                        "elapsed": getattr(j, "elapsed", None),
                        "note": getattr(j, "note", None),
                        "name": getattr(j, "name", None),
                        "exec_hosts": getattr(j, "exec_hosts", None),
                    }
                    for j in jobs
                ],
                indent=2,
            )
        )
    elif sys.stdout.isatty():
        _rich_table(jobs, host=host)
    else:
        if not jobs:
            click.echo("(no jobs)")
            return
        # Piped/plain output: same columns as the rich table, tab-friendly.
        click.echo(
            f"{'JOBID':<12} {'S':<1} {'QUEUE':<14} {'NODES':>5} {'WALL':>5} {'ELAP':>5} {'JOBNAME':<28} {'NOTE'}"
        )
        for j in jobs:
            smap2 = {
                "active": "R",
                "running": "R",
                "queued": "Q",
                "held": "H",
                "failed": "F",
                "completed": "F",
                "canceled": "F",
                "new": "Q",
            }
            s2 = smap2.get(
                j.state.lower(), j.state[:1].upper() if j.state else "?"
            )
            nodes2 = str(getattr(j, "nodes", None) or "—")
            wall2 = _short_wall(getattr(j, "walltime", None))
            elap2 = (
                _short_wall(getattr(j, "elapsed", None))
                if getattr(j, "elapsed", None)
                else (
                    "--"
                    if j.state.lower() in ("queued", "held", "new")
                    else "—"
                )
            )
            click.echo(
                f"{j.job_id.split('.')[0]:<12} {s2:<1} {(j.queue or '—'):<14} {nodes2:>5} {wall2:>5} {elap2:>5} {(getattr(j, 'name', None) or '—'):<28} {(getattr(j, 'note', None) or '')}"
            )


def _status_via_ssh_json(host, job_id, *, historical: bool, as_json: bool):
    from ezpz.remote import get_job_status_via_ssh

    st = get_job_status_via_ssh(host, job_id, historical=historical)
    _render_single(st, as_json=as_json, via="ssh")


def _apply_list_filters(jobs, *, queue, states, limit):
    """Apply --queue / --states / --limit post-fetch for SSH path."""
    if queue:
        qlow = queue.lower()
        jobs = [j for j in jobs if (j.queue or "").lower() == qlow]
    if states:
        wanted = {s.strip().lower() for s in states.split(",") if s.strip()}
        if wanted:
            jobs = [j for j in jobs if j.state.lower() in wanted]
    if limit is not None and limit > 0:
        jobs = jobs[:limit]
    return jobs


def _status_via_ssh_json_list(host, *, historical, limit, owner, as_json):
    from ezpz.remote import list_jobs_via_ssh

    jobs = list_jobs_via_ssh(
        host, historical=historical, limit=limit, owner=owner
    )
    if as_json is None:
        return jobs
    _render_list(jobs, host=host, as_json=as_json)
    return jobs


def _status_via_ssh(host, job_id, owner=None):  # noqa: ARG001 — owner kept for compat; JSON path uses it, text path ignores
    import subprocess

    cmd = (
        f"/opt/pbs/bin/qstat -x {job_id}"
        if job_id
        else "/opt/pbs/bin/qstat -x"
    )
    argv = ["ssh", "-o", "BatchMode=yes", host, cmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    # If not TTY (piped), keep raw qstat for grep/awk compat; rich only on TTY
    if (
        not sys.stdout.isatty()
        or proc.returncode != 0
        or not proc.stdout.strip()
    ):
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
        jobs.append(
            _RJS(
                job_id=jid,
                state=state,
                queue=queue,
                raw={"name": name},
                owner=user,
            )
        )
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
            argv = [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                f"/opt/pbs/bin/qstat -xf {j.job_id}",
            ]
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=8
            )
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
                    note = note[len("Not Running:") :].strip()
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
@click.option(
    "--remote",
    default=None,
    metavar="HOST",
    help="Host to cancel on (SSH name or ALCF resource). Omit for local qdel. Also positional: ezpz cancel aurora JOBID.",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "ssh", "iri"]),
    help="Remote backend with --remote (default: auto).",
)
@click.option(
    "--scheduler",
    default=None,
    type=click.Choice(["PBS", "SLURM"]),
    help="Override scheduler auto-detection.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would run without cancelling.",
)
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
        click.echo(
            f"would cancel {job_id} on {remote} via {backend} (scheduler={scheduler or 'auto'})"
        )
        return
    try:
        result = cancel_remote(
            remote, job_id, backend=backend, scheduler=scheduler
        )
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    if result.returncode == 0:
        click.echo(f"cancelled {job_id} (via {result.backend})")
        return
    click.echo(
        f"error: cancel {job_id} failed via {result.backend} (exit {result.returncode}): {result.stderr}",
        err=True,
    )
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
