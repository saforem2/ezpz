"""Extra remote commands: extend, resources, incidents."""
from __future__ import annotations
import json, shlex, sys, urllib.parse
import click

def _parse_walltime(s: str) -> int:
    s=s.strip()
    if ":" not in s:
        return int(s)
    parts=[int(p) for p in s.split(":")]
    if len(parts)==3:
        h,m,sec=parts
    elif len(parts)==2:
        h=0; m,sec=parts
    else:
        raise ValueError(f"bad walltime {s!r}")
    return h*3600+m*60+sec

@click.command("extend")
@click.argument("job_id")
@click.option("--remote", "-r", required=True, help="Target host (aurora/polaris/crux or SSH alias).")
@click.option("--time", "-t", "walltime", default=None, help="New walltime HH:MM:SS or seconds.")
@click.option("--queue", "-q", default=None, help="Move to queue/partition.")
@click.option("--account", "-A", default=None, help="Charge account.")
@click.option("--backend", type=click.Choice(["auto","iri","ssh"]), default="auto", show_default=True)
@click.option("--dry-run", is_flag=True, help="Print what would be sent without sending.")
def extend_cmd(job_id, remote, walltime, queue, account, backend, dry_run):
    """Extend or retarget a queued job (IRI PUT / SSH qalter)."""
    dur=_parse_walltime(walltime) if walltime else None
    if not any([dur, queue, account]):
        raise click.UsageError("nothing to update: pass --time / --queue / --account")
    if dry_run:
        click.echo(f"would PUT /api/v1/compute/job/{{rid}}/{job_id} {{duration:{dur}, queue:{queue}, account:{account}}} backend={backend}")
        return
    from ezpz.remote import update_remote, RemoteSubmitError
    try:
        st=update_remote(remote, job_id, backend=backend, duration_seconds=dur, queue=queue, account=account)
        click.echo(f"{st.job_id} {st.state} queue={st.queue} (via {st.raw.get(chr(34)+chr(118)+chr(105)+chr(97)+chr(34),chr(34)+chr(105)+chr(114)+chr(105)+chr(34)) if st.raw else chr(34)+chr(105)+chr(114)+chr(105)+chr(34)})" if False else f"{st.job_id} {st.state} queue={st.queue} (via {st.raw.get(chr(118)+chr(105)+chr(97),chr(105)+chr(114)+chr(105)) if st.raw else chr(105)+chr(114)+chr(105)})")
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

# alias
update_job_cmd = click.Command("update-job", callback=extend_cmd.callback, params=extend_cmd.params, help="Alias for extend.")

@click.command("resources")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--all/--no-all", "all_resources", is_flag=True, default=True, help="Use /status/resources (all) vs /compute/resources (501).")
@click.option("--limit", type=int, default=100, show_default=True)
def resources_cmd(as_json, all_resources, limit):
    """List IRI resources (Aurora/Polaris/Crux/...). Uses /status/resources by default."""
    import os
    from ezpz.remote import get_compute_resources, get_all_resources, RemoteSubmitError
    try:
        data=get_all_resources(limit=limit) if all_resources else get_compute_resources()
    except RemoteSubmitError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    # data is list[dict] with varying shapes
    norm=[]
    for item in (data if isinstance(data, list) else data.get("resources", data.get("data", []))):
        if not isinstance(item, dict): continue
        rid=str(item.get("id") or item.get("resource_id") or "")
        name=str(item.get("name") or rid).lower()
        st=item.get("current_status") or item.get("status") or item.get("state") or "unknown"
        if isinstance(st, dict): st=st.get("state") or st.get("status") or "unknown"
        st=str(st).lower()
        norm.append({"name":name,"id":rid,"status":st,"raw":item})
    if as_json:
        clean=[{k:v for k,v in d.items() if k!="raw"} for d in norm]
        if os.environ.get("ALCF_IRI_JSON_RAW"): clean=norm
        click.echo(json.dumps(clean, indent=2))
        return
    if not norm:
        click.echo("(no resources)")
        return
    if sys.stdout.isatty():
        from rich.console import Console
        from rich.table import Table
        tbl=Table(show_header=True, header_style="bold", box=None, padding=(0,1))
        tbl.add_column("Resource", style="bold", no_wrap=True)
        tbl.add_column("ID", style="dim", no_wrap=True)
        tbl.add_column("Status", justify="center")
        cmap={"up":"green","degraded":"yellow","down":"red","unknown":"dim","active":"green"}
        for r in sorted(norm, key=lambda x: x["name"]):
            c=cmap.get(r["status"], "")
            st=f"[{c}]{r["status"]}[/{c}]" if c else r["status"]
            short=r["id"][:8]+"…" if len(r["id"])>12 else r["id"]
            tbl.add_row(r["name"], short, st)
        Console().print(tbl)
        c2=Console()
        for r in sorted(norm, key=lambda x: x["name"]):
            c2.print(f"  [dim]{r["name"]}  {r["id"]}[/dim]")
    else:
        click.echo(f"{"RESOURCE":<14} {"ID":<40} STATUS")
        for r in sorted(norm, key=lambda x: x["name"]):
            click.echo(f"{r["name"]:<14} {r["id"]:<40} {r["status"]}")

@click.command("incidents")
@click.option("--resource", default=None, help="Resource filter (aurora/polaris or UUID).")
@click.option("--status", "status_", default=None, type=click.Choice(["up","down","degraded","unknown"]), help="Filter by status.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
@click.option("--limit", type=int, default=20, show_default=True)
def incidents_cmd(resource, status_, as_json, limit):
    """List IRI incidents (maintenance/outages). Public, no auth required."""
    from ezpz.remote import list_incidents as _li
    incidents=_li(resource_id=resource if resource and "-" in resource else None, status=status_, limit=limit)
    # If alias passed, resolve via resources
    if resource and "-" not in resource and not any(i.get("resource_id")==resource for i in incidents):
        pass  # best-effort filter already
    if as_json:
        click.echo(json.dumps(incidents, indent=2))
        return
    if not incidents:
        click.echo("(no incidents)")
        return
    # rich
    if sys.stdout.isatty():
        from rich.console import Console
        from rich.table import Table
        colmap={"down":"red","degraded":"yellow","up":"green","unknown":"dim"}
        t=Table(show_header=True, header_style="bold", box=None, padding=(0,1))
        t.add_column("Name", overflow="fold", max_width=40)
        t.add_column("Status", justify="center")
        t.add_column("Type")
        t.add_column("Time", no_wrap=True)
        for inc in incidents:
            st=(inc.get("status") or "").lower()
            c=colmap.get(st,"")
            cell=f"[{c}]{st}[/{c}]" if c else st
            ts=(inc.get("start") or inc.get("last_modified") or "")
            ts=ts.replace("T"," ")[:16] if ts else "—"
            t.add_row(inc.get("name") or inc.get("id","")[:24], cell, inc.get("type") or "—", ts)
        Console().print(t)
    else:
        click.echo(f"{"NAME":<40} {"STATUS":<10} {"TYPE":<12} TIME")
        for inc in incidents:
            ts=(inc.get("start") or "")[:16].replace("T"," ")
            click.echo(f"{inc.get(chr(110)+chr(97)+chr(109)+chr(101),)[:40]:<40} {inc.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115),):<10} {inc.get(chr(116)+chr(121)+chr(112)+chr(101),) or chr(8212):<12} {ts}")
