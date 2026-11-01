"""ezpz status --watch / ezpz watch — poll and notify on transitions."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time

import click

from ezpz.cli.status_cancel_cmd import _rich_table


def _notify(title: str, body: str) -> None:
    """Best-effort desktop notification; never raises."""
    try:
        sysname = platform.system()
        if sysname == "Darwin" and shutil.which("terminal-notifier"):
            subprocess.run(
                ["terminal-notifier", "-title", title, "-message", body],
                timeout=5,
                capture_output=True,
            )
            return
        if shutil.which("notify-send"):
            subprocess.run(
                ["notify-send", title, body], timeout=5, capture_output=True
            )
            return
    except Exception:
        pass
    # Fallback: audible bell
    try:
        click.echo("\a", nl=False)
    except Exception:
        pass


def _poll(loop_kwargs: dict, interval: float, once: bool = False) -> int:
    """Poll loop — factored so both `status --watch` and `watch` share it."""
    from ezpz.remote import list_jobs_via_ssh

    host = loop_kwargs.get("host")
    if not host:
        click.echo("status --watch requires --remote HOST", err=True)
        return 2
    # initial snapshot for diffing — empty map means all jobs are "new"
    prev: dict[tuple[str, str], str] = {}  # (host, job_id) -> state
    first = True
    try:
        while True:
            try:
                jobs = list_jobs_via_ssh(
                    host,
                    **{
                        k: v
                        for k, v in loop_kwargs.items()
                        if k in ("historical", "limit", "owner")
                    },
                )
            except Exception as exc:
                click.echo(f"[watch] poll failed: {exc}", err=True)
                jobs = []
            # Print table each tick
            if jobs:
                _rich_table(jobs, host=host)
            else:
                click.echo(
                    f"[watch] no jobs on {host} (polling every {interval:.0f}s) — Ctrl-C to stop"
                )
            # Diff by (host, job_id)
            cur: dict[tuple[str, str], str] = {
                (host, j.job_id): j.state for j in jobs
            }
            if not first:
                for key, state in cur.items():
                    old = prev.get(key)
                    if old is None:
                        continue
                    ol = old.lower()
                    nl = state.lower()
                    # Q->R and R->F
                    q_states = {"queued", "held", "new", "q", "h"}
                    r_states = {"active", "running", "r"}
                    f_states = {
                        "completed",
                        "failed",
                        "canceled",
                        "exiting",
                        "f",
                        "e",
                    }
                    jid = key[1].split(".")[0]
                    if ol in q_states and nl in r_states:
                        msg = f"{jid} {old} -> {state} on {host}"
                        click.echo(f"[watch] {msg}")
                        if not loop_kwargs.get("no_notify"):
                            _notify("ezpz watch", msg)
                    elif ol in r_states and nl in f_states:
                        msg = f"{jid} {old} -> {state} on {host}"
                        click.echo(f"[watch] {msg}")
                        if not loop_kwargs.get("no_notify"):
                            _notify("ezpz watch", msg)
                # Detect finished (disappeared) — treat as F
                for key, old in prev.items():
                    if key not in cur and old.lower() in {
                        "active",
                        "running",
                        "r",
                    }:
                        jid = key[1].split(".")[0]
                        msg = f"{jid} {old} -> finished (left queue) on {host}"
                        click.echo(f"[watch] {msg}")
                        if not loop_kwargs.get("no_notify"):
                            _notify("ezpz watch", msg)
            prev = cur
            first = False
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\n[watch] stopped.")
    return 0


@click.command(name="watch")
@click.option(
    "--remote",
    "host",
    required=True,
    metavar="HOST",
    help="Host to watch (SSH alias or IRI resource).",
)
@click.option(
    "--interval",
    type=float,
    default=30,
    show_default=True,
    help="Poll interval in seconds.",
)
@click.option("--historical/--no-historical", default=False)
@click.option("--limit", type=int, default=20)
@click.option("--owner", default=None)
def watch_cmd(host, interval, historical, limit, owner):
    """Poll job status and notify on Q->R / R->F transitions."""
    # Resolve owner via ssh -G default (same qmert convention as status)
    if owner is None:
        from ezpz.remote import resolve_remote_owner

        owner = resolve_remote_owner(host)
    sys.exit(
        _poll(
            {
                "host": host,
                "historical": historical,
                "limit": limit,
                "owner": owner,
            },
            interval,
        )
    )


# Hook for `ezpz status --watch` — added by monkey-patching status_cmd params
# See cli/__init__.py wiring below.


def status_watch_poll(**kw):
    """Entry point used by `status --watch` path."""
    return _poll(kw, kw.pop("interval", 30))


# ── test-friendly aliases ──
notify = _notify


def _diff_jobs(prev, cur):
    """Test helper: (prev_map, cur_job_list) -> (msgs, cur_map)."""
    cur_map = {
        (
            j.raw.get("_fanout_host", "")
            if hasattr(j, "raw") and j.raw
            else "",
            j.job_id,
        ): j.state
        for j in cur
    }
    # reuse _poll diff logic in miniature: caller checks Q->R/R->F like _poll does
    msgs = []
    for k, st in cur_map.items():
        old = prev.get(k)
        if old is None:
            msgs.append(f"new: {k}")
            continue
        ol, nl = old.lower(), st.lower()
        if ol in {"queued", "held", "new", "q", "h"} and nl in {
            "active",
            "running",
            "r",
        }:
            msgs.append(f"{k} {old}->{st}")
        elif ol in {"active", "running", "r"} and nl in {
            "completed",
            "failed",
            "canceled",
            "exiting",
            "f",
            "e",
        }:
            msgs.append(f"{k} {old}->{st}")
    # disappeared finished
    for k, old in prev.items():
        if k not in cur_map and old.lower() in {"active", "running", "r"}:
            msgs.append(f"{k} {old}->finished (left queue)")
    return msgs, cur_map


def _fetch_jobs(*a, **kw):
    from ezpz.remote import list_jobs_via_ssh as _lj

    return _lj(*a, **kw)
