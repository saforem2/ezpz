"""Happy-path for ezpz status --watch (mocked SSH)."""

import importlib.util

spec = importlib.util.spec_from_file_location(
    "watch_mod", "src/ezpz/cli/watch_cmd.py"
)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

from ezpz.remote import RemoteJobStatus


def test_diff_q_to_r_and_r_to_f():
    j_q = RemoteJobStatus(
        job_id="1.aurora-pbs-0001",
        state="queued",
        raw={"_fanout_host": "aurora"},
    )
    j_r = RemoteJobStatus(
        job_id="1.aurora-pbs-0001",
        state="active",
        raw={"_fanout_host": "aurora"},
    )
    j_f = RemoteJobStatus(
        job_id="1.aurora-pbs-0001",
        state="completed",
        raw={"_fanout_host": "aurora"},
    )
    _, m = w._diff_jobs({}, [j_q])
    assert m
    msgs, m2 = w._diff_jobs(m, [j_r])
    assert any("->" in x for x in msgs)
    msgs2, _ = w._diff_jobs(m2, [j_f])
    assert any("->" in x for x in msgs2)


def test_poll_once_triggers_notify_callback(monkeypatch):
    q = RemoteJobStatus(
        job_id="10.aurora-pbs-0001",
        state="queued",
        queue="debug",
        raw={"_fanout_host": "aurora"},
    )
    r = RemoteJobStatus(
        job_id="10.aurora-pbs-0001",
        state="active",
        queue="debug",
        raw={"_fanout_host": "aurora"},
    )
    calls = {"n": 0}

    def fake_list(host, **kw):
        calls["n"] += 1
        return [q] if calls["n"] == 1 else [r]

    monkeypatch.setattr(w, "list_jobs_via_ssh", fake_list, raising=False)
    import ezpz.remote as rem

    monkeypatch.setattr(rem, "list_jobs_via_ssh", fake_list)
    notified = []
    monkeypatch.setattr(w, "_notify", lambda t, b: notified.append((t, b)))
    _, prev = w._diff_jobs({}, [q])
    fake_list("aurora")
    notified.clear()
    msgs, _ = w._diff_jobs(prev, [r])
    for m in msgs:
        w._notify("ezpz watch", m)
    assert notified
    assert any("10" in b for _, b in notified)
