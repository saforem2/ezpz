# Node-kill postmortem: what `--auto-retry` actually proved

A real 4-node Sunspot job, 13 ranks `kill -9`'d on one active node
mid-training, driven end-to-end by
[`ezpz launch --auto-retry`](../cli/launch/index.md#auto-retry-on-bad-node-failure-auto-retry).

The headline is honest but narrow: **the failover mechanism works — the
next attempt really did land on a different node set.** The identification
does not. This page walks the whole run from the captured logs, because
three of the four interesting findings only show up when you stop reading
the summary and start reading the evidence.

!!! info "Scope"
    This is the on-node counterpart to
    [Fault injection](fault-injection.md), which drives the same loop
    against a synthetic trainer on a laptop, and to
    [Checkpoint restart](checkpoint-restart.md), which measures what a
    restart *costs*. This page is about whether the loop's decisions are
    right when the hardware is real.

## The run

Sunspot job **12473704**, `experiments/fault-injection/autoretry_nodekill.pbs`:

| | |
|---|---|
| allocation | `select=4` → **2 active + 2 spare** (`--nhosts 2`) |
| workload | `ezpz.examples.fsdp_tp`, agpt-2b, 24 ranks (`world_size=24`) |
| schedule | `--train-iters 400 --save-interval 20` |
| fault | `pbsdsh -n 0` → `kill -9` on every `fsdp_tp` pid on one node (the receipt reported **13**), once a checkpoint past step 40 landed |

The killer deliberately skips any pid whose cmdline mentions `ezpz.launch`,
so the retry supervisor survives and the *ranks* are what die — an earlier
job killed the supervisor instead and produced one attempt, an empty
`bad_nodes.txt`, and nothing to fail over.

## The decision trace

Nine lines, start to finish. This is the whole story the loop tells about
itself:

```text
[17:28:14][I][ezpz/launch:768:_resolve_auto_retry_allocation] [auto-retry] 4 total / 2 active / 2 spare. Active hostfile: <expt>/logs/failover-12473704/active.hostfile
[17:28:14][I][ezpz/launch_autoretry:769:run_with_auto_retry] [auto-retry] attempt 1 — active=2 hosts, spare=2 hosts
[17:29:56][W][ezpz/launch_autoretry:863:run_with_auto_retry] [auto-retry] blind rotation: x1921c1s0b0n0-hsn0... -> x1921c1s4b0n0-hsn0...
[17:29:56][W][ezpz/launch_autoretry:761:run_with_auto_retry] [auto-retry] attempt 2 (prior rc=-9, sleeping 5s)...
[17:30:01][I][ezpz/launch_autoretry:769:run_with_auto_retry] [auto-retry] attempt 2 — active=2 hosts, spare=1 hosts
[17:33:53][W][ezpz/launch_autoretry:863:run_with_auto_retry] [auto-retry] blind rotation: x1921c1s4b0n0-hsn0... -> x1921c1s5b0n0-hsn0...
[17:33:53][W][ezpz/launch_autoretry:761:run_with_auto_retry] [auto-retry] attempt 3 (prior rc=143, sleeping 10s)...
[17:34:03][I][ezpz/launch_autoretry:769:run_with_auto_retry] [auto-retry] attempt 3 — active=2 hosts, spare=0 hosts
[17:35:35][E][ezpz/launch_autoretry:828:run_with_auto_retry] [auto-retry] FAILOVER STOP: exhausted (no spare nodes left, rc=143)
```

Read the word **`blind`** twice. That is finding #2, and everything below
is the explanation.

## What each attempt did

| | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|
| ran on | `s0b0n0`, `s1b0n0` | `s1b0n0`, **`s4b0n0`** | `s1b0n0`, **`s5b0n0`** |
| resumed from | — | step 40 | step 280 |
| last iter logged | 42 | 300 | 300 |
| last durable save | step 40 | step 280 | — |
| died of | 13 ranks `kill -9`'d | `OSError: [Errno 28]` mid-save | `[Errno 28]` + truncated-shard `RuntimeError` |
| shell rc | `-9` | `143` | `143` |
| scraper output | *(empty)* | *(empty)* | *(empty)* |
| verdict | `BAD_NODE_BLIND` | `BAD_NODE_BLIND` | `EXHAUSTED` |
| evicted | `s0b0n0` (= `active[0]`) | `s4b0n0` (= `active[0]`) | — |

Attempt 2 is the result worth having: it ran on a host set that did not
include the node whose ranks were killed, resumed from step 40, and trained
260 more iterations. **Node-swapping works on real hardware.** The rest of
this page is about how it got there.

## Finding 1 — a killed *process* leaves nothing to scrape

The last five lines of `attempt-1.log`:

```text
[17:29:51][I][ezpz/history:1765:log_metrics] iter=40 epoch=0 bidx=39 loss=14.167702(±0.11) dt=0.372866(±4.8…
[17:29:54][I][examples/_checkpoint:175:save_checkpoint] saved checkpoint: …/ckpt/step-40
[17:29:54][I][examples/fsdp_tp:3797:train] train/ckpt_save_seconds=2.9756 (sync save @ step 40)
[17:29:55][I][ezpz/history:1765:log_metrics] iter=41 epoch=0 bidx=40 loss=14.174600(±0.10) dt=0.377319(±3.7…
[17:29:55][I][ezpz/history:1765:log_metrics] iter=42 epoch=0 bidx=41 loss=13.925915(±0.074) dt=0.375328(±5.…
```

That is the **end of the file**. No `shepherd died from signal 9`, no
hostname-prefixed PALS line, no traceback. A `SIGKILL` gives the process no
chance to say anything, and PALS reported the aggregate without naming the
node. The log simply stops mid-training.

!!! warning "This is not a node death, and the distinction matters"

    The heading originally read "a hard *node* death leaves nothing to
    scrape". That was wrong, and it took asking PBS to notice:

    ```console
    $ pbsnodes x1921c1s0b0n0      # the node whose ranks were killed
    state = job-exclusive
    $ pbsnodes x1921c1s4b0n0      # a healthy node in the same job
    state = job-exclusive
    ```

    Identical. The node was never faulty — the experiment killed
    *processes*, and `palsd` on that host stayed up and answering. So
    PALS had nothing to report, and the silence is a property of the
    injected fault, not of node failure:

    | | `kill -9` the ranks | a real node loss |
    | --- | --- | --- |
    | processes | gone | gone |
    | `palsd` on that host | **up** | **down** |
    | PALS emits | *nothing* | `<host>: shepherd died from signal 9` |
    | `pbsnodes` | unchanged | eventually `down,offline` |

    That shepherd line is already the registered scrape pattern, so for
    a genuine node loss the named path may well work — this run simply
    never exercised it. The harness gained a `KILL_MODE=pals` that
    routes the kill through `palsig` so the shepherd observes the app
    dying, which is the only mode that can test named attribution
    ([#234](https://github.com/saforem2/ezpz/issues/234)).

Run the real scraper against it and you get the same answer:

```console
$ python3 -m ezpz.failover --machine sunspot attempt-1.log
$          # empty — exit 0, zero hosts named
```

For contrast, the scraper is not broken — feed it a signature it *does*
know and it names the host:

```console
$ printf 'x1921c1s4b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: shepherd died from signal 9\n' > /tmp/pos.log
$ python3 -m ezpz.failover --machine sunspot /tmp/pos.log
x1921c1s4b0n0.hsn.cm.sunspot.alcf.anl.gov
```

All three attempt logs scrape empty. Nothing was ever attributed to a host
in this entire run.

!!! note "Grepping these logs yourself"

    `attempt-*.log` is ANSI-colorized, and the escapes land *inside* the
    tokens (`\e[36miter\e[0m=42`), so `grep 'iter=42'` finds nothing.
    Strip first — which is what `classify_attempt` does before any
    matcher runs:

    ```bash
    sed -e 's/\x1b\[[0-9;]*m//g' attempt-1.log | grep -oE 'iter=[0-9]+' | tail -1
    # iter=42
    ```

## Finding 2 — so both swaps were guesses, and the test passed by luck

With the scraper empty, the classifier reaches the last row of the
[termination matrix](../cli/launch/index.md#termination-matrix) — *any other
non-zero, scraper found nothing* — and the loop calls `swap_one_blind`,
which always evicts `active[0]`:

```python
# src/ezpz/launch_autoretry.py — swap_one_blind
bad = self.active[0]
spare = self.spare.popleft()
self.active[0] = spare
```

`pbsdsh -n 0` targets the **first node in the allocation**, which is also
`active[0]`. The guess was right by construction of the test, not because
the loop knew anything.

!!! danger "Kill `active[1]` and today's code evicts a healthy node"

    Had the fault landed on the second active host, the blind rotation
    would have retired the *healthy* `active[0]` and left the dead node in
    the active set — and the next attempt would have tried to launch ranks
    on a node that has none.

    So this run proves **"the mechanism can swap a node out and relaunch
    elsewhere"** — real, and worth having — but **not** "the loop identifies
    which node died." Tracked in
    [#234](https://github.com/saforem2/ezpz/issues/234); the fix is for the
    experiment to kill a node the blind path would *not* pick (`pbsdsh -n 1`),
    which today's code would fail.

Attempt 2 makes the cost concrete: it evicted `s4b0n0`, a node that was
perfectly healthy and had just been rotated *in* one attempt earlier. Its
only offence was sitting at index 0. That is
[#233](https://github.com/saforem2/ezpz/issues/233) — the loop does not
record *why* a host was retired, so `bad_nodes.txt` reads the same whether
a host was named by the scraper or picked by position.

## Finding 3 — the scraper was right to stay silent

It is tempting to read "scraper found nothing" as a scraper bug. It is not.
Attempt 2's real failure ends like this:

```text
[rank23]:   File ".../torch/distributed/checkpoint/filesystem.py", line 422, in _write_files_from_queue
[rank23]:     with create_stream(file_name, "wb") as stream:
[rank23]:   File ".../torch/distributed/checkpoint/filesystem.py", line 520, in create_stream
[rank23]:     with path.open(mode) as stream:
[rank23]: OSError: [Errno 28] No space left on device

x1921c1s1b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: rank 16 exited with code 1
x1921c1s4b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov: rank 0 died from signal 15
```

Two hostnames sit right there in the teardown. `src/ezpz/failover/patterns/sunspot.py`
deliberately declines both, and says so in its module docstring
(lines 17-20):

> We DO NOT match `rank N died from signal {11,15}` — those are almost
> always cascading deaths downstream of a primary kill on a *different*
> node, so including them would falsely tag innocent nodes.

The registered Sunspot patterns (`sunspot.py:138`) are exactly two:
`shepherd_signal_9` and `gloo_connection_closed`. Neither matches a PALS
teardown cascade, so the scraper looked at a failure with two hostnames in
it and correctly refused to blame either. **The healthy node was retired by
`swap_one_blind`, not by a misattribution.**

The same holds one machine over — the documented Aurora `UR_OOM` example's
`rank 7 exited with code 1` line is *not* scraped either:

```console
$ printf 'x4610c4s3b0n0.hsn.cm.aurora.alcf.anl.gov: rank 7 exited with code 1\n' > /tmp/ur.log
$ python3 -m ezpz.failover --machine aurora /tmp/ur.log
$          # empty
```

That line does trip `_CRASH_PATTERNS_RX` — which is what makes the attempt
*retryable* rather than a clean walltime — but tripping the crash matcher
and being scraped as a named host are two different things.

## Finding 4 — "no space left on device" did not mean the disk was full

The obvious reading of attempt 2's `[Errno 28]` is that `/lus/tegu` filled
up. It had not:

```console
$ df -h /lus/tegu
1.2P  116T  1.1P  10% /lus/tegu          # <-- 10% full

$ lfs df -h /lus/tegu
tegu-OST0000_UUID   581.4T  35.4T  540.2T   7% [OST:0]
tegu-OST0001_UUID   581.4T  30.8T  544.7T   6% [OST:1]
tegu-OST0002_UUID    25.2T  24.7T   254.7G 100% [OST:2]   <--
tegu-OST0003_UUID    25.2T  24.6T   398.0G  99% [OST:3]   <--

$ lfs getstripe -d .../out
stripe_count: 1
```

Two of four OSTs were full; the other two were nearly empty — and they are
not even the same size. With `stripe_count: 1`, each DCP shard file lands
*wholly* on one round-robin-selected OST, so a write's success depends
entirely on which OST it drew. Some ranks' shards landed on a full OST and
raised `[Errno 28]`; others landed on a healthy one and succeeded.

(The traceback appears 144 times across all 24 ranks in `attempt-2.log`,
but that count is *not* the evidence for the split — DCP's
`distW.all_reduce("write", ...)` re-raises the originating rank's failure
on every rank, which is why ranks that wrote fine still print
`Traceback (most recent call last): (RANK 2)`. The OST table above is the
evidence.)

Two consequences worth carrying away:

- **This ENOSPC was retryable.** The same write reissued has a real chance
  of landing on a healthy OST. A classifier that matched `Errno 28` and
  declared the job terminal would have converted a recoverable failure into
  a dead one. (That was the original proposal in
  [#231](https://github.com/saforem2/ezpz/issues/231), since withdrawn.)
- **On Lustre, check `lfs df` before `df`.** A filesystem at 10% can be out
  of space for your job. Widening `stripe_count` spreads a shard across
  OSTs and makes a single full OST survivable.

What actually filled those two OSTs was the experiment itself: agpt-2b
writes ~22 GB per checkpoint, **nothing prunes them**, and 400 iters at
`--save-interval 20` is 15 saves ≈ 331 GB. The harness now runs 100 iters
saving every 25 (4 saves, ~88 GB peak) with an `EXIT` trap that reclaims
them, plus a preflight that refuses to start without room.

## What this run does and does not establish

| claim | status |
|---|---|
| the next attempt's `mpiexec` lands on a different host set | **proven** — attempt 2 ran on `s4b0n0`, resumed from step 40 |
| a spare is drawn from the allocation and the hostfile is rewritten in place | **proven** — `active=2` held across all three attempts as `spare` went 2 → 1 → 0 |
| the loop terminates cleanly when spares run out | **proven** — `FAILOVER STOP: exhausted` |
| training resumes from the last durable checkpoint across a node swap | **proven** — step 40, then step 280 |
| the loop identifies **which** node died | **not proven** — [#234](https://github.com/saforem2/ezpz/issues/234) |
| a healthy node is not retired | **disproven** — `s4b0n0`, [#233](https://github.com/saforem2/ezpz/issues/233) |
| an unattributable failure does not consume a spare | **disproven** — both ENOSPC attempts burned one, [#231](https://github.com/saforem2/ezpz/issues/231) |

The last three are why `--auto-retry` is worth running with a spare pool you
can afford to spend, and worth reading the postmortem for afterwards rather
than trusting `bad_nodes.txt` as a hardware verdict.

## Reproducing it

```bash
# From an ezpz checkout on Sunspot. select=4 gives 2 active + 2 spare.
qsub experiments/fault-injection/autoretry_nodekill.pbs
```

The script prints `PASS`/`FAIL` per check and an `OVERALL:` verdict. Its
first assertion is that the kill *landed* (`killed N` in the receipt file),
checked before anything that presumes it — an earlier job reported three
`FAIL`s that all traced back to a kill that never happened, and a job before
that reported `PASS` vacuously for the same reason.

To inspect a run afterwards, see
[Reading the postmortem log](../cli/launch/index.md#reading-the-postmortem-log).
