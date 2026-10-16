#!/usr/bin/env bash
# One authoritative monitor for round 3 (per the operating contract:
# a single poller, >=60s interval, connection reuse).
prev=""
for _ in $(seq 1 30); do
    out=""
    a=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes aurora \
        'cat /home/foremans/ezpz-v3.log 2>/dev/null | tail -30' 2>/dev/null | grep -vE "bind: warning")
    [ -n "$a" ] && out="$out
[AURORA]
$a"
    s=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes sunspot \
        'cat /home/foremans/ezpz-v3.log 2>/dev/null | tail -30' 2>/dev/null | grep -vE "bind: warning")
    [ -n "$s" ] && out="$out
[SUNSPOT]
$s"
    p=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes perlmutter \
        'cat /pscratch/sd/f/foremans/pm-diag-58827488.log 2>/dev/null | tail -20' 2>/dev/null)
    [ -n "$p" ] && out="$out
[PERLMUTTER-DIAG]
$p"
    if [ "$out" != "$prev" ] && [ -n "$out" ]; then
        echo "$out"
        prev="$out"
    fi
    n=$(printf '%s' "$out" | grep -c -E "PBS_DONE|DIAG_DONE")
    [ "$n" -ge 3 ] && {
        echo "ALL DONE"
        break
    }
    sleep 75
done
