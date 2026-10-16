#!/usr/bin/env bash
# Watch both stack-bisect jobs on Aurora debug.
#   8870519 = frameworks/2025.3.1        (torch 2.10, stack confounded)
#   8870547 = custom .venv torch 2.13    (old stack, confound removed)
prev=""
for _ in $(seq 1 40); do
    out=""
    for spec in "2025.3.1-t210:8870519:fsdptp-dbg2025.log" \
        "custom-t213:8870547:fsdptp-v213-2025.log"; do
        name="${spec%%:*}"
        rest="${spec#*:}"
        job="${rest%%:*}"
        log="${rest##*:}"
        st=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes aurora \
            "bash -lc 'qstat -xf ${job} 2>/dev/null | grep -oE \"job_state = .\" | head -1'" \
            2>/dev/null | grep -vE "bind: warning")
        body=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes aurora \
            "grep -hE '^RESULT|^torch=|device_count|FATAL|PBS_DONE' ~/${log} 2>/dev/null | tail -10" \
            2>/dev/null | grep -vE "bind: warning")
        out="$out
[${name} ${st:-?}]
${body:-  (nothing yet)}"
    done
    if [ "$out" != "$prev" ]; then
        echo "$out"
        prev="$out"
    fi
    [ "$(printf '%s' "$out" | grep -c 'PBS_DONE')" -ge 2 ] && {
        echo "BOTH DONE"
        break
    }
    sleep 80
done
