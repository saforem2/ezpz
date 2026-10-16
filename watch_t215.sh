#!/usr/bin/env bash
# Watch the torch 2.15 test (Aurora next-eval, 8868889).
JOB=8870519
prev=""
for _ in $(seq 1 40); do
    st=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes aurora \
        "bash -lc 'qstat -xf ${JOB} 2>/dev/null | grep -oE \"job_state = .\" | head -1'" \
        2>/dev/null | grep -vE "bind: warning")
    body=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes aurora \
        "grep -hE '^RESULT|^===== BEGIN|^torch=|^CCL_KVS|PBS_DONE' ~/fsdptp-dbg2025.log 2>/dev/null | tail -14" \
        2>/dev/null | grep -vE "bind: warning")
    out="[${st:-?}]
${body:-  (nothing yet)}"
    if [ "$out" != "$prev" ]; then
        echo "$out"
        prev="$out"
    fi
    printf '%s' "$out" | grep -q "PBS_DONE" && {
        echo "T215 DONE"
        break
    }
    sleep 75
done
