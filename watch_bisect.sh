#!/usr/bin/env bash
# Watch the fsdp_tp bisect (Sunspot 12478669).
#
# Keys on the job id and reads the job-stamped log, so a stale file from
# an earlier run cannot be reported as this round's result.
JOB=12478677
prev=""
for _ in $(seq 1 40); do
    st=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes sunspot \
        "bash -lc 'qstat -xf ${JOB} 2>/dev/null | grep -oE \"job_state = .\" | head -1'" \
        2>/dev/null | grep -vE "bind: warning")
    body=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes sunspot \
        "grep -hE '^RESULT|^===== BEGIN|PBS_DONE' ~/fsdptp-ccl.log 2>/dev/null | tail -14" \
        2>/dev/null | grep -vE "bind: warning")
    out="[${st:-?}]
${body:-  (no arms finished yet)}"
    if [ "$out" != "$prev" ]; then
        echo "$out"
        prev="$out"
    fi
    printf '%s' "$out" | grep -q "PBS_DONE" && {
        echo "BISECT DONE"
        break
    }
    sleep 75
done
