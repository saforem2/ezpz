# agpt-2b checkpoint-restart: sync vs async (real Sunspot)

**Baseline:** 240 steps in 2.03 min.
**Sync restart:** reached step 240 in 5.42 min across 3 restart(s).
**Async restart:** reached step 240 in 5.65 min across 3 restart(s).

**Per-save training-thread stall (median):**

- sync `ckpt_save_seconds` = **3.543s** (n=11) — blocking write of the full checkpoint.
- async `ckpt_stage_seconds` = **0.295s** (n=8) — CPU stage only (the cheap half).
- async `ckpt_drain_seconds` = **5.175s** (n=8) — blocking /tmp→shared-FS fan-out at the next step (previously untimed).
- async TRUE total (stage+drain) = **5.470s** — 1.54x the sync stall (async is SLOWER here).

| phase | # | resume@step | lost steps | restart_seconds |
|-------|---|-------------|-----------|-----------------|
| sync | 1 | 61 | 3 | 43.21 |
| sync | 2 | 121 | 0 | 39.08 |
| sync | 3 | 181 | 2 | 39.13 |
| async | 1 | 61 | 0 | 37.48 |
| async | 2 | 121 | 0 | 39.58 |
| async | 3 | 181 | 0 | 38.64 |
