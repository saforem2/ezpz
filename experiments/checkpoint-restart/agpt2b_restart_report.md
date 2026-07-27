# agpt-2b checkpoint-restart: sync vs async (real Sunspot)

**Baseline:** 2000 steps in 16.94 min.
**Sync restart:** reached step 2000 in 20.94 min across 3 restart(s).
**Async restart:** reached step 2000 in 20.75 min across 3 restart(s).

**Per-save training-thread stall (median):**

- sync `ckpt_save_seconds` = **3.754s** (n=19) — blocking write of the full checkpoint.
- async `ckpt_stage_seconds` = **0.310s** (n=19) — CPU stage only (the cheap half).
- async `ckpt_drain_seconds` = **0.731s** (n=18) — blocking /tmp→shared-FS fan-out at the next step (previously untimed).
- async TRUE total (stage+drain) = **1.048s** — 3.58x less than sync.

| phase | # | resume@step | lost steps | restart_seconds |
|-------|---|-------------|-----------|-----------------|
| sync | 1 | 501 | 4 | 39.77 |
| sync | 2 | 1001 | 2 | 40.34 |
| sync | 3 | 1501 | 2 | 43.86 |
| async | 1 | 501 | 6 | 42.19 |
| async | 2 | 1001 | 11 | 42.12 |
| async | 3 | 1501 | 9 | 42.40 |
