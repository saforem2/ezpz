# agpt-2b checkpoint-restart: sync vs async (real Sunspot)

**Baseline:** 240 steps in 2.03 min.
**Sync restart:** reached step 240 in 5.41 min across 3 restart(s).
**Async restart:** reached step 240 in 5.70 min across 3 restart(s).

**Per-save training-thread stall (median):**

- sync `ckpt_save_seconds` = **3.618s** (n=11) — blocking write of the full checkpoint.
- async `ckpt_stage_seconds` = **0.322s** (n=12) — CPU stage only (the cheap half).
- async `ckpt_drain_seconds` = **0.652s** (n=12) — blocking /tmp→shared-FS fan-out at the next step (previously untimed).
- async TRUE total (stage+drain) = **0.976s** — 3.71x less than sync.

| phase | # | resume@step | lost steps | restart_seconds |
|-------|---|-------------|-----------|-----------------|
| sync | 1 | 61 | 3 | 40.41 |
| sync | 2 | 121 | 3 | 39.48 |
| sync | 3 | 181 | 2 | 39.29 |
| async | 1 | 61 | 19 | 37.70 |
| async | 2 | 121 | 21 | 41.48 |
| async | 3 | 181 | 19 | 41.79 |
