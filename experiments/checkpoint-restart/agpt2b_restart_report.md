# agpt-2b checkpoint-restart: sync vs async (real Sunspot)

**Baseline:** 240 steps in 2.03 min.
**Sync restart:** reached step 240 in 5.54 min across 3 restart(s).
**Async restart:** reached step 240 in 5.79 min across 3 restart(s).

**Per-step save stall (median):** sync `ckpt_save_seconds`=3.567s (n=11) vs async `ckpt_stage_seconds`=0.301s (n=11) — 11.9x less stall.

| phase | # | resume@step | lost steps | restart_seconds |
|-------|---|-------------|-----------|-----------------|
| sync | 1 | 61 | 3 | 39.21 |
| sync | 2 | 121 | 1 | 43.35 |
| sync | 3 | 181 | 3 | 43.03 |
| async | 1 | 61 | 3 | 37.95 |
| async | 2 | 121 | 2 | 37.86 |
| async | 3 | 181 | 2 | 41.68 |
