# agpt-2b checkpoint-restart: sync vs async (real Sunspot)

**Baseline:** 300 steps in 20.56 min.
**Sync restart:** reached step 300 in 24.51 min across 2 restart(s).
**Async restart:** reached step 300 in 24.81 min across 2 restart(s).

**Per-save training-thread stall (median):**

- sync `ckpt_save_seconds` = **23.573s** (n=3) — blocking write of the full checkpoint.
- async `ckpt_stage_seconds` = **1.729s** (n=5) — CPU stage only (the cheap half).
- async `ckpt_drain_seconds` = **3.692s** (n=3) — blocking /tmp→shared-FS fan-out at the next step (previously untimed).
- async TRUE total (stage+drain) = **5.415s** — 4.35x less than sync.

| phase | # | resume@step | lost steps | restart_seconds |
|-------|---|-------------|-----------|-----------------|
| sync | 1 | 101 | 0 | 55.78 |
| sync | 2 | 201 | 0 | 55.25 |
| async | 1 | 101 | 4 | 63.47 |
| async | 2 | 201 | 5 | 55.62 |
