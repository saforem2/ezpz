#!/bin/bash
# GPU smoke: train fsdp_tp with Zephon, checkpoint, resume.
# Requires: torch + Zephon, 1+ ranks (works single-rank for CI).
set -euo pipefail
out=${1:-"./outputs/zephon-training-smoke"}
extra=("${@:2}")
echo "Output: $out"
mkdir -p "$out"
echo "Phase 1: train 2 steps, checkpoint every step"
ezpz launch --nproc 2 --nproc_per_node 2 --nhosts 1 -- \
  python3 -m ezpz.examples.fsdp_tp \
  --model debug --tp 2 --seq-len 32 --vocab-size 50257 --dataset random --dataloader zephon \
  --train-iters 2 --save-interval 1 --ckpt-dir "$out/ckpt" \
  "${extra[@]}" 2>&1 | tee "$out/phase1.log" | tail -20
python3 - "$out/ckpt/step-2/meta.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    meta = json.load(f)
assert meta["step"] == 2, meta
PY

echo "Phase 2: resume from step 2 and train to step 3"
ezpz launch --nproc 2 --nproc_per_node 2 --nhosts 1 -- \
  python3 -m ezpz.examples.fsdp_tp \
  --model debug --tp 2 --seq-len 32 --vocab-size 50257 --dataset random --dataloader zephon \
  --train-iters 3 --save-interval 1 --ckpt-dir "$out/ckpt" \
  "${extra[@]}" 2>&1 | tee "$out/phase2.log" | tail -20
grep -q "RESUMED from" "$out/phase2.log"
python3 - "$out/ckpt/step-3/meta.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    meta = json.load(f)
assert meta["step"] == 3, meta
PY
echo "Smoke OK: resumed from step 2 and saved step 3"
