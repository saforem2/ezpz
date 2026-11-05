#!/bin/bash
# GPU smoke: train fsdp_tp with Zephon, checkpoint, resume.
# Requires: torch + Zephon, 1+ ranks (works single-rank for CI).
set -euo pipefail
out=${1:-"./outputs/zephon-training-smoke"}
extra=("${@:2}")
echo "Output: $out"
mkdir -p "$out"
echo "Phase 1: train 2 steps, checkpoint every step"
ezpz launch python3 -m ezpz.examples.fsdp_tp --dataloader zephon --train-steps 2 --ckpt-dir "$out/ckpt" "${extra[@]}" 2>&1 | tail -20
echo "Phase 2: resume + 1 more step (if checkpoint exists)"
if [ -d "$out/ckpt" ]; then
  ezpz launch python3 -m ezpz.examples.fsdp_tp --dataloader zephon --train-steps 3 --ckpt-dir "$out/ckpt" "${extra[@]}" 2>&1 | tail -20
  echo "Smoke OK: resumed to step 3"
else
  echo "No ckpt dir — smoke only ran phase 1"
fi
