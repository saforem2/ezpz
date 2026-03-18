#!/bin/bash
# capture_recipe_outputs.sh
#
# Run each docs/recipes.md snippet on Aurora and save ANSI-stripped output.
# Usage (on an interactive compute node):
#
#   qsub -l select=1 -l walltime=00:10:00 -A <project> -q debug -I
#   cd /path/to/ezpz
#   bash scripts/capture_recipe_outputs.sh
#
# Output files are written to ./recipe_outputs/

set -euo pipefail

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

OUTDIR="${1:-./recipe_outputs}"
SCRIPTS_DIR="${SCRIPTS_DIR:-"$(pwd)/tmp-scripts"}"
mkdir -p "${SCRIPTS_DIR}"
mkdir -p "$OUTDIR"

strip_ansi() { perl -pe 's/\e\[[0-9;]*[mGKHF]//g'; }

# ── recipe scripts ──────────────────────────────────────────────────────

cat > "$SCRIPTS_DIR/recipe_fsdp.py" << 'PYEOF'
import torch
import ezpz

rank = ezpz.setup_torch()
model = torch.nn.Linear(32, 16).to(ezpz.get_torch_device())
model = ezpz.wrap_model(model, use_fsdp=True)  # use_fsdp=False for DDP
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(f"[rank {rank}] model wrapped, optimizer ready")
ezpz.cleanup()
PYEOF

cat > "$SCRIPTS_DIR/recipe_wandb.py" << 'PYEOF'
import ezpz

rank = ezpz.setup_torch()
if rank == 0:
    ezpz.setup_wandb(project_name="ezpz-wandb-recipe")

history = ezpz.History()
num_steps = 10
for step in range(num_steps):
    loss_val = 1.0 / (step + 1)
    lr_val = 1e-3
    history.update({"loss": loss_val, "lr": lr_val})

if rank == 0:
    history.finalize(outdir="wandb-recipe-outputs", plot=False)
PYEOF

cat > "$SCRIPTS_DIR/recipe_timing.py" << 'PYEOF'
import time
import torch
import ezpz

rank = ezpz.setup_torch()
model = torch.nn.Linear(32, 16).to(ezpz.get_torch_device())
batch = torch.randn(8, 32, device=ezpz.get_torch_device())

ezpz.synchronize()
t0 = time.perf_counter()

output = model(batch)
loss = output.sum()
loss.backward()

ezpz.synchronize()
dt = time.perf_counter() - t0
print(f"[rank {rank}] step time: {dt:.4f}s")
ezpz.cleanup()
PYEOF

cat > "$SCRIPTS_DIR/recipe_no_dist_history.py" << 'PYEOF'
import ezpz

rank = ezpz.setup_torch()
history = ezpz.History(distributed_history=False)

for step in range(5):
    history.update({"loss": 1.0 / (step + 1)})

print(f"[rank {rank}] distributed_history={history.distributed_history}")
ezpz.cleanup()
PYEOF

# ── run each recipe ─────────────────────────────────────────────────────

RECIPES=(recipe_fsdp recipe_wandb recipe_timing recipe_no_dist_history)

for recipe in "${RECIPES[@]}"; do
    echo "── Running ${recipe} ──"
    WANDB_MODE=disabled \
        ezpz launch python3 "$SCRIPTS_DIR/${recipe}.py" \
        2>&1 | strip_ansi > "$OUTDIR/${recipe}.txt"
    echo "   → saved to $OUTDIR/${recipe}.txt"
done

echo ""
echo "All outputs saved to $OUTDIR/"
ls -l "$OUTDIR"/*.txt

# clean up temp scripts
rm -rf "$SCRIPTS_DIR"
