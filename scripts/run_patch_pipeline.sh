#!/usr/bin/env bash
# One GPU job at a time: slice geometry (x, z) -> JEPA training -> patch localisation evaluation.
set -e
cd "$(dirname "$0")/.."
cap() { systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 -p CPUQuota=400% -- nice -n 10 env OMP_NUM_THREADS=4 "$@"; }
cap .venv/bin/python scripts/09_sbr_slice_positions.py > results/sbr_slice_positions.log 2>&1
cap .venv/bin/python scripts/12_train_patch_jepa.py --steps 4000 > results/patch_jepa.log 2>&1
cap .venv/bin/python scripts/13_patch_localisation.py > results/patch_localisation.log 2>&1
