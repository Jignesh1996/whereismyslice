#!/usr/bin/env bash
# Smoothness-weight sweep for the deformable step (one GPU job at a time).
cd "$(dirname "$0")/.."
cap() { systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 -p CPUQuota=400% -- nice -n 10 env OMP_NUM_THREADS=4 "$@"; }
for lam in 0.01 0.05 0.2; do
  cap .venv/bin/python scripts/15_stress_improved.py --n 6 --lam $lam --out results/sweep/lam_$lam.csv \
      --conditions clean "elastic 10%" "cuts + tears" "all 10%" > results/sweep/lam_$lam.log 2>&1
done
