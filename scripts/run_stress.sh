#!/usr/bin/env bash
# synthetic stress test, one GPU job, resumes from results/stress_inplane.csv if it exits early
cd "$(dirname "$0")/.."
for attempt in 1 2 3 4; do
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=400% -- nice -n 10 \
    env OMP_NUM_THREADS=4 .venv/bin/python scripts/11_synthetic_stress.py --n 30 >> results/stress.log 2>&1 && exit 0
  echo "attempt $attempt exited early, restarting" >> results/stress.log
done
