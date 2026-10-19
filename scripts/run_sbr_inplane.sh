#!/usr/bin/env bash
# Run the SbR in-plane test for both datasets, ONE GPU job at a time, restarting (resume) if a run exits early.
cd "$(dirname "$0")/.."
run() {  # dataset, extra args
  for attempt in 1 2 3 4; do
    systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=400% -- nice -n 10 \
      env OMP_NUM_THREADS=4 .venv/bin/python scripts/08_sbr_inplane.py --dataset "$@" >> "results/sbr_$1.log" 2>&1 && return 0
    echo "attempt $attempt for $1 exited early, restarting" >> "results/sbr_$1.log"
  done
  return 1
}
run BigBrain && run Allen_labels --mri-px-mm 1.0
