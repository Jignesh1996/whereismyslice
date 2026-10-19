#!/usr/bin/env bash
# SbR slice matching (one GPU job), resumable.
cd "$(dirname "$0")/.."
systemd-inhibit --what=idle:sleep --who=wms --why="slice matching" \
  systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 -p CPUQuota=400% -- nice -n 10 env OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/18_sbr_slice_matching.py --resume "$@"
