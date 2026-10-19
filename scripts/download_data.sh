#!/usr/bin/env bash
# Download BigBrain histology + MRI for Where's My Slice. Resumable (curl -C -); safe to re-run.
# Usage: scripts/download_data.sh [--full100]   (adds the 100 um volumes, +1.7 GB)
set -euo pipefail
B=https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015
D=$(cd "$(dirname "$0")/.." && pwd)/data
get() { mkdir -p "$D/$2"; echo ">> $1"; curl -sfSL --retry 5 -C - -o "$D/$2/$(basename "$1")" "$B/$1"; }

# 1. BigBrain 3D histology, native histological space (registered to its own MRI)
for r in 200 400 1000; do get 3D_Volumes/Histological_Space/nii/full8_${r}um_optbal.nii.gz bigbrain/histological; done
# 2. BigBrain's own post-mortem MRI, aligned to the histology blocks
get 3D_MRI/nii/mri_to_blk_aligned.nii.gz bigbrain/mri
# 3. BigBrain warped to MNI ICBM152 2009b sym + the matching ICBM152 T1 (the cross-subject MRI target)
for r in 200 400; do get 3D_Volumes/MNI-ICBM152_Space/nii/full8_${r}um_2009b_sym.nii.gz bigbrain/icbm2009b_sym; done
get 3D_Volumes/MNI-ICBM152_Space/nii/mni_icbm152_t1_tal_nlin_sym_09b_hires.nii.gz mri/icbm2009b_sym
for f in bigbrain_to_icbm2009b_lin.xfm bigbrain_to_icbm2009b_nl.xfm bigbrain_to_icbm2009b_nl_grid_0.mnc; do
  get 3D_Volumes/MNI-ICBM152_Space/transformation/$f bigbrain/transforms; done
if [[ "${1:-}" == "--full100" ]]; then
  get 3D_Volumes/Histological_Space/nii/full8_100um_optbal.nii.gz bigbrain/histological
  get 3D_Volumes/MNI-ICBM152_Space/nii/full8_100um_2009b_sym.nii.gz bigbrain/icbm2009b_sym
fi

# 4. Real 2D coronal sections (the reconstruction inputs): all 7404 thumbnails (~263 um/px, 581 MB)
mkdir -p "$D/sections/thumbnail"
seq -f "pm%04go.png" 1 7404 | while read f; do [[ -s "$D/sections/thumbnail/$f" ]] || echo "$B/2D_Final_Sections/Coronal/Png/Thumbnail/$f"; done \
  | xargs -r -P 4 -n 50 sh -c 'cd "$0" && curl -sfSL --retry 5 --remote-name-all "$@"' "$D/sections/thumbnail"
# plus a few full-resolution (20 um) sections for held-out tests
mkdir -p "$D/sections/full_res"
for n in 1000 2000 3000 3500 4000 5000 6000; do f=$(printf "pm%04do.png" $n)
  [[ -s "$D/sections/full_res/$f" ]] || curl -sfSL --retry 5 -o "$D/sections/full_res/$f" "$B/2D_Final_Sections/Coronal/Png/Full_Resolution/$f"; done
echo "thumbnails: $(ls "$D/sections/thumbnail" | wc -l) / 7404"; du -sh "$D"
