# Where's My Slice? — working repo

Find where a 2D human histology section sits in a 3D reference (BigBrain histology or MRI): its plane
position and cutting angle (pose), then its local, non-rigid alignment. Two families of methods share
one plane sampler and one evaluation:

* **Classical**: grid search with FFT normalised cross-correlation → batched gradient refinement of the
  7-parameter pose through a differentiable plane sampler (NCC same-modality, soft mutual information
  histology→MRI) → 2D B-spline registration (SimpleITK, Mattes MI).
* **Embeddings**: slice bank rendered from the reference over y × pitch × yaw, embedded and indexed with
  faiss. Frozen DINOv2 baseline, and a histology↔MRI dual encoder trained on synthetic pairs resliced on
  the fly. Retrieval candidates can be passed to the classical refinement (hybrid).

## Setup

```bash
uv sync                          # Python 3.12, torch cu128 (RTX 50xx needs CUDA 12.8+)
scripts/download_data.sh         # ~1.7 GB, resumable; --full100 adds the 100 um volumes
```

On the shared laptop run every job capped, e.g.
`systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 -p CPUQuota=400% -- nice -n 10 env OMP_NUM_THREADS=4 .venv/bin/python scripts/...`

## Data (`data/`, from ftp.bigbrainproject.org BigBrainRelease.2015, CC BY 4.0)

| what | file | notes |
|---|---|---|
| BigBrain 3D histology, histological space | `bigbrain/histological/full8_{200,400,1000}um_optbal.nii.gz` | |
| BigBrain's own post-mortem MRI | `bigbrain/mri/mri_to_blk_aligned.nii.gz` | 0.44 mm; sits 1.0 mm off in y, corrected via `offset_mm.json` (script 02) |
| BigBrain in ICBM152 2009b sym | `bigbrain/icbm2009b_sym/full8_{200,400}um_2009b_sym.nii.gz` | cross-subject setting |
| ICBM152 2009b sym T1 | `mri/icbm2009b_sym/mni_icbm152_t1_tal_nlin_sym_09b_hires.nii.gz` | includes skull |
| BigBrain → ICBM transforms | `bigbrain/transforms/` | MINC xfm + nl grid |
| Real coronal sections | `sections/thumbnail/pm0001o.png … pm7404o.png` (all), `sections/full_res/` (7 at 20 um) | pm0001 = occipital pole |

All images are handled tissue-bright on a zero background (histology is inverted on load).

## Code (`src/wms/`)

| module | |
|---|---|
| `volumes.py` | volume registry/loader (RAS, [0, 1], offset fix) |
| `planes.py` | `Pose` (x, y, z, pitch, yaw, roll, log-scale), differentiable `sample`, random poses, `pose_error` |
| `sections.py` | real BigBrain sections and their pixel sizes |
| `metrics.py` | batched NCC, Parzen mutual information, blur, resize |
| `classical.py` | `coarse_search`, `refine`, `localise`, `local_align` |
| `augment.py` | stain jitter, tears, missing chunks, folds, blur/noise (numpy and batched torch) |
| `embed.py` | canonical views (fixed mm scale, tissue-centred), DINOv2 `Embedder` |
| `retrieval.py` | slice bank, faiss index, query → poses (roll/flip test-time search) |
| `dual_encoder.py` | shared DINOv2 + per-modality heads, soft-target InfoNCE, pose head |
| `inplane.py` | 2D affine of a section inside its MRI slice: rotation × scale grid + FFT translation, MI refinement |

## Scripts (run in order)

1. `01_check_alignment.py` — histology vs MRI overlays + NMI (`figures/alignment_*.png`)
2. `02_fix_mri_offset.py` — estimate the MRI↔histology translation
3. `03_classical_synthetic.py` — classical pose on synthetic oblique, artefacted sections
4. `04_real_sections.py` — classical pose of real sections; fits the 20 um cutting line used as truth
5. `05_embedding_retrieval.py [--bank bigbrain_mri] [--model models/dual_encoder.pt]` — retrieval + hybrid
6. `06_train_dual_encoder.py` — train the histology↔MRI encoder (y ∈ [0, 20] mm held out)
7. `07_local_alignment.py` — pose + B-spline alignment of full-resolution sections
8. `08_sbr_inplane.py --dataset BigBrain|Allen_labels` — in-plane position of SbR histology inside its MRI slice (`run_sbr_inplane.sh` runs both)
9. `09_sbr_slice_positions.py` — 3D position and pixel size of the SbR MRI slices
10. `10_sbr_figures.py` — SbR summary figures
11. `11_synthetic_stress.py` — synthetic damage stress test (`run_stress.sh`)
12. `12_train_patch_jepa.py` — JEPA-style patch localiser
13. `13_patch_localisation.py` — real patches in their MRI slice (2D) and in the whole MRI volume (3D)  (`run_patch_pipeline.sh` runs 09 → 12 → 13)
14. `14_hierarchical_patch.py` — slice first, then pose: patch-only hierarchical search vs placing the parent section first
15. `15_stress_improved.py` — stress test with the regularised deformable step (`run_stress_sweep.sh` tunes λ)
16. `16_sbr_examples_figure.py` — reference vs ours overlays for chosen SbR slices (edge and medial examples)
17. `17_damage_examples_figure.py` — what each synthetic damage type looks like on one real section
18. `18_sbr_slice_matching.py` — which MRI slice: raw SbR section → histology volume → MRI, vs MRI directly (`run_slice_matching.sh`)
19. `19_oblique_test.py` — sections resliced at 0–45° tilt: is the cutting angle recovered?
20. `20_slice_matching_figure.py`, `21_oblique_figure.py` — figures for 18 and 19
22. `22_dual_encoder_figure.py` — training pairs and loss curve of the dual encoder
23. `23_patch_pipeline_figure.py` — one real histology piece: section → slice → spot in the MRI
24. `24_oblique_rotation_test.py` — oblique (0–45°) AND rotated (360°) sections: tilt, rotation and alignment scored
25. `25_3d_figures.py patch|oblique` — 3D views of the found plane / spot inside the brain
26. `26_patch_improve.py` — sharper piece localisation: piece found in its own section + section in-plane alignment

`pytest tests` checks the sampler against the voxel grid, rotations, scale and gradients.

## Results so far

All numbers from `results/` (2026-10-18). Truth for real sections: y = −70.01 mm + 20.00 µm × n, the line
fitted by script 04 (68/69 inliers; the 20.00 µm slope matches the cutting thickness).

**Classical pose (no position prior)**

| queries | target | median offset | within 2 mm | tilt | time |
|---|---|---|---|---|---|
| 69 real sections (thumbnails) | BigBrain histology | 0.01 mm | 100 % | 0.0° | ~1 s |
| 69 real sections | BigBrain MRI (offset-corrected) | 0.10 mm | 100 % | 0.5° | ~1.4 s |
| 30 synthetic oblique + artefacts | BigBrain histology | 0.02 mm | 100 % | 0.04° | ~1 s |
| 30 synthetic oblique + artefacts | BigBrain MRI | 0.09 mm | 100 % | 0.4° | |

**Embedding retrieval, MRI bank (histology query → MRI planes), 1 mm / 5° bank grid**

| encoder | queries | top-1 within 3 mm | top-5 contains ≤3 mm | hybrid (top-5 → classical) within 2 mm | retrieval |
|---|---|---|---|---|---|
| frozen DINOv2 ViT-S | 69 real | 38 % | 78 % | 86 % | 19 ms |
| dual encoder (trained 3000 steps, 10 min) | 59 real, trained y range | 100 % | 100 % | 100 % | 11 ms |
| dual encoder | 10 real, held-out slab y ∈ [0, 20] mm | 90 % | 100 % | 100 % | 10 ms |

Frozen DINOv2 against the *histology* bank: 88 % top-1 within 3 mm, hybrid 96 %.

**Local alignment (full-res sections at 0.2 mm, B-spline 10 mm mesh, Mattes MI)**: against the MRI,
tissue Dice 0.936–0.941 → 0.946–0.964 and NCC +0.01–0.03; against the histology volume the warp is ≈0.02–0.04 mm
(nothing to correct, as expected).

**In-plane position on paired slices (SbR datasets, Casamitjana et al. 2021, github.com/acasamitjana/SynthByReg)**

`data/sbr/` (sparse clone of the repo's `data/`). `slice_N` in `histo/` and `mri/` are a pair; histology is the
raw section (rotated up to ±90°, slide edges, labels). We estimate the MRI-pixel → histology-pixel affine from the
image pair and tissue masks only, with no initial transform (`wms/inplane.py`, script 08, ~5.5 s/slice on the GPU), and
compare it with the dataset's reference affine.

| dataset | reference | n | median distance to reference | within 1 / 2 (mm or px) | tissue Dice ours vs reference |
|---|---|---|---|---|---|
| BigBrain (MRI px = 0.4 mm) | NiftyReg files `histo/affine` (reproduce `images_affine` at NCC 0.999) | 344 | 0.48 mm | 87 % / 94 % (mm) | 0.920 vs 0.919 |
| BigBrain, trusted reference (Dice ≥ 0.9) | same | 256 | 0.41 mm | 98 % / 100 % (mm) | 0.925 vs 0.924 |
| Allen (pixel size not in the repo) | recovered from `images_affine` per piece (NCC ≥ 0.95; 35 of 127 items not recoverable) | 92 | 2.5 px | 2 % / 29 % (px), 85 % < 4 px | 0.966 vs 0.961 |

All 21 BigBrain slices more than 2 mm from the reference have higher Dice with our placement (+0.03 to +0.55); the
largest are visible reference failures (`figures/sbr_BigBrain_reference_failures.png`). Same for all 14 Allen items
beyond 4 px. Summary figures: `figures/sbr_BigBrain_inplane.png`, `figures/sbr_Allen_labels_inplane.png`.

The SbR BigBrain MRI slices are exact coronal planes of BigBrain's MRI: y = −71.19 + 0.400 × N mm, 0.4007 mm/px
(script 09, |residual| ≤ 0.02 mm), i.e. SbR slice N ≈ BigBrain section 20 N − 59. This is the ground truth for the
next step, histology → MRI slice matching.

**Synthetic stress test (script 11, 30 real BigBrain SbR slices × 9 damage types, exact ground truth)**

Damage is applied to the raw histology with a known backward map (`wms/distort.py`); localisation runs with no
initial transform and a 360° rotation search. Median per-slice mean error (mm). "added" = distance from our own
result on the undamaged slice; points whose tissue was cut away are not scored.

| damage | affine vs truth | affine added | B-spline added | best of the two within 2 mm | failures > 5 mm |
|---|---|---|---|---|---|
| none | 0.40 | – | – | 100 % | 0 |
| rotation 360° | 0.39 | 0.05 | 0.58 | 100 % | 0 |
| affine 5 % / 10 % | 0.43 / 0.43 | 0.06 / 0.06 | 0.52 / 0.47 | 100 % | 0 |
| elastic 5 % | 1.28 | 1.14 | 0.96 | 93 % | 0 |
| elastic 10 % | 2.49 | 2.45 | 1.99 | 37 % | 0 |
| cuts + tears | 0.81 | 0.65 | 0.80 | 97 % | 0 |
| all 5 % | 1.43 | 1.33 | 0.96 | 97 % | 0 |
| all 10 % | 2.32 | 2.34 | 1.59 | 57 % | 1 |

Rotation, stretch/shear and cuts are absorbed by the global affine search. Elastic deformation is the limit: the
B-spline (40 px mesh, Mattes MI, masked to overlapping tissue) removes ~20–30 % of it but adds ~0.5 mm of drift
when there is nothing to correct. Next: regularised / diffeomorphic local step, applied only when it improves the
match.

**Improved whole-section alignment (script 15, same 270 cases, same damage seeds)**

`inplane.deform`: GPU displacement field on a 6 → 12 control grid (bicubic), maximising MI + mask Dice with a
strain penalty (λ = 0.05, chosen on a 6-slice sweep, `run_stress_sweep.sh`), on top of the affine. Median shift
from our own undamaged result (mm):

| damage | affine | old B-spline | **new deformable** |
|---|---|---|---|
| rotation 360° | 0.05 | 0.58 | **0.05** |
| affine 5 % / 10 % | 0.06 / 0.06 | 0.52 / 0.47 | **0.07 / 0.05** |
| elastic 5 % | 1.14 | 0.96 | **0.51** |
| elastic 10 % | 2.45 | 1.99 | **1.58** |
| cuts + tears | 0.65 | 0.80 | **0.30** |
| all 5 % | 1.33 | 0.96 | **0.63** |
| all 10 % | 2.34 | 1.59 | **1.50** |

* No drift where there is nothing to fix (0.05 mm vs 0.5 mm for the B-spline), about half the elastic and cut
  error removed; cases moved > 2 mm by damage: 44 affine, 28 B-spline, 18 deformable. ~9 s per section in total.
* Versus the reference affine on undamaged sections it sits 0.77 mm away (affine 0.40): it also bends the section
  where histology and MRI genuinely differ, which an affine reference cannot judge.
* Did not help: re-refining the affine after removing uncovered (cut) MRI tissue from the mask (`localise_cut_aware`)
  changed nothing; the improvement on cuts comes from the deformation. The gain gate (0.01) almost never rejects.
* Elastic 10 % is still 1.5 mm: the strain penalty limits large smooth warps; a diffeomorphic (SyN-like) step is
  the next thing to try.

**JEPA-style patch localisation (scripts 12–13, 80 real raw SbR BigBrain patches, ~28 mm)**

Trained 4000 steps (~9 min) on synthetic patches from the 3D BigBrain histology vs MRI; held-out synthetic median
5.9 mm. On real patches, fraction within 10 mm (median error):

| method | clean | damaged |
|---|---|---|
| classical template search, 2D | 30 % (28.1 mm) | 20 % (29.6 mm) |
| frozen DINOv2 dense features, 2D | 12 % (35.8 mm) | 0 % (49.3 mm) |
| JEPA + sub-cell peak, 2D (patch in its MRI slice) | **85 % (4.7 mm)** | 48 % (10.7 mm) |
| JEPA + sub-cell peak, 3D (patch in the whole MRI volume, 1395-plane bank) | **52 % (9.1 mm)** | 28 % (20.7 mm) |
| JEPA + classical local refine, 2D / 3D | 45 % / 28 % | 20 % / 5 % |

The classical local refinement makes JEPA worse (cross-modal template matching picks wrong nearby spots); damage
halves the hit rate. Damaged = rotation 360°, 10 % affine, 5 % elastic, cuts. The first 8 slices used a different
patch draw (the run was resumed after a laptop suspend).

### Which MRI slice? (`scripts/18_sbr_slice_matching.py`, all 344 SbR BigBrain sections, no prior)

| method | typical distance from true slice | within ±1 slice (0.4 mm) | within 1 mm |
|---|---|---|---|
| **histology volume first** (NCC), MRI slice read off | **0.24 mm** | **89 %** | **96 %** |
| then refined on the MRI (MI) | 0.34 mm | 69 % | 84 % |
| MRI directly (MI) | 0.41 mm | 65 % | 78 % |

* Every method finds the sections +1.0 mm from SbR's slice numbering (spread 0.5 mm): exactly the 1.0 mm offset of
  the released BigBrain MRI (scripts/02), so SbR paired histology with the uncorrected MRI. Truth = SbR plane +
  1.0 mm (an independent correction, not fitted here).
* Similarity to neighbouring MRI planes peaks at the true slice (`figures/sbr_slice_profile.png`,
  `figures/sbr_slice_neighbours.png`).
* Some SbR tissue masks cover the whole glass slide (e.g. slices 99, 104, 127, 189).

### Oblique cuts (`scripts/19_oblique_test.py`, 10 sections per tilt, coarse grid ±48°)

Histology resliced at 0/15/30/45° from coronal (random direction, roll ±10°), searched with no prior: 100 % within
5° and 2 mm at every tilt, in the MRI (median angle error 0.29–0.46°, plane 0.06–0.13 mm) and in the histology
volume (0.02–0.03°, ≤ 0.06 mm). ~4–5 s each. 90° (sagittal/axial) cuts would need the grid around those axes.

### Sharper piece localisation (`scripts/26_patch_improve.py`, same 320 pieces, truth + 1.0 mm SbR offset)

| section placed first, then … | within 10 mm (clean / damaged) | within 2 mm | median |
|---|---|---|---|
| JEPA in that MRI slice | 80 % / 52 % | 8 % / 2 % | 5.7 / 9.6 mm |
| JEPA on a 5.6 mm grid (4 half-cell shifted planes) | 78 % / 54 % | 9 % / 1 % | 5.5 / 9.2 mm |
| piece found in its own section (NCC), carried by the 3D pose | 98 % / 99 % | 60 % / 57 % | 1.7 / 1.8 mm |
| **+ section aligned in-plane inside its MRI plane** | **98 % / 99 %** | **82 % / 78 %** | **0.8 / 1.1 mm** |

Needs the piece's own section (same-contrast matching is nearly exact); isolated blocks rely on the piece-alone route.

### Full pose of oblique, rotated sections (`scripts/24_oblique_rotation_test.py`, 10 per tilt)

Histology resliced at 0/15/30/45° from coronal and rotated anywhere in 360°, searched in the MRI with no prior
(coarse: tilt ±48° step 12, rotation step 15°; then refinement), ~12 s each. Within 5° angle, 5° rotation and 2 mm
alignment (mean 3D distance over tissue pixels): 39 / 40 (0°: 10/10, 15°: 10/10, 30°: 9/10, 45°: 10/10); typical
errors 0.4°, 0.4°, 0.4 mm. The one failure (30°) landed in the wrong part of the brain.

### Slice first, then pose (`scripts/14_hierarchical_patch.py`)

320 new real patches (80 slices x 2 x clean/damaged), all methods on the same patches; within 10 mm (median):

| 3D method | clean | damaged |
|---|---|---|
| flat: JEPA joint argmax over the 1 mm bank (scripts/13) | 52 % (9.5 mm) | 24 % (21.1 mm) |
| flat over a denser bank (0.5 mm, 25 tilts) | 50 % (10.0 mm) | 26 % (20.9 mm) |
| patch-only hierarchical: slice from y-smoothed scores, then pose around it | 52 % (9.6 mm) | 26 % (20.8 mm) |
| + 8-fold rotation/flip averaging of the patch embedding | 55 % (8.8 mm) | 29 % (17.8 mm) |
| **parent section placed in 3D first (classical MI), patch found in that plane, + TTA** | **81 % (5.7 mm)** | **51 % (9.7 mm)** |
| oracle: patch in its true MRI slice, + TTA | 79 % (5.1 mm) | 53 % (9.1 mm) |

* From the patch alone, the slice is the bottleneck: the right slice (|y| <= 2 mm) is picked for only ~20 % of
  patches, and ranking slices first does not change that; a 28 mm patch carries too little evidence.
* Placing the whole raw section first works: 89 % of the 80 sections within 2 mm in y (median 1.1 mm, worst 5.6
  mm; median 3D tissue error 2.3 mm, ~2 s each), and patch localisation in that estimated plane matches the true-slice oracle.
* Confidence becomes useful in the section-first route: in the most confident half (2D peak margin) 93 % of clean
  and 72 % of damaged patches are within 10 mm; for the patch-only routes the confidence barely separates.

**Caveats**
* GPU jobs must run one at a time on this laptop: two concurrent CUDA processes triggered `NVRM: NV_ERR_NO_MEMORY`
  and both hung (this caused the training hang). Script 08 writes results per slice and has a stall watchdog;
  `scripts/run_sbr_inplane.sh` runs both datasets sequentially with resume.
* Same brain throughout: the real sections are the ones BigBrain was reconstructed from, and the dual encoder
  trained on that brain's histology and MRI. The held-out slab tests unseen anatomy, not an unseen brain or
  stain. Next: ICBM152 / other brains (DHARANI, NextBrain).
* Real-section truth comes from a 2-parameter line fit to the classical histology results, so the histology row
  is near-circular; the MRI and embedding rows are independent of it.
* The MRI B-spline warp overfits in places (pm4000: 95th-percentile displacement 4.6 mm, visible distortion in
  the temporal lobe in `figures/local_4000.png`); it needs bending-energy regularisation or a coarser mesh.
* Bank roll is 0 with test-time rolls ±10°; sections rotated further need a wider roll search.
* A first training run hung at step ~2850 with the GPU idle (cause unknown); the trainer now checkpoints every
  500 steps and dumps a stack trace if a step takes >120 s.
