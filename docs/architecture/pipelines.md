# Pipeline Comparison

This project has 3 pipelines for rat segmentation and tracking.
All use YOLO (detection + keypoints) and SAM2 (segmentation), but
differ in how they handle identity and what happens when YOLO fails.

---

## Comparison Table

| Aspect | sam2_yolo | sam2_video | reference |
|--------|-----------|------------|-----------|
| **SAM2** | ImagePredictor (stateless) | VideoPredictor (temporal memory) | ImagePredictor (stateless) |
| **YOLO** | detect + track (BoT-SORT) | detect-only | detect-only |
| **Tracking** | SlotTracker (YOLO IDs + Hungarian) | SAM2 temporal memory | IdentityMatcher (Hungarian + state machine) |
| **Centroid fallback** | No | Yes (SAM2 temporal) | Yes (SAM2 point prompt from previous centroid) |
| **Batch/chunks** | Yes | No | Yes |
| **Contacts** | Yes | Yes | Yes |
| **Speed** | Fast (~25 FPS) | Slow (~5-10 FPS) | Fast (~25 FPS) |
| **Occlusion robustness** | Medium | High | Medium-High |
| **Recommended** | Videos without frequent interactions | Short videos with heavy occlusion | **Default choice** — best balance |

---

## 1. sam2_yolo

**Module:** `src/pipelines/sam2_yolo/`
**Configs:** `configs/local_quick.yaml`, `configs/hpc_full.yaml`
**Slurm:** `slurm/run_infer.sbatch`, `slurm/run_chunks.sbatch`

### Per-frame flow

```
Frame → YOLO.track() (BoT-SORT) → boxes + track_ids + keypoints
      → SAM2ImagePredictor.predict(box=...) → masks
      → SlotTracker (YOLO IDs + Hungarian) → fixed slots
      → ContactTracker (optional)
      → Render overlay
```

### How tracking works

1. YOLO uses BoT-SORT internally → assigns track_id to each detection
2. SlotTracker maps track_ids to fixed slots (slot 0 = green, slot 1 = red)
3. If YOLO swaps IDs, a swap guard compares straight vs swapped cost
4. Fallback: Hungarian assignment with soft cost (distance + IoU + area)

### Batch support

```bash
# Job array with 4 parallel chunks
sbatch --array=0-3 slurm/run_chunks.sbatch

# Each chunk processes a range of frames
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0

# Merge at the end
python scripts/merge_chunks.py outputs/runs/*_chunk*/ -o outputs/runs/merged
```

### Weaknesses

- Depends on YOLO every frame. If YOLO loses a rat → empty slot, rat "disappears"
- BoT-SORT can cause ID switches during crossings/interactions
- No fallback: if YOLO doesn't detect, SAM2 has no prompt

### When to use

- Videos without frequent rat interactions
- When you need batch processing for long videos
- When speed is the priority

---

## 2. sam2_video

**Module:** `src/pipelines/sam2_video/`
**Configs:** `configs/local_sam2video.yaml`, `configs/hpc_sam2video.yaml`
**Slurm:** `slurm/run_sam2video.sbatch`

### Per-segment flow (2000 frames)

```
Segment:
  1. Extract frames to temporary JPEGs
  2. YOLO detect_only() on first frame → boxes for initialization
  3. SAM2VideoPredictor.init_state(frames_dir)
  4. SAM2.add_new_points_or_box(frame_idx=0, box=...) per rat
  5. SAM2.propagate_in_video() → generates masks for all frames
     Per frame:
       a. YOLO detect_only() → keypoints
       b. Match keypoints ↔ SAM2 masks (nose-in-mask or centroid)
       c. ContactTracker (optional)
       d. Render overlay
  6. Clean up JPEGs + GPU memory
```

### How tracking works

- SAM2VideoPredictor has a memory bank of 7 frames (memory_attention + memory_encoder)
- Once initialized with boxes, propagates masks automatically
- Doesn't need YOLO on intermediate frames (only for keypoints)
- Identity maintained by SAM2 internally (object pointers)
- Between segments: centroid matching to preserve colors

### Batch support

**No batch support.** Processes the full video at once (divided into
internal segments of ~2000 frames for memory management, but cannot
be parallelized across GPUs).

### Weaknesses

- Slow: ~3-5x slower than other pipelines
- Requires extracting frames to JPEG (additional I/O)
- Higher memory usage (GPU + RAM for frames)
- Cannot parallelize with chunks

### When to use

- Short videos with heavy interactions/occlusions
- When tracking quality matters more than speed
- When other pipelines fail on specific frames

---

## 3. reference (recommended)

**Module:** `src/pipelines/reference/`
**Configs:** `configs/local_reference.yaml`, `configs/hpc_reference.yaml`
**Slurm:** `slurm/run_reference.sbatch`

### Per-frame flow

```
Frame → YOLO detect_only() → boxes + keypoints (NO BoT-SORT)
      → SAM2ImagePredictor.set_image(frame)
      → For each detection: SAM2.predict(box=...) → mask
      → If YOLO detected fewer rats than active slots:
          For each unmatched slot:
            SAM2.predict(point_coords=prev_centroid) → fallback mask
      → filter_duplicates() → remove redundant masks
      → IdentityMatcher.match() → assign to fixed slots
      → ContactTracker (optional)
      → Render overlay
```

### How tracking works

1. IdentityMatcher uses Hungarian assignment with 3-component soft cost:
   - Distance cost (weight 0.4): normalized by `proximity_threshold`
   - Mask IoU cost (weight 0.4): shape continuity between frames
   - Area ratio cost (weight 0.2): size consistency
2. Velocity prediction with EMA smoothing (0.7/0.3 blend)
3. Swap guard for N=2 crossings: compares straight vs swapped total cost
4. **State machine** (SEPARATE ↔ MERGED) for close interaction handling:
   - MERGED: freezes identity, coasts with velocity, skips centroid fallback
   - Split resolution: 2×2 Hungarian using pre-merge evidence

See [Identity Matcher Design](identity_matcher.md) for full details.

### Centroid fallback (key feature)

When YOLO detects only 1 rat but IdentityMatcher knows there are 2 active slots:

```python
# The lost slot has a previous centroid
prev_centroid = matcher.prev_centroids[slot_idx]

# Used as point prompt for SAM2
SAM2.predict(point_coords=prev_centroid, point_labels=[1])
```

This recovers the lost rat's mask using SAM2 with a point instead of a box.
This is what makes the pipeline work during occlusions.

### Batch support

```bash
# Same chunk args as sam2_yolo
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0

# Or use run_parallel.sh (recommended):
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

### Weaknesses

- No SAM2 temporal memory → identity can flip with very fast movement
- proximity_threshold needs tuning per resolution
- Centroid fallback depends on previous centroid being close to current position

### When to use

- **Default choice** for most videos
- Videos with interactions where YOLO loses rats momentarily
- When you need batch processing + robustness
- Direct alternative to sam2_yolo when ID switches are a problem

---

## Component Comparison

### Models

| Component | sam2_yolo | sam2_video | reference |
|-----------|-----------|------------|-----------|
| YOLO | `model.track()` | `model()` detect-only | `model()` detect-only |
| SAM2 | `SAM2ImagePredictor` | `SAM2VideoPredictor` | `SAM2ImagePredictor` |
| SAM2 build | `build_sam2()` | `build_sam2_video_predictor()` | `build_sam2()` |

### Tracking / Identity

| Component | sam2_yolo | sam2_video | reference |
|-----------|-----------|------------|-----------|
| Class | `SlotTracker` | SAM2 internal | `IdentityMatcher` |
| File | `src/common/tracking.py` | N/A | `src/pipelines/reference/identity_matcher.py` |
| Identity based on | YOLO track IDs + Hungarian | SAM2 object pointers | Hungarian + soft cost + state machine |
| Uses YOLO track IDs | Yes (primary) | No | No |
| Missing frames | 5 frame tolerance | N/A (SAM2 propagates) | Centroid fallback + velocity coasting |
| Swap guard | Yes | N/A | Yes |

### Config Files

| Pipeline | Local | HPC | Slurm |
|----------|-------|-----|-------|
| sam2_yolo | `configs/local_quick.yaml` | `configs/hpc_full.yaml` | `slurm/run_infer.sbatch`, `slurm/run_chunks.sbatch` |
| sam2_video | `configs/local_sam2video.yaml` | `configs/hpc_sam2video.yaml` | `slurm/run_sam2video.sbatch` |
| reference | `configs/local_reference.yaml` | `configs/hpc_reference.yaml` | `slurm/run_reference.sbatch` |

### Shared Modules

All pipelines reuse these common modules:

| Module | File | Purpose |
|--------|------|---------|
| Video I/O | `src/common/io_video.py` | Read/write video, iterate frames |
| Config | `src/common/config_loader.py` | YAML + CLI overrides |
| Metrics | `src/common/metrics.py` | mask_iou, compute_centroid |
| Visualization | `src/common/visualization.py` | Overlays, keypoints, centroids |
| Contacts | `src/common/contacts.py` | ContactTracker with bout detection |
| Data classes | `src/common/utils.py` | Detection, Keypoint dataclasses |

---

## Quick Commands

### Local test (10s clip)

```bash
# sam2_yolo
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml

# sam2_video
python -m src.pipelines.sam2_video.run --config configs/local_sam2video.yaml

# reference (recommended)
python -m src.pipelines.reference.run --config configs/local_reference.yaml
```

### Bunya HPC (2 min video)

```bash
# reference (recommended)
bash scripts/run_parallel.sh data/raw/original_120s.avi

# sam2_yolo
sbatch --export=ALL,CONFIG=configs/hpc_full.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_infer.sbatch

# sam2_video
sbatch --export=ALL,CONFIG=configs/hpc_sam2video.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_sam2video.sbatch
```

### With contacts enabled

```bash
# Add contacts.enabled=true
python -m src.pipelines.reference.run --config configs/local_reference.yaml contacts.enabled=true

# Or on Bunya:
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_reference.yaml "contacts.enabled=true"
```
