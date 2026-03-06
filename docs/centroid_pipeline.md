# Centroid Pipeline — Technical Documentation

> Production pipeline for laboratory rat tracking and social contact classification.
> YOLO26 + SAM2 hybrid architecture with mask-anchored keypoints.

---

## Overview

The centroid pipeline tracks 2 laboratory rats in video using two AI models:

| Model | Role |
|-------|------|
| **YOLO26** | Detects each rat: bounding box + 7 anatomical keypoints |
| **SAM2** | Segments each rat: pixel-precise mask (silhouette) |

**Key principle**: SAM2 masks are the backbone — they drive identity and spatial tracking.
YOLO provides anatomical detail (keypoints) but its positions are stabilized against the masks.

---

## Architecture Diagram

```
Frame 0 (initialization):
  YOLO detect → 2 boxes → SAM2 segment (box prompts) → 2 masks → centroids

Frame 1..N (steady state):
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. YOLO detect (every frame)                                   │
  │    → 7 keypoints per rat (boxes ignored after init)            │
  │                                                                │
  │ 2. SAM2 segment (centroid prompts always)                      │
  │    centroid(+) + other centroid(-) → multimask → best score    │
  │                                                                │
  │ 3. Compute centroids from masks                                │
  │ 4. Resolve overlapping pixels (nearest centroid)               │
  │ 5. Assign YOLO keypoints to masks (spatial overlap)            │
  │ 6. Carry-over missing keypoints (centroid delta shift)         │
  │ 7. Stabilize keypoints against masks (EMA smoothing)           │
  │ 8. Contact classification (6 types)                            │
  │ 9. Render overlay video                                        │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Processing Steps (Detail)

### Step 1: YOLO Detection

```python
detections = detect_only(yolo, frame_rgb, confidence=0.25)
```

- Runs every frame on the full image
- Returns bounding boxes + 7 keypoints per rat
- **No tracking** (no BoT-SORT) — independent detection per frame
- Keypoints: `tail_tip`, `tail_base`, `tail_start`, `mid_body`, `nose`, `right_ear`, `left_ear`
- Used for: keypoints for contact classification (boxes ignored after init)

### Step 2: SAM2 Segmentation (Centroid Prompts)

Every frame uses the same prompting strategy — centroid points with positive/negative labels:

```python
SAM2.predict(
    point_coords = [[this_centroid], [other_centroid]],
    point_labels = [1, 0],           # positive, negative
    box = None,
    multimask_output = True,         # 3 candidates → pick best score
)
```

- Previous frame's centroid as positive prompt ("segment here")
- Other rat's centroid as negative prompt ("NOT here")
- `multimask_output=True` generates 3 candidate masks; best selected by confidence
- Works at all distances — negative prompts help SAM2 distinguish rats even when close
- `resolve_overlaps` handles any remaining mask bleeding between rats

#### Why centroid prompts only (no YOLO box prompts)?

An earlier version switched to YOLO box prompts when rats were close. This caused
**identity swaps** because YOLO detection order is arbitrary — detection[0] might be
either rat. When unordered YOLO boxes were fed as SAM2 prompts, the wrong box went
to the wrong slot, swapping masks and propagating the swap permanently.

SAM2 centroid prompts are inherently slot-stable: each slot's centroid propagates
from the previous frame, so identity is never disrupted by YOLO ordering.

### Step 3: Centroid Update

```python
centroid = compute_centroid(mask)  # center of mass of mask pixels
```

- Each mask produces a centroid (center of mass)
- If a mask is empty → keep previous frame's centroid (prevents drift)
- Centroids are the "identity anchor" — they propagate frame to frame

### Step 4: Overlap Resolution

```python
resolve_overlaps(slot_masks, slot_centroids)
```

- When two masks share pixels, each contested pixel goes to the nearest centroid
- Modifies masks in-place
- Ensures no pixel belongs to both rats

### Step 5: Keypoint Assignment

```python
slot_dets = _assign_keypoints_to_masks(detections, slot_masks, slot_centroids)
```

Two-pass assignment:
1. **Containment**: YOLO detection center inside a mask → assign to that mask's slot
2. **Fallback**: nearest centroid within 200px

This decouples YOLO detection order from identity — the mask determines which rat gets which keypoints.

### Step 6: Temporal Carry-Over

```python
slot_dets = _carry_over_keypoints(slot_dets, prev_slot_dets, prev_centroids, new_centroids)
```

- If YOLO misses a rat (no detection), copy previous frame's keypoints
- Shift positions by centroid delta: `new_pos = old_pos + (curr_centroid - prev_centroid)`
- Marked with `is_carried_over=True` → contacts get `quality_flag="stale_keypoints"`

### Step 7: Keypoint Stabilization (Mask-Anchored)

```python
slot_dets = _stabilize_keypoints(slot_dets, slot_masks, prev_slot_dets, prev_centroids, new_centroids)
```

**Problem**: YOLO detects each frame independently → keypoints jump frame-to-frame.
SAM2 masks are much more stable (centroid propagation).

**Solution**: Anchor keypoints to the mask:

| Keypoint position | Action |
|-------------------|--------|
| **Inside** mask | EMA smooth: `pos = 0.5 * yolo + 0.5 * (prev + delta)` |
| **Outside** mask | Reject YOLO, use: `pos = prev + centroid_delta` |

The EMA alpha (0.5) balances reactivity vs smoothness:
- Higher (→1.0) = more YOLO influence, faster reaction, more jitter
- Lower (→0.0) = more mask tracking, smoother, slower to react

### Step 8: Contact Classification

```python
contact_events = contact_tracker.update(slot_dets, slot_masks, slot_centroids, frame_idx)
```

6 social contact types with body-length normalized thresholds:

| Code | Contact Type | Detection Method |
|------|-------------|------------------|
| N2N | Nose-to-nose | Both noses within contact zone |
| N2AG | Nose-to-anogenital | Nose near other's tail base |
| T2T | Tail-to-tail | Both tail tips close |
| N2B | Nose-to-body | Nose near other's mid-body |
| FOL | Following | One rat trails another (speed + alignment) |
| SBS | Side-by-side | Parallel movement with mask overlap |

Contacts always run (no merge detection skip) — this is critical because social contacts
happen precisely when rats are close, which is when previous pipelines would skip analysis.

### Step 9: Video Overlay

Each frame renders:
- Semi-transparent masks (green = rat 0, red = rat 1)
- Keypoint dots on each rat
- Centroid markers
- Yellow line between centroids during active contacts
- Status bar: `Animals: 2/2 | Centroid | BOX (2 det, 2 assigned, 0 carried) | N2N`

---

## Identity Management

**No IdentityMatcher needed.** Identity is inherent from SAM2 centroid propagation:

- Frame 0: slot 0 = first YOLO detection (green), slot 1 = second (red)
- Frame 1+: each slot's centroid propagates → same mask region → same identity
- Keypoints are assigned to masks, not to YOLO detection order

This eliminates the complex SEPARATE/MERGED state machine from the reference pipeline.

---

## Configuration Reference

```yaml
# --- Input ---
video_path: data/clips/output-10s.mp4
output_dir: outputs/runs

# --- Models ---
models:
  yolo_path: models/yolo/best.pt
  sam2_checkpoint: models/sam2/.../sam2.1_hiera_tiny.pt  # tiny=local, large=HPC
  sam2_config: configs/sam2.1/sam2.1_hiera_t.yaml
  device: auto                    # auto, cuda, cpu

# --- YOLO Detection ---
detection:
  confidence: 0.25                # detection confidence threshold
  max_animals: 2                  # number of animals to track
  keypoint_names:                 # 7 anatomical landmarks
    - tail_tip
    - tail_base
    - tail_start
    - mid_body
    - nose
    - right_ear
    - left_ear
  keypoint_min_conf: 0.3          # minimum keypoint confidence
  filter_class: null              # filter by class (null = all)
  nms_iou: 0.5                   # NMS threshold (no-op with YOLO26)

# --- SAM2 Segmentation ---
segmentation:
  sam_threshold: 0.0              # mask binarization threshold

# --- Contact Classification ---
contacts:
  enabled: false                  # enable with contacts.enabled=true
  min_keypoint_conf: 0.3          # keypoint confidence for contacts
  contact_zone_bl: 0.3            # contact zone in body lengths
  proximity_zone_bl: 1.0          # proximity zone in body lengths
  fallback_body_length_px: 120    # body length estimate in pixels
  sbs_mask_iou_min: 0.02          # side-by-side mask overlap threshold
  sbs_max_velocity_bl: 0.04       # max velocity for SBS (BL/frame)
  sbs_parallel_cos_min: 0.7       # cosine similarity for parallel movement
  follow_radius_bl: 0.5           # following detection radius
  follow_min_speed_bl: 0.025      # minimum speed for following
  follow_alignment_cos: 0.7       # alignment threshold for following
  follow_min_frames: 30           # minimum frames for following bout
  bout_max_gap_frames: 3          # max gap to bridge between bouts
  bout_min_duration_frames: 9     # minimum bout duration
  mask_overlap_warning: 0.5       # warn if masks overlap this much
  det_slot_match_radius: 200.0    # max distance for keypoint assignment

# --- Output ---
output:
  video_codec: XVID
  overlay_colors:
    - [0, 255, 0, 150]            # Rat 0: Green
    - [255, 0, 0, 150]            # Rat 1: Red

# --- Limits ---
scan:
  max_frames: null                # null = process entire video
```

---

## Running

### Local (short clips)

```bash
# Without contacts
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml

# With contacts
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml contacts.enabled=true

# Custom video
python -m src.pipelines.centroid.run --config configs/local_centroid.yaml \
    video_path=data/clips/my_video.mp4 contacts.enabled=true
```

### HPC (Bunya — multi-GPU parallel)

```bash
ssh s4948012@bunya.rcc.uq.edu.au
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# Request 3 GPUs
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:3 --cpus-per-task=8 --mem=64G --time=24:00:00
srun --pty bash

# Run (splits video into 3 chunks, 1 per GPU, merges at end)
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

Results are saved to `outputs/runs/<timestamp>_centroid_merged/` and the script prints
an `scp` command for downloading.

---

## Output Files

```
outputs/runs/<timestamp>_centroid/
├── overlays/
│   └── centroid_2026-03-06.avi      # Overlay video with masks + keypoints
├── contacts/                         # (only if contacts.enabled=true)
│   ├── contacts_raw.csv              # Per-frame raw contact classifications
│   ├── contacts_real_events.csv      # Filtered event table
│   ├── session_summary_real.json     # Summary statistics
│   ├── event_log.txt                 # Chronological event log
│   ├── contacts_timeline.png         # Timeline chart
│   └── contacts_report.tar.gz       # All contact files compressed
└── logs/
    └── run.log                       # Full pipeline log
```

---

## Key Design Decisions

### Why SAM2 drives identity (not YOLO)
YOLO detects independently each frame — detection order can flip randomly.
SAM2 centroid propagation is stable: the same mask region stays in the same slot.

### Why YOLO runs every frame (not just init)
YOLO provides anatomical keypoints (nose, ears, tail) needed for contact classification.
Without keypoints, we can't distinguish N2N from N2AG.

### Why keypoints are stabilized against masks
YOLO keypoints jitter 5-15px frame-to-frame because each detection is independent.
SAM2 masks move smoothly. The EMA stabilization makes keypoints follow the mask
while still updating from YOLO when the detection lands inside the mask.

### Why centroid prompts always (no YOLO box prompts)
An earlier version switched to YOLO box prompts when rats were close (<100px).
This caused identity swaps because YOLO detection order is arbitrary — the wrong
box would go to the wrong slot, swapping masks permanently. SAM2 centroid prompts
with positive/negative labels + `resolve_overlaps` handle all distances reliably.

### Why no IdentityMatcher
The reference pipeline used a Hungarian assignment + SEPARATE/MERGED state machine.
The centroid pipeline doesn't need this because SAM2 propagation inherently maintains
identity. Removing it simplifies the code and eliminates a class of identity swap bugs.

---

## Lessons Learned — Identity Swap Prevention

This section documents the root causes of identity swaps we identified and fixed.
**Do not reintroduce YOLO box prompts or YOLO-ordered inputs to SAM2.**

### Key insight: SAM2 alone tracks identity perfectly

SAM2 centroid propagation (positive + negative point prompts) maintains stable identity
across all distances, including close interactions. This was validated by running SAM2
with centroid prompts only — no swaps occurred. The swaps were always introduced by
YOLO interfering with SAM2's prompting.

### Root cause: YOLO detection order is arbitrary

YOLO runs independently each frame with no tracking. The order of `detections[]` is
arbitrary — detection[0] might be rat A in one frame and rat B in the next. This is
fine when YOLO is only used for keypoints (assigned to masks by spatial overlap). But
it becomes catastrophic when YOLO ordering touches SAM2 identity:

| What went wrong | How it caused swaps |
|----------------|-------------------|
| YOLO boxes as SAM2 prompts (box prompt path) | Wrong box → wrong slot → mask swap → permanent |
| YOLO detection order assumed to match slots | detection[0] might be slot 1's rat |
| Switching prompting strategy based on proximity | Transition itself could introduce swap |

### What is safe (YOLO does NOT touch identity)

- YOLO keypoints assigned to SAM2 masks by spatial overlap (`_assign_keypoints_to_masks`)
- Temporal carry-over using centroid delta (not YOLO ordering)
- Keypoint stabilization against masks (EMA smoothing)
- Contact classification using slot-assigned keypoints

### What is NOT safe (never do these)

- Using YOLO boxes as SAM2 box prompts (detection order is random)
- Using YOLO detection index to determine slot assignment for SAM2
- Any path where YOLO ordering influences which mask goes to which slot
- Switching between prompting strategies mid-video (transition = swap risk)

### Design rule

**SAM2 centroid prompts are the sole source of identity.** YOLO provides keypoints only.
YOLO outputs must never be used in a way that can reorder, swap, or override SAM2 slots.
