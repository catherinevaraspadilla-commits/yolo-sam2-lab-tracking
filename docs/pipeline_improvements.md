# Pipeline Architecture

This document describes the final pipeline architecture, what each component does,
why it reduces ID switching / color flicker, and how to tune parameters.

---

## Pipeline Flow

```
Frame (RGB)
  │
  ├─ [Optional] Mirror-pad borders (yolo_border_padding_px)
  │
  ▼
YOLO model.track() (BoT-SORT / ByteTrack)
  │  → boxes + keypoints + class names + TRACK IDs
  │  → filter by class (filter_class)
  │  → correct coordinates if padded
  │  BoT-SORT provides: Kalman prediction, motion compensation,
  │  track buffer (survives lost detections), optional ReID
  │
  ▼
SAM2 Segmentation (per box, on original unpadded frame)
  │  → predictor.set_image(frame)
  │  → predict(box=..., multimask_output=False)
  │  → mask = masks[0] > sam_threshold
  │
  ▼
Smart Duplicate Filter (only if num_masks > max_animals)
  │  → sort by area (largest first)
  │  → keep if IoU with all kept < mask_iou_threshold
  │
  ▼
Slot Tracker (Hungarian assignment + soft costs)
  │  → N fixed color slots (slot 0 = green, slot 1 = red, ...)
  │
  │  Step 1: Match by YOLO track ID (O(1) lookup)
  │    If track ID already assigned to a slot → instant match
  │
  │  Step 2: Hungarian assignment for unmatched masks ↔ slots
  │    Cost matrix = w_dist × centroid_dist + w_iou × (1-mask_iou) + w_area × area_change
  │    scipy.optimize.linear_sum_assignment → globally optimal
  │    Area change is SOFT cost (not hard veto)
  │    Only veto if area ratio > 5x (truly pathological)
  │
  │  Step 3: Unmatched masks → first free slot
  │  Step 4: Unmatched slots → missing_count++
  │    Release slot after max_missing_frames consecutive misses
  │
  ▼
Render Overlay (persistent buffer)
  │  → colored masks (by slot index)
  │  → bounding boxes + labels + confidence
  │  → keypoints (7-point rat pose)
  │  → centroids
  │
  ▼
Output Video (.avi)
```

---

## What Changed (vs. Previous Version) and Why

### 1. Replaced `model()` with `model.track()` (BoT-SORT)

**Before:** `model(frame, conf=...)` — per-frame detection, no temporal state.
**After:** `model.track(frame, persist=True, tracker="botsort.yaml")` — BoT-SORT
maintains a Kalman-filter state for each tracked object across frames.

**Why this helps:**
- BoT-SORT predicts where each object will be in the next frame. Even if YOLO
  misses a detection for 1-2 frames, the Kalman filter keeps the track alive
  (controlled by `track_buffer` in the tracker YAML, default=30 frames).
- The track ID assignment uses motion prediction, not just nearest-neighbor,
  so it's more robust to crossing paths and close contact.
- Global Motion Compensation (sparseOptFlow) handles any camera vibration.

**Where:** [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py) — `detect_and_track()`.

---

### 2. Replaced greedy matching with Hungarian assignment

**Before:** For each slot, greedily pick the nearest mask that passes distance
+ area gates. This is sequential — slot 0 picks first, slot 1 gets leftovers.
With 2 objects, if slot 0 grabs the wrong mask, slot 1 is forced to swap.

**After:** Build a cost matrix (masks × slots) and solve with
`scipy.optimize.linear_sum_assignment()` for the globally optimal assignment
that minimizes total cost.

**Why this helps:**
For 2 objects, the Hungarian algorithm compares both possible pairings:
- `cost(mask_A→slot_0) + cost(mask_B→slot_1)` vs
- `cost(mask_A→slot_1) + cost(mask_B→slot_0)`
and picks whichever is cheaper overall. This eliminates the cascading
errors of greedy matching during crossings.

**Where:** [tracking.py](../src/common/tracking.py) — `SlotTracker._hungarian_assign()`.

---

### 3. Removed hard area gating (replaced with soft cost)

**Before:** `area_change_threshold: 0.4` — if a mask's area changed by >40%
from the previous frame, the match was **vetoed** (hard reject). Near borders
or during contact, legitimate area changes exceed 40% (partial cropping,
occlusion), causing the correct match to be rejected → the other rat fills
the slot → color swap.

**After:** Area change is a **soft cost** added to the assignment matrix with
weight `w_area=0.2`. Large area changes make the match more expensive but
NOT impossible. Only truly pathological cases (area ratio > 5x) are vetoed.

**Why this helps:**
The Hungarian solver can still pick the correct assignment even when area
changes are large, as long as the centroid distance and mask IoU costs
favor the correct pairing. In practice, this eliminates the most common
cause of ID switches at frame borders.

**Where:** [tracking.py](../src/common/tracking.py) — `SlotTracker._compute_cost()`.

---

### 4. YOLO track ID → slot mapping (primary identity)

**Before:** Identity came entirely from centroid matching between frames.
**After:** YOLO track IDs (from BoT-SORT) serve as the primary identity signal.
Each YOLO track ID is mapped to a slot index via `_track_to_slot` dict.
When a mask arrives with a known track ID, it's instantly assigned to the
correct slot without any cost computation.

Hungarian assignment is only used as fallback for:
- New tracks not yet in the mapping
- Track IDs that change (rare with BoT-SORT but can happen)

**Where:** [tracking.py](../src/common/tracking.py) — `SlotTracker._match_by_track_id()`.

---

## BoT-SORT Tracker Configuration

The Ultralytics BoT-SORT tracker has its own config. Default values work
well for most cases. Key parameters you might tune:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `track_buffer` | 30 | Frames to keep a lost track alive. Raise to 60-120 for slow animals. |
| `track_high_thresh` | 0.25 | Min confidence for first-pass matching. |
| `track_low_thresh` | 0.1 | Min confidence for second-pass (low-conf) matching. |
| `match_thresh` | 0.8 | Max cost for IoU-based association. |
| `with_reid` | false | Enable appearance ReID. Adds compute but helps after occlusion. |
| `gmc_method` | sparseOptFlow | Global Motion Compensation. |

To customize, create a YAML file (e.g., `configs/trackers/custom_botsort.yaml`):

```yaml
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 60          # raised for slow-moving rats
match_thresh: 0.8
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: false
```

Then reference it in your config:
```yaml
tracking:
  tracker_config: configs/trackers/custom_botsort.yaml
```

Or via CLI: `tracking.tracker_config=configs/trackers/custom_botsort.yaml`

---

## Parameter Recommendations

### Default settings (start here)

```yaml
detection:
  confidence: 0.25
  filter_class: null
  yolo_border_padding_px: 0

tracking:
  tracker_config: botsort.yaml
  max_centroid_distance: 150.0
  max_missing_frames: 5
  w_dist: 0.4
  w_iou: 0.4
  w_area: 0.2
  cost_threshold: 0.85
```

### If you see border flicker

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    detection.yolo_border_padding_px=64 \
    detection.confidence=0.20 \
    tracking.max_missing_frames=10
```

### If you see ID swaps during close contact

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    tracking.w_iou=0.5 \
    tracking.w_dist=0.3 \
    tracking.w_area=0.2 \
    tracking.max_centroid_distance=100
```

This prioritizes mask shape (IoU) over centroid distance, which helps when
centroids are close but mask shapes are distinguishable.

### If detections are noisy

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    detection.confidence=0.40 \
    detection.filter_class=ratas
```

---

## Debug Signals

Every 50 frames, the log shows slot status:

```
Processed 100 frames | slot0:tid=1(miss=0) | slot1:tid=2(miss=0)
```

| Signal | Meaning | Action |
|--------|---------|--------|
| `tid=1(miss=0)` | Track 1 assigned to slot 0, matched this frame | Stable |
| `tid=2(miss=3)` | Track 2 not matched for 3 frames | Check if near border |
| `slot1:free` | Slot released after max_missing exceeded | Rat left frame or tracking lost |
| `tid` changes (e.g., tid=1 → tid=3) | BoT-SORT created a new track | Check `track_buffer` in tracker config |

---

## Known Risks & Limitations

1. **BoT-SORT ID switches during prolonged occlusion:** If two rats overlap
   for many frames and then separate, BoT-SORT may assign new track IDs.
   The slot tracker handles this via Hungarian fallback, but colors may swap.
   **Mitigation:** Raise `track_buffer` in BoT-SORT config to 60-120.

2. **Border padding + large values:** Padding >128px may create "ghost"
   reflected detections. Stick to 32-64px.

3. **Soft area cost with extreme partial views:** If a rat is 90% off-screen,
   the area change is huge. The soft cost will be high but the 5x hard veto
   threshold will still allow it. Combined with mask IoU cost, the assignment
   should still be correct.

4. **Cost weight tuning:** The defaults (0.4/0.4/0.2) work for lab videos.
   For very different scenarios (different resolution, speed, occlusion patterns),
   you may need to re-tune. Use the CLI overrides for quick experiments.

5. **No cross-gap re-identification:** If a slot is released and re-assigned,
   the new occupant gets the same color. There's no appearance-based re-ID
   across long gaps (could enable `with_reid: true` in BoT-SORT but this
   adds compute and may not help for visually identical rats).

---

## Future Improvements

- **YOLO-only mode:** Skip SAM2 for fast experiments. Use YOLO boxes for tracking.
- **Frame skip / stride:** Process every Nth frame for speed.
- **Temporal smoothing:** EMA on box/keypoint coordinates.
- **SAM2 VideoPredictor:** Use temporal mask propagation for even more stable
  mask tracking (major architectural change).
- **YOLO 26:** Upgrade model from YOLOv8 to YOLO26 for better detections.
  Retrain on same dataset. YOLO26 supports NMS-free end-to-end inference.
