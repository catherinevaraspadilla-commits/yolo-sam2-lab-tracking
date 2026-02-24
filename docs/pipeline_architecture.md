# Pipeline Architecture — Final Design

This document describes the current pipeline architecture, what each technique
does, why it reduces flicker at borders, and how to tune each parameter.

---

## Pipeline Flow

```
Frame (RGB)
  │
  ├─ [Optional] Mirror-pad borders (yolo_border_padding_px)
  │
  ▼
YOLO Pose Detection
  │  → boxes + keypoints + class names
  │  → filter by class (filter_class)
  │  → correct box coordinates (subtract padding, clamp to frame)
  │
  ▼
SAM2 Segmentation (per box, on original unpadded frame)
  │  → predictor.set_image(original frame)
  │  → predict(box=..., multimask_output=False)
  │  → mask = masks[0] > sam_threshold
  │
  ▼
Smart Duplicate Filter (only if num_masks > max_animals)
  │  → sort by area (largest first)
  │  → keep mask if IoU with all kept masks < mask_iou_threshold
  │  → stop at max_animals masks
  │
  ▼
Fixed-Slot Tracker (FixedSlotTracker)
  │  → N fixed slots (slot 0 = green, slot 1 = red, ...)
  │  → match current masks to slots using:
  │      gate 1: centroid distance < max_centroid_distance
  │      gate 2: area change < area_change_threshold
  │      rank by: mask IoU with previous frame (if use_mask_iou=True)
  │  → unmatched slots: increment missing_count
  │  → release slot only after max_missing_frames consecutive misses
  │  → new masks → assign to first free slot
  │
  ▼
Render Overlay (persistent buffer)
  │  → masks (colored by slot index)
  │  → bounding boxes + labels
  │  → keypoints
  │  → centroids
  │  → status text
  │
  ▼
Output Video (.avi)
```

---

## Technique Details

### A. YOLO Class Filter (`detection.filter_class`)

**What:** Only keeps YOLO detections whose `class_name` matches the filter.

**Why:** Prevents non-rat detections (e.g., a hand, cage edge artifact) from
entering the pipeline and wasting a tracking slot or confusing SAM2.

**Where in code:** [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py) —
`detect_boxes()` checks `if filter_class is not None and class_name != filter_class: continue`.

**Config:**
```yaml
detection:
  filter_class: "ratas"   # or null to keep all
```

**CLI override:** `detection.filter_class=ratas`

---

### B. Border Padding (`detection.yolo_border_padding_px`)

**What:** Before running YOLO, the frame is mirror-padded by N pixels on
all sides (`cv2.copyMakeBorder` with `BORDER_REFLECT_101`). YOLO sees a
larger frame where rats at the edge are "completed" by their reflection.
After detection, box coordinates are shifted back and clamped to the
original frame boundaries. Keypoint coordinates are also corrected.

**Why:** Rats partially outside the frame produce unstable YOLO boxes
(position jumps, confidence fluctuates). With padding, YOLO sees a
"complete" object and produces more stable boxes. SAM2 still operates
on the original (unpadded) frame using the corrected boxes.

**Where in code:** [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py) —
`pad_frame()` and `correct_boxes_after_padding()`.

**Config:**
```yaml
detection:
  yolo_border_padding_px: 0    # 0=off, 32 or 64 recommended for border issues
```

**CLI override:** `detection.yolo_border_padding_px=64`

**Trade-offs:**
- Small cost: YOLO processes a slightly larger image (adds 2*N pixels per dimension)
- Padding too large (>128) may introduce hallucinated reflections of the rat

---

### C. Smart Duplicate Filter

**What:** If YOLO produces more than `max_animals` detections (e.g., 3 boxes
for 2 rats), SAM2 will generate 3 masks. The filter removes duplicates:

1. If `num_masks <= max_animals`: **no filtering** — return all masks as-is.
2. Sort masks by area (largest first).
3. For each mask, check IoU against all already-kept masks.
4. If IoU > `mask_iou_threshold` with any kept mask, discard as duplicate.
5. Stop when `max_animals` masks are kept.

**Why:** Without this, the tracker may receive 3+ masks and waste slots.
The "skip if <= max_animals" rule avoids incorrectly removing a valid mask
when exactly 2 are detected.

**Where in code:** [tracking.py](../src/common/tracking.py) — `filter_masks()`.

**Config:**
```yaml
segmentation:
  mask_iou_threshold: 0.5    # raise to 0.7-0.8 if rats often overlap
detection:
  max_animals: 2             # number of rats in the video
```

---

### D. Fixed-Slot Tracker (`FixedSlotTracker`)

**What:** Maintains N persistent "slots," each representing one animal identity.
Slot 0 always renders in color 0 (green), slot 1 in color 1 (red), etc.
The old centroid-only tracker is replaced by this more robust system.

**Where in code:** [tracking.py](../src/common/tracking.py) — `FixedSlotTracker` class.
Created in [postprocess.py](../src/pipelines/sam2_yolo/postprocess.py) — `create_tracker()`.

**Matching algorithm (per frame):**

1. Compute centroid and area for each current mask.
2. Build distance matrix: current masks × active slots.
3. For each active slot, try to find the best matching mask:
   - **Gate 1 (distance):** `centroid_dist < max_centroid_distance` — skip if too far.
   - **Gate 2 (area):** `|area_current - area_prev| / area_prev < area_change_threshold` — skip if size changed too much.
   - **Ranking:** If `use_mask_iou=True` and a previous mask is stored, all candidates
     passing both gates are ranked by mask IoU (highest wins). Otherwise, nearest by
     centroid distance wins.
4. **Unmatched slots:** `missing_count += 1`. Slot keeps its identity (color) until
   `missing_count > max_missing_frames`, then it is released.
5. **Unmatched masks:** assigned to the first free (released) slot.

**Why this reduces flicker:**

| Problem | How fixed slots help |
|---------|---------------------|
| YOLO misses a rat for 1-2 frames at border | Slot stays active (missing tolerance), same color when rat reappears |
| Centroid jumps because box was partially cropped | Area gate rejects the false match, mask IoU ranks the correct match higher |
| Two detections swap order in YOLO output | Tracker matches by position/area/IoU, not by YOLO array order |
| Rat fully leaves frame and re-enters | After `max_missing_frames`, slot is released; re-entering rat gets the first free slot |

**Config:**
```yaml
tracking:
  max_centroid_distance: 150.0    # pixels — depends on resolution
  area_change_threshold: 0.4      # 0.4 = ±40% area change tolerated
  max_missing_frames: 5           # frames before releasing a slot
  use_mask_iou: true              # rank by mask IoU when available
```

**CLI overrides:**
```bash
tracking.max_centroid_distance=100 tracking.area_change_threshold=0.3 tracking.max_missing_frames=8
```

---

### E. Render Buffer Fix

**What:** The render pipeline uses a persistent `np.ndarray` buffer
(`frame_buffer`) allocated once at video start. Each frame copies RGB data
into this buffer instead of creating new arrays per `draw_*` call.

**Why:** Reduces memory allocation pressure and GC pauses, especially
for long videos (thousands of frames).

**Where in code:** [run.py](../src/pipelines/sam2_yolo/run.py) — `frame_buffer = np.empty(...)`,
then `np.copyto(frame_buffer, frame_rgb)` per frame.

---

## Parameter Recommendations

### Default settings (start here)

```yaml
detection:
  confidence: 0.25
  filter_class: null
  yolo_border_padding_px: 0

segmentation:
  sam_threshold: 0.0
  mask_iou_threshold: 0.5

tracking:
  max_centroid_distance: 150.0
  area_change_threshold: 0.4
  max_missing_frames: 5
  use_mask_iou: true
```

### If you see border flicker

```yaml
detection:
  yolo_border_padding_px: 64    # try 32 first, then 64
  confidence: 0.20              # slightly lower to avoid threshold flickering

tracking:
  max_missing_frames: 5         # or raise to 8-10 for very choppy borders
  area_change_threshold: 0.5    # slightly more permissive for partial views
```

### If you see ID swaps during close contact

```yaml
tracking:
  max_centroid_distance: 100.0  # lower to prevent cross-matching
  use_mask_iou: true            # helps distinguish overlapping rats
  area_change_threshold: 0.3    # stricter — rats shouldn't change size much

segmentation:
  mask_iou_threshold: 0.7       # higher to keep both masks even when overlapping
```

### If detections are noisy (phantom boxes)

```yaml
detection:
  confidence: 0.40              # stricter
  filter_class: "ratas"         # reject non-rat classes
```

---

## Debug Signals

The pipeline logs tracking slot status every 50 frames:

```
Processed 100 frames | slot0:active(miss=0,area=12340) | slot1:active(miss=2,area=9876)
```

**What to watch for:**

| Signal | Meaning | Action |
|--------|---------|--------|
| `miss=0` on both slots | Stable tracking | All good |
| `miss` oscillating 0→2→0 | Rat briefly lost, but slot held | Normal for border movement |
| `miss` reaching `max_missing_frames` | Slot about to be released | Raise `max_missing_frames` or lower `confidence` |
| Both slots show same `area` | Possible merge | Check `mask_iou_threshold` |
| `slot0:free` after being active | Slot was released | Check if rat left frame or was lost |

---

## Known Risks & Limitations

1. **Stale slots:** If `max_missing_frames` is too high, a slot may hold
   a "ghost" identity for too long after the animal truly left the frame.
   The next new detection fills this stale slot instead of creating fresh.
   **Mitigation:** keep `max_missing_frames` at 3-5.

2. **Area gate too strict for partial views:** A rat entering/leaving the
   frame has rapidly changing mask area. The area gate may reject valid
   matches. **Mitigation:** raise `area_change_threshold` to 0.5-0.6 or
   combine with border padding.

3. **Mask IoU ranking adds compute:** Storing previous masks and computing
   IoU adds overhead. For very high-res frames, this may be noticeable.
   Disable with `use_mask_iou: false` if speed is critical.

4. **Border padding + large padding values:** Very large padding (>128px)
   may cause YOLO to detect "mirrored rats" in the padding region.
   Stick to 32-64px.

5. **Single-class model assumption:** `filter_class` only works if the
   YOLO model outputs class names. With a single-class model (like ours),
   all detections are the same class, so `filter_class` is optional.

6. **No re-identification after release:** Once a slot is released and
   re-assigned, the new occupant gets the same color as the old one.
   There is no cross-gap re-identification.

7. **Two rats very close + both at border:** The combination of
   overlapping centroids and partial views may still cause brief swaps.
   Border padding + mask IoU ranking + stricter area gate is the best
   defense, but not perfect.

---

## Future Improvements (Not Yet Implemented)

These are potential additions that are **not currently in the code**:

- **YOLO-only mode:** Skip SAM2 entirely for fast experiments. Track using
  YOLO box centers instead of mask centroids. Add `pipeline.mode: "full" | "yolo_only"`.

- **Frame skip / stride:** Process every Nth frame. Add `scan.frame_stride: 1`.

- **Temporal smoothing:** EMA on box coordinates and keypoints to reduce
  visual jitter. Add `tracking.temporal_smoothing: 0.0`.

- **SAM2 video predictor:** Use `SAM2VideoPredictor` for temporal mask
  propagation with built-in track IDs, removing the need for our
  centroid-based tracker entirely. Major architectural change.
