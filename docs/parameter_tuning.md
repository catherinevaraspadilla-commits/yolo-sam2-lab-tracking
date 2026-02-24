# Parameter Tuning Guide — SAM2 + YOLO Pipeline

This document lists every tunable parameter in the pipeline, where it lives in code,
what it controls, and how to adjust it based on common error symptoms.

---

## 1. YOLO Parameters

### 1.1 `detection.confidence`

| | |
|---|---|
| **Config key** | `detection.confidence` |
| **Default** | `0.25` |
| **Range** | `0.0` – `1.0` |
| **Where in code** | `run.py` → `infer_yolo.detect_boxes(model, frame, confidence)` → Ultralytics `model(frame, conf=confidence)` |
| **What it does** | Minimum confidence score for YOLO to keep a detection. Boxes below this are discarded. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Missed detections (rat not detected) | **Lower** to 0.15–0.20 | `detection.confidence=0.15` |
| Too many false positives (phantom boxes) | **Raise** to 0.35–0.50 | `detection.confidence=0.4` |
| Boxes appear/disappear between frames (flicker) | **Lower** slightly to stabilize detections near the threshold | `detection.confidence=0.20` |
| Slow/partial rats at borders missed | **Lower** to 0.15 | `detection.confidence=0.15` |

**Trade-off:** Lower confidence = more detections but more noise. Higher = cleaner but may miss rats in difficult poses.

---

### 1.2 `detection.max_animals`

| | |
|---|---|
| **Config key** | `detection.max_animals` |
| **Default** | `2` |
| **Where in code** | `run.py` → `postprocess.py` → `tracking.filter_masks(masks, iou_threshold, max_count=max_animals)` |
| **What it does** | Caps the maximum number of masks kept after filtering. Also sets the number of tracking slots. |

**Tuning:**

| Symptom | Action |
|---------|--------|
| Third rat appears temporarily, confuses tracking | Keep at `2` (or however many rats are in the video) |
| One rat consistently missing | Check if it's filtered out — try raising to `3` temporarily to diagnose |
| Video has more animals | Set to actual animal count |

---

### 1.3 `detection.filter_class`

| | |
|---|---|
| **Config key** | `detection.filter_class` |
| **Default** | `null` (keep all classes) |
| **Where in code** | `infer_yolo.py` — `detect_boxes()` skips detections where `class_name != filter_class` |
| **What it does** | Only keeps YOLO detections of this class name. Rejects everything else. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Phantom boxes on non-rat objects | Set to `"ratas"` | `detection.filter_class=ratas` |
| All detections disappearing | Check class names in YOLO output match the filter | Set to `null` to debug |

---

### 1.4 `detection.yolo_border_padding_px`

| | |
|---|---|
| **Config key** | `detection.yolo_border_padding_px` |
| **Default** | `0` (disabled) |
| **Range** | `0` – `128` (recommended: 32 or 64) |
| **Where in code** | `infer_yolo.py` — `pad_frame()` + `correct_boxes_after_padding()` |
| **What it does** | Mirror-pads the frame before YOLO inference. Helps detect rats at borders. Boxes and keypoints are corrected back to original coordinates. SAM2 operates on the unpadded frame. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Rats at borders not detected or unstable | Set to 32–64 | `detection.yolo_border_padding_px=64` |
| False detections in padding area | **Lower** or disable | `detection.yolo_border_padding_px=0` |

**Trade-off:** Adds ~5-10% computation time for YOLO. Values >128 may cause hallucinated reflections.

---

### 1.5 `detection.keypoint_names`

| | |
|---|---|
| **Config key** | `detection.keypoint_names` |
| **Default** | `["tail_tip", "tail_base", "tail_start", "mid_body", "nose", "right_ear", "left_ear"]` |
| **Where in code** | `infer_yolo.py` — `DEFAULT_KEYPOINT_NAMES`, passed to `detect_boxes()` |
| **What it does** | Assigns human-readable names to each keypoint index. Order must match training annotation order. |

**Tuning:** Only change if you retrain the model with different/reordered keypoints.

---

### 1.6 `detection.keypoint_min_conf`

| | |
|---|---|
| **Config key** | `detection.keypoint_min_conf` |
| **Default** | `0.3` |
| **Range** | `0.0` – `1.0` |
| **Where in code** | `run.py` → `visualization.draw_keypoints(frame, detections, min_conf=kpt_min_conf)` |
| **What it does** | Minimum confidence to render a keypoint on the overlay. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Keypoints jumping to wrong positions | **Raise** to 0.5–0.7 | `detection.keypoint_min_conf=0.5` |
| Missing keypoints that should be visible | **Lower** to 0.1–0.2 | `detection.keypoint_min_conf=0.15` |
| Want to see all keypoints for debugging | Set to `0.0` | `detection.keypoint_min_conf=0.0` |

---

### 1.7 YOLO NMS (Non-Maximum Suppression)

| | |
|---|---|
| **Config key** | Not currently exposed in config |
| **Default** | Ultralytics internal default (`iou=0.7`) |
| **Where in code** | `model(frame, conf=..., iou=0.5)` in `infer_yolo.py` |
| **What it does** | Controls how aggressively YOLO merges overlapping boxes. |

**To expose:** Add `detection.nms_iou` to config, pass as `model(frame, conf=confidence, iou=nms_iou)`.

| Symptom | Action |
|---------|--------|
| Two boxes for the same rat | Lower NMS IoU to 0.4–0.5 |
| Two overlapping rats merged into one box | Raise NMS IoU to 0.8–0.9 |

---

## 2. SAM2 Parameters

### 2.1 `models.sam2_checkpoint` + `models.sam2_config`

| | |
|---|---|
| **Config keys** | `models.sam2_checkpoint`, `models.sam2_config` |
| **Where in code** | `models_io.py` → `build_sam2(config_name, checkpoint_path, device=device)` |
| **What it does** | Selects the SAM2 model size. Larger = better mask quality but slower. |

**Available sizes:**

| Size | Checkpoint | Config | Speed | Quality | VRAM |
|------|-----------|--------|-------|---------|------|
| Tiny | `sam2.1_hiera_tiny.pt` | `sam2.1_hiera_t.yaml` | Fastest | Good enough for testing | ~2 GB |
| Small | `sam2.1_hiera_small.pt` | `sam2.1_hiera_s.yaml` | Fast | Better | ~3 GB |
| Base+ | `sam2.1_hiera_base_plus.pt` | `sam2.1_hiera_b+.yaml` | Medium | Good | ~5 GB |
| Large | `sam2.1_hiera_large.pt` | `sam2.1_hiera_l.yaml` | Slowest | Best | ~8 GB |

**Important:** Checkpoint and config must match! Mixing sizes will crash.

---

### 2.2 `segmentation.sam_threshold`

| | |
|---|---|
| **Config key** | `segmentation.sam_threshold` |
| **Default** | `0.0` |
| **Range** | Typically `-2.0` to `+2.0` (raw logits) |
| **Where in code** | `infer_sam2.py` → `m = raw_masks[0] > sam_threshold` |
| **What it does** | Threshold applied to SAM2's raw mask logits. Higher = tighter masks. Lower = larger masks. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Masks too large / bleed into background | **Raise** to 0.5–2.0 | `segmentation.sam_threshold=1.0` |
| Masks too small / don't cover the whole rat | **Lower** to -1.0 or -2.0 | `segmentation.sam_threshold=-1.0` |

---

### 2.3 `segmentation.mask_iou_threshold`

| | |
|---|---|
| **Config key** | `segmentation.mask_iou_threshold` |
| **Default** | `0.5` |
| **Range** | `0.0` – `1.0` |
| **Where in code** | `postprocess.py` → `tracking.filter_masks(masks, iou_threshold=mask_iou_thr)` |
| **What it does** | When two masks overlap more than this, the smaller is removed as duplicate. Only applied when num_masks > max_animals. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Two masks for the same rat (double-masking) | **Lower** to 0.3 | `segmentation.mask_iou_threshold=0.3` |
| Two overlapping rats lose one mask | **Raise** to 0.7–0.8 | `segmentation.mask_iou_threshold=0.7` |

---

## 3. Tracking Parameters

### 3.1 `tracking.max_centroid_distance`

| | |
|---|---|
| **Config key** | `tracking.max_centroid_distance` |
| **Default** | `150.0` (pixels) |
| **Range** | `50.0` – `500.0` |
| **Where in code** | `tracking.py` → `FixedSlotTracker.__init__(max_distance=...)` |
| **What it does** | Gate 1: maximum pixel distance a centroid can move between frames for a valid match. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Colors swap between rats (ID switching) | **Lower** to 80–100 | `tracking.max_centroid_distance=100` |
| Fast-moving rat loses identity constantly | **Raise** to 200–300 | `tracking.max_centroid_distance=250` |

**Note:** Resolution-dependent. 150px at 640x480 is ~23% of frame width.

---

### 3.2 `tracking.area_change_threshold`

| | |
|---|---|
| **Config key** | `tracking.area_change_threshold` |
| **Default** | `0.4` |
| **Range** | `0.0` – `1.0` |
| **Where in code** | `tracking.py` → `FixedSlotTracker.update()` — `area_ratio = abs(curr - prev) / prev` |
| **What it does** | Gate 2: rejects a match if the mask area changed by more than this fraction. 0.4 = +/-40% change tolerated. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| ID swaps when rat partially enters/leaves frame | **Raise** to 0.5–0.6 (more permissive) | `tracking.area_change_threshold=0.6` |
| Rats swap identities when similar size | **Lower** to 0.2–0.3 (stricter) | `tracking.area_change_threshold=0.25` |

---

### 3.3 `tracking.max_missing_frames`

| | |
|---|---|
| **Config key** | `tracking.max_missing_frames` |
| **Default** | `5` |
| **Range** | `1` – `30` |
| **Where in code** | `tracking.py` → `FixedSlotTracker.update()` — releases slot when `missing_count > max_missing_frames` |
| **What it does** | Number of consecutive frames a slot can go without a match before its identity is released. Prevents color swaps from brief detection gaps at borders. |

**Tuning:**

| Symptom | Action | Example |
|---------|--------|---------|
| Color flickers when rat touches border | **Raise** to 8–10 | `tracking.max_missing_frames=10` |
| Ghost identity sticks around too long after rat leaves | **Lower** to 2–3 | `tracking.max_missing_frames=2` |

**Trade-off:** Higher = more robust to gaps but slower to free up stale slots.

---

### 3.4 `tracking.use_mask_iou`

| | |
|---|---|
| **Config key** | `tracking.use_mask_iou` |
| **Default** | `true` |
| **Where in code** | `tracking.py` → `FixedSlotTracker.update()` — ranks candidates by mask IoU when enabled |
| **What it does** | When multiple mask candidates pass both gates (distance + area), rank them by mask IoU with the previous frame's mask instead of just using nearest distance. |

**Tuning:**

| Symptom | Action |
|---------|--------|
| IDs swap when centroids are close but shapes differ | Keep `true` |
| Need maximum speed | Set `false` to skip IoU computation |

---

## 4. Device & Performance Parameters

### 4.1 `models.device`

| | |
|---|---|
| **Config key** | `models.device` |
| **Options** | `"auto"`, `"cuda"`, `"cpu"` |

See [running_hpc_bunya.md](running_hpc_bunya.md) for detailed GPU setup.

### 4.2 `scan.max_frames`

| | |
|---|---|
| **Config key** | `scan.max_frames` |
| **Default** | `150` (local), `null` (HPC) |
| **What it does** | Limits how many frames are processed. `null` = entire video. |

---

## 5. Common Problems → What to Tune

### Problem: Color flicker / ID switching near frame borders

**Cause:** Rats at frame borders → unstable YOLO detections → centroid jumps → tracker reassigns identity.

**Knobs (try in order):**
1. `detection.yolo_border_padding_px=64` — stabilize YOLO detections at edges
2. `tracking.max_missing_frames=8` — hold slot through brief gaps
3. `detection.confidence=0.20` — avoid detections appearing/disappearing
4. `tracking.area_change_threshold=0.5` — tolerate area changes from partial views
5. `tracking.use_mask_iou=true` — better matching when centroids jump

---

### Problem: Missed detections (rat not detected in some frames)

1. `detection.confidence=0.15` — lower threshold
2. `detection.yolo_border_padding_px=64` — helps at borders
3. Check if rat is partially occluded — model limitation
4. If consistently missed: more training data needed

---

### Problem: Too many false positives

1. `detection.confidence=0.4` — raise threshold
2. `detection.filter_class=ratas` — reject non-rat classes
3. `detection.max_animals=2` — filter keeps only top N by area

---

### Problem: Track fragmentation (new IDs every few frames)

1. `tracking.max_missing_frames=8` — tolerate gaps
2. `tracking.max_centroid_distance=200` — tolerate bigger jumps
3. `detection.confidence=0.15` — ensure continuous detections

---

### Problem: Two rats merged during close contact

1. `segmentation.mask_iou_threshold=0.8` — keep both overlapping masks
2. `tracking.max_centroid_distance=100` — prevent cross-matching
3. `tracking.use_mask_iou=true` — distinguish by shape
4. Try larger SAM2 model for better mask separation

---

### Problem: Slow runtime / GPU memory issues

| Knob | Effect |
|------|--------|
| SAM2 tiny instead of large | 3-5x faster, less VRAM |
| `scan.max_frames=300` | Process fewer frames |
| `models.device=cuda` | 10-50x faster than CPU for SAM2 |
| `tracking.use_mask_iou=false` | Skip IoU computation |
| `detection.yolo_border_padding_px=0` | Skip padding overhead |

---

## 6. Quick Reference — All Config Keys

```yaml
# Models
models.device                       # "auto" | "cuda" | "cpu"
models.yolo_path                    # Path to YOLO .pt weights
models.sam2_checkpoint              # Path to SAM2 checkpoint
models.sam2_config                  # SAM2 Hydra config name

# YOLO detection
detection.confidence                # 0.0–1.0 (default: 0.25)
detection.max_animals               # integer (default: 2)
detection.filter_class              # string or null (default: null)
detection.yolo_border_padding_px    # integer (default: 0)
detection.keypoint_names            # list of 7 strings
detection.keypoint_min_conf         # 0.0–1.0 (default: 0.3)

# SAM2 segmentation
segmentation.sam_threshold          # float (default: 0.0)
segmentation.mask_iou_threshold     # 0.0–1.0 (default: 0.5)

# Tracking (FixedSlotTracker)
tracking.max_centroid_distance      # pixels (default: 150.0)
tracking.area_change_threshold      # 0.0–1.0 (default: 0.4)
tracking.max_missing_frames         # integer (default: 5)
tracking.use_mask_iou               # true | false (default: true)

# Scan limits
scan.max_frames                     # integer or null

# Video output
output.video_codec                  # "XVID" | "avc1" | "mp4v"
output.overlay_colors               # list of [R, G, B, A]

# Not yet in config (requires code change):
# YOLO NMS IoU threshold (internal default: 0.7)
# YOLO input image size (internal default: 640)
# Frame skip / stride
```
