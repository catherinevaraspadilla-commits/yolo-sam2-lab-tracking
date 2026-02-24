# Reference Codebase Comparison

Comparison of our SAM2+YOLO pipeline against the reference `yolov8_sam2_toolkit`
documented in `reference/`. Focus on techniques that address our two main failure
modes: **close-contact ID switches** and **wall/border detection instability**.

---

## 1. Architecture Differences

| Aspect | Our pipeline | Reference toolkit |
|--------|-------------|-------------------|
| Pipeline pattern | Sequential function calls in a loop | Data Bus: shared `frame_data` dict passed through processor chain |
| YOLO inference | `model.track()` with BoT-SORT (temporal state) | `model()` per-frame (stateless detection) |
| Identity source | YOLO track IDs (primary) + Hungarian assignment (fallback) | SAM2-side `IdentityMatcher` with distance + area gating |
| Tracking location | `SlotTracker` in `src/common/tracking.py` | `IdentityMatcher` in `tracking/identity_matcher.py` |
| Mask dedup | IoU-based, sort by area, keep top N | Same algorithm (IoU-based, sort by area, keep top N) |
| YOLO filtering | Class name filter + border padding | 7-step filter chain: confidence, class, area (fixed+adaptive), edge margin, ROI, custom NMS, entity count |
| Config system | YAML + CLI overrides | Constructor parameters (code-level) |

**Key takeaway:** The reference has **much richer YOLO-side filtering** that could
directly help with our border and crowding issues, but uses simpler (greedy)
tracking. Our tracking is better (Hungarian + BoT-SORT), but our YOLO filtering
is minimal.

---

## 2. What They Do Differently (and Why It Matters)

### 2.1 Adaptive area filtering (reference: `area` + `area_error`)

**Reference approach:**
```python
YOLOProcessor(area=None, area_error=0.2)
# First frame: learns the "reference area" from the first valid detection
# All subsequent frames: rejects boxes where area deviates >20% from reference
```

**Why this helps us:**
- When mice touch the wall, YOLO sometimes produces a **partial bounding box**
  (half the normal size) or a **wall artifact** (abnormally large box covering
  the wall region). Both deviate heavily from the expected animal area.
- Adaptive area filtering catches these at the YOLO level, **before** they reach
  SAM2 and the tracker. This prevents the ghost masks that cause ID confusion.

**Our current approach:** We have no YOLO-level area filtering. Area is only
used as a soft cost in Hungarian assignment (weight `w_area=0.2`), which means
bad detections still produce SAM2 masks that then need to be resolved by the
tracker.

**Risk:** A static ±20% tolerance is too tight for our videos where mice can be
partially occluded. The reference uses this for fixed-camera scenarios with
stable object sizes. We'd want ±40-50% tolerance, or make the reference area
a running average rather than first-frame-only.

---

### 2.2 Edge margin filter (reference: `edge_margin`)

**Reference approach:**
```python
YOLOProcessor(edge_margin=20)
# Rejects any detection whose centroid is within 20px of the frame border
```

```python
def _is_near_edge(self, box, frame_shape):
    h, w = frame_shape[:2]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (cx < self.edge_margin or cx > w - self.edge_margin or
            cy < self.edge_margin or cy > h - self.edge_margin)
```

**Why this helps us:**
- Wall-adjacent detections are our biggest source of instability. When a mouse
  walks along the wall, YOLO produces flickering partial boxes at the border.
- An edge margin filter **suppresses these outright** rather than letting them
  propagate through SAM2 and confuse the tracker.

**Interaction with our border padding:**
We already have `yolo_border_padding_px` which mirror-pads the frame to *improve*
border detections. These are **complementary strategies**:
- Border padding: helps YOLO detect objects partially outside the frame
- Edge margin: suppresses detections that are *too* close to the edge (likely
  artifacts or partial views not worth tracking)

The right approach is: pad first (so real objects near borders get detected
properly), then edge-margin-filter (to remove residual border artifacts).

**Risk:** If set too aggressively, edge margin will suppress real mice that
walk along the wall. For our 640x480 videos, 10-15px is a safe starting
point (the mouse body is ~100-150px across, so a 10px margin won't suppress
a mouse whose body is still mostly in frame).

---

### 2.3 Entity count enforcement (reference: `entities`, `min_entities`, `max_entities`)

**Reference approach:**
```python
YOLOProcessor(entities=2)       # Keep EXACTLY 2 (by confidence)
YOLOProcessor(max_entities=2)   # Keep AT MOST 2
YOLOProcessor(min_entities=2)   # If < 2, lower confidence 5% and retry (recursive)
```

The `min_entities` retry is particularly interesting:
- If YOLO finds < N objects at the configured confidence, it **recursively
  lowers the threshold by 5%** and re-runs until it finds enough or hits a
  floor.
- This prevents "losing" a mouse when it's partially occluded and YOLO's
  confidence drops below the threshold.

**Why this helps us:**
- We use `max_animals=2` but only apply it at the mask deduplication stage (after
  SAM2). If YOLO finds 0 or 1 detection due to occlusion, we can't recover.
- The `min_entities` retry would catch the case where a mouse is partially
  behind the other one and YOLO's confidence drops to, say, 0.18 (below our
  0.25 threshold).

**Risk:** The retry loop can produce false positives at very low confidence.
Need a floor (e.g., never go below 0.10). Also adds latency from re-running
YOLO, but this is rare (only triggers when detections are missing).

**Our current approach:** We set `detection.confidence=0.25` globally. If one
mouse is occluded and drops below 0.25, it simply disappears. BoT-SORT's
`track_buffer` keeps the slot alive for ~30 frames, but the mask is gone.

---

### 2.4 Custom NMS with `max_overlap` (reference)

**Reference approach:**
```python
YOLOProcessor(max_overlap=0.5)
# After all filtering, apply custom NMS: for each pair of remaining boxes,
# if IoU > max_overlap, keep the one with higher confidence.
```

**Why this helps us:**
- When two mice are touching, YOLO often produces **3 boxes**: one for each mouse
  plus a larger box encompassing both. Standard YOLO NMS may not catch this if
  the merged box has a different class label or is just below the NMS threshold.
- Custom NMS with a lower IoU threshold (0.3-0.5) would catch these merged
  detections.

**Our current approach:** We rely on Ultralytics' built-in NMS (typically IoU=0.7)
and then our mask-level deduplication in `postprocess.py`. The mask dedup works
but adds unnecessary SAM2 computation for boxes that should have been filtered
at the YOLO stage.

---

### 2.5 ROI filter (reference: `roi`)

**Reference approach:**
```python
YOLOProcessor(roi=[100, 100, 800, 600])
# Only keep detections whose centroid is within this rectangle
```

**Why this helps us:**
- If the camera setup has a known arena region, ROI filtering would instantly
  eliminate all detections outside the arena (e.g., equipment visible at the
  edges, reflections).
- Less relevant if the full frame IS the arena, but useful for specific setups.

**Relevance:** Low priority for our current videos (the full frame is the arena),
but worth having as a config option for future experiments.

---

### 2.6 SAM2 manual mode / centroid-based tracking (reference)

**Reference approach:**
The reference `SAM2Processor` has a **manual mode** where you click initial
points in frame 1, then SAM2 uses the previous frame's **centroid as a point
prompt** (not a box prompt) for subsequent frames:

```python
# Frame N+1: use centroid from frame N as point prompt
predictor.predict(
    point_coords=np.array([prev_centroid]),
    point_labels=np.array([1]),
    multimask_output=False
)
```

**Why this is interesting:**
- Point prompts produce **tighter masks** than box prompts when the object is
  partially occluded — the point is on the visible part of the animal, while
  a box prompt may include occluding objects.
- When two mice are touching, a box prompt covering both mice will confuse
  SAM2. A point prompt at the known centroid position is more precise.

**Why we don't need it now:**
- Our pipeline uses YOLO boxes as SAM2 prompts, which is the standard approach
  and works well when YOLO produces clean boxes.
- The manual mode requires clicking initial points, which doesn't scale for
  automated processing.
- However, the **hybrid idea** is worth noting: use YOLO boxes normally, but
  when two boxes overlap significantly, switch to point prompts (centroid from
  previous frame) for the overlapping detections.

---

### 2.7 Identity matching algorithm comparison

**Reference `IdentityMatcher`:**
```
For each slot (with known previous centroid):
  1. Compute distance to all current detections
  2. Sort by distance (ascending)
  3. For the nearest: check distance < proximity_threshold AND area_change < area_tolerance
  4. If both pass → assign. If not → slot stays empty.
  5. Unassigned detections → fill remaining empty slots in order.
```

**Our `SlotTracker`:**
```
Step 1: Match by YOLO track ID (O(1) lookup)
Step 2: Build cost matrix (unmatched masks x unmatched slots)
        cost = w_dist * dist + w_iou * (1-IoU) + w_area * area_change
        Solve with Hungarian algorithm (globally optimal)
Step 3: Remaining masks → free slots
Step 4: Missing slots → increment counter, release after max_missing_frames
```

**Comparison:**

| Feature | Reference | Ours |
|---------|-----------|------|
| Matching strategy | Greedy per-slot (nearest first) | Hungarian (globally optimal) |
| Area gating | Hard veto (±40%) | Soft cost (weight 0.2), hard veto only at 5x |
| Identity signal | Centroid distance only | YOLO track ID (primary) + centroid+IoU+area (fallback) |
| IoU in matching | Not used | Yes (mask IoU, weight 0.4) |
| Missing tolerance | None (slot just stays empty) | `max_missing_frames` with counter |
| Multi-frame state | Previous centroids + areas | Previous centroids + areas + masks + track ID mapping |

**Verdict:** Our tracking is **significantly better** for close-contact scenarios.
The reference's greedy matching with hard area veto is exactly the approach we
already replaced because it caused cascading ID switches. We should NOT adopt
the reference's tracking algorithm.

---

## 3. Techniques Worth Adopting

### Priority 1 (High impact, low effort)

#### 3.1 Edge margin filter

**What:** Add `edge_margin` parameter to `detect_and_track()`. After YOLO
produces detections, reject any whose centroid is within `edge_margin` pixels
of the frame border.

**Where to add:** [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py) —
in `detect_and_track()`, after coordinate correction but before returning.

**Config:**
```yaml
detection:
  edge_margin: 0    # 0 = disabled, 10-15 = good starting point
```

**Interaction with border padding:** Apply edge margin filter on the *original*
(unpadded) frame coordinates, i.e., after `correct_detections_after_padding()`.

**Effort:** ~15 lines of code. Add a centroid check + filter to `detect_and_track()`.

---

#### 3.2 `max_entities` cap at YOLO level

**What:** After all YOLO filtering, keep only the top N detections by confidence.
Currently we cap at the mask level (in `postprocess.py`), but capping earlier
avoids wasted SAM2 computation.

**Where to add:** [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py) —
at the end of `detect_and_track()`, sort by confidence and truncate.

**Config:** Already have `detection.max_animals`. Just enforce it earlier.

**Effort:** ~5 lines.

---

### Priority 2 (Medium impact, medium effort)

#### 3.3 Adaptive area filtering

**What:** Learn a "reference area" from early detections. Reject YOLO boxes
whose area deviates beyond a tolerance (e.g., ±40%).

**Implementation notes:**
- Use a running average (EMA) of valid detection areas rather than first-frame-only,
  because mouse apparent size changes as they move around the arena.
- Only update the reference from high-confidence (>0.5) detections to avoid
  learning from artifacts.
- Need separate reference areas per class if using multi-class.

**Where to add:** New function in [infer_yolo.py](../src/pipelines/sam2_yolo/infer_yolo.py),
or a small stateful class that wraps the detection step.

**Config:**
```yaml
detection:
  area_adaptive: false       # Enable/disable
  area_tolerance: 0.4        # ±40% from reference
  area_warmup_frames: 10     # Frames to build reference before filtering
```

**Effort:** ~40-50 lines. Needs a stateful object (persists across frames)
to track the reference area.

---

#### 3.4 Min-entities confidence retry

**What:** If YOLO finds fewer than `max_animals` detections, recursively lower
the confidence threshold by 5% and re-run until enough are found or a floor
is hit.

**Implementation notes:**
- This is the most impactful technique for **partial occlusion** scenarios.
- Must be careful: re-running `model.track()` with different confidence could
  interfere with BoT-SORT's internal state. Safer approach: run once at a
  low confidence (0.10), then apply our own confidence threshold + retry logic
  on the results without re-running the model.
- Alternative: just always run YOLO at a lower confidence (0.10-0.15) and
  filter downstream. The reference's retry is needed because they use per-frame
  `model()`, but we use `model.track()` which already handles low-confidence
  associations internally via `track_low_thresh`.

**Key insight:** BoT-SORT's `track_low_thresh=0.1` in the tracker config already
does a similar thing — it has a second-pass matching for low-confidence detections.
So we may already have this benefit without explicit retry logic. Worth verifying
by lowering `detection.confidence` to 0.15 and testing.

**Effort:** If reimplemented: ~30 lines. If we just lower `detection.confidence`:
0 lines (config change only).

---

### Priority 3 (Nice to have, defer)

#### 3.5 Custom NMS threshold

**What:** Post-YOLO NMS with configurable IoU threshold (lower than default).

**Why defer:** Our mask-level deduplication already handles the merged-box
problem, and adding another NMS step risks over-suppression. Worth trying
only if we see specific merged-detection artifacts.

**Config:**
```yaml
detection:
  nms_iou: null   # null = use YOLO default (~0.7). Lower = more aggressive.
```

**Effort:** ~10 lines. YOLO supports `iou=` parameter in `model.track()`.

---

#### 3.6 ROI filter

**What:** Restrict detections to a defined rectangle.

**Why defer:** Our current videos use the full frame as the arena. Only useful
if we have videos with equipment/objects outside the tracking area.

**Effort:** ~10 lines.

---

#### 3.7 SAM2 point prompts as fallback

**What:** When two YOLO boxes overlap significantly, use point prompts (previous
centroid) instead of box prompts for SAM2 segmentation.

**Why defer:** This is a significant architectural change to `infer_sam2.py` and
requires the tracker to feed centroid predictions back to the segmentation step.
High potential but high complexity. Better suited as a future iteration after
the simpler filters are validated.

**Effort:** ~80-100 lines + API changes across multiple files.

---

## 4. What NOT to Adopt

### 4.1 Reference tracking algorithm (IdentityMatcher)

**Do not adopt.** Their greedy matching with hard area gating is exactly the
approach we replaced because it caused cascading ID switches. Our Hungarian +
soft cost + YOLO track ID system is strictly superior.

### 4.2 Data Bus architecture

**Do not adopt.** The `frame_data` dictionary pattern is a valid design choice
but would require rewriting the entire pipeline for no functional benefit. Our
direct function-call approach is simpler and equally effective.

### 4.3 Per-frame `model()` instead of `model.track()`

**Do not adopt.** The reference uses stateless per-frame detection. We use
`model.track()` with BoT-SORT which provides temporal state, Kalman prediction,
and stable track IDs. This is a major advantage.

### 4.4 NumPy < 2.0 constraint

**Note only.** The reference requires `numpy < 2.0.0`. We should verify our
pipeline works with NumPy 2.x and pin if needed, but this is not a technique
to adopt.

---

## 5. Recommended Adoption Order

```
Phase 1 (quick wins, test immediately):
  1. Edge margin filter (edge_margin=10)
  2. Max entities cap at YOLO level
  3. Lower detection.confidence to 0.15 (leverage BoT-SORT's low_thresh)

Phase 2 (after Phase 1 is validated):
  4. Adaptive area filtering (with EMA, ±40% tolerance)
  5. Custom NMS threshold (iou=0.5 in model.track())

Phase 3 (future, if needed):
  6. ROI filter
  7. SAM2 point prompt fallback for overlapping boxes
```

**Testing each phase:**
- Run on the 6s clip (`data/clips/output-6s.mp4`) with `local_quick.yaml`
- Compare overlay video: count ID switches, border flicker events
- Check the slot debug log every 50 frames for stability

---

## 6. Parameter Mapping

Quick reference for translating between the reference's parameters and our config:

| Reference parameter | Our equivalent | Notes |
|---------------------|---------------|-------|
| `confidence` | `detection.confidence` | Same meaning |
| `entities` | `detection.max_animals` | Theirs is exact count, ours is max |
| `max_entities` | `detection.max_animals` | Same |
| `min_entities` | (not implemented) | Could adopt with retry logic |
| `area_min` / `area_max` | (not implemented) | Could adopt as config params |
| `area` / `area_error` | (not implemented) | Adaptive area — Priority 2 |
| `classes` | `detection.filter_class` | Theirs uses class IDs, ours uses class name |
| `edge_margin` | (not implemented) | Priority 1 |
| `max_overlap` | (not implemented, YOLO default NMS) | Priority 3 |
| `roi` | (not implemented) | Priority 3 |
| `proximity_threshold` | `tracking.max_centroid_distance` | Same concept, different usage (theirs=hard gate, ours=normalization) |
| `area_tolerance` | `tracking.w_area` | Theirs=hard ±40% gate, ours=soft cost weight |
| `iou_threshold` | `segmentation.mask_iou_threshold` | Same — mask dedup IoU |
| `max_entities` (SAM2) | `detection.max_animals` | Same concept |
