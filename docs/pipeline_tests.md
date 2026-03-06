# Pipeline Tests — Centroid Pipeline (2026-03-06)

## How the Centroid Pipeline Works

```
Frame 0 (init):
  YOLO detect → boxes → SAM2 segment (box prompts) → 2 masks + centroids

Frame 1..N:
  1. YOLO detect (every frame) → keypoints + boxes
  2. SAM2 segment:
     - Centroids FAR apart (>100px) → centroid point prompts + negative points
     - Centroids CLOSE (<100px)     → YOLO box prompts + negative centroid points
  3. Compute centroids from masks
  4. Resolve overlapping pixels (centroid proximity)
  5. Assign YOLO keypoints to masks (spatial overlap)
  6. Carry-over missing keypoints (centroid delta shift)
  7. Contact classification (always — no merge skip)
  8. Render overlay
```

### Step-by-step detail

**Step 1: YOLO detection**
- Runs `detect_only()` every frame on the full image
- Returns bounding boxes + 7 keypoints per rat
- No tracking (no BoT-SORT) — just single-frame detection
- Used for: (a) keypoints for contacts, (b) boxes for SAM2 when close

**Step 2: SAM2 segmentation (hybrid prompting)**

When rats are **far apart** (centroid distance > `box_prompt_proximity_px`):
```
SAM2.predict(
    point_coords = [[this_centroid], [other_centroid]],
    point_labels = [1, 0],  # positive, negative
    box = None,
    multimask_output = True  # 3 masks → select best by score
)
```
- Works well: clear spatial separation, negative point is far from target
- SAM2 cleanly separates the two animals
- `multimask_output=True` generates 3 candidate masks; best selected by confidence score

When rats are **close** (centroid distance < `box_prompt_proximity_px`):
```
SAM2.predict(
    point_coords = [[this_kp1], [this_kp2], ...,   # positive (this rat's keypoints)
                     [other_centroid], [other_kp1], ...],  # negative
    point_labels = [1, 1, ..., 0, 0, ...],
    box = [x1, y1, x2, y2],  # YOLO box for THIS rat
    multimask_output = False
)
```
- Box prompt gives SAM2 the spatial extent of each rat
- This rat's YOLO keypoints (nose, ears, body) as positive prompts anchor the mask
- Other rat's centroid + keypoints as negative prompts exclude the neighbor
- YOLO boxes work even during close interaction because YOLO detects each rat separately

**Step 3: Centroid update**
- Compute mask centroid for each slot
- If mask produced no centroid → keep previous frame's centroid

**Step 4: Overlap resolution**
- `resolve_overlaps()` assigns contested pixels to nearest centroid
- Modifies masks in-place

**Step 5: Keypoint assignment**
- YOLO detections matched to SAM2 masks by containment (box center inside mask)
- Fallback: nearest centroid within 200px

**Step 6: Carry-over**
- If YOLO misses a rat, copy previous frame's keypoints shifted by centroid delta
- Marked with `is_carried_over=True` → `quality_flag="stale_keypoints"` in contacts

**Step 7: Contact classification**
- Always runs (no merge detection skip)
- 6 types: N2N, N2AG, T2T, FOL, SBS, N2B
- Body-length normalized thresholds

---

## Problem: Mask Loss During Interaction (before fix)

### Symptom
When rats interact (close proximity), only ONE rat gets a SAM2 mask. The other
rat's mask disappears — it either gets absorbed by the first mask or SAM2
produces an empty/tiny mask.

### Root cause
`_segment_from_centroids()` uses 1 positive point + 1 negative point per rat.
When centroids are ~50px apart:
- Both points land on the same visual blob
- SAM2 can't distinguish the two similar objects
- One mask absorbs the entire scene around both rats
- The other mask gets nothing (the "positive" point is inside the first mask)

### Previous workaround (removed)
The `identity_ambiguous` flag + `write_merged_placeholder()` **skipped contact
classification** when masks overlapped. This was counterproductive:
- Contacts were skipped *exactly* when rats were interacting
- The blind spot was the most important period for social phenotyping

### Fix: Hybrid prompting
When centroids are close, switch to YOLO box + negative point prompts:
- YOLO boxes correctly delineate each rat even during interaction
- Box prompts give SAM2 spatial extent, not just a point
- Negative centroid from the other rat helps exclude the neighbor
- Result: two separate masks even during close contact

---

## Configuration

```yaml
segmentation:
  sam_threshold: 0.0
  box_prompt_proximity_px: 100.0  # threshold for switching to box prompts
```

### Tuning `box_prompt_proximity_px`

| Value | Effect |
|-------|--------|
| 50 | Only switch to boxes when very close (less aggressive) |
| 100 | Default — switch when rats are within ~1 body length |
| 150 | Switch to boxes earlier (more aggressive, may reduce centroid stability) |

The threshold is in pixels. For a typical 640x480 video with rats ~120px long,
100px is approximately 0.8 body lengths.

---

## Test Checklist

### Test 1: Rats far apart (independent)
- [ ] Both masks clean, separate
- [ ] Status bar shows `CENTROID`
- [ ] Keypoints assigned correctly to each mask

### Test 2: Rats approaching (100-200px apart)
- [ ] Masks remain separate
- [ ] Transition from CENTROID → BOX prompting visible in status bar

### Test 3: Rats interacting (< 100px, touching)
- [ ] **Both rats have masks** (the key improvement)
- [ ] Status bar shows `BOX`
- [ ] Contact classification fires (N2N, N2AG, etc.)
- [ ] Keypoints assigned to correct masks

### Test 4: Rats separating after interaction
- [ ] Masks recover to normal centroid prompting
- [ ] No identity swap (same color stays on same rat)
- [ ] Status bar returns to `CENTROID`

### Test 5: Long interaction period (30+ frames close)
- [ ] Masks remain stable throughout
- [ ] Contact bouts detected with proper duration
- [ ] No quality_flag spam

### Test 6: Full video (120s)
- [ ] Pipeline completes without crash
- [ ] Contacts CSV has events during interaction periods
- [ ] Overlay video shows masks on both rats at all times
