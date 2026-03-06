# Pipeline Tests — Centroid Pipeline (2026-03-06)

## How the Centroid Pipeline Works

```
Frame 0 (init):
  YOLO detect → boxes → SAM2 segment (box prompts) → 2 masks + centroids

Frame 1..N:
  1. YOLO detect (every frame) → keypoints only (boxes ignored)
  2. SAM2 segment: centroid(+) + other centroid(-) → multimask → best score
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
- Used for: keypoints for contact classification (boxes ignored after init)

**Step 2: SAM2 segmentation (centroid prompts always)**

```
SAM2.predict(
    point_coords = [[this_centroid], [other_centroid]],
    point_labels = [1, 0],  # positive, negative
    box = None,
    multimask_output = True  # 3 masks → select best by score
)
```
- Previous frame's centroid as positive prompt ("segment here")
- Other rat's centroid as negative prompt ("NOT here")
- `multimask_output=True` generates 3 candidate masks; best selected by confidence score
- Works at all distances — negative prompts help SAM2 distinguish rats even when close
- `resolve_overlaps` handles any remaining mask bleeding

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

## Historical: Why YOLO Box Prompts Were Removed

### Problem with hybrid prompting (previous design)
An earlier version switched to YOLO box prompts when rats were close (<100px).
This caused **identity swaps** because YOLO detection order is arbitrary:

1. Rats get close → pipeline switches to YOLO box prompts
2. YOLO detection[0] might be slot 1's rat (random ordering)
3. Wrong box goes to wrong slot → masks swap → centroids swap
4. **Permanent identity swap** propagates for the rest of the video

### Why centroid-only works
- SAM2 centroid prompts are inherently slot-stable (each slot's centroid propagates from previous frame)
- Positive + negative points help SAM2 distinguish rats even at close range
- `resolve_overlaps` handles any mask bleeding when rats touch
- No YOLO ordering ever touches SAM2 prompts → no swap mechanism exists

---

## Configuration

```yaml
segmentation:
  sam_threshold: 0.0
```

---

## Test Checklist

### Test 1: Rats far apart (independent)
- [ ] Both masks clean, separate
- [ ] Status bar shows `CENTROID`
- [ ] Keypoints assigned correctly to each mask

### Test 2: Rats approaching
- [ ] Masks remain separate
- [ ] Prompting stays `CENTROID` throughout

### Test 3: Rats interacting (touching/overlapping)
- [ ] **Both rats have masks** (resolve_overlaps handles bleeding)
- [ ] Status bar shows `CENTROID`
- [ ] Contact classification fires (N2N, N2AG, etc.)
- [ ] Keypoints assigned to correct masks
- [ ] **No identity swap** (same color stays on same rat)

### Test 4: Rats separating after interaction
- [ ] Masks recover cleanly
- [ ] No identity swap (same color stays on same rat)

### Test 5: Long interaction period (30+ frames close)
- [ ] Masks remain stable throughout
- [ ] Contact bouts detected with proper duration
- [ ] No quality_flag spam

### Test 6: Full video (120s)
- [ ] Pipeline completes without crash
- [ ] Contacts CSV has events during interaction periods
- [ ] Overlay video shows masks on both rats at all times
