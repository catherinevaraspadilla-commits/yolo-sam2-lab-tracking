# Lessons Learned — SAM2 Tracking for Laboratory Rats

> Key findings from iterative testing of YOLO + SAM2 pipelines.
> These are validated empirically — do not revert these decisions without re-testing.

---

## Context

We track 2 laboratory rats in video. The rats:
- **Change posture rapidly** (running, rearing, grooming, curling)
- **Interact for prolonged periods** (sniffing, following, side-by-side)
- **Cross paths** and temporarily overlap in the frame
- Look visually similar (same species, same size, same color)

The goal: **once a rat is identified, follow it without confusing it with the other rat**, even during crossings or prolonged close interactions.

---

## Finding 1: `multimask_output=False` is critical

### What we tested

| Setting | Behavior | Result |
|---------|----------|--------|
| `multimask_output=True` | SAM2 generates 3 candidate masks, pipeline picks the one with highest confidence score | **Poor tracking** — masks degrade, identity becomes unstable |
| `multimask_output=False` | SAM2 generates 1 direct mask | **Stable tracking** — masks stay clean, identity maintained |

### Why True fails

When `multimask_output=True`, SAM2 produces 3 candidates of varying quality. The "best score" candidate is not always the best for tracking continuity. The score reflects SAM2's confidence in mask quality for that single frame, not temporal consistency. This means:

- The selected mask can jump between different interpretations frame-to-frame
- During close interactions, the "best" candidate might include parts of both rats
- The mask shape can change drastically between frames even for a stationary rat

### Rule

**Always use `multimask_output=False`** for centroid propagation tracking. Do not change this.

---

## Finding 2: YOLO every frame destroys identity

### What we tested

| Approach | Result |
|----------|--------|
| YOLO every frame → boxes as SAM2 prompts | **Identity swaps** — permanent, unrecoverable |
| YOLO every frame → keypoints only | Workable but unnecessary complexity |
| YOLO init only (frame 0) → SAM2 centroid propagation | **Best results** — stable identity throughout |

### Why YOLO every frame fails

YOLO runs independently each frame with no tracking memory. The detection order is arbitrary:
- Frame N: `detection[0]` = rat A, `detection[1]` = rat B
- Frame N+1: `detection[0]` = rat B, `detection[1]` = rat A

This is invisible to the pipeline — there's no way to know which detection is which rat.

When YOLO boxes were used as SAM2 prompts:
1. Wrong box → wrong slot
2. SAM2 segments the wrong region for that slot
3. Centroid updates to wrong position
4. **Permanent identity swap** — propagates forever

### Rule

**YOLO runs ONLY on frame 0 for initialization.** After that, SAM2 centroid propagation alone drives tracking. YOLO detection order must never influence SAM2 prompting.

---

## Finding 3: SAM2 centroid propagation maintains identity

### How it works

```
Frame 0 (init):
  YOLO detect → 2 boxes → SAM2 segment (box prompts) → 2 masks → 2 centroids

Frame 1..N:
  For each rat:
    SAM2.predict(
      point_coords = [[this_centroid], [other_centroid]],
      point_labels = [1, 0],    # positive, negative
      box = None,
      multimask_output = False,
    )
  → update centroids from new masks
  → resolve_overlaps for contested pixels
```

### Why it works

- **Positive prompt** (own centroid): tells SAM2 "segment the object at this location"
- **Negative prompt** (other rat's centroid): tells SAM2 "NOT this object"
- Centroids propagate frame-to-frame: each slot's centroid stays on the same rat
- No external ordering (YOLO) can disrupt the slot-centroid association
- `resolve_overlaps` handles any mask bleeding when rats touch

### What it handles well

- Rats crossing paths (centroids track through the crossing)
- Prolonged close interactions (negative prompts keep masks separate)
- Rapid posture changes (SAM2 adapts the mask shape each frame)
- Temporary occlusion (centroid stays at last known position, mask recovers)

---

## Finding 4: What we removed and why

| Removed Feature | Why |
|----------------|-----|
| YOLO box prompts (hybrid prompting) | Caused identity swaps via YOLO ordering |
| `multimask_output=True` | Degraded mask quality and tracking stability |
| YOLO every-frame detection for SAM2 | Unnecessary — SAM2 alone tracks better |
| IdentityMatcher (Hungarian assignment) | SAM2 propagation inherently maintains identity |
| SEPARATE/MERGED state machine | No longer needed without hybrid prompting |
| Keypoint assignment to masks | Removed with YOLO (can be re-added later if needed for contacts) |
| Keypoint stabilization (EMA) | Removed with YOLO |
| Temporal keypoint carry-over | Removed with YOLO |

---

## Current Pipeline (Minimal, Validated)

```python
# Frame 0: YOLO init
detections = detect_only(yolo, frame_rgb, confidence)
masks, scores = _segment_from_boxes(sam, frame_rgb, detections, threshold)
centroids = [compute_centroid(m) for m in masks]

# Frame 1..N: SAM2 centroid propagation only
masks, scores = _segment_from_centroids(sam, frame_rgb, prev_centroids, threshold)
# update centroids, resolve overlaps
```

This is the baseline that works. Any additions (YOLO keypoints, contacts, etc.) should be built on top of this without modifying the core SAM2 prompting logic.

---

## Rules for Future Development

1. **Never use `multimask_output=True`** for centroid propagation
2. **Never use YOLO boxes as SAM2 prompts** after initialization
3. **Never let YOLO detection order influence slot/identity assignment for SAM2**
4. **SAM2 centroid prompts are the sole source of identity**
5. Any feature additions (keypoints, contacts) must work *on top of* SAM2 masks — they must not alter how SAM2 is prompted
