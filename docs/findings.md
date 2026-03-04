# Debug Findings

Experimental results from running debug scripts on Bunya HPC.
Video: `data/raw/original_120s.avi` (960x960, 30 FPS, 3600 frames, 120s).

---

## Baseline: Reference Pipeline Problems

The **reference pipeline** (`src/pipelines/reference/`) is the recommended pipeline.

**How it works per frame:**
1. **YOLO detect-only** — runs YOLO26 without BoT-SORT tracking. Returns N
   bounding boxes + 7 keypoints (tail_tip, tail_base, tail_start, mid_body,
   nose, right_ear, left_ear) per rat. No temporal consistency — each frame
   is an independent detection.
2. **SAM2 segment** — for each YOLO box, SAM2 ImagePredictor generates a
   pixel-level mask using the box as prompt. Keypoints from the same rat are
   positive prompts; keypoints from other rats are negative prompts (to help
   separate overlapping boxes). If YOLO detects fewer rats than expected and
   the state is SEPARATE, a centroid fallback uses the previous frame's mask
   centroid as a point prompt.
3. **IdentityMatcher** — assigns masks to persistent identities (rat 0, rat 1)
   using Hungarian matching with a soft cost function (distance 40%, mask IoU
   40%, area 20%). Has a SEPARATE/MERGED state machine: when rats are close
   (`proximity_threshold=150px`), enters MERGED state which coasts tracks with
   velocity prediction for up to `max_merged_frames=30` frames (1 second).
4. **ContactTracker** — detects social contacts (nose-to-nose, nose-to-body,
   etc.) based on keypoint distances from the assigned identities.

It works reasonably but has significant problems during rat interactions. After
running the full 120s video on Bunya, we observed 4 recurring failure modes:

### 1. Mask bleeding

After close interactions, one rat's mask "leaks" onto the other — both rats
get covered by a single color, or the mask boundary shifts to include parts
of the neighboring rat.

**Why it happens:** When two rats are close, YOLO produces overlapping bounding
boxes. SAM2 receives two boxes that cover roughly the same area, so it segments
both rats into one blob. The negative keypoint prompts (designed to separate
overlapping boxes) help in some cases but fail when the boxes overlap heavily
or keypoints land on ambiguous positions.

**When it happens:** Predominantly during nose-to-nose or side-by-side
interactions where boxes overlap >50%.

### 2. ID swaps

During prolonged interactions (>1 second), the identity of rat A and rat B
suddenly swap — green becomes red and vice versa. The masks themselves look
fine, but the assignment is wrong.

**Why it happens:** IdentityMatcher enters MERGED state when rats are close.
`max_merged_frames=30` means it holds the merged state for 30 frames (1 second
at 30 FPS). If the interaction lasts longer, the state machine times out and
attempts split resolution. At that point, YOLO's box ordering may have changed
(YOLO doesn't maintain identity), so the 2x2 Hungarian assignment can match
incorrectly — especially if both rats moved during the interaction.

**When it happens:** Nose-to-nose interactions lasting >1 second, which are
common in social behavior experiments.

### 3. Rat disappears

One rat's mask vanishes entirely for stretches of 10-50+ frames. The video
shows only one colored mask; the other rat is visible but untracked.

**Why it happens:** YOLO intermittently fails to detect one of the rats
(detection count drops to 1 or 0). The centroid fallback mechanism is supposed
to compensate by using the previous frame's centroid as a SAM2 point prompt,
but it gets skipped during MERGED state (to avoid re-segmenting a merged blob
as a duplicate). Additionally, `filter_duplicates` can remove fallback masks
if they overlap too much with the remaining YOLO-prompted mask.

**When it happens:** Particularly when one rat is partially occluded, at frame
edges, or in low-contrast positions against the cage floor.

### 4. Intermittency

Masks flicker on and off every few frames — one frame shows 2 masks, next
shows 1, next shows 2 again. The overlay video looks unstable and jittery.

**Why it happens:** YOLO's detection count fluctuates between 1 and 2 on
consecutive frames. Each frame gets a different number of SAM2 prompts,
producing inconsistent segmentation. Since SAM2 ImagePredictor has no temporal
memory (each frame is independent), there is no smoothing between frames —
whatever YOLO gives, SAM2 segments.

**When it happens:** Throughout the video, but worse during transitions
(rat entering/leaving an interaction, turning, or changing posture).

### Common root cause

All 4 problems trace back to **YOLO's per-frame instability**. YOLO26 in
detect-only mode (no BoT-SORT tracking) provides no temporal consistency —
each frame is an independent detection that may produce 0, 1, or 2 boxes in
any order. This inconsistency cascades through SAM2 (bad prompts → bad masks)
and IdentityMatcher (unstable inputs → wrong assignments).

---

## 2026-03-04: Confirming YOLO as the bottleneck

To confirm that YOLO is the root cause (and not SAM2 or IdentityMatcher),
we created debug scripts that isolate each component.

### Experiment 1: YOLO-only debug

**Script:** `debug_yolo_only.py` on Bunya (full 120s video, no SAM2)

Renders only YOLO bounding boxes and keypoints on every frame. No segmentation,
no tracking — pure YOLO detection quality.

**Result:** YOLO detection is terrible.
- Intermittent: detections appear and disappear every second
- Swaps rat identities frame to frame (box 0 and box 1 flip constantly)
- During close interactions, boxes overlap heavily or merge into one
- Detection count fluctuates between 0, 1, and 2 unpredictably
- Even when both rats are clearly visible and separated, YOLO sometimes
  detects only 1

**Conclusion:** YOLO26 detect-only mode is unreliable for frame-by-frame rat
detection in this video. The intermittency is not limited to interactions —
it happens even when rats are clearly separated, suggesting the model's
confidence fluctuates near the detection threshold.

**Pending investigation:** YOLO is designed for moving objects and works well
in other domains (vehicles, people, sports). Why is it intermittent here?
Possible causes to investigate:
- **Posture changes**: rats contract, stretch, curl up — their silhouette
  changes drastically between frames. A rat curled up looks very different
  from one stretched out, and the model may have been trained mostly on
  one posture.
- **Similar appearance**: both rats are the same color/size. YOLO may
  fluctuate on which pixels belong to which detection.
- **Training data bias**: the YOLO26 model may have been trained on images
  where rats are more clearly separated or in different environments.
- **Confidence threshold**: detections near the threshold (e.g., 0.25) may
  alternate between appearing and disappearing. A lower threshold might
  help but could also add false positives.
- **No tracking**: detect-only mode has zero temporal smoothing. BoT-SORT
  was removed because it caused other problems, but some form of temporal
  filtering on detections might help stabilize.

### Experiment 2: SAM2 without YOLO (centroid propagation)

**Script:** `debug_sam2_no_yolo.py` on Bunya (full 120s video, pure mode)

Uses YOLO only on frame 0 to get initial bounding boxes. After that, SAM2
segments using previous-frame centroids as point prompts — no YOLO involvement.
Each rat's centroid is used as a positive prompt, and the other rat's centroid
as a negative prompt (to help SAM2 distinguish them when close).

**Result:** Almost perfect.
- YOLO used only once (frame 0 initialization)
- SAM2 propagated masks via centroids for all 3600 frames
- Only 1 blink observed in the entire 120s video
- Masks were stable and consistent throughout — no swaps, no flickering
- Both rats tracked correctly even during close interactions
- Tails not always fully covered (expected with a single centroid point prompt;
  SAM2 tends to segment the body core rather than thin extremities)

**Conclusion:** SAM2 ImagePredictor with centroid propagation is robust enough
to track 2 rats across 3600 frames without per-frame YOLO. When given consistent
prompts (centroids that move smoothly frame-to-frame), SAM2 produces stable,
reliable masks. The per-frame YOLO interference was the root cause of all
pipeline instability.

### Key Insight

| Approach | YOLO usage | Mask stability | Tail coverage |
|----------|-----------|----------------|---------------|
| Reference pipeline (YOLO every frame) | Every frame | Bad — intermittent, swaps, bleeding | N/A (masks too unstable to evaluate) |
| SAM2 no-YOLO (centroid propagation) | Frame 0 only | Almost perfect — 1 blink in 3600 frames | Partial (body ok, tail sometimes missing) |

**Root cause confirmed:** YOLO's per-frame intermittency causes SAM2 mask
instability in the reference pipeline. SAM2 itself is capable of stable
tracking when given consistent prompts.

---

### Rejected: `--reinit-every` (periodic YOLO re-initialization)

**Do NOT use `--reinit-every`.** Since YOLO is intermittent and swaps rat
identities every second, any re-initialization from YOLO risks:
- Swapping mask identities (YOLO box 0 ≠ previous mask 0)
- Losing a rat (YOLO detects only 1 at the reinit frame)
- Breaking the stable centroid propagation that SAM2 was maintaining

The whole point of the no-YOLO approach is to **escape YOLO's instability**.
Re-injecting YOLO periodically defeats that purpose.

### Tail coverage: accepted limitation

Tails are not fully covered by centroid-only prompts. We tested `multimask_output=True`
(pick largest of 3 SAM2 masks) but it included background/cage and made things worse.
Tail coverage is an inherent SAM2 limitation with point prompts — acceptable because
contact detection uses keypoints, not mask shape.

### Open problem: YOLO is still needed for contacts

The centroid-propagation approach solves mask stability, but **contact detection
still depends on YOLO**. ContactTracker needs per-rat keypoints (nose, ears,
tail_base, etc.) to classify contacts (nose-to-nose, nose-to-body, etc.).
SAM2 masks alone don't provide keypoint positions — only YOLO does.

This creates a contradiction: YOLO's keypoints are unreliable (intermittent,
swapping identities every second), but we need them for contacts. Stable masks
from centroid propagation don't help if the keypoints assigned to each rat
are wrong. This problem needs further investigation before building a
production pipeline.

---

## Next Steps

- [ ] Evaluate centroid-propagation approach on longer videos (10+ minutes)
- [ ] Investigate how to get reliable per-rat keypoints despite YOLO instability
