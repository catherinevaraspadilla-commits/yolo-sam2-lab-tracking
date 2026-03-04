# Debug Findings

Experimental results from running debug scripts on Bunya HPC.
Video: `data/raw/original_120s.avi` (960x960, 30 FPS, 3600 frames, 120s).

---

## 2026-03-04: YOLO is the bottleneck, not SAM2

### Experiment 1: YOLO-only debug

**Script:** `debug_yolo_only.py` on Bunya (full 120s video)

**Result:** YOLO detection is terrible.
- Intermittent: detections appear and disappear every second
- Swaps rat identities frame to frame (box 0 and box 1 flip)
- During close interactions, boxes overlap heavily or merge
- Detection count fluctuates between 0, 1, and 2

**Conclusion:** YOLO26 detect-only mode (no BoT-SORT tracking) is unreliable
for frame-by-frame rat detection. The intermittency cascades into SAM2 and
IdentityMatcher, causing all downstream failures.

### Experiment 2: SAM2 without YOLO (centroid propagation)

**Script:** `debug_sam2_no_yolo.py` on Bunya (full 120s video, pure mode)

**Result:** Almost perfect.
- YOLO used only on frame 0 for initialization
- SAM2 propagated masks via centroid point prompts for all 3600 frames
- Only 1 blink observed in the entire video
- Masks were stable and consistent throughout
- Tails not always fully covered (expected with single point prompt)

**Conclusion:** SAM2 ImagePredictor with centroid propagation is robust enough
to track 2 rats without per-frame YOLO. The per-frame YOLO interference was
the root cause of pipeline instability.

### Key Insight

| Approach | YOLO usage | Mask stability | Tail coverage |
|----------|-----------|----------------|---------------|
| Reference pipeline (YOLO every frame) | Every frame | Bad (intermittent) | N/A (masks too unstable) |
| SAM2 no-YOLO (centroid propagation) | Frame 0 only | Almost perfect | Partial (body ok, tail sometimes missing) |

**Root cause confirmed:** YOLO's per-frame intermittency causes SAM2 mask instability
in the reference pipeline. SAM2 itself is capable of stable tracking when given
consistent prompts (centroids from previous frame).

---

## Next Steps

- [ ] Test `--multimask` flag: SAM2 proposes 3 masks at different scales, pick largest for better tail coverage
- [ ] Test `--reinit-every 300`: periodic YOLO re-initialization as safety net for long videos
- [ ] Evaluate centroid-propagation approach on longer videos (10+ minutes)
- [ ] Consider building production pipeline based on centroid propagation
