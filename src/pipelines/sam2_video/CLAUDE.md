# sam2_video pipeline

Uses SAM2 VideoPredictor for temporal mask propagation. Best for short videos with heavy occlusion.

## Files

- `run.py` — Entry point. Processes video in segments (~2000 frames each).
- `match_keypoints.py` — Matches YOLO keypoints to SAM2 propagated masks

## Key Design

- SAM2VideoPredictor maintains temporal memory (7-frame memory bank)
- Initialized with YOLO boxes on first frame, then propagates automatically
- YOLO only needed for keypoints on intermediate frames (not for tracking)
- Requires extracting frames to JPEG (temporary directory)

## Limitations

- Slow (~3-5x slower than other pipelines)
- No batch/chunk support — processes full video sequentially
- Higher memory usage (GPU + disk for frame extraction)
