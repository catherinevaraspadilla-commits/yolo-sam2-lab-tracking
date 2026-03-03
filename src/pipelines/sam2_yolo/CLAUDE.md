# sam2_yolo pipeline

Original pipeline using YOLO BoT-SORT tracking + SAM2 segmentation.

## Files

- `run.py` — Entry point
- `models_io.py` — Model loading (YOLO + SAM2)
- `infer_yolo.py` — YOLO detection with `detect_only()` (also used by reference/sam3)
- `infer_sam2.py` — SAM2 segmentation
- `postprocess.py` — Filtering + SlotTracker tracking

## Key Design

- Uses `YOLO.track()` with BoT-SORT — YOLO assigns track IDs
- SlotTracker maps YOLO track IDs to fixed display slots
- No centroid fallback — if YOLO misses a rat, it disappears

## Known Issues

- BoT-SORT causes ID switches during close interactions
- No fallback when YOLO fails to detect
- Consider using `reference` pipeline instead for most use cases
