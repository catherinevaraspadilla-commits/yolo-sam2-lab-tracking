# reference pipeline

Recommended pipeline. YOLO detect-only + SAM2 segmentation + IdentityMatcher.

## Files

- `run.py` — Entry point and main loop. Supports `--start-frame`/`--end-frame` for chunked processing.
- `identity_matcher.py` — Hungarian assignment + SEPARATE/MERGED state machine. Also used by sam3 pipeline.
- `sam2_processor.py` — SAM2 prompts: box + positive keypoints + negative keypoints from other detections. Centroid fallback when YOLO misses a rat.

## Key Design

- No YOLO track IDs — identity is managed entirely by IdentityMatcher
- Centroid fallback: when YOLO detects fewer rats than expected, uses previous centroid as SAM2 point prompt
- State machine handles close interactions (MERGED state freezes identity assignment)

## Configs

- `configs/local_reference.yaml` — Local testing (tiny SAM2, auto device)
- `configs/hpc_reference.yaml` — HPC production (large SAM2, cuda)
