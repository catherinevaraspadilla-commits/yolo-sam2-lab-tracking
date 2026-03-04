# Debugging: Isolating YOLO vs SAM2 Problems

When the reference pipeline produces bad output (ID swaps, disappearing masks,
intermittent masks), use these debug scripts to identify the root cause.

## Decision Tree

```
Problem in overlay video?
  ├─ Is YOLO detecting both rats? → Run debug_yolo_only.py
  │    ├─ YOLO misses a rat often → YOLO problem (retrain, lower confidence)
  │    ├─ YOLO boxes overlap a lot → Boxes confuse SAM2 → verify with sam2_only
  │    └─ YOLO detections look clean → SAM2 or IdentityMatcher problem
  │
  └─ Are SAM2 masks correct? → Run debug_sam2_only.py
       ├─ Masks merge into one blob → SAM2 can't separate with current boxes
       ├─ Masks flicker/disappear → Centroid fallback unreliable
       ├─ Masks look good → IdentityMatcher is the problem (tune params)
       └─ Try --no-fallback → Isolates box-only vs centroid fallback quality
```

## YOLO-Only Debug

Shows only YOLO bounding boxes and keypoints, no segmentation.

```bash
# Local
python scripts/debug_yolo_only.py --config configs/local_reference.yaml

# HPC
python scripts/debug_yolo_only.py --config configs/hpc_reference.yaml \
    video_path=data/raw/original_120s.avi

# Specific frame range
python scripts/debug_yolo_only.py --config configs/local_reference.yaml \
    --start-frame 100 --end-frame 200
```

**What to look for:**
- Status bar shows `YOLO: N det` — should be 2 when both rats are visible
- Frames where detection count drops to 0 or 1
- Boxes that overlap heavily during interactions
- Keypoints landing on the wrong rat

## SAM2-Only Debug

Shows raw SAM2 masks without identity matching. Colors are by detection order
(mask 0 = green, mask 1 = red), NOT by identity.

```bash
# With centroid fallback (default)
python scripts/debug_sam2_only.py --config configs/local_reference.yaml

# Without centroid fallback (pure YOLO boxes)
python scripts/debug_sam2_only.py --config configs/local_reference.yaml --no-fallback

# HPC
python scripts/debug_sam2_only.py --config configs/hpc_reference.yaml \
    video_path=data/raw/original_120s.avi
```

**What to look for:**
- Status bar shows `N masks (X box + Y fallback)` — see how many come from boxes vs centroid
- Masks merging into a single blob during interactions
- Mask quality when using `--no-fallback` vs default
- Score values: low scores (<0.5) indicate SAM2 is uncertain

## Output

Both scripts create a run directory under `outputs/runs/` with:
- `overlays/yolo_debug_*.avi` or `overlays/sam2_debug_*.avi`
- `logs/run.log`
