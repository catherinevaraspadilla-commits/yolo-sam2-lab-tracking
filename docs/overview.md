# Project Overview

## Purpose

This project develops an AI-based workflow to analyze **laboratory mouse movement** from real-world lab videos. The near-term objective is to detect and track mouse movement accurately, with emphasis on **body-part localization** (nose, head, tail) to enable robust trajectory extraction and higher-level behavior inference (e.g., nose-touching, tail contact).

## Key Deliverable

A practical, repeatable pipeline that turns raw video into structured outputs (detections, masks, tracks, trajectories, and summary plots).

## Success Criteria

### Primary (movement + parts)
- For each frame, accurately identify mouse instances (bounding box / silhouette)
- Localize key body parts: **nose, head, tail** (optionally: tail base, ears, torso)
- Maintain consistent identity across time (tracking), especially during interactions

### Secondary (research readiness)
- Reproducible runs on laptop (short clips) and UQ Bunya HPC (full videos)
- Clear outputs: overlay videos, trajectory plots, experiment logs

## Models

### YOLOv8 (Ultralytics)
- **Role:** Fast detection of objects and/or body parts
- **Use case:** Detect mouse + body parts (nose/head/tail) from labeled images
- Provides bounding boxes and keypoint-like part detection

### SAM2 (Meta)
- **Role:** High-quality promptable segmentation for images and video
- **Use case:** Turn YOLO detections (boxes/points) into clean silhouette masks
- Stabilizes shapes across frames and improves tracking during motion

### SAM3 (exploration)
- **Role:** Potential improvement for segmentation/tracking during interactions
- **Use case:** Run as alternative pipeline and compare against SAM2

## Pipeline Overview

```
1. Video -> frames (or sampled frames)
2. YOLO inference -> mouse + body-part detections
3. SAM inference (SAM2 or SAM3) using YOLO prompts -> segmentation masks
4. Tracking & smoothing -> consistent trajectories over time
5. Outputs -> overlays + trajectories + plots
```

### Two Parallel Tracks
- **SAM2+YOLO pipeline** (`src/pipelines/sam2_yolo/`)
- **SAM3 pipeline** (`src/pipelines/sam3/`) — placeholder for future

Both share common utilities from `src/common/`.

## Mental Model

- **YOLO** finds where and what (mouse + parts)
- **SAM** refines into high-quality shapes (masks)
- **Tracking** turns per-frame results into clean movement trajectories
- **Outputs + evaluation** tell us whether we're improving
