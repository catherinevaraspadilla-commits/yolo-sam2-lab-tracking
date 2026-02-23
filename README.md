# Mouse Tracking Vision Pipeline (YOLOv8 + SAM2)

AI-based workflow for analyzing laboratory mouse movement from real-world lab videos. Combines YOLOv8 detection with SAM2 segmentation for accurate body-part localization (nose, head, tail), trajectory extraction, and behavior analysis.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run SAM2+YOLO pipeline on a short clip
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml

# Extract close-contact frames for labeling
python scripts/extract_frames.py all --config configs/local_quick.yaml

# Upload frames to Roboflow
export ROBOFLOW_API_KEY="your-key"
python scripts/upload_to_roboflow.py --config configs/local_quick.yaml --frames-root outputs/runs/<run>/frames
```

## How It Works

1. **YOLOv8** detects mouse bounding boxes and body parts
2. **SAM2** segments each detection into pixel-accurate masks
3. **Tracking** maintains consistent identity across frames via centroid matching
4. **Outputs** — overlay videos, extracted frames, run logs

## Project Structure

```
yolo-sam2-lab-tracking/
  configs/                      # YAML configs (local_quick, hpc_full)
  docs/                         # Documentation
  data/
    raw/                        # Full videos (gitignored)
    clips/                      # Short test clips
    roboflow_export/            # Annotated dataset exports
  models/
    yolo/                       # YOLOv8 weights
    sam2/                       # SAM2 checkpoints
  src/
    common/                     # Shared utilities
      config_loader.py          #   YAML loading, run setup, logging
      io_video.py               #   Video read/write/iterate
      metrics.py                #   IoU, centroids, closeness
      visualization.py          #   Mask overlays, annotations
      tracking.py               #   Mask filtering, identity matching
      utils.py                  #   Shared data classes
    pipelines/
      sam2_yolo/                # SAM2+YOLO pipeline
        run.py                  #   Entry point
        models_io.py            #   Model loading
        infer_yolo.py           #   YOLO detection
        infer_sam2.py           #   SAM2 segmentation
        postprocess.py          #   Filtering + tracking
      sam3/                     # SAM3 pipeline (placeholder)
  scripts/
    extract_frames.py           # Close-contact frame extraction
    upload_to_roboflow.py       # Roboflow upload
    extract_clip.py             # Cut clips from raw video
    export_results.py           # Export run artifacts
  slurm/                        # HPC job scripts
  outputs/runs/                 # Structured run outputs
```

## Configuration

All parameters are in YAML configs — no hardcoded thresholds:

- **`configs/local_quick.yaml`** — 6s clip, SAM2 tiny, CPU/GPU auto
- **`configs/hpc_full.yaml`** — Full video, SAM2 large, CUDA

Override any parameter via CLI:
```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml detection.confidence=0.4
```

## Documentation

- [Project Overview](docs/overview.md)
- [Data Notes](docs/data_notes.md)
- [Running Locally](docs/running_local.md)
- [Running on HPC (Bunya)](docs/running_hpc_bunya.md)
- [Labeling Guide](docs/labeling_guide.md)
- [Evaluation Plan](docs/evaluation.md)
- [Refactoring Plan](docs/refactoring_plan.md)

## Environments

| Environment | Config | Video | SAM2 Model | Device |
|-------------|--------|-------|-----------|--------|
| Local laptop | `local_quick.yaml` | 5-10s clips | tiny | auto |
| UQ Bunya HPC | `hpc_full.yaml` | ~20min full | large | cuda |

## Output Structure

```
outputs/runs/<timestamp>/
  config_used.yaml      # Reproducibility snapshot
  logs/run.log          # Execution log
  overlays/overlay.mp4  # Annotated video
  scan/                 # Frame analysis artifacts
  frames/               # Exported frames
```
