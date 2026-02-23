# Running Locally

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, CPU works for short clips)
- ffmpeg (optional, for clip extraction)

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd yolo-sam2-lab-tracking

# Create virtual environment
python -m venv .venv

#Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate   # Windows

# Install PyTorch (match your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Or with pyproject.toml:
pip install -e .
```

## Quick Test: SAM2+YOLO Pipeline

Process a 6-second clip with overlay output:

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml
```

Output is written to `outputs/runs/<timestamp>_sam2_yolo/overlays/overlay.mp4`.

Override parameters on the fly:

```bash
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml \
    detection.confidence=0.3 scan.max_frames=50
```

## Quick Test: Frame Extraction

Find close-contact frames and export them:

```bash
# Run all steps (scan + encounters + plan + export)
python scripts/extract_frames.py all --config configs/local_quick.yaml

# Or run steps individually
python scripts/extract_frames.py scan --config configs/local_quick.yaml
python scripts/extract_frames.py encounters --config configs/local_quick.yaml --run-dir outputs/runs/<your-run-dir>
python scripts/extract_frames.py plan --config configs/local_quick.yaml --run-dir outputs/runs/<your-run-dir>
python scripts/extract_frames.py export --config configs/local_quick.yaml --run-dir outputs/runs/<your-run-dir>
```

## Extract a Clip from Raw Video

```bash
python scripts/extract_clip.py \
    --input data/raw/original.mp4 \
    --output data/clips/test-5s.mp4 \
    --start 0 --duration 5
```

## Upload Frames to Roboflow

```bash
export ROBOFLOW_API_KEY="your-key"

python scripts/upload_to_roboflow.py \
    --config configs/local_quick.yaml \
    --frames-root outputs/runs/<run-dir>/frames
```

## Output Structure

Each run creates a timestamped directory:

```
outputs/runs/<timestamp>_<tag>/
  config_used.yaml          # Config snapshot for reproducibility
  logs/
    run.log                 # Full execution log
  overlays/
    overlay.mp4             # Annotated video (pipeline runs)
  scan/
    close_frames.jsonl      # Per-frame analysis (extract_frames)
    scan_metadata.json
    encounters_summary.json
    sampling_plan.json
  frames/
    encounter_000/
      rat_close_e000_f000123.jpg
    encounter_001/
      ...
```
