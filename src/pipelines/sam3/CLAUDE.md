# sam3 pipeline

SAM3 evaluation pipeline. Mirrors the reference pipeline architecture but uses SAM3 instead of SAM2 for segmentation.

## Files

- `run.py` — Entry point (mirrors `reference/run.py`)
- `models_io.py` — Loads YOLO + SAM3 (`build_sam3_image_model` + `Sam3Processor`)
- `sam3_processor.py` — SAM3 prompts with coordinate normalization

## Key Difference: Coordinate Normalization

SAM3 uses [0,1] normalized coordinates instead of pixel coordinates.
`sam3_processor.py` converts all YOLO-space pixel prompts before sending to SAM3:
- Box: `[x1, y1, x2, y2]` pixels → `[x1/w, y1/h, x2/w, y2/h]`
- Points: `[[x, y]]` pixels → `[[x/w, y/h]]`

## Dependencies

- Reuses `identity_matcher.py` from `reference/` pipeline
- Requires `sam3` package (Python 3.12, PyTorch 2.7+, CUDA 12.6)
- Checkpoint: `models/sam3/sam3.pt` (~3.2GB)

## Configs

- `configs/local_sam3.yaml` — Local testing
- `configs/hpc_sam3.yaml` — HPC production
