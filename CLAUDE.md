# Rat Tracking Vision Pipeline

## Project

YOLO26 + SAM2/SAM3 pipeline for laboratory rat tracking and social contact classification.
Runs locally (short clips) and on UQ Bunya HPC (full videos, multi-GPU parallel).

## Active Pipeline

**Reference pipeline** (`src/pipelines/reference/`) is the recommended pipeline.
- YOLO26 detect-only (no BoT-SORT) → SAM2 segment → IdentityMatcher → ContactTracker
- SAM3 pipeline (`src/pipelines/sam3/`) mirrors reference but swaps SAM2 for SAM3
- IdentityMatcher uses Hungarian assignment + SEPARATE/MERGED state machine
- 7 keypoints: tail_tip, tail_base, tail_start, mid_body, nose, right_ear, left_ear

## Key Paths

| What | Path |
|------|------|
| Reference pipeline | `src/pipelines/reference/` |
| SAM3 pipeline | `src/pipelines/sam3/` |
| Identity matcher | `src/pipelines/reference/identity_matcher.py` |
| SAM2 processor | `src/pipelines/reference/sam2_processor.py` |
| SAM3 processor | `src/pipelines/sam3/sam3_processor.py` |
| Contact tracker | `src/common/contacts.py` |
| Shared geometry | `src/common/geometry.py` |
| Shared metrics | `src/common/metrics.py` |
| YOLO26 weights | `models/yolo/modelyolo26.pt` |
| Reference configs | `configs/local_reference.yaml`, `configs/hpc_reference.yaml` |
| SAM3 configs | `configs/local_sam3.yaml`, `configs/hpc_sam3.yaml` |
| Parallel runner | `scripts/run_parallel.sh` |
| Chunk merger | `scripts/merge_chunks.py` |

## Running

```bash
# Local (10s clip) — reference pipeline
python -m src.pipelines.reference.run --config configs/local_reference.yaml

# Local — SAM3 pipeline
python -m src.pipelines.sam3.run --config configs/local_sam3.yaml

# Local with contacts
python -m src.pipelines.reference.run --config configs/local_reference.yaml contacts.enabled=true

# HPC (multi-GPU parallel)
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

## Documentation

See `docs/README.md` for the full index. Key docs:

| Folder | Content |
|--------|---------|
| `docs/setup/` | Local and HPC setup |
| `docs/architecture/` | System design, pipeline comparison, identity matcher |
| `docs/guides/` | Parameter tuning, labeling, evaluation |
| `docs/models/` | YOLO26 migration plan |
| `docs/contacts/` | Social contact design and output format |
| `docs/data/` | Data notes and output structure |
| `docs/changes/` | Observation log — pipeline improvements tracked by date |
| `docs/archive/` | Historical documents |

## Current Status

- SAM3 pipeline added (2026-03-03)
- YOLO26 integrated across all 8 configs (2026-03-03)
- IdentityMatcher state machine implemented (base version)
- 5 robustness fixes designed but **not applied** — waiting for test results
- ContactTracker with 5 contact types operational
- Parallel execution on Bunya working (scripts/run_parallel.sh)

## Conventions

- All configs in `configs/` — no hardcoded thresholds in code
- CLI overrides: `key.subkey=value` (e.g., `detection.confidence=0.4`)
- Output to `outputs/runs/<timestamp>_<tag>/`
- Logs go to `outputs/runs/<run>/logs/run.log`
