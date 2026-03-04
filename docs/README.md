# Documentation Index

## Setup

| Document | Description |
|----------|-------------|
| [Local Setup](setup/local.md) | Prerequisites, venv, quick tests |
| [HPC Setup (Bunya)](setup/hpc.md) | Cluster setup, salloc, parallel runs, file transfer, troubleshooting |

## Architecture

| Document | Description |
|----------|-------------|
| [Overview](architecture/overview.md) | Project purpose, models, success criteria |
| [Pipeline Comparison](architecture/pipelines.md) | sam2_yolo vs sam2_video vs reference vs sam3 — flow, tracking, recommendation |
| [Risks & Limitations](architecture/risks.md) | Known failure modes, workarounds, and risk priorities |
| [Identity Matcher](architecture/identity_matcher.md) | Hungarian matching + SEPARATE/MERGED state machine design |
| [Tracking Analysis](architecture/tracking_analysis.md) | BoT-SORT and ByteTrack parameter deep-dive |
| [Pipeline Improvements](architecture/pipeline_improvements.md) | sam2_yolo pipeline design, BoT-SORT config, debug signals |
| [Reference Comparison](architecture/reference_comparison.md) | What we adopted/rejected from the reference toolkit |

## Guides

| Document | Description |
|----------|-------------|
| [Parameter Tuning](guides/parameter_tuning.md) | Config reference with tuning guidance and problem-solution mapping |
| [Labeling](guides/labeling.md) | Minimum viable labels, hard cases, Roboflow workflow |
| [Evaluation](guides/evaluation.md) | Qualitative checks, quantitative metrics, comparison framework |
| [Debugging](guides/debugging.md) | YOLO-only and SAM2-only debug scripts to isolate pipeline problems |

## Models

| Document | Description |
|----------|-------------|
| [YOLO26 Migration](models/yolo26_migration.md) | Migration from YOLOv8 to YOLO26 — changes, retraining, NMS-free impact |

## Contacts

| Document | Description |
|----------|-------------|
| [Contact Design](contacts/design.md) | 5 contact types (N2N, N2AG, N2B, SBS, FOL), detection rules |
| [Output Format](contacts/output_format.md) | CSV columns, bout format, session JSON, PDF report |

## Data

| Document | Description |
|----------|-------------|
| [Data Notes](data/notes.md) | Input format, execution modes, domain gap, key challenges |
| [Output Structure](data/output_structure.md) | Run directories, chunk outputs, merged results, contact files |

## Changes (Observation Tracking)

| Document | Description |
|----------|-------------|
| [Observation Log](changes/observation_log.md) | Chronological log of video observations, problems, and fixes |
| [Pipeline Comparison](changes/pipeline_comparison.md) | Side-by-side comparison of pipeline runs on the same video |

## Archive

| Document | Description |
|----------|-------------|
| [Refactoring Plan](archive/refactoring_plan.md) | Historical record of the initial codebase restructuring |
