# Mouse Tracking Vision Pipeline (YOLOv8 + SAM2)

A modular computer-vision pipeline for detecting, segmenting, and tracking multiple rodents in video recordings.
The project demonstrates practical AI engineering, model integration, and software architecture applied to a real laboratory workflow.

## What this project demonstrates

### Computer Vision Engineering

* Custom YOLOv8 model for small-animal detection.
* Precise segmentation using Segment Anything 2 (SAM2).
* Hybrid tracking: centroid matching, IoU filtering, and color-consistent identity assignment.

### Model Integration & GPU Optimization

* Coordinated inference between YOLO and SAM2.
* Dynamic device selection (CPU/GPU) for performance.
* Efficient frame-by-frame video processing.

### Software Engineering

* Clean, modular architecture (config, models_io, strategies, pipeline, video_utils).
* Strategy pattern for switching between tracking modes.
* Proper package structure compatible with VS Code, local dev, and Colab.
* Single command-line entry point using python -m src.run.

### Automation & Data Processing

* Automatic detection, segmentation, mask filtering, and overlay generation.
* Video reader/writer utilities for long recordings.
* Deterministic, reusable pipeline for lab environments.

## High-Level Architecture

src/  
├─ config.py           # Centralized paths & parameters  
├─ models_io.py        # Loads YOLO & SAM2 with device management  
├─ strategies.py       # Simple tracking / consistent-color tracking  
├─ video_utils.py      # IoU, centroids, overlays, frame handling  
├─ pipeline.py         # Runs the video processing loop  
└─ run.py              # CLI entry point  

external/  
└─ segment-anything-2/ # Integrated SAM2 repo  

models/  
└─ yolov8lrata.pt      # Custom YOLOv8 detector

## How the system works

1. YOLOv8 detects rat bounding boxes.
2. SAM2 segments each box, producing pixel-accurate masks.
3. Smart filtering removes duplicate or overlapping masks.
4. Tracking strategy reorders identities per frame, ensuring consistent colors.
5. Pipeline writes processed frames into an output video.

Each stage is decoupled and testable.

## Usage

`python -m src.run \`
`  --strategy tracking \`
`  --input data/videos/input.mp4 \`
`  --output data/results/output.mp4`

### Strategies:

* simple: YOLO + SAM2 with smart mask filtering
* tracking: YOLO + SAM2 + identity-consistent tracking

## Why this project matters

This project shows end-to-end ownership of a vision system:

* Designing an architecture for real-world experimental data
* Integrating heterogeneous models
* Building a robust, configurable processing pipeline
* Ensuring reproducibility and performance on GPU environments

It reflects experience across AI engineering, software engineering, computer vision, and automation, applicable to roles in ML, backend, and R&D teams.
