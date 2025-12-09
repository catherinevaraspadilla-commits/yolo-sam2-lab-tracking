# Close-Contact Detection for Lab Rats (YOLOv8 + Bunya)

## Overview
This module detects moments where rats are close to each other in a video and prepares a balanced set of frames for annotation in Roboflow. The workflow is step-by-step and designed for exploration on Bunya.

## Folder Structure
yolo-sam2-lab-tracking/
  data/
    videos/
      original.mp4
  models/
    yolov8lrata.pt
  src/
    roboflow/
      close_contact.py
  bunya/
    close_contact/
  frames/
    encounter_000/
    encounter_001/
  requirements.txt


## Simple Execution
python -m src.roboflow.close_contact scan
python -m src.roboflow.close_contact encounters
python -m src.roboflow.close_contact plan --target-total-frames 200
python -m src.roboflow.close_contact export

## Detailed Workflow

### Step 1: Scan Video
python -m src.roboflow.close_contact scan \
    --video data/videos/original.mp4 \
    --model models/yolov8lrata.pt \
    --conf 0.25 \
    --distance-thr 0.15 \
    --iou-thr 0.10

Outputs:
* bunya/close_contact/close_frames.jsonl
* bunya/close_contact/scan_metadata.json

### Step 2: Build Encounters
python -m src.roboflow.close_contact encounters \
    --max-gap-seconds 2.0 \
    --min-duration-seconds 0.5

Outputs:
* bunya/close_contact/encounters_summary.json

### Step 3: Build Sampling Plan
python -m src.roboflow.close_contact plan \
    --target-total-frames 200 \
    --min-per-encounter 10

Outputs:
* bunya/close_contact/sampling_plan.json

### Step 4: Export Frames
python -m src.roboflow.close_contact export \
    --video data/videos/original.mp4 \
    --prefix rat_close

Outputs go to:
frames/
  encounter_000/
  encounter_001/

## Interpretation of Closeness
A frame is considered “close” if:
* At least two rats are detected.
* Either the normalized centroid distance is below the threshold, or IoU is above the threshold.

## Tuning Strategy
* Increase distance threshold if too few frames are detected as close.
* Decrease it if too many are detected.
* Adjust max-gap-seconds and min-duration-seconds to merge or split encounters.

## Full Example
python -m src.roboflow.close_contact scan \
    --video data/videos/original.mp4 \
    --model models/yolov8lrata.pt \
    --conf 0.30 \
    --distance-thr 0.14 \
    --iou-thr 0.10

python -m src.roboflow.close_contact encounters \
    --max-gap-seconds 2.0 \
    --min-duration-seconds 0.5

python -m src.roboflow.close_contact plan \
    --target-total-frames 150 \
    --min-per-encounter 15

python -m src.roboflow.close_contact export \
    --video data/videos/original.mp4 \
    --prefix rat_close

## Important Note
The process stops after generating frames. Frames are not automatically uploaded to Roboflow.
