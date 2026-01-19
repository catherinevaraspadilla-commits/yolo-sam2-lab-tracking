# src/roboflow/close_contact.py

"""
Close-contact detection workflow for lab rats, using YOLOv8.

This module is designed for step-by-step exploration on Bunya:
1) scan      -> run YOLO over the video and classify frames as "rats close / not close"
2) encounters-> group close frames into encounters and summarize them
3) plan      -> build a sampling plan for frames to export to Roboflow
4) export    -> actually save selected frames to the frames/ folder

All intermediate artifacts go to bunya/close_contact/.
Final extracted frames go to frames/.
"""

"""
SAM3 (segmentation) + YOLO (body parts) for close-contact analysis (counting, duration, type) for lab rats.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
from ultralytics import YOLO


# -------------------------------------------------------------------
# Paths and basic setup
# -------------------------------------------------------------------

# This file lives at: ROOT/src/roboflow/close_contact.py
ROOT = Path(__file__).resolve().parents[2]

VIDEOS_DIR = ROOT / "data" / "videos"
MODELS_DIR = ROOT / "models"

# All intermediate / analysis outputs go here (Bunya-friendly, no UI needed)
BUNYA_DIR = ROOT / "bunya"
REVIEW_DIR = BUNYA_DIR / "close_contact"
FRAMES_DIR = ROOT / "frames"

DEFAULT_VIDEO_PATH = VIDEOS_DIR / "original.mp4"
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov8lrata.pt"

REVIEW_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass
class RatDetection:
    """Single rat detection bounding box and confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    def center(self) -> Tuple[float, float]:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        return cx, cy

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class FrameInfo:
    """Summary of a single frame for close-contact analysis."""
    frame_idx: int
    time_sec: float
    num_rats: int
    detections: List[RatDetection]
    is_close: bool
    min_pair_distance_norm: float
    max_iou: float


@dataclass
class Encounter:
    """Continuous period where rats are close."""
    encounter_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    num_close_frames: int


# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------

def compute_iou(a: RatDetection, b: RatDetection) -> float:
    """Compute IoU between two bounding boxes."""
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    union_area = a.area() + b.area() - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def evaluate_closeness(
    detections: List[RatDetection],
    frame_width: int,
    frame_height: int,
    distance_thr_norm: float,
    iou_thr: float,
) -> Tuple[bool, float, float]:
    """
    Decide if rats are "close" in a frame:

    - We normalize distances by the diagonal of the frame.
    - A frame is "close" if:
      * there are at least 2 rats, and
      * either min normalized distance < distance_thr_norm
        OR max IoU between any pair > iou_thr.
    """
    n = len(detections)
    if n < 2:
        return False, 1.0, 0.0

    diag = math.sqrt(frame_width ** 2 + frame_height ** 2)
    if diag <= 0:
        return False, 1.0, 0.0

    min_dist_norm = 1.0
    max_iou = 0.0

    for i in range(n):
        cx1, cy1 = detections[i].center()
        for j in range(i + 1, n):
            cx2, cy2 = detections[j].center()
            dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            dist_norm = dist / diag
            if dist_norm < min_dist_norm:
                min_dist_norm = dist_norm

            iou = compute_iou(detections[i], detections[j])
            if iou > max_iou:
                max_iou = iou

    is_close = (min_dist_norm < distance_thr_norm) or (max_iou > iou_thr)
    return is_close, min_dist_norm, max_iou


# -------------------------------------------------------------------
# STEP 1: Scan video with YOLO and classify frames
# -------------------------------------------------------------------

def scan_video_for_closeness(
    video_path: Path = DEFAULT_VIDEO_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    conf_thr: float = 0.25,
    distance_thr_norm: float = 0.15,
    iou_thr: float = 0.1,
    max_frames: Optional[int] = None,
) -> None:
    """
    Run YOLO on the video and classify each frame as "rats close" or not.
    Outputs:
      - review/close_contact/close_frames.jsonl   (one JSON per frame)
      - review/close_contact/scan_metadata.json   (fps, counts, thresholds, etc.)
    """
    video_path = video_path.resolve()
    model_path = model_path.resolve()

    print(f"[scan] Video: {video_path}")
    print(f"[scan] Model: {model_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[scan] Video properties: {width}x{height} @ {fps:.2f} FPS, total frames: {total_frames}")

    model = YOLO(str(model_path))
    print("[scan] YOLO model loaded")

    frames_file = REVIEW_DIR / "close_frames.jsonl"
    meta_file = REVIEW_DIR / "scan_metadata.json"

    num_processed = 0
    num_close_frames = 0

    with frames_file.open("w", encoding="utf-8") as f_out:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = num_processed
            if max_frames is not None and frame_idx >= max_frames:
                break

            # YOLO inference
            result = model(frame, verbose=False)[0]
            detections: List[RatDetection] = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for bbox, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = bbox.tolist()
                    if conf < conf_thr:
                        continue
                    detections.append(
                        RatDetection(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            conf=float(conf),
                        )
                    )

            time_sec = frame_idx / fps
            is_close, min_dist_norm, max_iou = evaluate_closeness(
                detections,
                frame_width=width,
                frame_height=height,
                distance_thr_norm=distance_thr_norm,
                iou_thr=iou_thr,
            )

            if is_close:
                num_close_frames += 1

            frame_info = FrameInfo(
                frame_idx=frame_idx,
                time_sec=time_sec,
                num_rats=len(detections),
                detections=detections,
                is_close=is_close,
                min_pair_distance_norm=min_dist_norm,
                max_iou=max_iou,
            )

            # Convert dataclasses into JSON-serializable dict
            out_dict: Dict[str, Any] = asdict(frame_info)
            f_out.write(json.dumps(out_dict) + "\n")

            num_processed += 1
            if num_processed % 100 == 0:
                print(
                    f"[scan] Processed {num_processed}/{total_frames} frames "
                    f"({num_close_frames} close so far)"
                )

    cap.release()

    metadata = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "num_processed_frames": num_processed,
        "num_close_frames": num_close_frames,
        "distance_thr_norm": distance_thr_norm,
        "iou_thr": iou_thr,
        "conf_thr": conf_thr,
        "frames_file": str(frames_file),
    }

    with meta_file.open("w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print(f"[scan] Done. Frames file: {frames_file}")
    print(f"[scan] Metadata file: {meta_file}")
    print(f"[scan] Close frames: {num_close_frames} / {num_processed}")


# -------------------------------------------------------------------
# STEP 2: Group close frames into encounters
# -------------------------------------------------------------------

def build_encounters(
    frames_file: Path | None = None,
    meta_file: Path | None = None,
    max_gap_seconds: float = 2.0,
    min_duration_seconds: float = 0.5,
) -> None:
    """
    Group consecutive close frames into encounters.
    Outputs:
      - review/close_contact/encounters_summary.json
    """
    frames_file = frames_file or (REVIEW_DIR / "close_frames.jsonl")
    meta_file = meta_file or (REVIEW_DIR / "scan_metadata.json")

    if not frames_file.exists():
        raise FileNotFoundError(f"Frames file not found: {frames_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    with meta_file.open("r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)

    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])

    max_gap_frames = int(max_gap_seconds * fps)
    min_duration_frames = int(min_duration_seconds * fps)

    encounters: List[Encounter] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_count: int = 0
    last_close_frame: Optional[int] = None

    total_close_frames = 0

    with frames_file.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            fr = json.loads(line)
            if not fr.get("is_close", False):
                continue

            frame_idx = int(fr["frame_idx"])
            total_close_frames += 1

            if current_start is None:
                # Start new encounter
                current_start = frame_idx
                current_end = frame_idx
                current_count = 1
            else:
                gap = frame_idx - int(last_close_frame)
                if gap <= max_gap_frames:
                    # Still same encounter
                    current_end = frame_idx
                    current_count += 1
                else:
                    # Close previous encounter if long enough
                    duration_frames = int(current_end) - int(current_start) + 1
                    if duration_frames >= min_duration_frames:
                        encounter_id = len(encounters)
                        encounters.append(
                            Encounter(
                                encounter_id=encounter_id,
                                start_frame=int(current_start),
                                end_frame=int(current_end),
                                start_time_sec=current_start / fps,
                                end_time_sec=current_end / fps,
                                num_close_frames=current_count,
                            )
                        )
                    # Start new encounter
                    current_start = frame_idx
                    current_end = frame_idx
                    current_count = 1

            last_close_frame = frame_idx

    # Flush last encounter
    if current_start is not None:
        duration_frames = int(current_end) - int(current_start) + 1
        if duration_frames >= min_duration_frames:
            encounter_id = len(encounters)
            encounters.append(
                Encounter(
                    encounter_id=encounter_id,
                    start_frame=int(current_start),
                    end_frame=int(current_end),
                    start_time_sec=current_start / fps,
                    end_time_sec=current_end / fps,
                    num_close_frames=current_count,
                )
            )

    summary = {
        "video_path": meta["video_path"],
        "fps": fps,
        "total_frames": total_frames,
        "total_close_frames": total_close_frames,
        "num_encounters": len(encounters),
        "max_gap_seconds": max_gap_seconds,
        "min_duration_seconds": min_duration_seconds,
        "encounters": [asdict(e) for e in encounters],
    }

    out_file = REVIEW_DIR / "encounters_summary.json"
    with out_file.open("w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=2)

    print(f"[encounters] Found {len(encounters)} encounters")
    for e in encounters:
        print(
            f"  - Encounter {e.encounter_id}: "
            f"{e.start_time_sec:.1f}s – {e.end_time_sec:.1f}s "
            f"({e.num_close_frames} close frames)"
        )
    print(f"[encounters] Summary saved to: {out_file}")


# -------------------------------------------------------------------
# STEP 3: Build sampling plan for frames
# -------------------------------------------------------------------

def build_sampling_plan(
    target_total_frames: int,
    min_per_encounter: int = 10,
    frames_file: Path | None = None,
    encounters_file: Path | None = None,
    meta_file: Path | None = None,
) -> None:
    """
    Build a sampling plan: decide how many frames to export per encounter
    and which frame indices to use.

    Strategy:
      - Approximate equal share across encounters.
      - Respect min_per_encounter.
      - Do not exceed the number of available close frames per encounter.

    Output:
      - review/close_contact/sampling_plan.json
    """
    frames_file = frames_file or (REVIEW_DIR / "close_frames.jsonl")
    encounters_file = encounters_file or (REVIEW_DIR / "encounters_summary.json")
    meta_file = meta_file or (REVIEW_DIR / "scan_metadata.json")

    if not frames_file.exists():
        raise FileNotFoundError(f"Frames file not found: {frames_file}")
    if not encounters_file.exists():
        raise FileNotFoundError(f"Encounters file not found: {encounters_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    with meta_file.open("r", encoding="utf-8") as f_meta:
        meta = json.load(f_meta)
    fps = float(meta["fps"])

    with encounters_file.open("r", encoding="utf-8") as f_enc:
        encounters_summary = json.load(f_enc)

    encounters_data = encounters_summary["encounters"]
    num_encounters = len(encounters_data)

    if num_encounters == 0:
        raise RuntimeError("[plan] No encounters found. Run 'encounters' first and check thresholds.")

    # Map encounters to their frame ranges
    encounter_ranges = []
    for e in encounters_data:
        encounter_ranges.append(
            (
                int(e["encounter_id"]),
                int(e["start_frame"]),
                int(e["end_frame"]),
            )
        )

    # Build mapping encounter_id -> list of close frame indices
    frames_by_encounter: Dict[int, List[int]] = {eid: [] for (eid, _, _) in encounter_ranges}

    with frames_file.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            fr = json.loads(line)
            if not fr.get("is_close", False):
                continue
            frame_idx = int(fr["frame_idx"])

            # Find which encounter this frame belongs to
            for eid, start_f, end_f in encounter_ranges:
                if start_f <= frame_idx <= end_f:
                    frames_by_encounter[eid].append(frame_idx)
                    break

    # Compute desired frames per encounter
    base_per_encounter = max(min_per_encounter, target_total_frames // num_encounters)

    selected_frames: List[Dict[str, Any]] = []
    per_encounter_info: Dict[str, Any] = {}

    for eid, start_f, end_f in encounter_ranges:
        available = frames_by_encounter[eid]
        n_available = len(available)

        if n_available == 0:
            per_encounter_info[str(eid)] = {
                "available_close_frames": 0,
                "selected_frames": 0,
            }
            continue

        desired = min(base_per_encounter, n_available)
        if desired <= 0:
            per_encounter_info[str(eid)] = {
                "available_close_frames": n_available,
                "selected_frames": 0,
            }
            continue

        # Spread selection across available frames
        indices = []
        for i in range(desired):
            pos = int(round(i * (n_available - 1) / max(1, desired - 1)))
            indices.append(available[pos])

        indices = sorted(set(indices))

        per_encounter_info[str(eid)] = {
            "available_close_frames": n_available,
            "selected_frames": len(indices),
        }

        for frame_idx in indices:
            selected_frames.append(
                {
                    "encounter_id": eid,
                    "frame_idx": frame_idx,
                    "time_sec": frame_idx / fps,
                }
            )

    selected_frames = sorted(selected_frames, key=lambda d: d["frame_idx"])
    actual_total = len(selected_frames)

    plan = {
        "target_total_frames": target_total_frames,
        "estimated_base_per_encounter": base_per_encounter,
        "actual_total_frames": actual_total,
        "per_encounter": per_encounter_info,
        "selected_frames": selected_frames,
    }

    out_file = REVIEW_DIR / "sampling_plan.json"
    with out_file.open("w", encoding="utf-8") as f_out:
        json.dump(plan, f_out, indent=2)

    print(f"[plan] Sampling plan saved to: {out_file}")
    print(f"[plan] Target total frames: {target_total_frames}")
    print(f"[plan] Actual total frames selected: {actual_total}")
    for eid_str, info in per_encounter_info.items():
        print(
            f"  - Encounter {eid_str}: "
            f"{info['selected_frames']} / {info['available_close_frames']} close frames selected"
        )


# -------------------------------------------------------------------
# STEP 4: Export frames according to sampling plan
# -------------------------------------------------------------------

def export_frames_from_plan(
    video_path: Path = DEFAULT_VIDEO_PATH,
    plan_file: Path | None = None,
    output_root: Path = FRAMES_DIR,
    filename_prefix: str = "rat_close",
) -> None:
    """
    Export frames according to sampling_plan.json.

    Output structure:
      frames/
        encounter_000/
          rat_close_e000_f000123.jpg
          ...
        encounter_001/
          ...
    """
    video_path = video_path.resolve()
    plan_file = plan_file or (REVIEW_DIR / "sampling_plan.json")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not plan_file.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")

    with plan_file.open("r", encoding="utf-8") as f_plan:
        plan = json.load(f_plan)

    selected_frames = plan.get("selected_frames", [])
    if not selected_frames:
        print("[export] No frames in plan. Nothing to export.")
        return

    # Group by encounter
    frames_by_encounter: Dict[int, List[int]] = {}
    for item in selected_frames:
        eid = int(item["encounter_id"])
        frame_idx = int(item["frame_idx"])
        frames_by_encounter.setdefault(eid, []).append(frame_idx)

    for eid in frames_by_encounter:
        frames_by_encounter[eid] = sorted(set(frames_by_encounter[eid]))

    all_frame_indices = sorted({f for frames in frames_by_encounter.values() for f in frames})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    print(f"[export] Exporting {len(all_frame_indices)} frames from {video_path}")
    current_idx = 0
    total_saved = 0

    # Simple iteration: read frames sequentially and save the selected ones
    target_set = set(all_frame_indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            # Decide which encounter(s) this frame belongs to
            for eid, frame_list in frames_by_encounter.items():
                if current_idx in frame_list:
                    encounter_dir = output_root / f"encounter_{eid:03d}"
                    encounter_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"{filename_prefix}_e{eid:03d}_f{current_idx:06d}.jpg"
                    out_path = encounter_dir / out_name
                    cv2.imwrite(str(out_path), frame)
                    total_saved += 1

        current_idx += 1

    cap.release()
    print(f"[export] Done. Saved {total_saved} frames under: {output_root}")


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Close-contact workflow for lab rat interaction detection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Run YOLO over the video and classify frames.")
    p_scan.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Input video path.")
    p_scan.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO model path.")
    p_scan.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p_scan.add_argument("--distance-thr", type=float, default=0.15, help="Normalized distance threshold.")
    p_scan.add_argument("--iou-thr", type=float, default=0.1, help="IoU threshold for overlap.")
    p_scan.add_argument("--max-frames", type=int, default=None, help="Limit for frames (for quick tests).")

    # encounters
    p_enc = subparsers.add_parser("encounters", help="Group close frames into encounters.")
    p_enc.add_argument("--max-gap-seconds", type=float, default=2.0, help="Max gap between close frames.")
    p_enc.add_argument("--min-duration-seconds", type=float, default=0.5, help="Minimum encounter duration.")

    # plan
    p_plan = subparsers.add_parser("plan", help="Build sampling plan for frames.")
    p_plan.add_argument("--target-total-frames", type=int, required=True, help="Target total frames to export.")
    p_plan.add_argument("--min-per-encounter", type=int, default=10, help="Minimum frames per encounter.")

    # export
    p_export = subparsers.add_parser("export", help="Export frames according to sampling plan.")
    p_export.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Input video path.")
    p_export.add_argument("--plan-file", type=Path, default=None, help="Custom sampling_plan.json path.")
    p_export.add_argument("--output-root", type=Path, default=FRAMES_DIR, help="Root folder for exported frames.")
    p_export.add_argument("--prefix", type=str, default="rat_close", help="Filename prefix for saved frames.")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "scan":
        scan_video_for_closeness(
            video_path=args.video,
            model_path=args.model,
            conf_thr=args.conf,
            distance_thr_norm=args.distance_thr,
            iou_thr=args.iou_thr,
            max_frames=args.max_frames,
        )
    elif args.command == "encounters":
        build_encounters(
            max_gap_seconds=args.max_gap_seconds,
            min_duration_seconds=args.min_duration_seconds,
        )
    elif args.command == "plan":
        build_sampling_plan(
            target_total_frames=args.target_total_frames,
            min_per_encounter=args.min_per_encounter,
        )
    elif args.command == "export":
        export_frames_from_plan(
            video_path=args.video,
            plan_file=args.plan_file,
            output_root=args.output_root,
            filename_prefix=args.prefix,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
