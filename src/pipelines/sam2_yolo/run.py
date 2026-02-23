"""
SAM2+YOLO pipeline entry point.

Processes a video frame-by-frame: YOLO detection -> SAM2 segmentation ->
tracking -> overlay rendering -> output video.

Usage:
    python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml
    python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.io_video import open_video_reader, get_video_properties, create_video_writer, iter_frames
from src.common.visualization import apply_masks_overlay, draw_centroids, draw_detections, draw_text
from .models_io import load_models
from .infer_yolo import detect_boxes
from .infer_sam2 import segment_from_boxes
from .postprocess import postprocess_frame

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str | Path, cli_overrides: List[str] | None = None) -> Path:
    """Run the full SAM2+YOLO video pipeline.

    Args:
        config_path: Path to the YAML config file.
        cli_overrides: Optional list of "key=value" override strings.

    Returns:
        Path to the run directory containing outputs.
    """
    config = load_config(config_path, cli_overrides)
    run_dir = setup_run_dir(config, tag="sam2_yolo")
    setup_logging(run_dir)

    logger.info("Starting SAM2+YOLO pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)

    # Load models
    yolo, sam = load_models(config)

    # Open video
    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)
    logger.info(
        "Video: %s (%dx%d @ %.1f FPS, %d frames)",
        video_path, props["width"], props["height"],
        props["fps"], props["total_frames"],
    )

    # Setup output video
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    codec = config.get("output", {}).get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    out_video_path = overlays_dir / f"overlay{ext}"
    writer = create_video_writer(
        out_video_path, props["fps"], props["width"], props["height"], codec=codec,
    )

    # Parse config values
    det_conf = config.get("detection", {}).get("confidence", 0.25)
    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)
    iou_thr = config.get("segmentation", {}).get("mask_iou_threshold", 0.5)
    max_animals = config.get("detection", {}).get("max_animals", 2)
    strategy = config.get("tracking", {}).get("strategy", "tracking")
    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    max_frames = config.get("scan", {}).get("max_frames")

    # Tracking state
    prev_centroids: List[Tuple[float, float]] = []

    frame_count = 0
    for frame_idx, frame_bgr in iter_frames(cap, max_frames=max_frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Step 1: YOLO detection
        detections = detect_boxes(yolo, frame_rgb, det_conf)

        # Step 2: SAM2 segmentation
        masks = segment_from_boxes(sam, frame_rgb, detections, sam_thr, iou_thr)

        # Step 3: Post-processing (filtering + tracking)
        if strategy == "tracking":
            ordered_masks, centroids = postprocess_frame(masks, prev_centroids, config)
            prev_centroids = [c for c in centroids if c is not None]
        else:
            from src.common.tracking import filter_masks
            ordered_masks = filter_masks(masks, iou_thr, max_animals)
            centroids = []

        # Step 4: Render overlay
        frame_out = apply_masks_overlay(frame_rgb, ordered_masks, colors=colors)
        frame_out = draw_detections(frame_out, detections, colors=colors)
        if strategy == "tracking" and centroids:
            frame_out = draw_centroids(frame_out, centroids, colors=colors)
        frame_out = draw_text(
            frame_out,
            f"Animals: {len(ordered_masks)}/{max_animals}",
        )

        # Write output
        frame_out_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        writer.write(frame_out_bgr)

        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Processed %d frames", frame_count)

    cap.release()
    writer.release()

    logger.info("Pipeline complete. %d frames processed.", frame_count)
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM2+YOLO video inference pipeline.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/local_quick.yaml).",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Optional config overrides as key=value (e.g., detection.confidence=0.5).",
    )
    args = parser.parse_args()

    run_pipeline(args.config, args.overrides or None)


if __name__ == "__main__":
    main()
