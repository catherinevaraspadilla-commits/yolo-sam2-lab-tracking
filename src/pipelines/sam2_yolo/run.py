"""
SAM2+YOLO pipeline entry point.

Pipeline flow:
  YOLO model.track() (BoT-SORT) → stable track IDs + boxes + keypoints
  → SAM2 segmentation per box → Hungarian slot assignment (soft costs)
  → optional contact classification → overlay rendering → output video.

Usage:
    python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml
    python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.io_video import (
    open_video_reader, get_video_properties, create_video_writer, iter_frames,
)
from src.common.contacts import ContactTracker
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_detections, draw_keypoints, draw_text,
)
from .models_io import load_models
from .infer_yolo import detect_and_track
from .infer_sam2 import segment_from_boxes
from .postprocess import create_tracker, postprocess_frame

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    chunk_id: int | None = None,
) -> Path:
    """Run the full SAM2+YOLO video pipeline.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Optional key=value config overrides.
        start_frame: First frame to process (0-based). Default 0.
        end_frame: Stop before this frame (exclusive). None = process to end.
        chunk_id: If set, appended to run directory name for chunk parallelism.
    """
    config = load_config(config_path, cli_overrides)
    tag = f"sam2_yolo_chunk{chunk_id}" if chunk_id is not None else "sam2_yolo"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting SAM2+YOLO pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)
    if start_frame > 0 or end_frame is not None:
        logger.info("Frame range: %d → %s", start_frame, end_frame or "end")

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
    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    max_animals = det_cfg.get("max_animals", 2)
    kpt_names = det_cfg.get("keypoint_names")
    kpt_min_conf = det_cfg.get("keypoint_min_conf", 0.3)
    filter_class = det_cfg.get("filter_class")
    border_padding = det_cfg.get("yolo_border_padding_px", 0)

    edge_margin = det_cfg.get("edge_margin", 0)
    nms_iou = det_cfg.get("nms_iou")

    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)
    iou_thr = config.get("segmentation", {}).get("mask_iou_threshold", 0.5)

    # Tracker config
    tracking_cfg = config.get("tracking", {})
    tracker_yaml = tracking_cfg.get("tracker_config", "botsort.yaml")

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    max_frames = config.get("scan", {}).get("max_frames")

    # Log configuration
    logger.info(
        "Tracking: tracker=%s, max_dist=%.0f, max_missing=%d, "
        "costs(dist=%.2f,iou=%.2f,area=%.2f), threshold=%.2f",
        tracker_yaml,
        tracking_cfg.get("max_centroid_distance", 150.0),
        tracking_cfg.get("max_missing_frames", 5),
        tracking_cfg.get("w_dist", 0.4),
        tracking_cfg.get("w_iou", 0.4),
        tracking_cfg.get("w_area", 0.2),
        tracking_cfg.get("cost_threshold", 0.85),
    )
    if border_padding > 0:
        logger.info("YOLO border padding: %d px (mirror)", border_padding)
    if filter_class:
        logger.info("YOLO class filter: '%s'", filter_class)
    if edge_margin > 0:
        logger.info("YOLO edge margin: %d px", edge_margin)
    if nms_iou is not None:
        logger.info("YOLO custom NMS IoU: %.2f", nms_iou)

    # Create slot tracker
    tracker = create_tracker(config)

    # Contact classification (optional)
    contacts_enabled = config.get("contacts", {}).get("enabled", False)
    contact_tracker = None
    if contacts_enabled:
        contact_tracker = ContactTracker(
            output_dir=run_dir / "contacts",
            fps=props["fps"],
            num_slots=max_animals,
            video_path=str(video_path),
            config=config,
        )
        logger.info("Contact classification enabled → %s", run_dir / "contacts")

    # Persistent render buffer
    h, w = props["height"], props["width"]
    frame_buffer = np.empty((h, w, 3), dtype=np.uint8)

    frame_count = 0
    for frame_idx, frame_bgr in iter_frames(
        cap, max_frames=max_frames, start_frame=start_frame, end_frame=end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Step 1: YOLO detection+tracking (BoT-SORT gives stable track IDs)
        detections = detect_and_track(
            yolo, frame_rgb, det_conf,
            tracker_config=tracker_yaml,
            keypoint_names=kpt_names,
            filter_class=filter_class,
            border_padding_px=border_padding,
            edge_margin=edge_margin,
            nms_iou=nms_iou,
        )

        # Step 2: SAM2 segmentation (on original frame, with corrected boxes)
        masks = segment_from_boxes(sam, frame_rgb, detections, sam_thr, iou_thr)

        # Step 3: Slot tracking (YOLO track IDs + Hungarian assignment)
        slot_masks, slot_centroids = postprocess_frame(
            masks, detections, tracker, config,
        )

        # Step 3b: Contact classification (if enabled)
        contact_events = []
        if contact_tracker is not None:
            contact_events = contact_tracker.update(
                detections, slot_masks, slot_centroids, frame_idx,
            )

        # Build render lists (skip None slots)
        render_masks = []
        render_colors = []
        render_centroids = []
        for i in range(tracker.num_slots):
            if slot_masks[i] is not None:
                render_masks.append(slot_masks[i])
                if colors:
                    render_colors.append(colors[i % len(colors)])
                if slot_centroids[i] is not None:
                    render_centroids.append(slot_centroids[i])

        if not render_colors:
            render_colors = None

        # Step 4: Render overlay
        np.copyto(frame_buffer, frame_rgb)
        frame_out = apply_masks_overlay(frame_buffer, render_masks, colors=render_colors)
        frame_out = draw_detections(frame_out, detections, colors=render_colors)
        frame_out = draw_keypoints(
            frame_out, detections, colors=render_colors, min_conf=kpt_min_conf,
        )
        if render_centroids:
            frame_out = draw_centroids(frame_out, render_centroids, colors=render_colors)

        active_count = sum(1 for m in slot_masks if m is not None)
        status_text = f"Animals: {active_count}/{max_animals}"

        # Add contact info to overlay
        if contact_events:
            for ev in contact_events:
                if ev.contact_type is not None:
                    status_text += f"  | {ev.contact_type}"
                    # Draw line between centroids during contact
                    ca = slot_centroids[ev.rat_a_slot]
                    cb = slot_centroids[ev.rat_b_slot]
                    if ca is not None and cb is not None:
                        cv2.line(
                            frame_out,
                            (int(ca[0]), int(ca[1])),
                            (int(cb[0]), int(cb[1])),
                            (255, 255, 0), 2,
                        )

        frame_out = draw_text(frame_out, status_text)

        # Write output
        frame_out_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        writer.write(frame_out_bgr)

        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Processed %d frames | %s", frame_count, tracker.get_debug_info())

        # Periodic GPU memory cleanup to prevent OOM on long videos
        if frame_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    cap.release()
    writer.release()

    # Finalize contact analysis (write bouts CSV, session JSON, PDF report)
    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        total_bouts = sum(
            v.get("total_bouts", 0)
            for v in summary.get("contact_type_summary", {}).values()
        )
        logger.info("Contact analysis: %d bouts detected", total_bouts)

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
        "--start-frame", type=int, default=0,
        help="First frame to process (0-based). Default: 0.",
    )
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Stop before this frame (exclusive). Default: process to end.",
    )
    parser.add_argument(
        "--chunk-id", type=int, default=None,
        help="Chunk identifier for parallel processing. Appended to run dir name.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Optional config overrides as key=value (e.g., detection.confidence=0.5).",
    )
    args = parser.parse_args()

    run_pipeline(
        args.config,
        args.overrides or None,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        chunk_id=args.chunk_id,
    )


if __name__ == "__main__":
    main()
