"""
Reference pipeline entry point.

Architecture ported from the reference pipeline (reference/ folder):
  - YOLO detect-only (no BoT-SORT tracking)
  - SAM2ImagePredictor with centroid fallback
  - IdentityMatcher with fixed slots (centroid + area matching)
  - filter_duplicates before matching
  - Contact classification (optional)

The key difference vs sam2_yolo:
  - No YOLO track IDs → no ID switch problems
  - Centroid fallback → SAM2 keeps tracking when YOLO loses a rat
  - IdentityMatcher → simpler, more robust identity management

Usage:
    python -m src.pipelines.reference.run --config configs/local_reference.yaml
    python -m src.pipelines.reference.run --config configs/hpc_reference.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
from datetime import date
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
from src.pipelines.sam2_yolo.models_io import load_models
from src.pipelines.sam2_yolo.infer_yolo import detect_only
from .identity_matcher import IdentityMatcher, filter_duplicates, resolve_overlaps
from .sam2_processor import segment_frame

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    chunk_id: int | None = None,
) -> Path:
    """Run the reference-style pipeline.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Optional key=value config overrides.
        start_frame: First frame to process (0-based).
        end_frame: Stop before this frame (exclusive). None = process to end.
        chunk_id: If set, appended to run directory name for chunk parallelism.
    """
    config = load_config(config_path, cli_overrides)
    tag = f"reference_chunk{chunk_id}" if chunk_id is not None else "reference"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting Reference pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)
    if start_frame > 0 or end_frame is not None:
        logger.info("Frame range: %d → %s", start_frame, end_frame or "end")

    # Load models (SAM2ImagePredictor + YOLO)
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
    sam_ckpt = config.get("models", {}).get("sam2_checkpoint", "")
    model_name = Path(sam_ckpt).stem if sam_ckpt else "sam2"
    today = date.today().strftime("%Y-%m-%d")
    out_video_path = overlays_dir / f"reference_{model_name}_{today}{ext}"
    writer = create_video_writer(
        out_video_path, props["fps"], props["width"], props["height"], codec=codec,
    )

    # Parse config
    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    max_animals = det_cfg.get("max_animals", 2)
    kpt_names = det_cfg.get("keypoint_names")
    kpt_min_conf = det_cfg.get("keypoint_min_conf", 0.3)
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")

    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)
    iou_thr = config.get("segmentation", {}).get("mask_iou_threshold", 0.5)

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    max_frames = config.get("scan", {}).get("max_frames")

    # Create IdentityMatcher (replaces SlotTracker)
    im_cfg = config.get("identity_matcher", {})
    matcher = IdentityMatcher(
        max_entities=max_animals,
        proximity_threshold=im_cfg.get("proximity_threshold", 60.0),
        area_tolerance=im_cfg.get("area_tolerance", 0.4),
    )
    logger.info(
        "IdentityMatcher: max_entities=%d, proximity=%.0f px, area_tolerance=%.1f%%",
        max_animals,
        im_cfg.get("proximity_threshold", 60.0),
        im_cfg.get("area_tolerance", 0.4) * 100,
    )

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

        # Step 1: YOLO detect-only (no BoT-SORT tracking)
        detections = detect_only(
            yolo, frame_rgb, det_conf,
            keypoint_names=kpt_names,
            filter_class=filter_class,
            nms_iou=nms_iou,
        )

        # Step 2: SAM2 segmentation with centroid fallback + keypoint prompts
        all_masks, all_scores = segment_frame(
            sam, frame_rgb, detections, matcher,
            sam_threshold=sam_thr,
            max_entities=max_animals,
            kpt_min_conf=kpt_min_conf,
        )

        # Step 3: Filter duplicates
        unique_masks, unique_scores = filter_duplicates(
            all_masks, all_scores, max_animals, iou_thr,
        )

        # Step 4: Identity matching (assign to fixed slots)
        slot_masks, slot_centroids, slot_areas, slot_scores = matcher.match(
            unique_masks, unique_scores,
        )

        # Step 4b: Resolve overlapping pixels between masks
        resolve_overlaps(slot_masks, slot_centroids)

        # Step 5: Match YOLO detections to slots for keypoint association
        # Find which detection best matches each slot by centroid proximity
        slot_dets = _match_dets_to_slots(detections, slot_centroids, max_animals)

        # Set track_id on matched detections so labels show slot index
        for slot_idx, det in enumerate(slot_dets):
            if det is not None:
                det.track_id = slot_idx + 1

        # Step 6: Contact classification (if enabled)
        contact_events = []
        if contact_tracker is not None:
            contact_events = contact_tracker.update(
                slot_dets, slot_masks, slot_centroids, frame_idx,
            )

        # Step 7: Render overlay
        render_masks = []
        render_colors = []
        render_centroids = []
        for i in range(max_animals):
            if slot_masks[i] is not None:
                render_masks.append(slot_masks[i])
                if colors:
                    render_colors.append(colors[i % len(colors)])
                if slot_centroids[i] is not None:
                    render_centroids.append(slot_centroids[i])

        if not render_colors:
            render_colors = None

        np.copyto(frame_buffer, frame_rgb)
        frame_out = apply_masks_overlay(frame_buffer, render_masks, colors=render_colors)

        # Draw detections aligned to their slots
        ordered_render_dets = [d for d in slot_dets if d is not None]
        if ordered_render_dets:
            frame_out = draw_detections(frame_out, ordered_render_dets, colors=render_colors)
            frame_out = draw_keypoints(
                frame_out, ordered_render_dets, colors=render_colors,
                min_conf=kpt_min_conf,
            )

        if render_centroids:
            frame_out = draw_centroids(frame_out, render_centroids, colors=render_colors)

        active_count = sum(1 for m in slot_masks if m is not None)
        status_text = f"Animals: {active_count}/{max_animals} | Reference"

        # Add contact info to overlay
        if contact_events:
            for ev in contact_events:
                if ev.contact_type is not None:
                    status_text += f"  | {ev.contact_type}"
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
            logger.info("Processed %d frames", frame_count)

        # Periodic GPU memory cleanup
        if frame_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    cap.release()
    writer.release()

    # Finalize contact analysis
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


def _match_dets_to_slots(
    detections: list,
    slot_centroids: list,
    max_animals: int,
) -> list:
    """Match YOLO detections to identity slots by centroid proximity.

    Returns a list of length max_animals where each element is the best
    matching Detection (or None) for that slot.
    """
    result = [None] * max_animals
    if not detections:
        return result

    det_centers = [d.center() for d in detections]
    used = set()

    for slot_idx in range(max_animals):
        sc = slot_centroids[slot_idx]
        if sc is None:
            continue

        best_di = None
        best_dist = float("inf")
        for di, dc in enumerate(det_centers):
            if di in used:
                continue
            dist = ((sc[0] - dc[0]) ** 2 + (sc[1] - dc[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_di = di

        if best_di is not None and best_dist < 200.0:  # max 200px
            result[slot_idx] = detections[best_di]
            used.add(best_di)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reference-style pipeline (IdentityMatcher + centroid fallback).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/local_reference.yaml).",
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
        help="Chunk identifier for parallel processing.",
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
