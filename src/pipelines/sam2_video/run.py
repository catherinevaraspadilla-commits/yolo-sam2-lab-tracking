"""
SAM2 Video pipeline entry point.

Pipeline flow:
  SAM2VideoPredictor manages segmentation + identity with temporal memory.
  YOLO runs in detect-only mode (no tracking) for keypoints per frame.
  Keypoints are matched to SAM2 masks by nose-in-mask or centroid proximity.
  Optional contact classification runs on the matched results.

The video is processed in segments (default 2000 frames) to bound memory.
At each segment boundary, SAM2 is re-initialized with fresh YOLO boxes.

Usage:
    python -m src.pipelines.sam2_video.run --config configs/local_sam2video.yaml
    python -m src.pipelines.sam2_video.run --config configs/hpc_sam2video.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.contacts import ContactTracker
from src.common.io_video import open_video_reader, get_video_properties, create_video_writer
from src.common.metrics import compute_centroid
from src.common.utils import Detection
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_detections, draw_keypoints, draw_text,
)
from src.pipelines.sam2_yolo.infer_yolo import detect_only
from .models_io import load_models
from .infer_sam2_video import extract_frames_to_jpeg, init_segment, propagate_segment
from .match_keypoints import match_detections_to_masks

logger = logging.getLogger(__name__)


def _find_init_detections(
    yolo_model,
    cap: cv2.VideoCapture,
    abs_start_frame: int,
    max_animals: int,
    det_conf: float,
    kpt_names: Optional[List[str]],
    filter_class: Optional[str],
    nms_iou: Optional[float],
    max_retries: int = 10,
) -> Tuple[List, int]:
    """Find YOLO detections on the first few frames of a segment.

    Retries on subsequent frames if YOLO doesn't detect enough animals on
    the first frame (e.g., one rat is partially occluded).

    Args:
        max_retries: Maximum frames to try before giving up.

    Returns:
        (detections, frame_offset) — detections list and how many frames
        were skipped to find them.
    """
    dets = []
    last_offset = 0
    for offset in range(max_retries):
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_start_frame + offset)
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = detect_only(
            yolo_model, frame_rgb, det_conf,
            keypoint_names=kpt_names,
            filter_class=filter_class,
            nms_iou=nms_iou,
        )
        last_offset = offset
        if len(dets) >= max_animals:
            # Sort by x-position for consistent obj_id ordering
            dets.sort(key=lambda d: d.x1)
            return dets[:max_animals], offset
        logger.debug(
            "Frame %d: only %d/%d detections, retrying...",
            abs_start_frame + offset, len(dets), max_animals,
        )

    # Best effort: use whatever we got on the last attempt
    logger.warning(
        "Could not find %d animals in %d tries near frame %d. Using %d detections.",
        max_animals, max_retries, abs_start_frame, len(dets),
    )
    if dets:
        dets.sort(key=lambda d: d.x1)
        return dets, last_offset
    return [], 0


def _match_segment_identity(
    prev_centroids: Optional[List[Optional[Tuple[float, float]]]],
    new_detections: List,
    max_animals: int,
) -> List[int]:
    """Determine obj_id ordering for a new segment to preserve identity.

    Compares the last segment's mask centroids with the new segment's YOLO
    detection centroids to find the best mapping, so colors don't flip.

    Args:
        prev_centroids: Centroids from the last frame of the previous segment
            (indexed by obj_id). None on the first segment.
        new_detections: YOLO detections for the first frame of the new segment.
        max_animals: Maximum number of animals.

    Returns:
        List of obj_ids to assign to each detection (same length as new_detections).
    """
    n = len(new_detections)
    if prev_centroids is None or n <= 1:
        return list(range(n))

    det_centroids = [d.center() for d in new_detections]

    # For 2 animals: try straight vs swapped
    if n == 2 and len(prev_centroids) >= 2:
        pc_0 = prev_centroids[0]
        pc_1 = prev_centroids[1]
        dc_0 = det_centroids[0]
        dc_1 = det_centroids[1]

        if pc_0 is not None and pc_1 is not None:
            cost_straight = _dist(pc_0, dc_0) + _dist(pc_1, dc_1)
            cost_swapped = _dist(pc_0, dc_1) + _dist(pc_1, dc_0)
            if cost_swapped < cost_straight:
                return [1, 0]

    return list(range(n))


def _dist(a, b):
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
) -> Path:
    """Run the SAM2 Video pipeline.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Optional key=value config overrides.

    Returns:
        Path to the run output directory.
    """
    config = load_config(config_path, cli_overrides)
    run_dir = setup_run_dir(config, tag="sam2_video")
    setup_logging(run_dir)

    logger.info("Starting SAM2 Video pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)

    # Load models
    yolo, sam2_predictor = load_models(config)

    # Open video
    video_path = config["video_path"]
    cap = open_video_reader(video_path)
    props = get_video_properties(cap)
    total_frames = props["total_frames"]
    fps = props["fps"]
    h, w = props["height"], props["width"]
    logger.info(
        "Video: %s (%dx%d @ %.1f FPS, %d frames)",
        video_path, w, h, fps, total_frames,
    )

    # Parse config
    det_cfg = config.get("detection", {})
    det_conf = det_cfg.get("confidence", 0.25)
    max_animals = det_cfg.get("max_animals", 2)
    kpt_names = det_cfg.get("keypoint_names")
    kpt_min_conf = det_cfg.get("keypoint_min_conf", 0.3)
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")

    sam2_cfg = config.get("sam2_video", {})
    segment_size = sam2_cfg.get("segment_size", 2000)
    offload_video = sam2_cfg.get("offload_video_to_cpu", True)
    offload_state = sam2_cfg.get("offload_state_to_cpu", False)

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None

    # Setup output video
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    codec = config.get("output", {}).get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    sam_ckpt = config.get("models", {}).get("sam2_checkpoint", "")
    model_name = Path(sam_ckpt).stem if sam_ckpt else "sam2"
    today = date.today().strftime("%Y-%m-%d")
    out_video_path = overlays_dir / f"sam2_video_{model_name}_{today}{ext}"
    writer = create_video_writer(out_video_path, fps, w, h, codec=codec)

    # Contact classification (optional)
    contacts_enabled = config.get("contacts", {}).get("enabled", False)
    contact_tracker = None
    if contacts_enabled:
        contact_tracker = ContactTracker(
            output_dir=run_dir / "contacts",
            fps=fps,
            num_slots=max_animals,
            video_path=str(video_path),
            config=config,
        )
        logger.info("Contact classification enabled → %s", run_dir / "contacts")

    # Compute segments
    segments = []
    for seg_start in range(0, total_frames, segment_size):
        seg_end = min(seg_start + segment_size, total_frames)
        segments.append((seg_start, seg_end))
    logger.info(
        "Processing %d segments of %d frames each (%d total frames)",
        len(segments), segment_size, total_frames,
    )

    # Persistent render buffer
    frame_buffer = np.empty((h, w, 3), dtype=np.uint8)

    prev_segment_centroids: Optional[List[Optional[Tuple[float, float]]]] = None
    total_processed = 0

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        seg_frames = seg_end - seg_start
        logger.info(
            "=== Segment %d/%d: frames [%d, %d) (%d frames) ===",
            seg_idx + 1, len(segments), seg_start, seg_end, seg_frames,
        )

        # Step 1: Find YOLO detections for SAM2 initialization
        init_dets, init_offset = _find_init_detections(
            yolo, cap, seg_start, max_animals, det_conf,
            kpt_names, filter_class, nms_iou,
        )
        if not init_dets:
            logger.warning("No detections found for segment %d, skipping", seg_idx)
            # Write blank frames for this segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            for _ in range(seg_frames):
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                writer.write(frame_bgr)
                total_processed += 1
            continue

        # Determine obj_id ordering to preserve identity across segments
        obj_id_order = _match_segment_identity(
            prev_segment_centroids, init_dets, max_animals,
        )
        # Reorder detections to match obj_id assignment
        ordered_dets = [None] * len(init_dets)
        for det_idx, obj_id in enumerate(obj_id_order):
            ordered_dets[obj_id] = init_dets[det_idx]
        ordered_dets = [d for d in ordered_dets if d is not None]

        # Step 2: Extract JPEG frames for this segment
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_seg{seg_idx}_"))
        try:
            frames_dir, n_extracted, _ = extract_frames_to_jpeg(
                video_path, tmp_dir, seg_start, seg_end,
            )

            # Step 3: Initialize SAM2 on this segment
            state = init_segment(
                sam2_predictor, frames_dir, ordered_dets,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=offload_state,
            )

            # Step 4: Propagate masks and process each frame
            # We also need original frames for YOLO and rendering
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)

            segment_centroids: List[Optional[Tuple[float, float]]] = [None] * max_animals

            for local_idx, obj_ids, masks in propagate_segment(sam2_predictor, state):
                abs_frame_idx = seg_start + local_idx

                # Read original frame for YOLO + rendering
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Step 4a: YOLO detect-only for keypoints
                yolo_dets = detect_only(
                    yolo, frame_rgb, det_conf,
                    keypoint_names=kpt_names,
                    filter_class=filter_class,
                    nms_iou=nms_iou,
                )

                # Step 4b: Match YOLO detections to SAM2 masks
                det_map = match_detections_to_masks(
                    yolo_dets, obj_ids, masks,
                    nose_kp_index=4,
                    min_kp_conf=kpt_min_conf,
                )

                # Build slot-based outputs (indexed by obj_id)
                slot_masks: List[Optional[np.ndarray]] = [None] * max_animals
                slot_centroids: List[Optional[Tuple[float, float]]] = [None] * max_animals
                slot_dets: List[Optional[Detection]] = [None] * max_animals

                for oi, oid in enumerate(obj_ids):
                    if oi < max_animals:
                        slot_masks[oid] = masks[oi]
                        c = compute_centroid(masks[oi])
                        slot_centroids[oid] = c
                        segment_centroids[oid] = c
                        slot_dets[oid] = det_map.get(oid)

                # Step 4c: Contact classification
                contact_events = []
                if contact_tracker is not None:
                    # Build detection list aligned with slots for contact tracker
                    matched_dets = []
                    for si in range(max_animals):
                        det = slot_dets[si]
                        if det is not None:
                            matched_dets.append(det)
                    contact_events = contact_tracker.update(
                        matched_dets, slot_masks, slot_centroids, abs_frame_idx,
                    )

                # Step 4d: Render overlay
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

                # Draw detections aligned to their masks
                ordered_render_dets = []
                for i in range(max_animals):
                    if slot_dets[i] is not None and slot_masks[i] is not None:
                        ordered_render_dets.append(slot_dets[i])
                if ordered_render_dets:
                    frame_out = draw_detections(frame_out, ordered_render_dets, colors=render_colors)
                    frame_out = draw_keypoints(
                        frame_out, ordered_render_dets, colors=render_colors,
                        min_conf=kpt_min_conf,
                    )

                if render_centroids:
                    frame_out = draw_centroids(frame_out, render_centroids, colors=render_colors)

                active_count = sum(1 for m in slot_masks if m is not None)
                status_text = f"Animals: {active_count}/{max_animals} | SAM2Video"

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

                frame_out_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
                writer.write(frame_out_bgr)

                total_processed += 1
                if total_processed % 50 == 0:
                    logger.info(
                        "Processed %d/%d frames (segment %d/%d)",
                        total_processed, total_frames, seg_idx + 1, len(segments),
                    )

            # Save last centroids for identity continuity
            prev_segment_centroids = segment_centroids

            # Reset SAM2 state
            sam2_predictor.reset_state(state)

        finally:
            # Cleanup temp JPEG directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Free GPU memory between segments
        if torch.cuda.is_available():
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

    logger.info("Pipeline complete. %d frames processed.", total_processed)
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM2 Video inference pipeline (temporal memory tracking).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/local_sam2video.yaml).",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Optional config overrides as key=value (e.g., detection.confidence=0.5).",
    )
    args = parser.parse_args()

    run_pipeline(args.config, args.overrides or None)


if __name__ == "__main__":
    main()
