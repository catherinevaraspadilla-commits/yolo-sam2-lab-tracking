"""
Centroid pipeline — SAM2 centroid propagation + YOLO keypoints + contacts.

Architecture:
  - SAM2 centroid propagation drives masks and identity (stable backbone)
  - YOLO runs on the full image every frame but only for keypoints
  - Keypoints assigned to masks by spatial overlap (not YOLO box ordering)
  - Temporal carry-over fills in keypoints when YOLO misses a rat
  - ContactTracker runs on the stable masks + assigned keypoints

Key difference from reference pipeline:
  - YOLO only needed on frame 0 for initialization (boxes → SAM2 masks)
  - After init, SAM2 propagates using prev-frame centroids as point prompts
  - No IdentityMatcher, no SEPARATE/MERGED state machine
  - YOLO boxes are ignored after init — only keypoints extracted

Usage:
    python -m src.pipelines.centroid.run --config configs/local_centroid.yaml
    python -m src.pipelines.centroid.run --config configs/hpc_centroid.yaml \\
        contacts.enabled=true
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.contacts import ContactTracker
from src.common.io_video import (
    open_video_reader, get_video_properties, create_video_writer, iter_frames,
)
from src.common.metrics import compute_centroid
from src.common.geometry import euclidean_distance
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_keypoints, draw_text,
)
from src.common.geometry import resolve_overlaps
from src.common.yolo_inference import detect_only
from src.common.model_loaders import load_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2 segmentation helpers (from debug_sam2_no_yolo.py)
# ---------------------------------------------------------------------------

def _segment_from_boxes(predictor, frame_rgb, detections, sam_threshold):
    """Segment using YOLO boxes (initialization only)."""
    predictor.set_image(frame_rgb)
    masks, scores = [], []
    for det in detections:
        box = np.array([det.x1, det.y1, det.x2, det.y2])
        raw, sc, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=box, multimask_output=False,
        )
        masks.append(raw[0] > sam_threshold)
        scores.append(float(sc[0]))
    return masks, scores


def _segment_from_centroids(predictor, frame_rgb, centroids, sam_threshold):
    """Segment using previous-frame centroids as point prompts.

    Each rat gets its own centroid as a positive prompt and the other
    rat's centroid as a negative prompt to help SAM2 distinguish them.
    """
    predictor.set_image(frame_rgb)
    masks, scores = [], []
    for i, (cx, cy) in enumerate(centroids):
        pos = [[cx, cy]]
        neg = [[ox, oy] for j, (ox, oy) in enumerate(centroids) if j != i]
        all_points = pos + neg
        all_labels = [1] * len(pos) + [0] * len(neg)
        raw, sc, _ = predictor.predict(
            point_coords=np.array(all_points, dtype=np.float32),
            point_labels=np.array(all_labels, dtype=np.int32),
            box=None, multimask_output=True,
        )
        best_idx = int(np.argmax(sc))
        masks.append(raw[best_idx] > sam_threshold)
        scores.append(float(sc[best_idx]))
    return masks, scores


# ---------------------------------------------------------------------------
# Keypoint assignment and carry-over
# ---------------------------------------------------------------------------

def _assign_keypoints_to_masks(
    detections, masks, centroids,
) -> List[Optional["Detection"]]:
    """Assign YOLO detections to SAM2 masks by spatial overlap.

    For each detection, check if its bounding box center falls inside
    a mask. If yes, assign to that slot. If ambiguous or outside all
    masks, assign to the nearest mask centroid.

    Returns slot-aligned list (length = len(masks)).
    """
    n_slots = len(masks)
    slot_dets: List[Optional["Detection"]] = [None] * n_slots
    used_dets = set()

    # First pass: assign by mask containment
    for det in detections:
        cx, cy = det.center()
        ix, iy = int(round(cx)), int(round(cy))
        best_slot = None
        for i, mask in enumerate(masks):
            if slot_dets[i] is not None:
                continue  # slot already taken
            h, w = mask.shape
            if 0 <= iy < h and 0 <= ix < w and mask[iy, ix]:
                best_slot = i
                break
        if best_slot is not None:
            slot_dets[best_slot] = det
            used_dets.add(id(det))

    # Second pass: unassigned detections → nearest centroid (fallback)
    for det in detections:
        if id(det) in used_dets:
            continue
        cx, cy = det.center()
        best_slot = None
        best_dist = float("inf")
        for i in range(n_slots):
            if slot_dets[i] is not None:
                continue
            d = euclidean_distance((cx, cy), centroids[i])
            if d < best_dist:
                best_dist = d
                best_slot = i
        if best_slot is not None and best_dist < 200.0:
            slot_dets[best_slot] = det
            used_dets.add(id(det))

    # Set track_id for rendering
    for i, det in enumerate(slot_dets):
        if det is not None:
            det.track_id = i + 1

    return slot_dets


def _carry_over_keypoints(
    slot_dets,
    prev_slot_dets,
    prev_centroids,
    curr_centroids,
):
    """Fill missing detections with previous frame's keypoints.

    Shifts carried-over keypoint positions by the centroid delta
    (curr_centroid - prev_centroid) to approximate movement.
    Returns the updated slot_dets list (modified in place).
    """
    if prev_slot_dets is None:
        return slot_dets

    for i in range(len(slot_dets)):
        if slot_dets[i] is not None:
            continue  # has fresh keypoints
        if prev_slot_dets[i] is None:
            continue  # nothing to carry over
        if prev_centroids[i] is None or curr_centroids[i] is None:
            continue

        # Compute centroid delta for position shift
        dx = curr_centroids[i][0] - prev_centroids[i][0]
        dy = curr_centroids[i][1] - prev_centroids[i][1]

        # Deep copy previous detection and shift keypoints
        carried = copy.deepcopy(prev_slot_dets[i])
        carried.x1 += dx
        carried.y1 += dy
        carried.x2 += dx
        carried.y2 += dy
        carried.track_id = i + 1
        # Mark as carried-over so ContactTracker can flag stale keypoints
        carried.is_carried_over = True
        if carried.keypoints:
            for kp in carried.keypoints:
                kp.x += dx
                kp.y += dy
        slot_dets[i] = carried

    return slot_dets


def _stabilize_keypoints(
    slot_dets, slot_masks, prev_slot_dets, prev_centroids, curr_centroids,
    ema_alpha=0.5,
):
    """Anchor keypoints to SAM2 masks with EMA smoothing.

    YOLO keypoints jump frame-to-frame because each detection is independent.
    SAM2 masks are much more stable (centroid propagation). This function
    makes keypoints follow the mask instead of YOLO:

    - Keypoint INSIDE mask  → EMA blend with previous position (smooth)
    - Keypoint OUTSIDE mask → use previous position + centroid delta (reject YOLO)

    ema_alpha: weight for current YOLO position (0=pure tracking, 1=pure YOLO).
    """
    if prev_slot_dets is None:
        return slot_dets

    for i, det in enumerate(slot_dets):
        if det is None or not det.keypoints:
            continue
        mask = slot_masks[i]
        if mask is None:
            continue

        prev_det = prev_slot_dets[i] if i < len(prev_slot_dets) else None
        if prev_det is None or not prev_det.keypoints:
            continue

        # Centroid delta for shifting previous positions
        dx, dy = 0.0, 0.0
        if (prev_centroids and curr_centroids
                and prev_centroids[i] is not None
                and curr_centroids[i] is not None):
            dx = curr_centroids[i][0] - prev_centroids[i][0]
            dy = curr_centroids[i][1] - prev_centroids[i][1]

        h, w = mask.shape
        for k, kp in enumerate(det.keypoints):
            if k >= len(prev_det.keypoints):
                break
            prev_kp = prev_det.keypoints[k]
            shifted_x = prev_kp.x + dx
            shifted_y = prev_kp.y + dy

            ix, iy = int(round(kp.x)), int(round(kp.y))
            inside = 0 <= iy < h and 0 <= ix < w and mask[iy, ix]

            if inside:
                # EMA smooth: blend YOLO with previous shifted position
                kp.x = ema_alpha * kp.x + (1 - ema_alpha) * shifted_x
                kp.y = ema_alpha * kp.y + (1 - ema_alpha) * shifted_y
            else:
                # Outside mask → reject YOLO, use mask-tracked position
                kp.x = shifted_x
                kp.y = shifted_y

    return slot_dets


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path,
    cli_overrides: List[str] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    chunk_id: int | None = None,
) -> Path:
    """Run the centroid pipeline.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Optional key=value config overrides.
        start_frame: First frame to process (0-based).
        end_frame: Stop before this frame (exclusive). None = all.
        chunk_id: Chunk ID for parallel execution labeling.
    """
    config = load_config(config_path, cli_overrides)
    tag = f"centroid_chunk{chunk_id}" if chunk_id is not None else "centroid"
    run_dir = setup_run_dir(config, tag=tag)
    setup_logging(run_dir)

    logger.info("Starting Centroid pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Run directory: %s", run_dir)
    if start_frame > 0 or end_frame is not None:
        logger.info("Frame range: %d -> %s", start_frame, end_frame or "end")

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

    # Output video
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    codec = config.get("output", {}).get("video_codec", "XVID")
    ext = ".avi" if codec == "XVID" else ".mp4"
    today = date.today().strftime("%Y-%m-%d")
    out_video_path = overlays_dir / f"centroid_{today}{ext}"
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

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    max_frames = config.get("scan", {}).get("max_frames")

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
        logger.info("Contact classification enabled -> %s", run_dir / "contacts")

    # Centroid propagation state
    prev_centroids: Optional[List[Tuple[float, float]]] = None
    prev_slot_dets: Optional[List[Optional["Detection"]]] = None
    initialized = False
    yolo_uses = 0

    # Render buffer
    h, w = props["height"], props["width"]
    frame_buffer = np.empty((h, w, 3), dtype=np.uint8)

    frame_count = 0
    carried_count = 0  # frames where at least one slot was carried over

    for frame_idx, frame_bgr in iter_frames(
        cap, max_frames=max_frames, start_frame=start_frame, end_frame=end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        slot_masks: List[Optional[np.ndarray]] = [None] * max_animals
        slot_centroids: List[Optional[Tuple[float, float]]] = [None] * max_animals
        slot_dets: List[Optional["Detection"]] = [None] * max_animals
        mode = "none"
        has_carry_over = False

        # ------------------------------------------------------------------
        # Step 1: SAM2 masks (centroid propagation or YOLO init)
        # ------------------------------------------------------------------
        if not initialized:
            # YOLO initialization: find first frame with enough detections
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names, filter_class=filter_class,
                nms_iou=nms_iou,
            )[:max_animals]

            if len(detections) >= max_animals:
                masks, scores = _segment_from_boxes(sam, frame_rgb, detections, sam_thr)
                centroids = [compute_centroid(m) for m in masks]
                if all(c is not None for c in centroids):
                    for i in range(max_animals):
                        slot_masks[i] = masks[i]
                        slot_centroids[i] = centroids[i]
                    # Assign init keypoints directly (ordered by YOLO detection)
                    for i, det in enumerate(detections[:max_animals]):
                        det.track_id = i + 1
                        slot_dets[i] = det
                    prev_centroids = list(centroids)
                    prev_slot_dets = list(slot_dets)
                    initialized = True
                    yolo_uses += 1
                    mode = f"YOLO-INIT ({len(detections)} det)"
                    logger.info("Initialized at frame %d (%d detections)", frame_idx, len(detections))
            else:
                mode = f"WAITING ({len(detections)}/{max_animals} det)"
        else:
            # ----------------------------------------------------------
            # Step 1a: YOLO detection (every frame — for keypoints + boxes)
            # ----------------------------------------------------------
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names, filter_class=filter_class,
                nms_iou=nms_iou,
            )

            # ----------------------------------------------------------
            # Step 1b: SAM2 segmentation — centroid prompts always
            # ----------------------------------------------------------
            masks, scores = _segment_from_centroids(
                sam, frame_rgb, prev_centroids, sam_thr,
            )
            prompt_mode = "CENTROID"

            # Update centroids from new masks
            new_centroids = []
            for i, m in enumerate(masks):
                c = compute_centroid(m)
                if c is not None:
                    new_centroids.append(c)
                    slot_masks[i] = m
                    slot_centroids[i] = c
                elif prev_centroids[i] is not None:
                    new_centroids.append(prev_centroids[i])
                    slot_centroids[i] = prev_centroids[i]
                else:
                    new_centroids.append(None)

            # Resolve overlapping mask pixels by centroid proximity
            resolve_overlaps(slot_masks, slot_centroids)

            # ----------------------------------------------------------
            # Step 2: Assign keypoints to masks by spatial overlap
            # ----------------------------------------------------------
            valid_masks = [m for m in slot_masks if m is not None]
            valid_centroids = [c for c in slot_centroids if c is not None]
            if valid_masks and valid_centroids:
                slot_dets = _assign_keypoints_to_masks(
                    detections, slot_masks, slot_centroids,
                )

            # ----------------------------------------------------------
            # Step 3: Temporal carry-over for missing keypoints
            # ----------------------------------------------------------
            slot_dets = _carry_over_keypoints(
                slot_dets, prev_slot_dets, prev_centroids, new_centroids,
            )

            # ----------------------------------------------------------
            # Step 3b: Stabilize keypoints against SAM2 masks
            # ----------------------------------------------------------
            slot_dets = _stabilize_keypoints(
                slot_dets, slot_masks, prev_slot_dets,
                prev_centroids, new_centroids,
            )

            n_fresh = sum(1 for d in slot_dets if d is not None and id(d) in {id(x) for x in detections})
            n_total = sum(1 for d in slot_dets if d is not None)
            has_carry_over = n_total > n_fresh

            if has_carry_over:
                carried_count += 1

            prev_centroids = new_centroids
            prev_slot_dets = list(slot_dets)
            mode = f"{prompt_mode} ({len(detections)} det, {n_fresh} assigned, {n_total - n_fresh} carried)"

        # ------------------------------------------------------------------
        # Step 4: Contact classification (if enabled)
        # ------------------------------------------------------------------
        contact_events = []
        if contact_tracker is not None and initialized:
            contact_events = contact_tracker.update(
                slot_dets, slot_masks, slot_centroids, frame_idx,
            )

        # ------------------------------------------------------------------
        # Step 6: Render overlay
        # ------------------------------------------------------------------
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

        # Draw keypoints (no bounding boxes — YOLO boxes not used in this pipeline)
        ordered_dets = [d for d in slot_dets if d is not None]
        if ordered_dets:
            frame_out = draw_keypoints(
                frame_out, ordered_dets, colors=render_colors,
                min_conf=kpt_min_conf,
            )

        if render_centroids:
            frame_out = draw_centroids(frame_out, render_centroids, colors=render_colors)

        # Contact overlay
        if contact_events:
            for ev in contact_events:
                if ev.contact_type is not None:
                    ca = slot_centroids[ev.rat_a_slot]
                    cb = slot_centroids[ev.rat_b_slot]
                    if ca is not None and cb is not None:
                        cv2.line(
                            frame_out,
                            (int(ca[0]), int(ca[1])),
                            (int(cb[0]), int(cb[1])),
                            (255, 255, 0), 2,
                        )

        # Status bar
        active_count = sum(1 for m in slot_masks if m is not None)
        status = f"Animals: {active_count}/{max_animals} | Centroid | {mode}"
        if contact_events:
            for ev in contact_events:
                if ev.contact_type is not None:
                    status += f" | {ev.contact_type}"
        frame_out = draw_text(frame_out, status)

        writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))

        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Processed %d frames (mode=%s)", frame_count, mode)

        if frame_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    cap.release()
    writer.release()

    # Finalize contacts
    if contact_tracker is not None:
        summary = contact_tracker.finalize()
        total_bouts = sum(
            v.get("total_bouts", 0)
            for v in summary.get("contact_type_summary", {}).values()
        )
        logger.info("Contact analysis: %d bouts detected", total_bouts)

        # Post-process contacts into clean events
        try:
            from scripts.postprocess_contacts_simple import run_postprocess
            run_postprocess(run_dir / "contacts", fps=props["fps"])
        except Exception as e:
            logger.warning("Contact post-processing failed: %s", e)

    logger.info("Pipeline complete. %d frames processed. YOLO used %d times for init.", frame_count, yolo_uses)
    logger.info("Frames with keypoint carry-over: %d/%d (%.1f%%)",
                carried_count, frame_count,
                100 * carried_count / max(frame_count, 1))
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centroid pipeline: SAM2 centroid propagation + YOLO keypoints + contacts.",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Chunk identifier for parallel processing.")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides as key=value.")
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
