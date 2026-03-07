"""
Centroid pipeline — SAM2 centroid propagation (YOLO init only).

Architecture:
  - YOLO runs ONLY on the first frame to initialize (box prompts → 2 masks)
  - After init, SAM2 propagates using centroid point prompts (multimask_output=False)
  - Each rat: positive prompt (own centroid) + negative prompt (other rat's centroid)
  - No YOLO after init — SAM2 alone maintains identity and mask quality
  - resolve_overlaps handles contested pixels between masks

CRITICAL: multimask_output must be False. True generates 3 candidates and picks
by score, which degrades tracking quality. False gives a single direct mask that
tracks more reliably through posture changes, interactions, and crossings.

CRITICAL: YOLO must NOT run every frame for prompting. YOLO detection order is
arbitrary — detection[0] might be either rat. Using YOLO ordering for SAM2
prompts causes permanent identity swaps.

Usage:
    python -m src.pipelines.centroid.run --config configs/local_centroid.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.common.config_loader import load_config, setup_run_dir, setup_logging
from src.common.io_video import (
    open_video_reader, get_video_properties, create_video_writer, iter_frames,
)
from src.common.metrics import compute_centroid
from src.common.visualization import (
    apply_masks_overlay, draw_centroids, draw_text,
)
from src.common.geometry import resolve_overlaps
from src.common.yolo_inference import detect_only
from src.common.model_loaders import load_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2 segmentation helpers
# ---------------------------------------------------------------------------

def _segment_from_boxes(predictor, frame_rgb, detections, sam_threshold):
    """Segment using YOLO boxes (initialization only — frame 0)."""
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

    multimask_output=False is critical — True degrades tracking quality.
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
            box=None,
            multimask_output=False,
        )
        masks.append(raw[0] > sam_threshold)
        scores.append(float(sc[0]))

    return masks, scores


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
    filter_class = det_cfg.get("filter_class")
    nms_iou = det_cfg.get("nms_iou")
    sam_thr = config.get("segmentation", {}).get("sam_threshold", 0.0)

    colors_raw = config.get("output", {}).get("overlay_colors")
    colors = [tuple(c) for c in colors_raw] if colors_raw else None
    max_frames = config.get("scan", {}).get("max_frames")

    # Centroid propagation state
    prev_centroids: Optional[List[Tuple[float, float]]] = None
    initialized = False
    yolo_uses = 0

    frame_count = 0
    for frame_idx, frame_bgr in iter_frames(
        cap, max_frames=max_frames, start_frame=start_frame, end_frame=end_frame,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mode = "none"

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []

        # --------------------------------------------------------------
        # SAM2 segmentation
        # --------------------------------------------------------------
        if not initialized:
            # YOLO initialization: detect rats and segment with box prompts
            detections = detect_only(
                yolo, frame_rgb, det_conf,
                keypoint_names=kpt_names, filter_class=filter_class,
                nms_iou=nms_iou,
            )[:max_animals]

            if len(detections) >= max_animals:
                all_masks, all_scores = _segment_from_boxes(
                    sam, frame_rgb, detections, sam_thr,
                )
                yolo_uses += 1
                initialized = True
                mode = f"YOLO-INIT ({len(detections)} det)"
                logger.info(
                    "Initialized at frame %d (%d detections)",
                    frame_idx, len(detections),
                )
            else:
                mode = f"WAITING ({len(detections)}/{max_animals} det)"

        else:
            # Centroid propagation — SAM2 only, no YOLO
            all_masks, all_scores = _segment_from_centroids(
                sam, frame_rgb, prev_centroids, sam_thr,
            )
            mode = "CENTROID"

        # --------------------------------------------------------------
        # Update centroids from masks
        # --------------------------------------------------------------
        if all_masks:
            new_centroids = []
            for i, m in enumerate(all_masks):
                c = compute_centroid(m)
                if c is not None:
                    new_centroids.append(c)
                elif prev_centroids is not None and i < len(prev_centroids):
                    new_centroids.append(prev_centroids[i])
            if len(new_centroids) == max_animals:
                # Resolve overlapping mask pixels
                resolve_overlaps(all_masks, new_centroids)
                prev_centroids = new_centroids

        # --------------------------------------------------------------
        # Render overlay
        # --------------------------------------------------------------
        frame_out = np.copy(frame_rgb)

        if all_masks:
            render_colors = colors[:len(all_masks)] if colors else None
            frame_out = apply_masks_overlay(frame_out, all_masks, colors=render_colors)

            centroids = [compute_centroid(m) for m in all_masks]
            valid = [c for c in centroids if c is not None]
            if valid:
                c_colors = render_colors[:len(valid)] if render_colors else None
                frame_out = draw_centroids(frame_out, valid, colors=c_colors)

        scores_str = ", ".join(f"{s:.2f}" for s in all_scores) if all_scores else "-"
        n_masks = len(all_masks)
        status = (
            f"SAM2: {n_masks} masks | mode={mode} | "
            f"scores=[{scores_str}] | YOLO used {yolo_uses}x | frame {frame_idx}"
        )
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
    logger.info("Pipeline complete. %d frames processed. YOLO used %d times.", frame_count, yolo_uses)
    logger.info("Overlay video: %s", out_video_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centroid pipeline: SAM2 centroid propagation (YOLO init only).",
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
