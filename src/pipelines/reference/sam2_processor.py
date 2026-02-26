"""
SAM2 segmentation with centroid fallback — ported from reference pipeline.

Hybrid segmentation strategy:
  1. Primary: use YOLO detection boxes + keypoint prompts as SAM2 prompts
  2. When boxes overlap: add negative keypoints from other detections
  3. Fallback: use previous-frame centroid as SAM2 point prompt (when YOLO misses)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.common.metrics import compute_centroid
from src.common.utils import Detection
from .identity_matcher import IdentityMatcher

logger = logging.getLogger(__name__)


def _boxes_overlap(a: Detection, b: Detection) -> bool:
    """Check if two detection boxes have any intersection."""
    return a.x1 < b.x2 and a.x2 > b.x1 and a.y1 < b.y2 and a.y2 > b.y1


def _extract_keypoints(det: Detection, min_conf: float) -> List[List[float]]:
    """Extract high-confidence keypoint coordinates from a detection."""
    if det.keypoints is None:
        return []
    return [[kp.x, kp.y] for kp in det.keypoints if kp.conf >= min_conf]


def segment_frame(
    predictor: SAM2ImagePredictor,
    frame_rgb: np.ndarray,
    detections: List[Detection],
    matcher: IdentityMatcher,
    sam_threshold: float = 0.0,
    max_entities: int = 2,
    kpt_min_conf: float = 0.3,
) -> Tuple[List[np.ndarray], List[float]]:
    """Segment a frame with SAM2 using YOLO boxes + keypoints + centroid fallback.

    Flow:
        1. set_image(frame_rgb) — encode the frame once
        2. For each YOLO detection: predict(box + keypoints) → mask
           - If boxes overlap with another detection, add negative keypoints
             from the other detection to help SAM2 distinguish them
        3. If YOLO detected fewer objects than active slots:
           for each active slot without a YOLO mask nearby,
           predict(point_coords=prev_centroid) → fallback mask
        4. Return all masks + scores (to be filtered/matched by caller)

    Args:
        predictor: Loaded SAM2ImagePredictor.
        frame_rgb: Frame in RGB format (H, W, 3).
        detections: YOLO detections (detect-only, no track IDs).
        matcher: IdentityMatcher (provides prev_centroids for fallback).
        sam_threshold: Threshold for SAM2 logit masks.
        max_entities: Expected number of objects to track.
        kpt_min_conf: Minimum keypoint confidence for SAM2 prompts.

    Returns:
        (masks, scores) — all masks from boxes + fallback centroids.
    """
    if len(detections) == 0 and all(c is None for c in matcher.prev_centroids):
        return [], []

    predictor.set_image(frame_rgb)

    all_masks: List[np.ndarray] = []
    all_scores: List[float] = []

    # --- Step 1: Segment from YOLO detection boxes + keypoints ---
    active_dets = detections[:max_entities]
    for i, det in enumerate(active_dets):
        box = np.array([det.x1, det.y1, det.x2, det.y2])

        # Build point prompts from keypoints
        pos_points = _extract_keypoints(det, kpt_min_conf)
        neg_points: List[List[float]] = []

        # When boxes overlap, add negative points from the other rat's keypoints
        for j, other_det in enumerate(active_dets):
            if j == i:
                continue
            if _boxes_overlap(det, other_det):
                neg_points.extend(_extract_keypoints(other_det, kpt_min_conf))

        # Combine positive + negative prompts
        if pos_points or neg_points:
            all_points = pos_points + neg_points
            all_labels = [1] * len(pos_points) + [0] * len(neg_points)
            point_coords = np.array(all_points, dtype=np.float32)
            point_labels = np.array(all_labels, dtype=np.int32)
        else:
            point_coords = None
            point_labels = None

        raw_masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        m = raw_masks[0] > sam_threshold
        all_masks.append(m)
        all_scores.append(float(scores[0]))

    # --- Step 2: Centroid fallback for missing objects ---
    # If YOLO detected fewer objects than expected AND we have previous centroids,
    # use the previous centroids as point prompts for the missing objects.
    n_from_yolo = len(all_masks)
    if n_from_yolo < max_entities:
        # Compute centroids of masks we already have from YOLO
        yolo_centroids = [compute_centroid(m) for m in all_masks]

        for slot_idx in range(matcher.max_entities):
            prev_c = matcher.prev_centroids[slot_idx]
            if prev_c is None:
                continue

            # Check if this slot's centroid is already covered by a YOLO mask
            already_covered = False
            for yc in yolo_centroids:
                if yc is not None:
                    dist = ((prev_c[0] - yc[0]) ** 2 + (prev_c[1] - yc[1]) ** 2) ** 0.5
                    if dist < 100.0:  # within 100px = likely the same object
                        already_covered = True
                        break

            if already_covered:
                continue

            # This slot has no YOLO coverage — use centroid as point prompt
            if len(all_masks) >= max_entities:
                break

            point = np.array([[prev_c[0], prev_c[1]]], dtype=np.float32)
            label = np.array([1], dtype=np.int32)
            raw_masks, scores, _ = predictor.predict(
                point_coords=point,
                point_labels=label,
                box=None,
                multimask_output=False,
            )
            m = raw_masks[0] > sam_threshold
            all_masks.append(m)
            all_scores.append(float(scores[0]))

            logger.debug(
                "Centroid fallback for slot %d at (%.0f, %.0f) → area %d",
                slot_idx, prev_c[0], prev_c[1], m.sum(),
            )

    return all_masks, all_scores
