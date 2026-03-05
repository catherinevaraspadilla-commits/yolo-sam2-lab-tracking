"""
SAM3 segmentation with centroid fallback.

Hybrid segmentation strategy (same as reference pipeline, using SAM3 instead of SAM2):
  1. Primary: use YOLO detection boxes + keypoint prompts as SAM3 prompts
  2. When boxes overlap: add negative keypoints from other detections
  3. Fallback: use previous-frame centroid as SAM3 point prompt (when YOLO misses)

Key difference from SAM2 processor: SAM3 uses normalized [0,1] coordinates
instead of pixel coordinates. All prompts are converted before being passed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sam3.model.sam3_image_processor import Sam3Processor

from src.common.metrics import compute_centroid
from src.common.utils import Detection
from src.pipelines.reference.identity_matcher import IdentityMatcher

logger = logging.getLogger(__name__)


def _boxes_overlap(a: Detection, b: Detection) -> bool:
    """Check if two detection boxes have any intersection."""
    return a.x1 < b.x2 and a.x2 > b.x1 and a.y1 < b.y2 and a.y2 > b.y1


def _extract_keypoints(det: Detection, min_conf: float) -> List[List[float]]:
    """Extract high-confidence keypoint coordinates (pixel space) from a detection."""
    if det.keypoints is None:
        return []
    return [[kp.x, kp.y] for kp in det.keypoints if kp.conf >= min_conf]


def _normalize_points(points_px: List[List[float]], h: int, w: int) -> List[List[float]]:
    """Convert pixel-space points to SAM3 normalized [0,1] coordinates.

    Args:
        points_px: List of [x, y] in pixel coordinates.
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        List of [x_norm, y_norm] in [0, 1] range.
    """
    return [[x / w, y / h] for x, y in points_px]


def _normalize_box(box_px: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert pixel-space box [x1,y1,x2,y2] to SAM3 normalized [0,1] coordinates.

    Args:
        box_px: Array of [x1, y1, x2, y2] in pixel coordinates.
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        Array of [x1_norm, y1_norm, x2_norm, y2_norm] in [0, 1] range.
    """
    return np.array([box_px[0] / w, box_px[1] / h, box_px[2] / w, box_px[3] / h],
                    dtype=np.float32)


def segment_frame(
    processor: Sam3Processor,
    frame_rgb: np.ndarray,
    detections: List[Detection],
    matcher: IdentityMatcher,
    sam_threshold: float = 0.0,
    max_entities: int = 2,
    kpt_min_conf: float = 0.3,
    interaction_state: str = "SEPARATE",
) -> Tuple[List[np.ndarray], List[float]]:
    """Segment a frame with SAM3 using YOLO boxes + keypoints + centroid fallback.

    Flow:
        1. set_image(frame_rgb) — encode the frame once
        2. For each YOLO detection: predict(box + keypoints) → mask
           - If boxes overlap with another detection, add negative keypoints
             from the other detection to help SAM3 distinguish them
        3. If YOLO detected fewer objects than active slots AND state is SEPARATE:
           for each active slot without a YOLO mask nearby,
           predict(point_coords=prev_centroid) → fallback mask
           (Centroid fallback is skipped during MERGED state to avoid
           re-segmenting the merged blob as a duplicate.)
        4. Return all masks + scores (to be filtered/matched by caller)

    Args:
        processor: Loaded SAM3 Sam3Processor.
        frame_rgb: Frame in RGB format (H, W, 3).
        detections: YOLO detections (detect-only, no track IDs).
        matcher: IdentityMatcher (provides prev_centroids for fallback).
        sam_threshold: Threshold for SAM3 logit masks.
        max_entities: Expected number of objects to track.
        kpt_min_conf: Minimum keypoint confidence for SAM3 prompts.
        interaction_state: Current matcher state ('SEPARATE' or 'MERGED').
            Centroid fallback is disabled during MERGED to avoid duplicating
            the merged blob.

    Returns:
        (masks, scores) — all masks from boxes + fallback centroids.
    """
    if len(detections) == 0 and all(c is None for c in matcher.prev_centroids):
        return [], []

    h, w = frame_rgb.shape[:2]

    # Encode the frame once
    inference_state = processor.set_image(frame_rgb)

    all_masks: List[np.ndarray] = []
    all_scores: List[float] = []

    # --- Step 1: Segment from YOLO detection boxes + keypoints ---
    active_dets = detections[:max_entities]
    for i, det in enumerate(active_dets):
        box_px = np.array([det.x1, det.y1, det.x2, det.y2])
        box_norm = _normalize_box(box_px, h, w)

        # Build point prompts from keypoints
        pos_points_px = _extract_keypoints(det, kpt_min_conf)
        neg_points_px: List[List[float]] = []

        # When boxes overlap, add negative points from the other rat's keypoints
        for j, other_det in enumerate(active_dets):
            if j == i:
                continue
            if _boxes_overlap(det, other_det):
                neg_points_px.extend(_extract_keypoints(other_det, kpt_min_conf))

        # Combine positive + negative prompts (normalized)
        if pos_points_px or neg_points_px:
            all_points_px = pos_points_px + neg_points_px
            all_labels = [1] * len(pos_points_px) + [0] * len(neg_points_px)
            point_coords = np.array(
                _normalize_points(all_points_px, h, w), dtype=np.float32,
            )
            point_labels = np.array(all_labels, dtype=np.int32)
        else:
            point_coords = None
            point_labels = None

        raw_masks, scores, _ = processor.predict(
            inference_state=inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_norm,
            multimask_output=False,
        )
        m = raw_masks[0] > sam_threshold
        all_masks.append(m)
        all_scores.append(float(scores[0]))

    # --- Step 2: Centroid fallback for missing objects ---
    # If YOLO detected fewer objects than expected AND we have previous centroids,
    # use the previous centroids as point prompts for the missing objects.
    # Skip during MERGED state: centroid fallback near a merged blob would just
    # re-segment the same blob, producing a duplicate mask.
    n_from_yolo = len(all_masks)
    if n_from_yolo < max_entities and interaction_state != "MERGED":
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

            point_norm = np.array(
                [[prev_c[0] / w, prev_c[1] / h]], dtype=np.float32,
            )
            label = np.array([1], dtype=np.int32)
            raw_masks, scores, _ = processor.predict(
                inference_state=inference_state,
                point_coords=point_norm,
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
