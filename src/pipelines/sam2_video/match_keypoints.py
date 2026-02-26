"""
Match YOLO detections (with keypoints) to SAM2 Video masks.

Since SAM2VideoPredictor manages identity through temporal memory, the obj_ids
from SAM2 are the authoritative identity. YOLO detections need to be matched
to the correct SAM2 object so that keypoints are associated with the right mask.

Matching strategy:
  1. Primary: check if YOLO nose keypoint falls inside SAM2 mask
  2. Fallback: centroid distance Hungarian assignment (2x2 for 2 rats)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.metrics import compute_centroid
from src.common.utils import Detection

logger = logging.getLogger(__name__)


def match_detections_to_masks(
    detections: List[Detection],
    obj_ids: List[int],
    masks: List[np.ndarray],
    nose_kp_index: int = 4,
    min_kp_conf: float = 0.3,
) -> Dict[int, Optional[Detection]]:
    """Match YOLO detections to SAM2 masks.

    Returns a mapping from SAM2 obj_id to the best-matching Detection.
    Unmatched obj_ids map to None.

    Args:
        detections: YOLO detections with keypoints (track_id=None).
        obj_ids: SAM2 object IDs.
        masks: Boolean masks (H, W) corresponding to obj_ids.
        nose_kp_index: Index of the nose keypoint (default 4 for rat model).
        min_kp_conf: Minimum keypoint confidence to use nose check.

    Returns:
        Dict mapping obj_id → Detection (or None if unmatched).
    """
    if not detections or not obj_ids:
        return {oid: None for oid in obj_ids}

    n_dets = len(detections)
    n_objs = len(obj_ids)

    # Try nose-in-mask matching first
    nose_match = _match_by_nose_in_mask(
        detections, obj_ids, masks, nose_kp_index, min_kp_conf,
    )
    if nose_match is not None:
        return nose_match

    # Fallback: centroid distance Hungarian
    return _match_by_centroid_distance(detections, obj_ids, masks)


def _match_by_nose_in_mask(
    detections: List[Detection],
    obj_ids: List[int],
    masks: List[np.ndarray],
    nose_kp_index: int,
    min_kp_conf: float,
) -> Optional[Dict[int, Optional[Detection]]]:
    """Match by checking if each detection's nose keypoint falls inside a mask.

    Returns None if matching is ambiguous or incomplete (fallback needed).
    """
    # Collect nose positions
    nose_positions = []
    for det in detections:
        if det.keypoints and nose_kp_index < len(det.keypoints):
            kp = det.keypoints[nose_kp_index]
            if kp.conf >= min_kp_conf:
                nose_positions.append((int(kp.y), int(kp.x)))  # row, col
            else:
                nose_positions.append(None)
        else:
            nose_positions.append(None)

    # Build match matrix: which detections' noses fall in which masks
    result: Dict[int, Optional[Detection]] = {oid: None for oid in obj_ids}
    used_dets = set()

    for oi, (oid, mask) in enumerate(zip(obj_ids, masks)):
        h, w = mask.shape
        best_det_idx = None

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            np_pos = nose_positions[di]
            if np_pos is None:
                continue
            row, col = np_pos
            if 0 <= row < h and 0 <= col < w and mask[row, col]:
                if best_det_idx is None:
                    best_det_idx = di
                else:
                    # Multiple detections match same mask — ambiguous
                    return None

        if best_det_idx is not None:
            result[oid] = detections[best_det_idx]
            used_dets.add(best_det_idx)

    # Check if all objects got a match
    matched_count = sum(1 for v in result.values() if v is not None)
    if matched_count < min(len(detections), len(obj_ids)):
        return None  # Incomplete — fall back to centroid

    return result


def _match_by_centroid_distance(
    detections: List[Detection],
    obj_ids: List[int],
    masks: List[np.ndarray],
) -> Dict[int, Optional[Detection]]:
    """Match detections to masks by minimum centroid distance (Hungarian-like).

    For 2 rats, this is a simple 2x2 comparison: try both assignments and
    pick the one with lower total distance.
    """
    result: Dict[int, Optional[Detection]] = {oid: None for oid in obj_ids}

    # Compute mask centroids
    mask_centroids = []
    for mask in masks:
        c = compute_centroid(mask)
        mask_centroids.append(c)

    # Compute detection centroids
    det_centroids = [det.center() for det in detections]

    n_objs = len(obj_ids)
    n_dets = len(detections)

    if n_dets == 0:
        return result

    if n_objs <= 2 and n_dets <= 2:
        # Simple exhaustive for small cases
        valid_objs = [(i, mc) for i, mc in enumerate(mask_centroids) if mc is not None]
        valid_dets = list(range(n_dets))

        if len(valid_objs) == 0 or len(valid_dets) == 0:
            return result

        if len(valid_objs) == 1 and len(valid_dets) == 1:
            oi = valid_objs[0][0]
            result[obj_ids[oi]] = detections[valid_dets[0]]
            return result

        if len(valid_objs) >= 2 and len(valid_dets) >= 2:
            oi_a, mc_a = valid_objs[0]
            oi_b, mc_b = valid_objs[1]
            dc_0 = det_centroids[0]
            dc_1 = det_centroids[1]

            # Straight assignment
            cost_straight = _dist(mc_a, dc_0) + _dist(mc_b, dc_1)
            # Swapped assignment
            cost_swapped = _dist(mc_a, dc_1) + _dist(mc_b, dc_0)

            if cost_straight <= cost_swapped:
                result[obj_ids[oi_a]] = detections[0]
                result[obj_ids[oi_b]] = detections[1]
            else:
                result[obj_ids[oi_a]] = detections[1]
                result[obj_ids[oi_b]] = detections[0]
            return result

        # 1 obj, 2 dets: pick closest
        if len(valid_objs) == 1:
            oi, mc = valid_objs[0]
            d0 = _dist(mc, det_centroids[0])
            d1 = _dist(mc, det_centroids[1])
            result[obj_ids[oi]] = detections[0] if d0 <= d1 else detections[1]
            return result

        # 2 objs, 1 det: pick closest
        if len(valid_dets) == 1:
            d_a = _dist(valid_objs[0][1], det_centroids[0])
            d_b = _dist(valid_objs[1][1], det_centroids[0])
            winner = valid_objs[0][0] if d_a <= d_b else valid_objs[1][0]
            result[obj_ids[winner]] = detections[0]
            return result

    # General case: greedy nearest-neighbor (sufficient for max_animals=2)
    used_dets = set()
    for oi in range(n_objs):
        mc = mask_centroids[oi]
        if mc is None:
            continue
        best_di = None
        best_dist = float("inf")
        for di in range(n_dets):
            if di in used_dets:
                continue
            d = _dist(mc, det_centroids[di])
            if d < best_dist:
                best_dist = d
                best_di = di
        if best_di is not None:
            result[obj_ids[oi]] = detections[best_di]
            used_dets.add(best_di)

    return result


def _dist(a: Optional[Tuple[float, float]], b: Tuple[float, float]) -> float:
    """Euclidean distance, returns inf if a is None."""
    if a is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
