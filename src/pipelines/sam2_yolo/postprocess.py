"""
Post-processing for the SAM2+YOLO pipeline.

Handles mask filtering, identity tracking, and frame annotation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.common.tracking import filter_masks, match_by_centroid


def postprocess_frame(
    masks: List[np.ndarray],
    prev_centroids: List[Tuple[float, float]],
    config: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[Tuple[float, float] | None]]:
    """Filter and reorder masks for consistent identity tracking.

    Args:
        masks: Raw masks from SAM2 segmentation.
        prev_centroids: Centroid positions from the previous frame.
        config: Pipeline config dictionary.

    Returns:
        (ordered_masks, ordered_centroids) with identity-consistent ordering.
    """
    max_animals = config.get("detection", {}).get("max_animals", 2)
    mask_iou_thr = config.get("segmentation", {}).get("mask_iou_threshold", 0.5)
    max_dist = config.get("tracking", {}).get("max_centroid_distance", 150.0)

    # Remove duplicate/overlapping masks
    filtered = filter_masks(masks, iou_threshold=mask_iou_thr, max_count=max_animals)

    # Match to previous frame identities
    ordered_masks, ordered_centroids = match_by_centroid(
        filtered, prev_centroids, max_distance=max_dist,
    )

    return ordered_masks, ordered_centroids
