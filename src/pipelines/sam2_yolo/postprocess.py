"""
Post-processing for the SAM2+YOLO pipeline.

Handles mask filtering and delegates identity tracking to FixedSlotTracker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.tracking import filter_masks, FixedSlotTracker


def create_tracker(config: Dict[str, Any]) -> FixedSlotTracker:
    """Create a FixedSlotTracker from config.

    Args:
        config: Pipeline config dictionary.

    Returns:
        Configured tracker instance.
    """
    max_animals = config.get("detection", {}).get("max_animals", 2)
    tracking_cfg = config.get("tracking", {})

    return FixedSlotTracker(
        num_slots=max_animals,
        max_distance=tracking_cfg.get("max_centroid_distance", 150.0),
        area_change_threshold=tracking_cfg.get("area_change_threshold", 0.4),
        max_missing_frames=tracking_cfg.get("max_missing_frames", 5),
        use_mask_iou=tracking_cfg.get("use_mask_iou", True),
    )


def postprocess_frame(
    masks: List[np.ndarray],
    tracker: FixedSlotTracker,
    config: Dict[str, Any],
) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
    """Filter masks and update the tracker.

    Args:
        masks: Raw masks from SAM2 segmentation.
        tracker: The persistent FixedSlotTracker instance.
        config: Pipeline config dictionary.

    Returns:
        (slot_masks, slot_centroids) — lists of length num_slots.
        Slots without a detection in this frame have None entries.
    """
    max_animals = config.get("detection", {}).get("max_animals", 2)
    mask_iou_thr = config.get("segmentation", {}).get("mask_iou_threshold", 0.5)

    # Remove duplicate/overlapping masks (only if > max_animals)
    filtered = filter_masks(masks, iou_threshold=mask_iou_thr, max_count=max_animals)

    # Update tracker with filtered masks
    slot_masks, slot_centroids = tracker.update(filtered)

    return slot_masks, slot_centroids
