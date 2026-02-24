"""
Post-processing for the SAM2+YOLO pipeline.

Handles mask filtering and delegates identity tracking to SlotTracker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.tracking import filter_masks, SlotTracker
from src.common.utils import Detection


def create_tracker(config: Dict[str, Any]) -> SlotTracker:
    """Create a SlotTracker from config."""
    max_animals = config.get("detection", {}).get("max_animals", 2)
    tracking_cfg = config.get("tracking", {})

    return SlotTracker(
        num_slots=max_animals,
        max_distance=tracking_cfg.get("max_centroid_distance", 150.0),
        max_missing_frames=tracking_cfg.get("max_missing_frames", 5),
        w_dist=tracking_cfg.get("w_dist", 0.4),
        w_iou=tracking_cfg.get("w_iou", 0.4),
        w_area=tracking_cfg.get("w_area", 0.2),
        cost_threshold=tracking_cfg.get("cost_threshold", 0.85),
    )


def postprocess_frame(
    masks: List[np.ndarray],
    detections: List[Detection],
    tracker: SlotTracker,
    config: Dict[str, Any],
) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
    """Filter masks and update the tracker.

    Args:
        masks: Raw masks from SAM2 segmentation (one per detection).
        detections: YOLO detections with track_ids (parallel to masks).
        tracker: The persistent SlotTracker instance.
        config: Pipeline config dictionary.

    Returns:
        (slot_masks, slot_centroids) — lists of length num_slots.
    """
    max_animals = config.get("detection", {}).get("max_animals", 2)
    mask_iou_thr = config.get("segmentation", {}).get("mask_iou_threshold", 0.5)

    # Remove duplicate/overlapping masks (only if > max_animals)
    # Keep track of which detections survive filtering
    if len(masks) > max_animals:
        # filter_masks returns a subset — we need to know which indices survived
        areas = [m.sum() for m in masks]
        sorted_idx = list(np.argsort(areas)[::-1])

        from src.common.metrics import mask_iou as _mask_iou
        kept_orig_idx = []
        kept_masks = []
        for idx in sorted_idx:
            if all(_mask_iou(masks[idx], km) <= mask_iou_thr for km in kept_masks):
                kept_masks.append(masks[idx])
                kept_orig_idx.append(idx)
                if len(kept_masks) >= max_animals:
                    break

        filtered_masks = kept_masks
        filtered_track_ids = [
            detections[i].track_id if i < len(detections) else None
            for i in kept_orig_idx
        ]
    else:
        filtered_masks = masks
        filtered_track_ids = [
            d.track_id for d in detections[:len(masks)]
        ]

    # Update tracker
    slot_masks, slot_centroids = tracker.update(filtered_masks, filtered_track_ids)

    return slot_masks, slot_centroids
