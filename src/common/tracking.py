"""
Multi-object tracking utilities: mask filtering and centroid-based identity matching.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .metrics import mask_iou, compute_centroid


def filter_masks(
    masks: List[np.ndarray],
    iou_threshold: float,
    max_count: int,
) -> List[np.ndarray]:
    """Keep at most `max_count` masks, removing duplicates by IoU.

    Masks are sorted by area (largest first). A mask is kept only if its IoU
    with all previously kept masks is below the threshold.

    Args:
        masks: List of boolean masks.
        iou_threshold: IoU threshold above which masks are considered duplicates.
        max_count: Maximum number of masks to keep.

    Returns:
        Filtered list of masks (at most max_count).
    """
    if len(masks) <= max_count:
        return masks

    areas = [m.sum() for m in masks]
    sorted_idx = np.argsort(areas)[::-1]
    sorted_masks = [masks[i] for i in sorted_idx]

    unique: List[np.ndarray] = []
    for mask in sorted_masks:
        if all(mask_iou(mask, u) <= iou_threshold for u in unique):
            unique.append(mask)
            if len(unique) >= max_count:
                break

    return unique


def match_by_centroid(
    current_masks: List[np.ndarray],
    previous_centroids: List[Tuple[float, float]],
    max_distance: float,
) -> Tuple[List[np.ndarray], List[Tuple[float, float] | None]]:
    """Reorder masks to match previous identities based on centroid proximity.

    Uses a greedy nearest-neighbor assignment. Masks whose closest previous
    centroid exceeds max_distance are placed in remaining empty slots.

    Args:
        current_masks: Masks detected in the current frame.
        previous_centroids: Centroid positions from the previous frame.
        max_distance: Maximum pixel distance for a valid match.

    Returns:
        (ordered_masks, ordered_centroids) with identity-consistent ordering.
    """
    if len(current_masks) == 0:
        return [], []

    curr_centroids = [
        compute_centroid(m) or (0.0, 0.0)
        for m in current_masks
    ]

    if len(previous_centroids) == 0:
        return current_masks, curr_centroids

    dist_mat = cdist(curr_centroids, previous_centroids)

    ordered_masks = [None] * len(previous_centroids)
    ordered_centroids = [None] * len(previous_centroids)
    used = []

    for prev_idx in range(len(previous_centroids)):
        dists = dist_mat[:, prev_idx]
        sorted_idx = np.argsort(dists)

        for ci in sorted_idx:
            if ci in used:
                continue
            if dists[ci] < max_distance:
                ordered_masks[prev_idx] = current_masks[ci]
                ordered_centroids[prev_idx] = curr_centroids[ci]
                used.append(ci)
                break

    # Place unmatched masks in empty slots
    for ci, mask in enumerate(current_masks):
        if ci not in used:
            for slot in range(len(ordered_masks)):
                if ordered_masks[slot] is None:
                    ordered_masks[slot] = mask
                    ordered_centroids[slot] = curr_centroids[ci]
                    break

    final_masks = [m for m in ordered_masks if m is not None]
    final_centroids = [c for c in ordered_centroids if c is not None]
    return final_masks, final_centroids
