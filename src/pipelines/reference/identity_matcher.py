"""
Identity matcher with fixed slots — ported from reference pipeline.

Maintains stable object identity across frames using centroid proximity
and area tolerance. No dependency on YOLO track IDs.

Includes filter_duplicates() to remove redundant masks before matching.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from src.common.metrics import mask_iou, compute_centroid

logger = logging.getLogger(__name__)


def resolve_overlaps(
    slot_masks: List[Optional[np.ndarray]],
    slot_centroids: List[Optional[Tuple[float, float]]],
) -> None:
    """Resolve overlapping pixels between slot masks by centroid proximity.

    For each pair of active masks, contested pixels (where both masks are True)
    are assigned to the mask whose centroid is closer. Modifies masks in-place.

    Args:
        slot_masks: List of boolean masks per slot (None for inactive).
        slot_centroids: List of (x, y) centroids per slot (None for inactive).
    """
    active = [
        (i, slot_masks[i], slot_centroids[i])
        for i in range(len(slot_masks))
        if slot_masks[i] is not None and slot_centroids[i] is not None
    ]

    for a_idx in range(len(active)):
        i, mi, ci = active[a_idx]
        for b_idx in range(a_idx + 1, len(active)):
            j, mj, cj = active[b_idx]

            overlap = mi & mj
            if not overlap.any():
                continue

            n_overlap = int(overlap.sum())
            logger.debug(
                "Resolving %d overlapping pixels between slot %d and %d", n_overlap, i, j,
            )

            # For overlapping pixels, compute distance to each centroid
            ys, xs = np.where(overlap)
            di = (xs - ci[0]) ** 2 + (ys - ci[1]) ** 2
            dj = (xs - cj[0]) ** 2 + (ys - cj[1]) ** 2

            # Pixels closer to j → remove from i
            closer_to_j = dj < di
            mi[ys[closer_to_j], xs[closer_to_j]] = False

            # Pixels closer to i (or equal) → remove from j
            closer_to_i = ~closer_to_j
            mj[ys[closer_to_i], xs[closer_to_i]] = False


def filter_duplicates(
    masks: List[np.ndarray],
    scores: List[float],
    max_entities: int,
    iou_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[float]]:
    """Remove redundant masks by IoU, keeping the largest first.

    Sorts masks by area (descending), then greedily keeps masks that
    don't overlap with already-kept masks above the IoU threshold.

    Args:
        masks: Boolean masks (H, W).
        scores: Confidence scores parallel to masks.
        max_entities: Maximum number of masks to keep.
        iou_threshold: IoU above which a mask is considered a duplicate.

    Returns:
        (unique_masks, unique_scores) — at most max_entities elements.
    """
    if len(masks) <= max_entities:
        return masks, scores

    areas = [m.sum() for m in masks]
    sorted_idx = np.argsort(areas)[::-1]

    unique_masks: List[np.ndarray] = []
    unique_scores: List[float] = []

    for idx in sorted_idx:
        is_dup = any(mask_iou(masks[idx], um) > iou_threshold for um in unique_masks)
        if not is_dup:
            unique_masks.append(masks[idx])
            unique_scores.append(scores[idx])
            if len(unique_masks) >= max_entities:
                break

    return unique_masks, unique_scores


class IdentityMatcher:
    """Fixed-slot identity tracker using centroid proximity + area tolerance.

    Ported from the reference pipeline's tracking/identity_matcher.py.

    Algorithm per frame:
        1. Compute centroid and area for each incoming mask.
        2. If first frame: assign masks to slots in order.
        3. If subsequent frame:
           a. Compute distance matrix between current centroids and prev centroids.
           b. For each active slot (with known prev centroid):
              - Find closest current mask.
              - Accept if dist < proximity_threshold AND area_variation < area_tolerance.
           c. Unassigned masks fill empty slots.
        4. Update prev_centroids, prev_areas, prev_masks.

    No YOLO track IDs are used — identity is purely spatial.
    """

    def __init__(
        self,
        max_entities: int,
        proximity_threshold: float = 50.0,
        area_tolerance: float = 0.4,
    ):
        """
        Args:
            max_entities: Number of fixed identity slots.
            proximity_threshold: Max centroid distance (px) for a valid match.
            area_tolerance: Max relative area change for a valid match (e.g. 0.4 = ±40%).
        """
        self.max_entities = max_entities
        self.proximity_threshold = proximity_threshold
        self.area_tolerance = area_tolerance

        self.prev_centroids: List[Optional[Tuple[float, float]]] = [None] * max_entities
        self.prev_areas: List[Optional[float]] = [None] * max_entities
        self.prev_masks: List[Optional[np.ndarray]] = [None] * max_entities
        self._first_frame = True

    def match(
        self,
        masks: List[np.ndarray],
        scores: List[float],
    ) -> Tuple[
        List[Optional[np.ndarray]],
        List[Optional[Tuple[float, float]]],
        List[float],
        List[float],
    ]:
        """Assign masks to fixed identity slots.

        Args:
            masks: Deduplicated boolean masks (H, W).
            scores: Confidence scores parallel to masks.

        Returns:
            (slot_masks, slot_centroids, slot_areas, slot_scores) — each has
            exactly max_entities elements. None/0.0 for inactive slots.
        """
        n = len(masks)

        # Compute current centroids and areas
        curr_centroids = []
        curr_areas = []
        for m in masks:
            c = compute_centroid(m)
            curr_centroids.append(c)
            curr_areas.append(float(m.sum()) if c is not None else 0.0)

        # Initialize output arrays
        out_masks: List[Optional[np.ndarray]] = [None] * self.max_entities
        out_centroids: List[Optional[Tuple[float, float]]] = [None] * self.max_entities
        out_areas: List[float] = [0.0] * self.max_entities
        out_scores: List[float] = [0.0] * self.max_entities

        if n == 0:
            # No masks — keep prev state for centroid fallback, don't clear
            return out_masks, out_centroids, out_areas, out_scores

        # --- First frame: assign in order ---
        if self._first_frame:
            for i in range(min(n, self.max_entities)):
                out_masks[i] = masks[i]
                out_centroids[i] = curr_centroids[i]
                out_areas[i] = curr_areas[i]
                out_scores[i] = scores[i] if i < len(scores) else 0.0
            self._update_prev(out_masks, out_centroids, out_areas)
            self._first_frame = False
            return out_masks, out_centroids, out_areas, out_scores

        # --- Subsequent frames: match by proximity ---
        return self._match_to_previous(
            masks, scores, curr_centroids, curr_areas,
        )

    def _match_to_previous(
        self,
        masks: List[np.ndarray],
        scores: List[float],
        curr_centroids: List[Optional[Tuple[float, float]]],
        curr_areas: List[float],
    ) -> Tuple[
        List[Optional[np.ndarray]],
        List[Optional[Tuple[float, float]]],
        List[float],
        List[float],
    ]:
        """Match current masks to previous slots by centroid distance + area."""
        n = len(masks)
        out_masks: List[Optional[np.ndarray]] = [None] * self.max_entities
        out_centroids: List[Optional[Tuple[float, float]]] = [None] * self.max_entities
        out_areas: List[float] = [0.0] * self.max_entities
        out_scores: List[float] = [0.0] * self.max_entities

        # Which slots have a previous centroid (active slots)
        active_slots = [
            i for i in range(self.max_entities) if self.prev_centroids[i] is not None
        ]

        # Which current masks have a valid centroid
        valid_curr = [i for i in range(n) if curr_centroids[i] is not None]

        if not active_slots or not valid_curr:
            # No previous state or no valid current masks — assign in order
            for i, ci in enumerate(valid_curr):
                if i >= self.max_entities:
                    break
                out_masks[i] = masks[ci]
                out_centroids[i] = curr_centroids[ci]
                out_areas[i] = curr_areas[ci]
                out_scores[i] = scores[ci] if ci < len(scores) else 0.0
            self._update_prev(out_masks, out_centroids, out_areas)
            return out_masks, out_centroids, out_areas, out_scores

        # Build distance matrix: active_slots x valid_curr
        prev_pts = np.array([self.prev_centroids[i] for i in active_slots])
        curr_pts = np.array([curr_centroids[i] for i in valid_curr])
        dist_matrix = cdist(prev_pts, curr_pts)

        # Greedy matching: for each active slot, find closest valid mask
        used_curr = set()
        for slot_pos, slot_idx in enumerate(active_slots):
            dists = dist_matrix[slot_pos]
            sorted_curr = np.argsort(dists)

            for ci_pos in sorted_curr:
                ci = valid_curr[ci_pos]
                if ci in used_curr:
                    continue

                d = dists[ci_pos]
                # Check proximity threshold
                if d > self.proximity_threshold:
                    break  # All remaining are farther

                # Check area tolerance
                prev_area = self.prev_areas[slot_idx]
                if prev_area is not None and prev_area > 0 and curr_areas[ci] > 0:
                    area_var = abs(curr_areas[ci] - prev_area) / prev_area
                    if area_var > self.area_tolerance:
                        continue  # Try next closest

                # Match accepted
                out_masks[slot_idx] = masks[ci]
                out_centroids[slot_idx] = curr_centroids[ci]
                out_areas[slot_idx] = curr_areas[ci]
                out_scores[slot_idx] = scores[ci] if ci < len(scores) else 0.0
                used_curr.add(ci)
                break

        # Fill empty slots with unassigned masks
        unassigned = [ci for ci in valid_curr if ci not in used_curr]
        empty_slots = [i for i in range(self.max_entities) if out_masks[i] is None]

        for ci, slot_idx in zip(unassigned, empty_slots):
            out_masks[slot_idx] = masks[ci]
            out_centroids[slot_idx] = curr_centroids[ci]
            out_areas[slot_idx] = curr_areas[ci]
            out_scores[slot_idx] = scores[ci] if ci < len(scores) else 0.0

        self._update_prev(out_masks, out_centroids, out_areas)
        return out_masks, out_centroids, out_areas, out_scores

    def _update_prev(
        self,
        masks: List[Optional[np.ndarray]],
        centroids: List[Optional[Tuple[float, float]]],
        areas: List[float],
    ) -> None:
        """Update previous-frame state for next frame's matching."""
        for i in range(self.max_entities):
            if masks[i] is not None and centroids[i] is not None:
                self.prev_masks[i] = masks[i]
                self.prev_centroids[i] = centroids[i]
                self.prev_areas[i] = areas[i]
            # If slot was unmatched this frame, keep prev state for fallback
