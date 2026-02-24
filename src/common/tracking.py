"""
Multi-object tracking with fixed identity slots.

Provides:
- Smart duplicate mask filtering (only when num_masks > max_count).
- Fixed-slot tracker with centroid distance + area change gating,
  optional mask-IoU priority, and missing-frame tolerance to prevent
  color flicker at borders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .metrics import mask_iou, compute_centroid

logger = logging.getLogger(__name__)


def filter_masks(
    masks: List[np.ndarray],
    iou_threshold: float,
    max_count: int,
) -> List[np.ndarray]:
    """Keep at most `max_count` masks, removing duplicates by IoU.

    If num_masks <= max_count, returns all masks unmodified (no filtering).
    Otherwise, masks are sorted by area (largest first). A mask is kept only
    if its IoU with all previously kept masks is below the threshold.

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


@dataclass
class SlotState:
    """Persistent state for one tracking slot (one animal identity)."""
    centroid: Optional[Tuple[float, float]] = None
    area: int = 0
    mask: Optional[np.ndarray] = None
    missing_count: int = 0
    active: bool = False


class FixedSlotTracker:
    """Fixed-slot identity tracker with missing-frame tolerance.

    Each slot corresponds to one animal (slot 0 = color 0, slot 1 = color 1, etc.).
    Slots persist across frames. A slot is only released after max_missing consecutive
    frames without a valid match, preventing color swaps from transient detection gaps.

    Matching uses two gates:
    - centroid distance < max_distance (pixels)
    - area change < area_change_threshold (fractional, e.g., 0.4 = +/-40%)

    When mask IoU is available (use_mask_iou=True and previous masks stored),
    candidates that pass the distance+area gate are ranked by mask IoU (descending)
    instead of just distance. This helps when centroids jump due to border cropping.
    """

    def __init__(
        self,
        num_slots: int,
        max_distance: float = 150.0,
        area_change_threshold: float = 0.4,
        max_missing_frames: int = 5,
        use_mask_iou: bool = True,
    ):
        self.num_slots = num_slots
        self.max_distance = max_distance
        self.area_change_threshold = area_change_threshold
        self.max_missing_frames = max_missing_frames
        self.use_mask_iou = use_mask_iou
        self.slots: List[SlotState] = [SlotState() for _ in range(num_slots)]

    def update(
        self,
        masks: List[np.ndarray],
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[Tuple[float, float]]]]:
        """Match current masks to persistent slots.

        Args:
            masks: Filtered masks for the current frame (at most num_slots).

        Returns:
            (ordered_masks, ordered_centroids) — lists of length num_slots.
            Slots without a match in this frame have None entries.
        """
        # Compute current centroids and areas
        curr_centroids = []
        curr_areas = []
        for m in masks:
            c = compute_centroid(m)
            curr_centroids.append(c if c is not None else (0.0, 0.0))
            curr_areas.append(int(m.sum()))

        # Count active slots
        active_slots = [i for i in range(self.num_slots) if self.slots[i].active]

        # --- First frame or no active slots: assign masks to slots in order ---
        if len(active_slots) == 0:
            result_masks: List[Optional[np.ndarray]] = [None] * self.num_slots
            result_centroids: List[Optional[Tuple[float, float]]] = [None] * self.num_slots
            for mi, m in enumerate(masks):
                if mi >= self.num_slots:
                    break
                result_masks[mi] = m
                result_centroids[mi] = curr_centroids[mi]
                self.slots[mi].centroid = curr_centroids[mi]
                self.slots[mi].area = curr_areas[mi]
                self.slots[mi].mask = m
                self.slots[mi].missing_count = 0
                self.slots[mi].active = True
            return result_masks, result_centroids

        # --- Build distance matrix: masks (rows) x active slots (cols) ---
        if len(masks) == 0:
            # No masks this frame — increment missing counts
            for s in self.slots:
                if s.active:
                    s.missing_count += 1
                    if s.missing_count > self.max_missing_frames:
                        self._release_slot(s)
            return [None] * self.num_slots, [None] * self.num_slots

        prev_centroids_active = [self.slots[i].centroid for i in active_slots]
        dist_mat = cdist(curr_centroids, prev_centroids_active)

        # --- Greedy assignment: for each active slot, find best matching mask ---
        used_masks: set = set()
        matched_slot_indices: set = set()

        result_masks = [None] * self.num_slots
        result_centroids: List[Optional[Tuple[float, float]]] = [None] * self.num_slots

        for col_idx, slot_idx in enumerate(active_slots):
            slot = self.slots[slot_idx]
            # Get candidate masks sorted by distance
            dists = dist_mat[:, col_idx]
            sorted_mask_indices = np.argsort(dists)

            best_mi = None
            best_score = -1.0

            for mi in sorted_mask_indices:
                if mi in used_masks:
                    continue

                # Gate 1: distance
                if dists[mi] >= self.max_distance:
                    break  # All remaining are further

                # Gate 2: area change
                if slot.area > 0:
                    area_ratio = abs(curr_areas[mi] - slot.area) / slot.area
                    if area_ratio > self.area_change_threshold:
                        continue

                # If using mask IoU for ranking, collect all valid candidates
                if self.use_mask_iou and slot.mask is not None:
                    iou = mask_iou(masks[mi], slot.mask)
                    if iou > best_score:
                        best_score = iou
                        best_mi = mi
                    # Don't break — check more candidates for better IoU
                else:
                    # Without mask IoU, take the nearest valid candidate
                    best_mi = mi
                    break

            if best_mi is not None:
                used_masks.add(best_mi)
                matched_slot_indices.add(slot_idx)
                result_masks[slot_idx] = masks[best_mi]
                result_centroids[slot_idx] = curr_centroids[best_mi]
                slot.centroid = curr_centroids[best_mi]
                slot.area = curr_areas[best_mi]
                slot.mask = masks[best_mi]
                slot.missing_count = 0
            else:
                # No valid match found for this slot
                slot.missing_count += 1
                if slot.missing_count > self.max_missing_frames:
                    self._release_slot(slot)

        # --- Place unmatched masks into free slots ---
        for mi in range(len(masks)):
            if mi in used_masks:
                continue
            free_slot = self._find_free_slot()
            if free_slot is None:
                logger.debug("No free slot for unmatched mask %d", mi)
                break
            result_masks[free_slot] = masks[mi]
            result_centroids[free_slot] = curr_centroids[mi]
            self.slots[free_slot].centroid = curr_centroids[mi]
            self.slots[free_slot].area = curr_areas[mi]
            self.slots[free_slot].mask = masks[mi]
            self.slots[free_slot].missing_count = 0
            self.slots[free_slot].active = True

        return result_masks, result_centroids

    def get_debug_info(self) -> str:
        """Return a debug string summarizing slot states."""
        parts = []
        for i, s in enumerate(self.slots):
            status = f"slot{i}:"
            if s.active:
                status += f"active(miss={s.missing_count},area={s.area})"
            else:
                status += "free"
            parts.append(status)
        return " | ".join(parts)

    def _release_slot(self, slot: SlotState) -> None:
        """Release a slot, clearing its state."""
        slot.centroid = None
        slot.area = 0
        slot.mask = None
        slot.missing_count = 0
        slot.active = False

    def _find_free_slot(self) -> Optional[int]:
        """Find the first inactive slot index, or None if all are active."""
        for i, s in enumerate(self.slots):
            if not s.active:
                return i
        return None
