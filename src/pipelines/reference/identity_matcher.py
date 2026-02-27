"""
Identity matcher with fixed slots, Hungarian assignment, and merge/split handling.

Maintains stable object identity across frames using:
  - Hungarian (optimal) assignment with soft cost (distance + mask IoU + area)
  - Velocity prediction with gating and clamping
  - Swap guard for N=2 crossings
  - N=2 interaction state machine (SEPARATE ↔ MERGED) to handle close contacts

No dependency on YOLO track IDs — identity is purely spatial + shape-based.

Includes filter_duplicates() to remove redundant masks before matching.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.metrics import mask_iou, compute_centroid

logger = logging.getLogger(__name__)

# Cost for impossible assignments
INF_COST = 1e6
# Hard veto only for extreme area changes (5x = 400% change)
HARD_AREA_RATIO_VETO = 5.0


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
    """Fixed-slot identity tracker with Hungarian matching and merge/split handling.

    Algorithm per frame:
        1. Compute centroid and area for each incoming mask.
        2. If first frame: assign masks to slots in order.
        3. Check interaction state (SEPARATE or MERGED).
        4. If SEPARATE:
           a. Build soft cost matrix (distance + mask IoU + area).
           b. Run Hungarian assignment for optimal matching.
           c. Apply swap guard for N=2.
           d. Check for merge transition.
        5. If MERGED:
           a. Don't assign masks — coast tracks with velocity prediction.
           b. Check for split transition.
           c. On split: resolve IDs using pre-merge evidence.
        6. Update velocity, area average, prev state.

    No YOLO track IDs are used — identity is purely spatial + shape-based.
    """

    def __init__(
        self,
        max_entities: int,
        proximity_threshold: float = 150.0,
        area_tolerance: float = 0.4,
        max_merged_frames: int = 30,
    ):
        """
        Args:
            max_entities: Number of fixed identity slots.
            proximity_threshold: Normalization distance for soft cost (px).
            area_tolerance: Kept for compatibility (used as soft cost weight).
            max_merged_frames: Max frames in MERGED state before force-reset.
        """
        self.max_entities = max_entities
        self.proximity_threshold = proximity_threshold
        self.area_tolerance = area_tolerance

        # Cost function weights (proven in SlotTracker)
        self._w_dist = 0.4
        self._w_iou = 0.4
        self._w_area = 0.2
        self._cost_threshold = 0.85

        # Per-slot tracking state
        self.prev_centroids: List[Optional[Tuple[float, float]]] = [None] * max_entities
        self.prev_areas: List[Optional[float]] = [None] * max_entities
        self.prev_masks: List[Optional[np.ndarray]] = [None] * max_entities
        self._velocities: List[Optional[Tuple[float, float]]] = [None] * max_entities
        self._first_frame = True

        # Interaction state machine
        self._state: str = "SEPARATE"
        self._frames_merged: int = 0
        self._max_merged_frames: int = max_merged_frames

        # Pre-merge snapshot (saved at SEPARATE → MERGED transition)
        self._pre_merge_centroids: List[Optional[Tuple[float, float]]] = [None] * max_entities
        self._pre_merge_velocities: List[Optional[Tuple[float, float]]] = [None] * max_entities
        self._pre_merge_masks: List[Optional[np.ndarray]] = [None] * max_entities
        self._pre_merge_areas: List[Optional[float]] = [None] * max_entities

        # Running average of single-rat area (for merge detection)
        self._avg_area: float = 0.0
        self._area_samples: int = 0

    @property
    def state(self) -> str:
        """Current interaction state: 'SEPARATE' or 'MERGED'."""
        return self._state

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
        curr_centroids: List[Optional[Tuple[float, float]]] = []
        curr_areas: List[float] = []
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
            self._update_avg_area(out_areas)
            self._first_frame = False
            return out_masks, out_centroids, out_areas, out_scores

        # --- State machine: branch on current interaction state ---
        if self._state == "MERGED":
            return self._handle_merged(
                masks, scores, curr_centroids, curr_areas,
            )

        # --- SEPARATE state: normal matching with merge detection ---
        result = self._match_separate(
            masks, scores, curr_centroids, curr_areas,
        )

        # Check for merge transition after matching
        if self._detect_merge(masks, curr_centroids, curr_areas):
            self._transition_to_merged()

        return result

    # ------------------------------------------------------------------
    # SEPARATE state matching (Hungarian + soft cost + swap guard)
    # ------------------------------------------------------------------

    def _match_separate(
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
        """Match current masks to previous slots using Hungarian assignment."""
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
            self._update_avg_area(out_areas)
            return out_masks, out_centroids, out_areas, out_scores

        # --- Build cost matrix: active_slots × valid_curr ---
        n_slots = len(active_slots)
        n_masks = len(valid_curr)
        cost_matrix = np.full((n_slots, n_masks), INF_COST, dtype=np.float64)

        for row, slot_idx in enumerate(active_slots):
            for col, ci in enumerate(valid_curr):
                cost_matrix[row, col] = self._compute_cost(
                    curr_centroids[ci], curr_areas[ci], masks[ci], slot_idx,
                )

        # --- Hungarian (optimal) assignment ---
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # --- Swap guard for N=2 ---
        if len(row_ind) == 2 and n_slots == 2 and n_masks == 2:
            cost_straight = cost_matrix[0, 0] + cost_matrix[1, 1]
            cost_swapped = cost_matrix[0, 1] + cost_matrix[1, 0]

            if cost_swapped < cost_straight - 0.1:
                # Hungarian should have caught this, but as a safety check
                logger.info(
                    "Swap guard: straight=%.3f > swapped=%.3f, re-assigning",
                    cost_straight, cost_swapped,
                )
                row_ind = np.array([0, 1])
                col_ind = np.array([1, 0])
            elif abs(cost_straight - cost_swapped) < 0.1:
                # Ambiguous — prefer continuity (keep previous assignment order)
                logger.debug("Swap guard: ambiguous (diff=%.3f), keeping continuity",
                             abs(cost_straight - cost_swapped))
                row_ind = np.array([0, 1])
                col_ind = np.array([0, 1])

        # --- Accept matches below cost threshold ---
        used_curr: set = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self._cost_threshold:
                continue
            slot_idx = active_slots[r]
            ci = valid_curr[c]
            out_masks[slot_idx] = masks[ci]
            out_centroids[slot_idx] = curr_centroids[ci]
            out_areas[slot_idx] = curr_areas[ci]
            out_scores[slot_idx] = scores[ci] if ci < len(scores) else 0.0
            used_curr.add(ci)

        # --- Fill empty slots with unassigned masks ---
        unassigned = [ci for ci in valid_curr if ci not in used_curr]
        empty_slots = [i for i in range(self.max_entities) if out_masks[i] is None]

        for ci, slot_idx in zip(unassigned, empty_slots):
            out_masks[slot_idx] = masks[ci]
            out_centroids[slot_idx] = curr_centroids[ci]
            out_areas[slot_idx] = curr_areas[ci]
            out_scores[slot_idx] = scores[ci] if ci < len(scores) else 0.0

        self._update_prev(out_masks, out_centroids, out_areas)
        self._update_avg_area(out_areas)
        return out_masks, out_centroids, out_areas, out_scores

    def _compute_cost(
        self,
        centroid: Optional[Tuple[float, float]],
        area: float,
        mask: np.ndarray,
        slot_idx: int,
    ) -> float:
        """Compute soft assignment cost between a mask and a slot.

        Uses 3 components: centroid distance (with velocity prediction),
        mask IoU, and area change ratio.

        Returns a value in [0, ~1.0], or INF_COST for impossible assignments.
        """
        prev_c = self.prev_centroids[slot_idx]
        if prev_c is None or centroid is None:
            return INF_COST

        # --- Distance cost (with velocity prediction) ---
        # Use predicted position if velocity is available
        predicted = prev_c
        vel = self._velocities[slot_idx]
        if vel is not None:
            predicted = (prev_c[0] + vel[0], prev_c[1] + vel[1])

        dx = centroid[0] - predicted[0]
        dy = centroid[1] - predicted[1]
        dist = (dx * dx + dy * dy) ** 0.5
        dist_cost = min(dist / self.proximity_threshold, 1.0)

        # --- Mask IoU cost ---
        prev_mask = self.prev_masks[slot_idx]
        if prev_mask is not None:
            iou = mask_iou(mask, prev_mask)
            iou_cost = 1.0 - iou
        else:
            iou_cost = 0.5  # neutral if no previous mask

        # --- Area change cost (soft, not hard veto) ---
        prev_area = self.prev_areas[slot_idx]
        if prev_area is not None and prev_area > 0 and area > 0:
            ratio = max(area, prev_area) / min(area, prev_area)
            if ratio > HARD_AREA_RATIO_VETO:
                return INF_COST  # only veto extreme cases (5x change)
            area_cost = min(ratio - 1.0, 1.0)
        else:
            area_cost = 0.5

        return self._w_dist * dist_cost + self._w_iou * iou_cost + self._w_area * area_cost

    # ------------------------------------------------------------------
    # Merge detection + state machine
    # ------------------------------------------------------------------

    def _detect_merge(
        self,
        masks: List[np.ndarray],
        curr_centroids: List[Optional[Tuple[float, float]]],
        curr_areas: List[float],
    ) -> bool:
        """Detect if two objects have merged into one blob."""
        # Need baseline area to detect merge
        if self._avg_area <= 0:
            return False

        # Case 1: Single mask with area significantly larger than one object
        if len(masks) == 1:
            if curr_areas[0] > 1.5 * self._avg_area:
                logger.debug(
                    "Merge candidate: single mask area=%.0f > 1.5 × avg=%.0f",
                    curr_areas[0], self._avg_area,
                )
                return True

        # Case 2: Two masks but heavily overlapping (effectively merged)
        if len(masks) == 2:
            iou = mask_iou(masks[0], masks[1])
            if iou > 0.5:
                logger.debug(
                    "Merge candidate: two masks with IoU=%.3f > 0.5", iou,
                )
                return True

        return False

    def _transition_to_merged(self) -> None:
        """Save pre-merge snapshot and enter MERGED state."""
        self._state = "MERGED"
        self._frames_merged = 0

        # Snapshot current state for split resolution
        for i in range(self.max_entities):
            self._pre_merge_centroids[i] = self.prev_centroids[i]
            self._pre_merge_velocities[i] = self._velocities[i]
            self._pre_merge_masks[i] = self.prev_masks[i]
            self._pre_merge_areas[i] = self.prev_areas[i]

        logger.info(
            "State: SEPARATE → MERGED (avg_area=%.0f, centroids=%s)",
            self._avg_area,
            [(f"{c[0]:.0f},{c[1]:.0f}" if c else "None") for c in self.prev_centroids],
        )

    def _handle_merged(
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
        """Handle frame during MERGED state.

        Don't assign masks to slots (identity is ambiguous).
        Check if split has occurred and resolve IDs if so.
        """
        self._frames_merged += 1

        # Check if split has occurred: 2 distinct masks
        if len(masks) >= 2:
            # Verify masks are actually distinct (not still merged)
            iou = mask_iou(masks[0], masks[1])
            if iou < 0.3:
                logger.info(
                    "State: MERGED → SEPARATE after %d frames (mask IoU=%.3f)",
                    self._frames_merged, iou,
                )
                self._state = "SEPARATE"
                self._frames_merged = 0
                return self._resolve_split(
                    masks, scores, curr_centroids, curr_areas,
                )

        # Force-reset after max_merged_frames
        if self._frames_merged > self._max_merged_frames:
            logger.warning(
                "MERGED state exceeded %d frames — force-resetting to SEPARATE",
                self._max_merged_frames,
            )
            self._state = "SEPARATE"
            self._frames_merged = 0
            # Fall through to normal matching (no pre-merge evidence available)
            return self._match_separate(
                masks, scores, curr_centroids, curr_areas,
            )

        # Still merged: return empty outputs (don't corrupt identity state)
        out_masks: List[Optional[np.ndarray]] = [None] * self.max_entities
        out_centroids: List[Optional[Tuple[float, float]]] = [None] * self.max_entities
        out_areas: List[float] = [0.0] * self.max_entities
        out_scores: List[float] = [0.0] * self.max_entities

        # Coast tracks: predict centroids using pre-merge velocity
        for i in range(self.max_entities):
            pm_c = self._pre_merge_centroids[i]
            pm_v = self._pre_merge_velocities[i]
            if pm_c is not None:
                if pm_v is not None:
                    # Predict position (clamped velocity)
                    max_vel = self.proximity_threshold / 2
                    vx = max(-max_vel, min(pm_v[0], max_vel))
                    vy = max(-max_vel, min(pm_v[1], max_vel))
                    out_centroids[i] = (
                        pm_c[0] + vx * self._frames_merged,
                        pm_c[1] + vy * self._frames_merged,
                    )
                else:
                    out_centroids[i] = pm_c

        # Don't update prev state — keep pre-merge snapshot intact
        return out_masks, out_centroids, out_areas, out_scores

    def _resolve_split(
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
        """Assign IDs after a merge using pre-merge evidence.

        Uses velocity-predicted positions + mask IoU with pre-merge masks
        to resolve which mask belongs to which slot.
        """
        out_masks: List[Optional[np.ndarray]] = [None] * self.max_entities
        out_centroids: List[Optional[Tuple[float, float]]] = [None] * self.max_entities
        out_areas: List[float] = [0.0] * self.max_entities
        out_scores: List[float] = [0.0] * self.max_entities

        # Collect valid masks and active pre-merge slots
        valid_curr = [i for i in range(len(masks)) if curr_centroids[i] is not None]
        active_slots = [
            i for i in range(self.max_entities)
            if self._pre_merge_centroids[i] is not None
        ]

        if not active_slots or not valid_curr:
            # Fallback: assign in order
            for i, ci in enumerate(valid_curr):
                if i >= self.max_entities:
                    break
                out_masks[i] = masks[ci]
                out_centroids[i] = curr_centroids[ci]
                out_areas[i] = curr_areas[ci]
                out_scores[i] = scores[ci] if ci < len(scores) else 0.0
            self._update_prev(out_masks, out_centroids, out_areas)
            self._update_avg_area(out_areas)
            return out_masks, out_centroids, out_areas, out_scores

        # --- Build split cost matrix: active_slots × valid_curr ---
        n_slots = len(active_slots)
        n_masks = len(valid_curr)
        cost = np.full((n_slots, n_masks), INF_COST, dtype=np.float64)

        for row, slot_idx in enumerate(active_slots):
            pm_c = self._pre_merge_centroids[slot_idx]
            pm_v = self._pre_merge_velocities[slot_idx]
            pm_mask = self._pre_merge_masks[slot_idx]
            pm_area = self._pre_merge_areas[slot_idx]

            if pm_c is None:
                continue

            # Predict where this slot should be after merge
            predicted = pm_c
            if pm_v is not None:
                max_vel = self.proximity_threshold / 2
                vx = max(-max_vel, min(pm_v[0], max_vel))
                vy = max(-max_vel, min(pm_v[1], max_vel))
                predicted = (
                    pm_c[0] + vx * self._frames_merged,
                    pm_c[1] + vy * self._frames_merged,
                )

            for col, ci in enumerate(valid_curr):
                c = curr_centroids[ci]
                if c is None:
                    continue

                # Distance to predicted position
                dx = c[0] - predicted[0]
                dy = c[1] - predicted[1]
                dist = (dx * dx + dy * dy) ** 0.5
                dist_cost = min(dist / self.proximity_threshold, 1.0)

                # Mask IoU with pre-merge mask (strongest identity signal)
                if pm_mask is not None:
                    iou = mask_iou(masks[ci], pm_mask)
                    iou_cost = 1.0 - iou
                else:
                    iou_cost = 0.5

                # Area similarity with pre-merge area
                if pm_area is not None and pm_area > 0 and curr_areas[ci] > 0:
                    ratio = max(curr_areas[ci], pm_area) / min(curr_areas[ci], pm_area)
                    area_cost = min(ratio - 1.0, 1.0)
                else:
                    area_cost = 0.5

                # Weight IoU higher at split (0.4) — shape is best signal
                cost[row, col] = 0.3 * dist_cost + 0.4 * iou_cost + 0.3 * area_cost

        # --- Hungarian assignment ---
        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self._cost_threshold:
                continue
            slot_idx = active_slots[r]
            ci = valid_curr[c]
            out_masks[slot_idx] = masks[ci]
            out_centroids[slot_idx] = curr_centroids[ci]
            out_areas[slot_idx] = curr_areas[ci]
            out_scores[slot_idx] = scores[ci] if ci < len(scores) else 0.0

        logger.info(
            "Split resolved: %s",
            {active_slots[r]: valid_curr[c] for r, c in zip(row_ind, col_ind)
             if cost[r, c] <= self._cost_threshold},
        )

        self._update_prev(out_masks, out_centroids, out_areas)
        self._update_avg_area(out_areas)
        return out_masks, out_centroids, out_areas, out_scores

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------

    def _update_prev(
        self,
        masks: List[Optional[np.ndarray]],
        centroids: List[Optional[Tuple[float, float]]],
        areas: List[float],
    ) -> None:
        """Update previous-frame state and compute velocity."""
        for i in range(self.max_entities):
            if masks[i] is not None and centroids[i] is not None:
                # Compute velocity before updating centroid
                self._update_velocity(i, centroids[i])
                self.prev_masks[i] = masks[i]
                self.prev_centroids[i] = centroids[i]
                self.prev_areas[i] = areas[i]
            # If slot was unmatched this frame, keep prev state for fallback

    def _update_velocity(
        self, slot_idx: int, new_centroid: Tuple[float, float],
    ) -> None:
        """Update velocity for a slot with EMA smoothing and clamping."""
        prev_c = self.prev_centroids[slot_idx]
        if prev_c is None:
            self._velocities[slot_idx] = None
            return

        dx = new_centroid[0] - prev_c[0]
        dy = new_centroid[1] - prev_c[1]

        # Clamp velocity to prevent runaway prediction
        max_vel = self.proximity_threshold / 2
        dx = max(-max_vel, min(dx, max_vel))
        dy = max(-max_vel, min(dy, max_vel))

        # EMA smoothing: 0.7 * new + 0.3 * old
        prev_vel = self._velocities[slot_idx]
        if prev_vel is not None:
            dx = 0.7 * dx + 0.3 * prev_vel[0]
            dy = 0.7 * dy + 0.3 * prev_vel[1]

        self._velocities[slot_idx] = (dx, dy)

    def _update_avg_area(self, slot_areas: List[float]) -> None:
        """Update running average of single-object mask area.

        Only called during SEPARATE state so the average reflects
        individual object sizes, not merged blobs.
        """
        for area in slot_areas:
            if area > 0:
                self._area_samples += 1
                alpha = min(0.1, 1.0 / self._area_samples)
                self._avg_area = (1 - alpha) * self._avg_area + alpha * area
