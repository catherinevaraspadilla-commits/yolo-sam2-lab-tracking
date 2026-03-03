"""Geometric utilities for tracking computations."""

from __future__ import annotations

from typing import List, Optional, Tuple


def euclidean_distance(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
) -> float:
    """Euclidean distance between two 2D points.

    Returns float('inf') if either point is None.
    """
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def match_dets_to_slots(
    detections: list,
    slot_centroids: list,
    max_animals: int,
    max_distance: float = 200.0,
) -> List[Optional[object]]:
    """Match YOLO detections to identity slots by centroid proximity.

    Returns a list of length *max_animals* where each element is the best
    matching Detection (or ``None``) for that slot.

    Each detection must expose a ``.center()`` method returning ``(x, y)``.
    """
    result: List[Optional[object]] = [None] * max_animals
    if not detections:
        return result

    det_centers = [d.center() for d in detections]
    used: set = set()

    for slot_idx in range(max_animals):
        sc = slot_centroids[slot_idx]
        if sc is None:
            continue

        best_di = None
        best_dist = float("inf")
        for di, dc in enumerate(det_centers):
            if di in used:
                continue
            dist = ((sc[0] - dc[0]) ** 2 + (sc[1] - dc[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_di = di

        if best_di is not None and best_dist < max_distance:
            result[slot_idx] = detections[best_di]
            used.add(best_di)

    return result
