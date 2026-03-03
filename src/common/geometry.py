"""Geometric utilities for tracking computations."""

from __future__ import annotations

from typing import Optional, Tuple


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
