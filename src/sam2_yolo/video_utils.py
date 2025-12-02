# src/video_utils.py
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple

from .config import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_DIST,
    COLORS_RATS,
)


# -------------------------
# Geometry and metrics
# -------------------------

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes IoU between two boolean masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def compute_centroid(mask: np.ndarray) -> Tuple[float, float] | None:
    """
    Computes the (x, y) centroid of a mask.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


# -------------------------
# Smart mask filtering
# -------------------------

def filter_masks_smart(
    masks: List[np.ndarray],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_rats: int = 2,
) -> List[np.ndarray]:
    """
    Keeps at most 'max_rats' masks.
    If more masks exist, duplicates are removed using IoU.
    """
    if len(masks) <= max_rats:
        return masks

    areas = [m.sum() for m in masks]
    sorted_idx = np.argsort(areas)[::-1]
    sorted_masks = [masks[i] for i in sorted_idx]

    unique = []
    for mask in sorted_masks:
        if all(compute_iou(mask, u) <= iou_threshold for u in unique):
            unique.append(mask)
            if len(unique) >= max_rats:
                break

    return unique


# -------------------------
# Centroid-based matching
# -------------------------

def match_masks_by_centroid(
    current_masks: List[np.ndarray],
    previous_centroids: List[Tuple[float, float]],
    max_distance: float = DEFAULT_TRACKING_MAX_DIST,
):
    """
    Reorders current masks so they match previous rat identity
    based on centroid distance.
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

    # Fill empty slots with unused masks
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


# -------------------------
# Rendering utilities
# -------------------------

def apply_masks_overlay(frame_rgb, masks):
    """
    Applies colored RGBA masks on top of the RGB frame.
    """
    h, w, _ = frame_rgb.shape
    out = frame_rgb.astype(np.float32).copy()

    for i, m in enumerate(masks):
        r, g, b, a = COLORS_RATS[i % len(COLORS_RATS)]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[:, :, :3] = (r, g, b)
        overlay[:, :, 3] = m.astype(np.uint8) * a

        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        out = out * (1 - alpha) + overlay[:, :, :3] * alpha

    return out.astype(np.uint8)


def draw_centroids(frame_rgb, centroids):
    """
    Draws circles and text labels for rat centroids.
    """
    out = frame_rgb.copy()
    for i, c in enumerate(centroids):
        if c is None:
            continue
        cx, cy = int(c[0]), int(c[1])
        r, g, b, _ = COLORS_RATS[i]
        cv2.circle(out, (cx, cy), 6, (r, g, b), -1)
        cv2.putText(out, f"R{i+1}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    return out


def open_video_reader(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def create_video_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
