# src/strategies.py
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .config import (
    DEFAULT_YOLO_CONF,
    DEFAULT_SAM_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_DIST,
    COLORS_RATS,
)
from . import video_utils as vu


# -------------------------
# YOLO detection
# -------------------------

def _yolo_detect_boxes(
    model: YOLO,
    frame_rgb: np.ndarray,
    conf: float,
):
    """
    Runs YOLO and returns a list of [x1, y1, x2, y2] boxes.
    """
    results = model(frame_rgb, conf=conf, verbose=False)

    boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            boxes.append([x1, y1, x2, y2])

    print(f"   [YOLO] Boxes detected: {len(boxes)}")
    return boxes


# -------------------------
# SAM2 segmentation
# -------------------------

def _segment_with_boxes(
    predictor: SAM2ImagePredictor,
    frame_rgb: np.ndarray,
    boxes,
    sam_thresh: float,
    iou_thresh: float,
):
    """
    Runs SAM2 on each YOLO box and applies smart mask filtering.
    """
    if len(boxes) == 0:
        return []

    predictor.set_image(frame_rgb)

    masks = []
    for i, box in enumerate(boxes):
        raw_masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box),
            multimask_output=False,
        )
        m = raw_masks[0] > sam_thresh
        masks.append(m)
        print(f"   [SAM] Box {i} -> mask area: {m.sum()} px")

    return vu.filter_masks_smart(masks, iou_thresh)


# -------------------------
# Strategy 1: YOLO + SAM (no tracking)
# -------------------------

def strategy_yolo_sam_simple(
    yolo: YOLO,
    sam: SAM2ImagePredictor,
    frame_rgb: np.ndarray,
    state: Dict[str, Any] | None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    YOLO detection + SAM mask segmentation + duplicate filtering.
    No tracking between frames.
    """
    boxes = _yolo_detect_boxes(yolo, frame_rgb, DEFAULT_YOLO_CONF)

    masks = _segment_with_boxes(
        sam, frame_rgb, boxes,
        sam_thresh=DEFAULT_SAM_THRESHOLD,
        iou_thresh=DEFAULT_IOU_THRESHOLD,
    )

    frame_out = vu.apply_masks_overlay(frame_rgb, masks)

    cv2.putText(frame_out, f"Rats: {len(masks)}/2",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    return frame_out, state or {}
    

# -------------------------
# Strategy 2: YOLO + SAM + centroid tracking
# -------------------------

def strategy_yolo_sam_tracking(
    yolo: YOLO,
    sam: SAM2ImagePredictor,
    frame_rgb: np.ndarray,
    state: Dict[str, Any] | None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    YOLO detection + SAM segmentation + IoU filtering + centroid-based tracking.
    Maintains consistent rat identity/colors across frames.
    """
    if state is None:
        state = {}

    prev_centroids = state.get("prev_centroids", [])

    boxes = _yolo_detect_boxes(yolo, frame_rgb, DEFAULT_YOLO_CONF)

    masks_unordered = _segment_with_boxes(
        sam, frame_rgb, boxes,
        sam_thresh=DEFAULT_SAM_THRESHOLD,
        iou_thresh=DEFAULT_IOU_THRESHOLD,
    )

    masks_ordered, centroids = vu.match_masks_by_centroid(
        masks_unordered,
        prev_centroids,
        max_distance=DEFAULT_TRACKING_MAX_DIST,
    )

    frame_out = vu.apply_masks_overlay(frame_rgb, masks_ordered)
    frame_out = vu.draw_centroids(frame_out, centroids)

    cv2.putText(frame_out, f"Rats: {len(masks_ordered)}/2",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    new_state = {"prev_centroids": centroids}
    return frame_out, new_state
