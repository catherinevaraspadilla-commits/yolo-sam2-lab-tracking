"""
YOLO inference for the SAM2+YOLO pipeline.

Runs YOLOv8 detection and returns structured Detection objects.
Supports both standard detection models and pose models (with keypoints).
Includes optional border padding to improve detections at frame edges.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.common.utils import Detection, Keypoint

logger = logging.getLogger(__name__)

# Default keypoint names for the 7-point rat pose model.
# Order matches the training annotation order in Roboflow.
# Override via config: detection.keypoint_names
DEFAULT_KEYPOINT_NAMES = [
    "tail_tip",
    "tail_base",
    "tail_start",
    "mid_body",
    "nose",
    "right_ear",
    "left_ear",
]


def pad_frame(
    frame_rgb: np.ndarray,
    padding_px: int,
) -> np.ndarray:
    """Add mirror-border padding around a frame.

    Args:
        frame_rgb: Input frame (H, W, 3).
        padding_px: Pixels to pad on each side.

    Returns:
        Padded frame (H + 2*pad, W + 2*pad, 3).
    """
    return cv2.copyMakeBorder(
        frame_rgb, padding_px, padding_px, padding_px, padding_px,
        cv2.BORDER_REFLECT_101,
    )


def correct_boxes_after_padding(
    detections: List[Detection],
    padding_px: int,
    orig_h: int,
    orig_w: int,
) -> List[Detection]:
    """Shift detection coordinates back to the original frame and clamp to bounds.

    Also shifts keypoint coordinates if present.

    Args:
        detections: Detections from the padded frame.
        padding_px: Padding that was added.
        orig_h: Original frame height.
        orig_w: Original frame width.

    Returns:
        Corrected detections with coordinates in original frame space.
    """
    corrected = []
    for det in detections:
        x1 = max(0.0, det.x1 - padding_px)
        y1 = max(0.0, det.y1 - padding_px)
        x2 = min(float(orig_w), det.x2 - padding_px)
        y2 = min(float(orig_h), det.y2 - padding_px)

        # Skip boxes that ended up entirely outside the original frame
        if x2 <= x1 or y2 <= y1:
            continue

        kps = None
        if det.keypoints is not None:
            kps = []
            for kp in det.keypoints:
                kps.append(Keypoint(
                    x=max(0.0, min(float(orig_w), kp.x - padding_px)),
                    y=max(0.0, min(float(orig_h), kp.y - padding_px)),
                    conf=kp.conf,
                    name=kp.name,
                ))

        corrected.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            conf=det.conf, class_name=det.class_name, keypoints=kps,
        ))
    return corrected


def detect_boxes(
    model: YOLO,
    frame_rgb: np.ndarray,
    confidence: float,
    keypoint_names: Optional[List[str]] = None,
    filter_class: Optional[str] = None,
    border_padding_px: int = 0,
) -> List[Detection]:
    """Run YOLO detection on a single frame.

    Supports both standard detection and pose models. If the model outputs
    keypoints, they are attached to each Detection object.

    Args:
        model: Loaded YOLO model (detect or pose).
        frame_rgb: Frame in RGB format (H, W, 3).
        confidence: Minimum confidence threshold.
        keypoint_names: Names for each keypoint index. If None, uses
            DEFAULT_KEYPOINT_NAMES.
        filter_class: If set, only keep detections of this class name
            (e.g., "ratas"). None keeps all classes.
        border_padding_px: Mirror-pad the frame by this many pixels before
            running YOLO, then correct boxes back. 0 = no padding.

    Returns:
        List of Detection objects above the confidence threshold.
    """
    orig_h, orig_w = frame_rgb.shape[:2]

    # Optional border padding
    if border_padding_px > 0:
        frame_for_yolo = pad_frame(frame_rgb, border_padding_px)
    else:
        frame_for_yolo = frame_rgb

    results = model(frame_for_yolo, conf=confidence, verbose=False)

    if keypoint_names is None:
        keypoint_names = DEFAULT_KEYPOINT_NAMES

    detections: List[Detection] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

        # Extract keypoints if this is a pose model
        has_kpts = r.keypoints is not None and len(r.keypoints) > 0
        kpts_data = r.keypoints.data.cpu().numpy() if has_kpts else None

        for i, (bbox, conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = bbox.tolist()
            class_name = None
            if cls_ids is not None and r.names:
                class_name = r.names.get(int(cls_ids[i]))

            # Class filter: skip detections that don't match
            if filter_class is not None and class_name != filter_class:
                continue

            # Build keypoints list
            kps = None
            if kpts_data is not None:
                kps = []
                for kp_idx in range(kpts_data.shape[1]):
                    kx, ky, kc = kpts_data[i][kp_idx]
                    name = keypoint_names[kp_idx] if kp_idx < len(keypoint_names) else f"kp{kp_idx}"
                    kps.append(Keypoint(
                        x=float(kx), y=float(ky), conf=float(kc), name=name,
                    ))

            detections.append(Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf),
                class_name=class_name,
                keypoints=kps,
            ))

    # Correct coordinates if we used padding
    if border_padding_px > 0:
        detections = correct_boxes_after_padding(
            detections, border_padding_px, orig_h, orig_w,
        )

    logger.debug("YOLO detected %d boxes (conf >= %.2f)", len(detections), confidence)
    return detections
