"""
YOLO inference for the SAM2+YOLO pipeline.

Runs YOLOv8 detection and returns structured Detection objects.
Supports both standard detection models and pose models (with keypoints).
"""

from __future__ import annotations

import logging
from typing import List, Optional

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
    "back",
    "mid_body",
    "neck",
    "ear",
    "nose",
]


def detect_boxes(
    model: YOLO,
    frame_rgb: np.ndarray,
    confidence: float,
    keypoint_names: Optional[List[str]] = None,
) -> List[Detection]:
    """Run YOLO detection on a single frame.

    Supports both standard detection and pose models. If the model outputs
    keypoints, they are attached to each Detection object.

    Args:
        model: Loaded YOLO model (detect or pose).
        frame_rgb: Frame in RGB format (H, W, 3).
        confidence: Minimum confidence threshold.
        keypoint_names: Names for each keypoint index. If None, uses
            DEFAULT_KEYPOINT_NAMES. Length must match the model's keypoint count.

    Returns:
        List of Detection objects above the confidence threshold.
    """
    results = model(frame_rgb, conf=confidence, verbose=False)

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

    logger.debug("YOLO detected %d boxes (conf >= %.2f)", len(detections), confidence)
    return detections
