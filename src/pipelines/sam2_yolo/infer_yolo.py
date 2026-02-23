"""
YOLO inference for the SAM2+YOLO pipeline.

Runs YOLOv8 detection and returns structured Detection objects.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from ultralytics import YOLO

from src.common.utils import Detection

logger = logging.getLogger(__name__)


def detect_boxes(
    model: YOLO,
    frame_rgb: np.ndarray,
    confidence: float,
) -> List[Detection]:
    """Run YOLO detection on a single frame.

    Args:
        model: Loaded YOLO model.
        frame_rgb: Frame in RGB format (H, W, 3).
        confidence: Minimum confidence threshold.

    Returns:
        List of Detection objects above the confidence threshold.
    """
    results = model(frame_rgb, conf=confidence, verbose=False)

    detections: List[Detection] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        # Get class names if available
        cls_ids = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

        for i, (bbox, conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = bbox.tolist()
            class_name = None
            if cls_ids is not None and r.names:
                class_name = r.names.get(int(cls_ids[i]))
            detections.append(Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf),
                class_name=class_name,
            ))

    logger.debug("YOLO detected %d boxes (conf >= %.2f)", len(detections), confidence)
    return detections
