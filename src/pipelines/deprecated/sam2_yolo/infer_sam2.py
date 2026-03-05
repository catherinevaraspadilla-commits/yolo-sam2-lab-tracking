"""
SAM2 segmentation for the SAM2+YOLO pipeline.

Uses YOLO detection boxes as prompts for SAM2 mask prediction.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.common.utils import Detection

logger = logging.getLogger(__name__)


def segment_from_boxes(
    predictor: SAM2ImagePredictor,
    frame_rgb: np.ndarray,
    detections: List[Detection],
    sam_threshold: float,
    iou_threshold: float,
) -> List[np.ndarray]:
    """Run SAM2 segmentation using YOLO detection boxes as prompts.

    Args:
        predictor: Loaded SAM2ImagePredictor.
        frame_rgb: Frame in RGB format (H, W, 3).
        detections: YOLO detections to use as box prompts.
        sam_threshold: Threshold for raw SAM2 mask logits.
        iou_threshold: Not used directly here (reserved for postprocessing).

    Returns:
        List of boolean masks, one per detection.
    """
    if len(detections) == 0:
        return []

    predictor.set_image(frame_rgb)

    masks: List[np.ndarray] = []
    for i, det in enumerate(detections):
        box = np.array([det.x1, det.y1, det.x2, det.y2])
        raw_masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        m = raw_masks[0] > sam_threshold
        masks.append(m)
        logger.debug("SAM2 box %d -> mask area: %d px", i, m.sum())

    return masks
