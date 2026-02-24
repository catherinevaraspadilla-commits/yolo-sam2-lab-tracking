"""
YOLO inference for the SAM2+YOLO pipeline.

Uses model.track() with BoT-SORT/ByteTrack for stable track IDs across frames.
Supports pose models (keypoints), class filtering, and border padding.
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
    """Add mirror-border padding around a frame."""
    return cv2.copyMakeBorder(
        frame_rgb, padding_px, padding_px, padding_px, padding_px,
        cv2.BORDER_REFLECT_101,
    )


def correct_detections_after_padding(
    detections: List[Detection],
    padding_px: int,
    orig_h: int,
    orig_w: int,
) -> List[Detection]:
    """Shift detection coordinates back to the original frame and clamp."""
    corrected = []
    for det in detections:
        x1 = max(0.0, det.x1 - padding_px)
        y1 = max(0.0, det.y1 - padding_px)
        x2 = min(float(orig_w), det.x2 - padding_px)
        y2 = min(float(orig_h), det.y2 - padding_px)

        if x2 <= x1 or y2 <= y1:
            continue

        kps = None
        if det.keypoints is not None:
            kps = [
                Keypoint(
                    x=max(0.0, min(float(orig_w), kp.x - padding_px)),
                    y=max(0.0, min(float(orig_h), kp.y - padding_px)),
                    conf=kp.conf, name=kp.name,
                )
                for kp in det.keypoints
            ]

        corrected.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            conf=det.conf, class_name=det.class_name,
            keypoints=kps, track_id=det.track_id,
        ))
    return corrected


def _parse_results(
    results,
    keypoint_names: List[str],
    filter_class: Optional[str],
) -> List[Detection]:
    """Parse Ultralytics results into Detection objects."""
    detections: List[Detection] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

        # Track IDs from model.track() — may be None if using model()
        track_ids = None
        if r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().numpy()

        # Keypoints from pose model
        has_kpts = r.keypoints is not None and len(r.keypoints) > 0
        kpts_data = r.keypoints.data.cpu().numpy() if has_kpts else None

        for i, (bbox, conf) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = bbox.tolist()
            class_name = None
            if cls_ids is not None and r.names:
                class_name = r.names.get(int(cls_ids[i]))

            if filter_class is not None and class_name != filter_class:
                continue

            # Track ID
            tid = int(track_ids[i]) if track_ids is not None else None

            # Keypoints
            kps = None
            if kpts_data is not None:
                kps = []
                for kp_idx in range(kpts_data.shape[1]):
                    kx, ky, kc = kpts_data[i][kp_idx]
                    name = keypoint_names[kp_idx] if kp_idx < len(keypoint_names) else f"kp{kp_idx}"
                    kps.append(Keypoint(x=float(kx), y=float(ky), conf=float(kc), name=name))

            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                conf=float(conf), class_name=class_name,
                keypoints=kps, track_id=tid,
            ))

    return detections


def detect_and_track(
    model: YOLO,
    frame_rgb: np.ndarray,
    confidence: float,
    tracker_config: str = "botsort.yaml",
    keypoint_names: Optional[List[str]] = None,
    filter_class: Optional[str] = None,
    border_padding_px: int = 0,
) -> List[Detection]:
    """Run YOLO detection+tracking on a single frame using model.track().

    Uses Ultralytics built-in BoT-SORT or ByteTrack for stable track IDs.
    The tracker maintains state across calls when persist=True.

    Args:
        model: Loaded YOLO model (detect or pose).
        frame_rgb: Frame in RGB format (H, W, 3).
        confidence: Minimum confidence threshold.
        tracker_config: Tracker config YAML name (e.g., "botsort.yaml",
            "bytetrack.yaml", or path to custom YAML).
        keypoint_names: Names for each keypoint index.
        filter_class: Only keep detections of this class name.
        border_padding_px: Mirror-pad frame borders before YOLO.

    Returns:
        List of Detection objects with .track_id set by the tracker.
    """
    orig_h, orig_w = frame_rgb.shape[:2]

    if border_padding_px > 0:
        frame_for_yolo = pad_frame(frame_rgb, border_padding_px)
    else:
        frame_for_yolo = frame_rgb

    if keypoint_names is None:
        keypoint_names = DEFAULT_KEYPOINT_NAMES

    # model.track() maintains tracker state across frames with persist=True
    results = model.track(
        frame_for_yolo,
        conf=confidence,
        persist=True,
        tracker=tracker_config,
        verbose=False,
    )

    detections = _parse_results(results, keypoint_names, filter_class)

    if border_padding_px > 0:
        detections = correct_detections_after_padding(
            detections, border_padding_px, orig_h, orig_w,
        )

    logger.debug(
        "YOLO tracked %d boxes (conf >= %.2f), track_ids: %s",
        len(detections), confidence,
        [d.track_id for d in detections],
    )
    return detections
