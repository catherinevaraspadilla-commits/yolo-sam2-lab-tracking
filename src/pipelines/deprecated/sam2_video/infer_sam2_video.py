"""
SAM2 Video mode inference wrapper.

Handles JPEG frame extraction, SAM2VideoPredictor initialization with box
prompts, and mask propagation across frames with temporal memory.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor

from src.common.utils import Detection

logger = logging.getLogger(__name__)


def extract_frames_to_jpeg(
    video_path: str | Path,
    output_dir: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[Path, int, Tuple[int, int]]:
    """Extract video frames to a directory of JPEG files.

    SAM2VideoPredictor requires a directory of JPEG images named as
    '%05d.jpg' (e.g., 00000.jpg, 00001.jpg, ...).

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to write JPEG frames into.
        start_frame: First frame to extract (0-based).
        end_frame: Stop before this frame (exclusive). None = to end.

    Returns:
        (output_dir, num_frames_extracted, (height, width)) tuple.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    count = 0
    for abs_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # SAM2 expects sequential naming starting from 0
        out_path = output_dir / f"{count:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        count += 1

    cap.release()
    logger.info(
        "Extracted %d frames [%d:%d) → %s",
        count, start_frame, start_frame + count, output_dir,
    )
    return output_dir, count, (h, w)


def init_segment(
    predictor: SAM2VideoPredictor,
    frames_dir: str | Path,
    detections: List[Detection],
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
) -> Dict[str, Any]:
    """Initialize SAM2VideoPredictor on a segment of JPEG frames.

    Args:
        predictor: Loaded SAM2VideoPredictor.
        frames_dir: Directory of JPEG frames ('%05d.jpg').
        detections: YOLO detections from the first frame (used as box prompts).
        offload_video_to_cpu: Keep frames on CPU to save GPU memory.
        offload_state_to_cpu: Keep inference state on CPU (slower but less VRAM).

    Returns:
        inference_state dict for use with propagate/add_new_points_or_box.
    """
    frames_dir = str(Path(frames_dir))
    state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )

    for obj_id, det in enumerate(detections):
        box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            box=box,
        )
        logger.debug(
            "Added box prompt for obj_id=%d: [%.0f, %.0f, %.0f, %.0f] (conf=%.2f)",
            obj_id, det.x1, det.y1, det.x2, det.y2, det.conf,
        )

    logger.info(
        "SAM2 Video initialized with %d objects on %s",
        len(detections), frames_dir,
    )
    return state


def propagate_segment(
    predictor: SAM2VideoPredictor,
    state: Dict[str, Any],
) -> Iterator[Tuple[int, List[int], List[np.ndarray]]]:
    """Propagate SAM2 masks through all frames in the segment.

    Wraps predictor.propagate_in_video() and converts GPU mask tensors
    to boolean numpy arrays (H, W).

    Args:
        predictor: SAM2VideoPredictor instance.
        state: Inference state from init_segment().

    Yields:
        (local_frame_idx, obj_ids, masks) where:
            - local_frame_idx: 0-based index within this segment
            - obj_ids: list of object IDs
            - masks: list of boolean np.ndarray (H, W), one per object
    """
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
        # video_res_masks shape: (num_objects, 1, H, W) — logit scores
        masks = []
        for i in range(video_res_masks.shape[0]):
            mask_logit = video_res_masks[i, 0]  # (H, W)
            mask_bool = (mask_logit > 0.0).cpu().numpy()
            masks.append(mask_bool)

        yield frame_idx, list(obj_ids), masks
