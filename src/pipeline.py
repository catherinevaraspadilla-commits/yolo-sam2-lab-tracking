# src/pipeline.py
import cv2
import numpy as np
from typing import Callable, Dict, Any

from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .video_utils import (
    open_video_reader,
    create_video_writer,
)


StrategyFn = Callable[
    [YOLO, SAM2ImagePredictor, np.ndarray, Dict[str, Any] | None],
    tuple[np.ndarray, Dict[str, Any]]
]


class VideoPipeline:
    """
    Main video processing pipeline.
    Reads frames, applies a strategy, writes the output video.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        sam_predictor: SAM2ImagePredictor,
        strategy_fn: StrategyFn,
    ):
        self.yolo = yolo_model
        self.sam = sam_predictor
        self.strategy = strategy_fn
        self.state: Dict[str, Any] | None = None

    def run(self, video_input: str, video_output: str):
        """
        Runs the pipeline on the input video and writes the output video.
        """
        print(f"[Pipeline] Processing: {video_input}")

        cap = open_video_reader(video_input)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Pipeline] {width}x{height} @ {fps} FPS, {total_frames} frames\n")

        writer = create_video_writer(video_output, fps, width, height)

        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frame_out_rgb, self.state = self.strategy(
                self.yolo, self.sam, frame_rgb, self.state
            )

            frame_out_bgr = cv2.cvtColor(frame_out_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_out_bgr)

            frame_idx += 1

        cap.release()
        writer.release()

        print(f"[Pipeline] Completed. {frame_idx} frames processed.")
        print(f"[Pipeline] Output saved to: {video_output}")
