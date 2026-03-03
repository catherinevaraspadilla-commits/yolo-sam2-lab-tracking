"""Shared model loading utilities used by all pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_yolo(model_path: str | Path, device: str) -> YOLO:
    """Load a YOLO model. Auto-detects architecture (v8, 11, 26).

    Args:
        model_path: Path to the .pt weights file.
        device: Device string ("cuda" or "cpu").

    Returns:
        Loaded YOLO model on the specified device.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    logger.info("Loading YOLO model from %s", model_path)
    model = YOLO(str(model_path))
    model.to(device)
    return model
