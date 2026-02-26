"""
Model loading for the SAM2 Video pipeline.

Loads SAM2VideoPredictor (with temporal memory) and YOLO (detect-only mode).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from src.common.config_loader import get_device
from src.pipelines.sam2_yolo.models_io import load_yolo

logger = logging.getLogger(__name__)


def load_sam2_video_predictor(
    checkpoint_path: str | Path,
    config_name: str,
    device: str,
) -> SAM2VideoPredictor:
    """Load a SAM2 video predictor (with temporal memory).

    Args:
        checkpoint_path: Path to the SAM2 checkpoint .pt file.
        config_name: Hydra config name inside the sam2 package
            (e.g., "configs/sam2.1/sam2.1_hiera_t.yaml").
        device: Device string ("cuda" or "cpu").

    Returns:
        SAM2VideoPredictor ready for video inference.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
    logger.info("Loading SAM2 VideoPredictor from %s (config: %s)", checkpoint_path, config_name)
    predictor = build_sam2_video_predictor(config_name, str(checkpoint_path), device=device)
    return predictor


def load_models(config: Dict[str, Any]) -> Tuple[YOLO, SAM2VideoPredictor]:
    """Load both YOLO and SAM2VideoPredictor from a config dictionary.

    Args:
        config: Full pipeline config (must contain 'models' section).

    Returns:
        (yolo_model, sam2_video_predictor) tuple.
    """
    device = get_device(config)
    models_cfg = config["models"]
    logger.info("Using device: %s", device)

    yolo = load_yolo(models_cfg["yolo_path"], device)
    sam2 = load_sam2_video_predictor(
        models_cfg["sam2_checkpoint"],
        models_cfg["sam2_config"],
        device,
    )
    return yolo, sam2
