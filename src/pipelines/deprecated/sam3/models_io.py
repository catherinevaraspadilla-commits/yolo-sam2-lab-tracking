"""
Model loading for the SAM3 pipeline.

Loads SAM3 image processor (Sam3Processor) and YOLO (detect-only mode).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from ultralytics import YOLO

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from src.common.config_loader import get_device
from src.common.model_loaders import load_yolo

logger = logging.getLogger(__name__)


def load_sam3_processor(
    checkpoint_path: str | Path,
    device: str,
) -> Sam3Processor:
    """Load a SAM3 image processor.

    Args:
        checkpoint_path: Path to the SAM3 checkpoint .pt file.
        device: Device string ("cuda" or "cpu").

    Returns:
        Sam3Processor ready for inference.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    logger.info("Loading SAM3 from %s", checkpoint_path)
    model = build_sam3_image_model(checkpoint_path=str(checkpoint_path), device=device)
    return Sam3Processor(model)


def load_models(config: Dict[str, Any]) -> Tuple[YOLO, Sam3Processor]:
    """Load both YOLO and SAM3 models from a config dictionary.

    Args:
        config: Full pipeline config (must contain 'models' section).

    Returns:
        (yolo_model, sam3_processor) tuple.
    """
    device = get_device(config)
    models_cfg = config["models"]
    logger.info("Using device: %s", device)

    yolo = load_yolo(models_cfg["yolo_path"], device)
    sam3 = load_sam3_processor(
        models_cfg["sam3_checkpoint"],
        device,
    )
    return yolo, sam3
