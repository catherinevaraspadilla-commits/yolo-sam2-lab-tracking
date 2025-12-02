# src/models_io.py
import sys
import torch
from ultralytics import YOLO

from .config import (
    YOLO_MODEL_PATH,
    SAM2_CHECKPOINT_PATH,
    SAM2_CONFIG_NAME,
    EXTERNAL_DIR,
)

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Models] Using device: {DEVICE}")


def load_yolo():
    """
    Loads YOLO model and moves it to the selected device.
    """
    model = YOLO(str(YOLO_MODEL_PATH))
    model.to(DEVICE)
    return model


def load_sam2_predictor():
    """
    Loads SAM2 using Hydra config name and checkpoint path.
    """
    sam2_model = build_sam2(
        SAM2_CONFIG_NAME,             # config name inside `sam2` package
        str(SAM2_CHECKPOINT_PATH),    # checkpoint path
        device=DEVICE,
    )
    return SAM2ImagePredictor(sam2_model)


def load_models():
    """
    Loads YOLO and SAM2 predictor.
    """
    yolo = load_yolo()
    sam = load_sam2_predictor()
    return yolo, sam
