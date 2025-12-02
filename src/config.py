from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
RESULTS_DIR = DATA_DIR / "results"

EXTERNAL_DIR = ROOT / "external" / "segment-anything-2"

# Checkpoint file path (this *can* be absolute)
SAM2_CHECKPOINT_PATH = EXTERNAL_DIR / "checkpoints" / "sam2.1_hiera_large.pt"

# Hydra config name INSIDE the `sam2` package (not a filesystem path)
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"

YOLO_MODEL_PATH = ROOT / "models" / "yolov8lrata.pt"

DEFAULT_YOLO_CONF = 0.5
DEFAULT_SAM_THRESHOLD = 0.0
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_TRACKING_MAX_DIST = 150.0

COLORS_RATS = [
    (0, 255, 0, 150),
    (255, 0, 0, 150),
]
