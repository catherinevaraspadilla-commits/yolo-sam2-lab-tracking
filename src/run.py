# src/run.py
import argparse
from .models_io import load_models
from .pipeline import VideoPipeline, load_models
from . import strategies
from .config import VIDEOS_DIR, RESULTS_DIR


def build_pipeline(strategy_name: str) -> VideoPipeline:
    """
    Creates a VideoPipeline with the selected strategy.
    """
    yolo, sam = load_models()

    if strategy_name == "simple":
        strategy = strategies.strategy_yolo_sam_simple
    elif strategy_name == "tracking":
        strategy = strategies.strategy_yolo_sam_tracking
    else:
        raise ValueError(f"Unknown strategy '{strategy_name}'")

    return VideoPipeline(yolo, sam, strategy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default="tracking",
        choices=["simple", "tracking"],
        help="Which tracking strategy to use.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(VIDEOS_DIR / "output-6s.mp4"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "output_tracking.mp4"),
    )
    args = parser.parse_args()

    pipeline = build_pipeline(args.strategy)
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()

"""
parser.add_argument(
    "--strategy",
    type=str,
    default="tracking",
    choices=["simple", "tracking"],
    help="Which tracking strategy to use.",
)
TO RUN

python -m src.run

or

python -m src.run --strategy tracking --input data/videos/rat.mp4 --output data/results/out.mp4


"""