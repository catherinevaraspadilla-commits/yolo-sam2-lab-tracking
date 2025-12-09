# src/roboflow/upload.py

"""
Upload frames from the frames/ folder to a Roboflow project.

Usage example (from project root):

  # Export frames first with close_contact.py, then:
  python -m src.roboflow.upload run \
      --frames-root frames \
      --workspace modelos-yolo \
      --project pruebasratslabs-c02t9 \
      --split train \
      --batch-name close-contact-2025-12-09

API key can be passed via:
  - Environment variable: ROBOFLOW_API_KEY
  - Or CLI argument: --api-key
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from roboflow import Roboflow


class RoboflowUploader:
    """
    Helper class to upload images from a folder structure like:

        frames/
          encounter_000/
            rat_close_e000_f000123.jpg
          encounter_001/
            rat_close_e001_f000567.jpg
          ...

    to a Roboflow project.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project_id: str,
        split: str = "train",
        batch_name: Optional[str] = None,
        tag_names: Optional[List[str]] = None,
        num_retry_uploads: int = 2,
    ) -> None:
        """
        Initialize the uploader.

        Args:
            api_key: Roboflow API key (use env var ROBOFLOW_API_KEY or pass explicitly).
            workspace: Roboflow workspace slug (e.g., "modelos-yolo").
            project_id: Roboflow project slug (e.g., "pruebasratslabs-c02t9").
            split: Dataset split to upload to: "train", "valid", or "test".
            batch_name: Optional batch name to group these images in Roboflow.
            tag_names: Optional list of tags to attach to images (e.g., ["close_contact"]).
            num_retry_uploads: Number of times to retry uploads if they fail.
        """
        self.api_key = api_key
        self.workspace = workspace
        self.project_id = project_id
        self.split = split
        self.batch_name = batch_name
        self.tag_names = tag_names
        self.num_retry_uploads = num_retry_uploads

        print("[upload] Initializing Roboflow client...")
        rf = Roboflow(api_key=self.api_key)
        self.project = rf.workspace(self.workspace).project(self.project_id)
        print(f"[upload] Connected to workspace='{self.workspace}', project='{self.project_id}'")

    def _iter_image_paths(self, frames_root: Path) -> List[Path]:
        """
        Find all image files under frames_root.

        We look for typical image extensions: .jpg, .jpeg, .png, .bmp, .webp.
        """
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        frames_root = frames_root.resolve()
        if not frames_root.exists():
            raise FileNotFoundError(f"frames_root does not exist: {frames_root}")

        print(f"[upload] Scanning for images under: {frames_root}")
        image_paths: List[Path] = []
        for path in frames_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)

        image_paths = sorted(image_paths)
        print(f"[upload] Found {len(image_paths)} images to upload.")
        return image_paths

    def upload_folder(self, frames_root: Path) -> Dict[str, Any]:
        """
        Upload all images found under frames_root to the configured Roboflow project.

        Returns a small summary dictionary:
          {
            "total_images": N,
            "successful": K,
            "failed": M,
            "failed_paths": [...],
          }
        """
        image_paths = self._iter_image_paths(frames_root)
        total = len(image_paths)
        successful = 0
        failed = 0
        failed_paths: List[str] = []

        if total == 0:
            print("[upload] No images found. Nothing to upload.")
            return {
                "total_images": 0,
                "successful": 0,
                "failed": 0,
                "failed_paths": [],
            }

        print(f"[upload] Starting upload: {total} images")
        for idx, img_path in enumerate(image_paths, start=1):
            img_str = str(img_path)
            try:
                print(f"[upload] ({idx}/{total}) Uploading: {img_str}")
                self.project.upload(
                    image_path=img_str,
                    split=self.split,
                    num_retry_uploads=self.num_retry_uploads,
                    batch_name=self.batch_name,
                    tag_names=self.tag_names,
                )
                successful += 1
            except Exception as e:
                failed += 1
                failed_paths.append(img_str)
                print(f"[upload] ERROR uploading {img_str}: {e}")

        print("\n[upload] ========= SUMMARY =========")
        print(f"[upload] Total images   : {total}")
        print(f"[upload] Successful     : {successful}")
        print(f"[upload] Failed         : {failed}")
        if failed > 0:
            print("[upload] Failed paths:")
            for p in failed_paths:
                print(f"  - {p}")
        print("[upload] ============================\n")

        return {
            "total_images": total,
            "successful": successful,
            "failed": failed,
            "failed_paths": failed_paths,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload frames from local folder to a Roboflow project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Upload images from frames_root to Roboflow.")
    p_run.add_argument(
        "--frames-root",
        type=Path,
        default=Path("frames"),
        help="Root folder containing encounter_* subfolders with images.",
    )
    p_run.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (if not provided, uses ROBOFLOW_API_KEY env var).",
    )
    p_run.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Roboflow workspace slug (e.g., 'modelos-yolo').",
    )
    p_run.add_argument(
        "--project",
        type=str,
        required=True,
        help="Roboflow project slug (e.g., 'pruebasratslabs-c02t9').",
    )
    p_run.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to upload to.",
    )
    p_run.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Optional batch name for these uploads in Roboflow.",
    )
    p_run.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=None,
        help="Tag to apply to uploaded images (can be passed multiple times).",
    )
    p_run.add_argument(
        "--num-retry-uploads",
        type=int,
        default=2,
        help="Number of times to retry uploads on failure.",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "run":
        api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key provided. Use --api-key or set ROBOFLOW_API_KEY environment variable."
            )

        uploader = RoboflowUploader(
            api_key=api_key,
            workspace=args.workspace,
            project_id=args.project,
            split=args.split,
            batch_name=args.batch_name,
            tag_names=args.tags,
            num_retry_uploads=args.num_retry_uploads,
        )

        uploader.upload_folder(args.frames_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


"""
export ROBOFLOW_API_KEY="kD254mWPx6WlP2phIeC4"

python -m src.roboflow.upload run \
    --frames-root frames \
    --workspace modelos-yolo \
    --project pruebasratslabs-c02t9 \
    --split train \
    --batch-name close-contact-2025-12-09 \
    --tag close_contact


"""