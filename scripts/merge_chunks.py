"""Merge outputs from parallel chunk processing into a single run directory.

Combines per-chunk CSVs, overlay videos, and contact analysis outputs.

Usage:
    python scripts/merge_chunks.py outputs/runs/*_chunk*/

    # Custom output directory:
    python scripts/merge_chunks.py outputs/runs/*_chunk*/ -o outputs/runs/merged
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_chunk_dirs(paths: List[str]) -> List[Path]:
    """Sort chunk directories by chunk ID extracted from directory name."""
    dirs = []
    for p in paths:
        d = Path(p)
        if d.is_dir():
            dirs.append(d)
    # Sort by chunk ID (e.g., "2026-02-26_120000_sam2_yolo_chunk0" → 0)
    def chunk_key(d: Path) -> int:
        name = d.name
        if "_chunk" in name:
            try:
                return int(name.rsplit("_chunk", 1)[1])
            except ValueError:
                pass
        return 0
    dirs.sort(key=chunk_key)
    return dirs


def merge_csvs(chunk_dirs: List[Path], filename: str, output_dir: Path) -> Path | None:
    """Concatenate a CSV file from each chunk, keeping header from first chunk."""
    out_path = output_dir / filename
    header_written = False
    total_rows = 0

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        for cdir in chunk_dirs:
            csv_path = cdir / filename
            if not csv_path.exists():
                # Check in contacts/ subdirectory
                csv_path = cdir / "contacts" / filename
                if not csv_path.exists():
                    continue
            with csv_path.open("r", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                for i, row in enumerate(reader):
                    if i == 0:
                        if not header_written:
                            csv.writer(out_f).writerow(row)
                            header_written = True
                        continue
                    csv.writer(out_f).writerow(row)
                    total_rows += 1

    if total_rows == 0:
        out_path.unlink(missing_ok=True)
        return None
    logger.info("Merged %s: %d rows from %d chunks", filename, total_rows, len(chunk_dirs))
    return out_path


def merge_overlay_videos(chunk_dirs: List[Path], output_dir: Path) -> Path | None:
    """Concatenate overlay videos using ffmpeg (if available) or OpenCV."""
    video_paths = []
    for cdir in chunk_dirs:
        overlays = cdir / "overlays"
        if overlays.exists():
            # Find any video file in overlays/ (supports new naming: pipeline_model_date.avi)
            found = sorted(overlays.glob("*.avi")) + sorted(overlays.glob("*.mp4"))
            if found:
                video_paths.append(found[0])  # take the first video found

    if not video_paths:
        logger.warning("No overlay videos found to merge")
        return None

    if len(video_paths) == 1:
        out_path = output_dir / "overlays" / video_paths[0].name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_paths[0], out_path)
        return out_path

    # Try ffmpeg concat — use same name as chunk video but with _merged suffix
    ext = video_paths[0].suffix
    base_name = video_paths[0].stem  # e.g. "reference_sam2.1_hiera_large_2026-02-27"
    out_path = output_dir / "overlays" / f"{base_name}_merged{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write concat list file
    list_file = output_dir / "overlays" / "_concat_list.txt"
    with list_file.open("w") as f:
        for vp in video_paths:
            f.write(f"file '{vp.resolve()}'\n")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c", "copy", str(out_path)],
            check=True, capture_output=True, text=True,
        )
        list_file.unlink()
        logger.info("Merged %d overlay videos → %s (ffmpeg)", len(video_paths), out_path)
        return out_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not available or failed. Falling back to OpenCV concat.")
        list_file.unlink(missing_ok=True)

    # Fallback: OpenCV concat (re-encodes)
    import cv2
    first = cv2.VideoCapture(str(video_paths[0]))
    fps = first.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
    writer.release()
    logger.info("Merged %d overlay videos → %s (OpenCV)", len(video_paths), out_path)
    return out_path


def merge_session_summaries(chunk_dirs: List[Path], output_dir: Path) -> None:
    """Merge session_summary.json files from contact analysis chunks."""
    summaries = []
    for cdir in chunk_dirs:
        sp = cdir / "contacts" / "session_summary.json"
        if sp.exists():
            with sp.open() as f:
                summaries.append(json.load(f))

    if not summaries:
        return

    # Merge: sum up counts, combine bout lists
    merged = {
        "merged_from_chunks": len(summaries),
        "merge_timestamp": datetime.now().isoformat(),
        "total_frames": sum(s.get("total_frames", 0) for s in summaries),
        "contact_type_summary": {},
    }

    # Aggregate contact type counts
    all_types = set()
    for s in summaries:
        all_types.update(s.get("contact_type_summary", {}).keys())

    for ct in sorted(all_types):
        merged["contact_type_summary"][ct] = {
            "total_bouts": sum(
                s.get("contact_type_summary", {}).get(ct, {}).get("total_bouts", 0)
                for s in summaries
            ),
            "total_frames": sum(
                s.get("contact_type_summary", {}).get(ct, {}).get("total_frames", 0)
                for s in summaries
            ),
        }

    contacts_dir = output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = contacts_dir / "session_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    logger.info("Merged session summary from %d chunks → %s", len(summaries), out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge parallel chunk outputs into a single run.",
    )
    parser.add_argument(
        "chunk_dirs", nargs="+",
        help="Chunk output directories (e.g., outputs/runs/*_chunk*/).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory. Default: outputs/runs/<timestamp>_merged/",
    )
    args = parser.parse_args()

    chunk_dirs = find_chunk_dirs(args.chunk_dirs)
    if not chunk_dirs:
        logger.error("No valid chunk directories found")
        sys.exit(1)

    logger.info("Found %d chunk directories:", len(chunk_dirs))
    for d in chunk_dirs:
        logger.info("  %s", d.name)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("outputs/runs") / f"{timestamp}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", output_dir)

    # Copy config from first chunk
    first_config = chunk_dirs[0] / "config_used.yaml"
    if first_config.exists():
        shutil.copy2(first_config, output_dir / "config_used.yaml")

    # Merge overlay videos
    merge_overlay_videos(chunk_dirs, output_dir)

    # Merge contact CSVs
    contacts_dir = output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    merge_csvs(chunk_dirs, "contacts_per_frame.csv", contacts_dir)
    merge_csvs(chunk_dirs, "contact_bouts.csv", contacts_dir)

    # Merge session summaries
    merge_session_summaries(chunk_dirs, output_dir)

    # Merge logs
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "run.log").open("w", encoding="utf-8") as out_log:
        for cdir in chunk_dirs:
            log_path = cdir / "logs" / "run.log"
            if log_path.exists():
                out_log.write(f"\n{'='*60}\n")
                out_log.write(f"=== Chunk: {cdir.name} ===\n")
                out_log.write(f"{'='*60}\n")
                out_log.write(log_path.read_text(encoding="utf-8"))

    logger.info("Merge complete → %s", output_dir)

    # Print final video location for easy copy-paste
    merged_videos = list((output_dir / "overlays").glob("*_merged.*")) if (output_dir / "overlays").exists() else []
    if merged_videos:
        print(f"\n{'='*60}")
        print(f"  MERGED VIDEO: {merged_videos[0]}")
        print(f"  OUTPUT DIR:   {output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
