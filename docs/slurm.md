# Parallel Execution on Bunya HPC

## Quick Start

```bash
# 1. Connect to Bunya
ssh s4948012@bunya.rcc.uq.edu.au

# 2. Request interactive session with 4 GPUs
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=06:00:00
srun --pty bash

# 3. Setup environment
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 4. Run (one command does everything)
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

## `scripts/run_parallel.sh`

Automates the full pipeline: analyze video, split into chunks, run in parallel on multiple GPUs, merge results, and print download commands.

### Arguments

| Arg | Description | Default |
|-----|-------------|---------|
| `$1` | Video path | (required) |
| `$2` | Number of chunks/GPUs | `4` |
| `$3` | Config file | `configs/hpc_reference.yaml` |
| `$4` | Extra overrides (space-separated) | none |

### Examples

```bash
# Default: 4 chunks, reference pipeline
bash scripts/run_parallel.sh data/raw/original_120s.avi

# 2 chunks (if only 2 GPUs available)
bash scripts/run_parallel.sh data/raw/original_120s.avi 2

# Custom config
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_reference.yaml

# With extra overrides
bash scripts/run_parallel.sh data/raw/original_120s.avi 4 configs/hpc_reference.yaml "detection.confidence=0.3"
```

### What It Does (step by step)

1. **Creates batch directory** with timestamp: `outputs/runs/2026-02-27_141500_reference_batch/`
2. **Reads video metadata** (total frames, FPS, resolution) using OpenCV
3. **Divides frames** into N equal chunks (e.g., 30000 frames / 4 = 7500 per chunk)
4. **Launches N processes** in parallel, each with `CUDA_VISIBLE_DEVICES=<gpu_id>`
5. **Waits** for all chunks to finish, reports timing per chunk
6. **Finds run directories** by parsing the logs (no glob collisions with previous runs)
7. **Merges results** via `scripts/merge_chunks.py`:
   - Concatenates overlay videos (ffmpeg or OpenCV fallback)
   - Combines contact CSVs and session summaries
   - Merges logs
8. **Compresses contacts** into `contacts_<batch_id>.tar.gz`
9. **Prints download commands** (scp) for the merged video and contacts

### Output Structure

```
outputs/runs/2026-02-27_141500_reference_batch/
├── chunks/
│   ├── *_reference_chunk0/
│   │   ├── config_used.yaml
│   │   ├── overlays/
│   │   │   └── reference_sam2.1_hiera_large_2026-02-27.avi
│   │   ├── contacts/
│   │   │   ├── contacts_per_frame.csv
│   │   │   ├── contact_bouts.csv
│   │   │   └── session_summary.json
│   │   └── logs/
│   │       └── run.log
│   ├── *_reference_chunk1/
│   ├── *_reference_chunk2/
│   └── *_reference_chunk3/
├── overlays/
│   └── reference_sam2.1_hiera_large_2026-02-27_merged.avi   <- FINAL VIDEO
├── contacts/
│   ├── contacts_per_frame.csv      <- all frames combined
│   ├── contact_bouts.csv           <- all bouts combined
│   └── session_summary.json        <- aggregated totals
├── contacts_2026-02-27_141500_reference_batch.tar.gz        <- compressed contacts
├── config_used.yaml
└── logs/
    ├── chunk_0.log
    ├── chunk_1.log
    ├── chunk_2.log
    ├── chunk_3.log
    └── run.log                     <- concatenated logs from all chunks
```

### Sample Output

```
Analyzing video: data/raw/original_120s.avi
  Resolution: 640x480
  FPS: 25.0
  Total frames: 30000
  Duration: 1200.0s (20.0 min)

=== Parallel Execution Plan ===
  Batch ID:   2026-02-27_141500_reference_batch
  Batch dir:  outputs/runs/2026-02-27_141500_reference_batch
  Config:     configs/hpc_reference.yaml
  Video:      data/raw/original_120s.avi
  Frames:     30000
  Chunks:     4
  Per chunk:  ~7500 frames

  Chunk 0: frames 0 -> 7500 (GPU 0)
  Chunk 1: frames 7500 -> 15000 (GPU 1)
  Chunk 2: frames 15000 -> 22500 (GPU 2)
  Chunk 3: frames 22500 -> 30000 (GPU 3)

All 4 chunks launched. Waiting...
  Monitor: tail -f outputs/runs/.../logs/chunk_0.log

  Chunk 0: DONE (12m 34s)
  Chunk 1: DONE (11m 58s)
  Chunk 2: DONE (12m 15s)
  Chunk 3: DONE (11m 42s)

  Total time: 12m 34s

=== All chunks completed. Merging... ===
  ...

============================================================
  BATCH COMPLETE: 2026-02-27_141500_reference_batch
  TOTAL TIME:     12m 34s
  BATCH DIR:      /home/s4948012/Balbi/.../2026-02-27_141500_reference_batch
  MERGED VIDEO:   /home/s4948012/Balbi/.../reference_sam2.1_hiera_large_2026-02-27_merged.avi
  CONTACTS:       /home/s4948012/Balbi/.../contacts_2026-02-27_141500_reference_batch.tar.gz
============================================================

# Download results (PowerShell):
scp s4948012@bunya.rcc.uq.edu.au:/home/.../reference_sam2.1_hiera_large_2026-02-27_merged.avi \
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
scp s4948012@bunya.rcc.uq.edu.au:/home/.../contacts_2026-02-27_141500_reference_batch.tar.gz \
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

## Alternative: sbatch (non-interactive)

If you prefer batch mode from the login node (without salloc):

```bash
# From login node (NOT inside salloc):
cd ~/Balbi/yolo-sam2-lab-tracking

PIPELINE=src.pipelines.reference.run \
CONFIG=configs/hpc_reference.yaml \
TOTAL_FRAMES=30000 \
OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" \
  sbatch --array=0-3 slurm/run_chunks.sbatch
```

Monitor: `squeue -u $USER`

Note: sbatch array jobs require manual merge after completion:
```bash
python scripts/merge_chunks.py outputs/runs/*reference_chunk*/
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch a chunk's progress in real-time
tail -f outputs/runs/<batch_id>/logs/chunk_0.log

# Check GPU utilization
nvidia-smi
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `PD` in squeue | Waiting for GPU | Wait, or reduce `--gres=gpu:N` |
| `Exit 1` on a chunk | Python error | Check `logs/chunk_N.log` |
| `CUDA out of memory` | Too many chunks sharing 1 GPU | Use 1 chunk per GPU (default) |
| `git pull` conflicts | Local changes on Bunya | `git stash && git pull && git stash pop` |
| Video not found in merge | Old `overlay.avi` naming | Already fixed - finds any `*.avi`/`*.mp4` |
