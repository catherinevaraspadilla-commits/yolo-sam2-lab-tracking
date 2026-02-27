# Bunya HPC Guide

Complete guide for running pipelines on UQ's Bunya HPC cluster.

## Table of Contents

- [Quick Start (automated)](#quick-start-automated)
- [Quick Start (manual)](#quick-start-manual)
- [Single Run (1 GPU)](#single-run-1-gpu)
- [Parallel Batch (sbatch)](#parallel-batch-sbatch)
- [run_parallel.sh Reference](#run_parallelsh-reference)
- [Output Structure](#output-structure)
- [First-Time Setup](#first-time-setup)
- [Transfer Files](#transfer-files)
- [Monitoring](#monitoring)
- [Other Run Options](#other-run-options)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (automated)

The fastest way to process a full video. One script does everything.

```bash
# 1. Connect to Bunya
ssh s4948012@bunya.rcc.uq.edu.au

# 2. Request 4 GPUs
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=06:00:00
srun --pty bash

# 3. Setup environment
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 4. Run (one command does everything: split, process, merge, print download)
bash scripts/run_parallel.sh data/raw/original_120s.avi
```

---

## Quick Start (manual)

If you want to control each chunk manually within a salloc session.

```bash
# 1. Connect and get 4 GPUs
ssh s4948012@bunya.rcc.uq.edu.au
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:4 --cpus-per-task=16 --mem=64G --time=06:00:00
srun --pty bash

# 2. Setup
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate
mkdir -p outputs/slurm

# 3. Launch 4 chunks in parallel, each on its own GPU
CUDA_VISIBLE_DEVICES=0 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 0 --end-frame 7500 --chunk-id 0 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 7500 --end-frame 15000 --chunk-id 1 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 15000 --end-frame 22500 --chunk-id 2 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  --start-frame 22500 --end-frame 30000 --chunk-id 3 \
  video_path=data/raw/original_120s.avi contacts.enabled=true > outputs/slurm/chunk_3.log 2>&1 &

# 4. Wait for all to finish
wait
echo "All chunks done!"

# 5. Check which run directories were created
ls -d outputs/runs/*reference_chunk*

# 6. Merge (replace paths with actual directories from step 5)
python scripts/merge_chunks.py outputs/runs/*_reference_chunk*/
```

**Key**: `CUDA_VISIBLE_DEVICES=N` assigns each process to a different GPU. The `&` runs them in background. `wait` blocks until all finish.

---

## Single Run (1 GPU)

For quick tests or short videos. No chunking needed.

```bash
# 1. Connect and get 1 GPU
ssh s4948012@bunya.rcc.uq.edu.au
salloc --partition=gpu_cuda --qos=gpu --gres=gpu:1 --time=06:00:00 --mem=64G
srun --pty bash

# 2. Setup
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 3a. Quick test (2 min clip)
python scripts/trim_video.py data/raw/original.mp4 --duration 120
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  video_path=data/raw/original_120s.avi contacts.enabled=true

# 3b. Or run sam2_yolo pipeline instead
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
  video_path=data/raw/original_120s.avi \
  models.sam2_checkpoint=models/sam2/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt \
  models.sam2_config=configs/sam2.1/sam2.1_hiera_t.yaml \
  contacts.enabled=true
```

---

## Parallel Batch (sbatch)

Non-interactive mode. Submit from the login node and let Slurm schedule it.
No need to stay connected.

```bash
# 1. Connect (login node only, no salloc needed)
ssh s4948012@bunya.rcc.uq.edu.au

# 2. Setup
cd ~/Balbi/yolo-sam2-lab-tracking
git pull
module load python/3.10.4-gcccore-11.3.0
source .venv/bin/activate

# 3. Trim video if needed (runs on login node, no GPU)
python scripts/trim_video.py data/raw/original.mp4 --duration 120

# 4. Submit 4 parallel chunks (Slurm assigns GPUs automatically)
PIPELINE=src.pipelines.reference.run \
CONFIG=configs/hpc_reference.yaml \
TOTAL_FRAMES=30000 \
OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" \
  sbatch --array=0-3 slurm/run_chunks.sbatch

# 5. Monitor (can disconnect and check later)
squeue -u $USER
#  JOBID  PARTITION  NAME        ARRAY_TASK  STATE    NODE
#  12345  gpu_cuda   sam2-chunk  0           RUNNING  gpu-node-1
#  12345  gpu_cuda   sam2-chunk  1           RUNNING  gpu-node-2
#  12345  gpu_cuda   sam2-chunk  2           PENDING  (waiting for GPU)
#  12345  gpu_cuda   sam2-chunk  3           PENDING  (waiting for GPU)

# View chunk progress:
cat outputs/slurm/chunk_0_*.out

# 6. When all chunks finish, merge
python scripts/merge_chunks.py outputs/runs/*_reference_chunk*/

# 7. Download results (see Transfer Files section)
```

**Difference vs salloc**: sbatch runs in the background. You can disconnect from Bunya and come back later. Slurm handles GPU scheduling. But you need to merge manually after all chunks finish.

---

## `run_parallel.sh` Reference

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
3. **Divides frames** into N equal chunks (e.g., 30000 / 4 = 7500 per chunk)
4. **Launches N processes** in parallel, each with `CUDA_VISIBLE_DEVICES=<gpu_id>`
5. **Waits** for all chunks, reports timing per chunk
6. **Finds run directories** by parsing logs (no collisions with previous runs)
7. **Merges results** via `scripts/merge_chunks.py` (video + contacts + logs)
8. **Compresses contacts** into `contacts_<batch_id>.tar.gz`
9. **Prints download commands** (scp) for the merged video and contacts

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

---

## Output Structure

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
├── contacts_2026-02-27_141500_reference_batch.tar.gz        <- compressed for download
├── config_used.yaml
└── logs/
    ├── chunk_0.log
    ├── chunk_1.log
    ├── chunk_2.log
    ├── chunk_3.log
    └── run.log                     <- concatenated logs from all chunks
```

---

## First-Time Setup

Only needed once when setting up the project on Bunya.

### Python + venv

```bash
cd ~/Balbi/yolo-sam2-lab-tracking
module load python/3.10.4-gcccore-11.3.0
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### PyTorch (CUDA 11.8)

```bash
export TMPDIR=$HOME/tmp && mkdir -p $TMPDIR
python -m pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# Validate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### SAM2

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git models/sam2/segment-anything-2
python -m pip install ./models/sam2/segment-anything-2

# Install other dependencies (exclude SAM2 editable install)
grep -v "^-e ./models/sam2/segment-anything-2" requirements.txt > requirements.fixed.txt
python -m pip install -r requirements.fixed.txt
pip install ultralytics opencv-python roboflow

# Download SAM2 checkpoints
cd models/sam2/segment-anything-2/checkpoints && bash download_ckpts.sh
cd ~/Balbi/yolo-sam2-lab-tracking
```

---

## Transfer Files

### Upload video to Bunya (PowerShell)

```powershell
scp C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\data\raw\original.mp4 `
  s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/data/raw/
```

### Check available runs (Bunya)

```bash
ls -lt ~/Balbi/yolo-sam2-lab-tracking/outputs/runs/
```

### Download overlay video (PowerShell)

```powershell
# The scp command is printed automatically by run_parallel.sh
# Example:
scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-27_141500_reference_batch/overlays/reference_sam2.1_hiera_large_2026-02-27_merged.avi `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

### Download contacts (PowerShell)

```powershell
scp s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-27_141500_reference_batch/contacts_2026-02-27_141500_reference_batch.tar.gz `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

### Download entire batch (PowerShell)

```powershell
scp -r s4948012@bunya.rcc.uq.edu.au:~/Balbi/yolo-sam2-lab-tracking/outputs/runs/2026-02-27_141500_reference_batch `
  "C:\Users\CatherineVaras\Downloads\yolo-sam2-lab-tracking\outputs\"
```

---

## Monitoring

```bash
# Check job status (sbatch mode)
squeue -u $USER

# Watch a chunk's progress in real-time (salloc mode)
tail -f outputs/runs/<batch_id>/logs/chunk_0.log

# Check GPU utilization
nvidia-smi

# Check disk usage
du -sh outputs/runs/
```

---

## Other Run Options

```bash
# 5 min clip test
python scripts/trim_video.py data/raw/original.mp4 --duration 300
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  video_path=data/raw/original_300s.avi contacts.enabled=true

# Full video (no trimming, single GPU)
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
  video_path=data/raw/original.mp4 contacts.enabled=true

# sam2_yolo pipeline (uses BoT-SORT tracking instead of IdentityMatcher)
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
  video_path=data/raw/original_120s.avi contacts.enabled=true

# Contacts only analysis (fast, no SAM2)
python scripts/analyze_contacts.py --config configs/hpc_full.yaml contacts.enabled=true
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `PD` in squeue | Waiting for GPU | Wait, or reduce `--gres=gpu:N` |
| `Exit 1` on a chunk | Python error | Check `logs/chunk_N.log` |
| `CUDA out of memory` | Too many processes per GPU | Use 1 chunk per GPU (default) |
| `git pull` conflicts | Local changes on Bunya | `git stash && git pull && git stash pop` |
| `salloc` hangs | No GPUs available | Try fewer GPUs or off-peak hours |
| Video not found in merge | Old naming | Fixed: merge finds any `*.avi`/`*.mp4` |
| `sbatch: error: Unable to open file` | Wrong directory | `cd ~/Balbi/yolo-sam2-lab-tracking` first |
| `AttributeError: NoneType` | Old code on Bunya | `git pull` to get latest fixes |
